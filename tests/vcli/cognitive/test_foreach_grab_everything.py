# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""S4-3 — "grab everything, one by one" END-TO-END (Stage 4 acceptance).

This is the core Stage-4 acceptance: prove a FOREACH long-chain runs end-to-end
on the tabletop playground scenario, through the REAL kernel pieces:

  NL -> GoalDecomposer (mock backend, ARM-ONLY validation)
     -> GoalExecutor (real topo-sort + foreach expansion)
     -> GoalVerifier (real AST sandbox)
     over the MERGED PlaygroundWorld sim-oracle verify namespace
     against a DETERMINISTIC stub arm that advances state so each per-step
     verify (holding_object() / object_on(item, 'tray')) flips False->True as the
     chain progresses.

The decomposed chain is:

    scan -> detect_all -> foreach(obj in detected objects): pick(obj) -> place(obj)

ASSERTIONS (the Stage-4 acceptance bar):
  - decompose is ARM-ONLY: no go2/base strategies, no hallucinated skills.
  - the dynamic tree's leaf (pick/place) count EQUALS the detected object count.
  - every per-step verify passes (deterministic predicate, no visual override).
  - the run reaches verified-done (trace.success).
  - the observation snapshot is JSON-safe and reflects the EXPANDED steps.

Hermetic: mock backend, no live LLM, no MuJoCo. The stub arm/gripper mirror the
sim-oracle's grounding contract (get_object_positions / get_joint_positions / fk
/ gripper.is_holding) exactly as the real arm exposes them, so the SAME
playground predicates that run against MuJoCo run here unchanged.
"""
from __future__ import annotations

import json
import os
from typing import Any

import pytest

from vector_os_nano.playground.catalog import TABLETOP_TRAY
from vector_os_nano.playground.world import PlaygroundWorld
from vector_os_nano.vcli.cognitive.blackboard import Blackboard
from vector_os_nano.vcli.cognitive.goal_decomposer import GoalDecomposer
from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.observation import run_snapshot
from vector_os_nano.vcli.cognitive.strategy_selector import StrategyResult


# ---------------------------------------------------------------------------
# Deterministic stub arm + gripper + agent (the sim oracle's ground truth).
# ---------------------------------------------------------------------------
#
# The playground predicates read DETERMINISTIC ground truth off the arm via the
# same accessors the real SO-101 arm exposes:
#   - arm.get_object_positions() -> {name: [x, y, z]}
#   - arm.get_joint_positions()  -> [j0..j4]
#   - arm.fk(joints)             -> (ee_xyz, rot)
#   - agent._gripper.is_holding() -> bool
# This stub advances that ground truth as pick/place primitives run, so the same
# predicates flip False->True step by step. No MuJoCo, fully deterministic.

# Object centres resting on the table top (z 0.02); lifted is well clear of it.
# The tray (tabletop world manifest) is (0.20, 0.0, 0.50, 0.25) on the table; a
# placed object lands at its centre, resting.
_TABLE_Z = 0.04
_LIFT_Z = 0.20
_TRAY_CENTRE = (0.35, 0.12)


class _StubGripper:
    def __init__(self) -> None:
        self._holding = False

    def is_holding(self) -> bool:
        return self._holding


class _StubArm:
    """A deterministic stand-in for the connected SO-101 arm.

    Holds ground-truth object positions; pick lifts the held object and parks the
    EE on top of it; place drops it (resting) into the tray. FK simply returns the
    current EE position the stub tracks, so holding_object()'s near-EE check is
    deterministic.
    """

    def __init__(self, object_names: tuple[str, ...]) -> None:
        # All objects start resting on the table, spread along x so they are
        # distinct ground-truth points (exact spread is irrelevant to the test).
        self._objects: dict[str, list[float]] = {
            name: [0.10 + 0.05 * i, -0.10, _TABLE_Z]
            for i, name in enumerate(object_names)
        }
        # EE parked away from every object initially -> holding_object() False.
        self._ee: list[float] = [0.0, 0.30, 0.30]

    # --- grounding-contract reads (what the predicates call) ---

    def get_object_positions(self) -> dict[str, list[float]]:
        return {k: list(v) for k, v in self._objects.items()}

    def get_joint_positions(self) -> list[float]:
        # Joint values are irrelevant to holding_object()/object_on(); a
        # fixed 5-vector keeps fk() well-formed.
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    def fk(self, _joints: Any) -> tuple[list[float], None]:
        return list(self._ee), None

    # --- mutations driven by the pick/place primitives ---

    def lift(self, name: str) -> None:
        pos = self._objects[name]
        pos[2] = _LIFT_Z
        # Park the EE on the lifted object so holding_object() sees it at-gripper.
        self._ee = [pos[0], pos[1], pos[2]]

    def drop_in_tray(self, name: str) -> None:
        self._objects[name] = [_TRAY_CENTRE[0], _TRAY_CENTRE[1], _TABLE_Z]
        # Retract the EE away so a stale near-EE reading can't mask a place.
        self._ee = [0.0, 0.30, 0.30]


class _StubAgent:
    def __init__(self, object_names: tuple[str, ...]) -> None:
        self._arm = _StubArm(object_names)
        self._gripper = _StubGripper()
        from tests.harness.fake_gt import bind_fake_sim

        bind_fake_sim(self, "so101")  # the live sim session's GT oracles


# ---------------------------------------------------------------------------
# A selector that routes each (concrete) sub_goal's explicit strategy to a
# primitive that advances the stub arm. The foreach body's pick/place strategies
# resolve here after expansion.
# ---------------------------------------------------------------------------


def _make_primitives(agent: _StubAgent) -> dict[str, Any]:
    arm: _StubArm = agent._arm
    gripper: _StubGripper = agent._gripper

    def scan(**_: Any) -> dict[str, Any]:
        # Perception sweep — no state change; verify uses describe_scene().
        return {"scanned": True}

    # The detect PRODUCING step is the REAL playground primitive — it runs the
    # deterministic sim-oracle detection (the SAME ground truth the
    # detect_objects() verify predicate reports) and writes {"objects": [...]}.
    # No fabricated detect primitive: this is the real perception output the
    # foreach iterates.
    detect_producer = PlaygroundWorld(TABLETOP_TRAY).build_step_primitives(agent)[
        PlaygroundWorld.DETECT_STRATEGY
    ]

    def pick(object_label: str | None = None, **_: Any) -> dict[str, Any]:
        if object_label in arm.get_object_positions():
            arm.lift(object_label)
            gripper._holding = True
        return {"picked": object_label}

    def place(object_label: str | None = None, **_: Any) -> dict[str, Any]:
        if object_label in arm.get_object_positions():
            arm.drop_in_tray(object_label)
            gripper._holding = False
        return {"placed": object_label}

    # Keyed by the SAME *_skill strategy names the decomposer validates + emits,
    # so the selector routes each concrete step's strategy straight to its
    # primitive (the foreach body's pick_skill/place_skill resolve here too).
    return {
        "scan_skill": scan,
        PlaygroundWorld.DETECT_STRATEGY: detect_producer,
        "pick_skill": pick,
        "place_skill": place,
    }


class _DirectSelector:
    """Route the sub_goal's explicit strategy straight to a primitive call."""

    def select(self, sub_goal: Any) -> StrategyResult:
        return StrategyResult(
            "primitive", sub_goal.strategy, dict(sub_goal.strategy_params)
        )


# ---------------------------------------------------------------------------
# Mock LLM backend — returns the fixed "grab everything" plan as JSON.
# ---------------------------------------------------------------------------


class _MockBackend:
    def __init__(self, response: str) -> None:
        self._response = response

    def call(self, messages, tools, system, max_tokens, on_text=None):  # noqa: ANN001
        resp = self._response

        class _R:
            text = resp

        return _R()


# The arm-only world vocab the decomposer validates against. Single-sourced here
# as the arm skill set (no base): a go2/base strategy or a hallucinated skill must
# be rejected. The body's pick/place are real arm skills; scan/detect are too.
_ARM_STRATEGIES = frozenset(
    {"scan_skill", PlaygroundWorld.DETECT_STRATEGY, "pick_skill", "place_skill"}
)
_ARM_VERIFY_FNS = frozenset({"holding_object", "object_on", "released", "arm_at_home"})


def _grab_everything_plan() -> dict[str, Any]:
    """scan -> detect_all -> foreach(obj): pick(obj) -> place(obj)."""
    return {
        "goal": "grab everything",
        "sub_goals": [
            {
                "name": "scan",
                "description": "look over the whole tabletop",
                "verify": "True",
                "strategy": "scan_skill",
                "depends_on": [],
                "strategy_params": {},
            },
            {
                "name": "detect_all",
                "description": "detect every object on the table",
                "verify": "True",
                "strategy": PlaygroundWorld.DETECT_STRATEGY,
                "depends_on": ["scan"],
                "strategy_params": {},
            },
            {
                "name": "grab_each",
                "description": "pick up and place each detected object, one by one",
                "verify": "True",
                "strategy": "",
                "depends_on": ["detect_all"],
                "strategy_params": {},
                "foreach": {
                    "source_step": "detect_all",
                    "source_path": "objects",
                    "var": "obj",
                    "body": [
                        {
                            "name": "pick_obj",
                            "description": "pick up the current object",
                            "verify": "holding_object()",
                            "strategy": "pick_skill",
                            "depends_on": [],
                            "strategy_params": {"object_label": "${obj.name}"},
                        },
                        {
                            "name": "place_obj",
                            "description": "place the current object in the tray",
                            "verify": "object_on('${obj.name}', 'tray')",
                            "strategy": "place_skill",
                            "depends_on": ["pick_obj"],
                            "strategy_params": {"object_label": "${obj.name}"},
                        },
                    ],
                },
            },
        ],
        "context_snapshot": "Tabletop with six objects; SO-101 arm at home.",
    }


def _decompose_grab_everything(plan: dict[str, Any]):
    """Run the REAL decomposer (arm-only vocab) over the mock plan."""
    decomposer = GoalDecomposer(
        _MockBackend(json.dumps(plan)),
        strategies=_ARM_STRATEGIES,
        verify_functions=_ARM_VERIFY_FNS,
        fallback_verify="True",
        has_base=False,  # arm-only: no walk_forward/turn/scan_360 base primitives
    )
    return decomposer.decompose("grab everything", "tabletop scene")


def _executor(agent: _StubAgent) -> GoalExecutor:
    """Wire a GoalExecutor over the MERGED PlaygroundWorld sim-oracle namespace."""
    world = PlaygroundWorld(TABLETOP_TRAY)
    namespace = world.build_verify_namespace(agent)
    verifier = GoalVerifier(namespace)
    executor = GoalExecutor(
        strategy_selector=_DirectSelector(),
        verifier=verifier,
        primitives=_make_primitives(agent),
    )
    executor.blackboard = Blackboard()  # the harness normally attaches this
    return executor


# ---------------------------------------------------------------------------
# Decompose: ARM-ONLY (no go2/base, no hallucinated skills).
# ---------------------------------------------------------------------------


def test_grab_everything_decomposes_arm_only() -> None:
    tree = _decompose_grab_everything(_grab_everything_plan())

    assert [sg.name for sg in tree.sub_goals] == ["scan", "detect_all", "grab_each"]
    # A clean arm-only plan generates no validator complaints.
    assert tree.validation_notes == ()

    # Every top-level strategy is an arm skill — no base primitive leaked in.
    top_strategies = {sg.strategy for sg in tree.sub_goals if sg.strategy}
    assert top_strategies <= _ARM_STRATEGIES
    for banned in ("walk_forward", "turn", "scan_360", "explore_skill", "navigate_skill"):
        assert banned not in top_strategies

    # The foreach body is arm-only too.
    loop = tree.sub_goals[2]
    assert loop.foreach is not None
    body_strategies = [t.strategy for t in loop.foreach.body]
    assert body_strategies == ["pick_skill", "place_skill"]
    # Per-item binding preserved AS DATA — never evaluated/formatted at decompose.
    assert loop.foreach.body[0].strategy_params["object_label"] == "${obj.name}"


def test_grab_everything_rejects_go2_and_hallucinated_skills() -> None:
    plan = _grab_everything_plan()
    # Hallucinate a go2 base strategy at the top level and a bogus skill in body.
    plan["sub_goals"][0]["strategy"] = "walk_forward"  # base/go2 — not in arm world
    plan["sub_goals"][2]["foreach"]["body"][0]["strategy"] = "teleport_skill"  # bogus

    tree = _decompose_grab_everything(plan)

    # The go2 base strategy is cleared (not silently routed).
    assert tree.sub_goals[0].strategy == ""
    # The hallucinated body skill is cleared.
    assert tree.sub_goals[2].foreach.body[0].strategy == ""
    # Fail-loud feedback names both offenders + surfaces the valid arm set.
    notes = "\n".join(tree.validation_notes)
    assert "walk_forward" in notes and "not valid" in notes
    assert "teleport_skill" in notes
    assert "pick_skill" in notes  # valid arm set surfaced for the replan


# ---------------------------------------------------------------------------
# End-to-end: real executor + verifier over the merged sim-oracle namespace.
# ---------------------------------------------------------------------------


def test_grab_everything_runs_end_to_end_verified() -> None:
    object_names = TABLETOP_TRAY.object_names
    n = len(object_names)
    assert n == 6  # the bundled tabletop ships six graspables

    agent = _StubAgent(object_names)
    tree = _decompose_grab_everything(_grab_everything_plan())
    executor = _executor(agent)

    trace = executor.execute(tree)

    # The whole chain reached verified-done.
    assert trace.success is True

    # The detect step is a REAL producing step: its captured result_data carries
    # the deterministic sim-oracle objects list the foreach iterates (not a
    # fabricated primitive). Read it straight off the trace.
    detect_step = next(s for s in trace.steps if s.sub_goal_name == "detect_all")
    detected = detect_step.result_data["output"]["objects"]
    detected_count = len(detected)
    assert detected_count == n  # the producer detected every scene object
    assert {o["name"] for o in detected} == set(object_names)

    # Leaf count: scan + detect_all + (pick + place) per object.
    names = [s.sub_goal_name for s in trace.steps]
    assert names[0] == "scan"
    assert names[1] == "detect_all"
    foreach_steps = names[2:]
    # The dynamic tree's pick/place leaf count EQUALS the REAL detected object
    # count from the producing step (not a hand-set N).
    pick_steps = [n_ for n_ in foreach_steps if n_.endswith(".pick_obj")]
    place_steps = [n_ for n_ in foreach_steps if n_.endswith(".place_obj")]
    assert len(pick_steps) == detected_count
    assert len(place_steps) == detected_count
    assert len(foreach_steps) == 2 * detected_count  # one pick + one place per object

    # Expansion is ordered: pick_obj then place_obj, per item index.
    expected = []
    for i in range(n):
        expected.append(f"grab_each[{i}].pick_obj")
        expected.append(f"grab_each[{i}].place_obj")
    assert foreach_steps == expected

    # Every per-step verify passed DETERMINISTICALLY (no visual override).
    for s in trace.steps:
        assert s.success is True, f"step {s.sub_goal_name} failed: {s.error}"
        assert s.verify_result is True
        assert s.visual_override is False


def test_grab_everything_oracle_state_advances_per_item() -> None:
    # The sim-oracle predicates must FLIP per item as the chain progresses: at the
    # end every object is resting in the tray.
    object_names = TABLETOP_TRAY.object_names
    n = len(object_names)
    agent = _StubAgent(object_names)

    # Before the run nothing is placed in the tray and nothing is held.
    world = PlaygroundWorld(TABLETOP_TRAY)
    ns = world.build_verify_namespace(agent)
    assert not any(ns["object_on"](o, "tray") for o in object_names)
    assert ns["holding_object"]() is False

    tree = _decompose_grab_everything(_grab_everything_plan())
    executor = _executor(agent)
    trace = executor.execute(tree)

    assert trace.success is True
    # All n objects ended resting inside the tray drop-zone.
    assert sum(ns["object_on"](o, "tray") for o in object_names) == n
    # The gripper released the last object after placing it.
    assert ns["holding_object"]() is False


# ---------------------------------------------------------------------------
# The observation snapshot is JSON-safe and reflects the EXPANDED steps.
# ---------------------------------------------------------------------------


def test_grab_everything_snapshot_json_safe_and_reflects_expansion() -> None:
    object_names = TABLETOP_TRAY.object_names
    n = len(object_names)
    agent = _StubAgent(object_names)
    tree = _decompose_grab_everything(_grab_everything_plan())
    executor = _executor(agent)

    trace = executor.execute(tree)
    snapshot = run_snapshot(trace)

    # Round-trips through json.dumps (deterministic, no exotic objects leaked).
    encoded = json.dumps(snapshot)
    assert isinstance(encoded, str)

    # The snapshot reflects the EXPANDED dynamic steps (not just the 3-node tree).
    step_names = [sv["sub_goal_name"] for sv in snapshot["steps"]]
    assert step_names[0] == "scan"
    assert step_names[1] == "detect_all"
    assert sum(1 for s in step_names if s.endswith(".pick_obj")) == n
    assert sum(1 for s in step_names if s.endswith(".place_obj")) == n
    assert len(snapshot["steps"]) == 2 + 2 * n

    # Every exported step shows success + verify_result (the verified loop).
    assert all(sv["success"] and sv["verify_result"] for sv in snapshot["steps"])
    assert snapshot["success"] is True

    # The static goal tree still carries exactly the 3 authored nodes; the loop
    # node's foreach expansion lives in the executed steps, not the tree view.
    assert [sg["name"] for sg in snapshot["goal_tree"]["sub_goals"]] == [
        "scan",
        "detect_all",
        "grab_each",
    ]


# ---------------------------------------------------------------------------
# F-2 — the foreach reads a REAL detect-producing step (not a fabricated one).
#
# The detect step's strategy is the playground's own producing primitive
# (DETECT_STRATEGY). It runs the deterministic sim-oracle detection and writes
# {"objects": [...]} as its result_data; the executor captures that to the
# Blackboard so the foreach's source_step.objects resolves the REAL list. These
# pin that the produced list is exactly the scene's objects and that it drives
# the loop's leaf count.
# ---------------------------------------------------------------------------


def test_detect_producer_output_matches_oracle_predicate() -> None:
    object_names = TABLETOP_TRAY.object_names
    agent = _StubAgent(object_names)
    world = PlaygroundWorld(TABLETOP_TRAY)

    # The producing primitive lists the scene's objects; no verify oracle is a
    # listing (WP10: a GT object count is vacuous evidence).
    producer = world.build_step_primitives(agent)[PlaygroundWorld.DETECT_STRATEGY]
    assert "detect_objects" not in world.build_verify_namespace(agent)

    produced = producer()
    assert isinstance(produced, dict)
    assert produced["count"] == len(object_names)
    assert sorted(o["name"] for o in produced["objects"]) == sorted(object_names)


def test_foreach_iterates_real_produced_list_via_blackboard() -> None:
    # End-to-end through the REAL producer: the captured objects list on the
    # Blackboard is what the foreach iterates, so the leaf count equals the
    # producer's detected count — no fabricated detect primitive anywhere.
    object_names = TABLETOP_TRAY.object_names
    agent = _StubAgent(object_names)
    tree = _decompose_grab_everything(_grab_everything_plan())
    executor = _executor(agent)

    trace = executor.execute(tree)
    assert trace.success is True

    # The producing step's captured output is on the run blackboard under its name.
    captured = executor.blackboard.get("detect_all")
    assert captured is not None
    produced_objects = captured["output"]["objects"]
    assert {o["name"] for o in produced_objects} == set(object_names)

    # Each produced object yielded exactly one pick + one place leaf.
    pick_steps = [s for s in trace.steps if s.sub_goal_name.endswith(".pick_obj")]
    assert len(pick_steps) == len(produced_objects)


# ---------------------------------------------------------------------------
# Optional live-LLM smoke (deselected from the canonical gate).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("VECTOR_LIVE_LLM") != "1",
    reason="live-LLM smoke: set VECTOR_LIVE_LLM=1 to run (deselected by default)",
)
def test_grab_everything_live_llm_smoke() -> None:  # pragma: no cover - opt-in
    """Smoke: a live backend decomposes 'grab everything' into a foreach plan.

    Opt-in only. Asserts on STRUCTURE (a foreach loop emerges, body strategies are
    arm-only), never on exact wording. Skipped by default so the canonical suite
    stays hermetic and free. The provider/model default to OpenRouter +
    gemini-2.5-flash; the key comes from VECTOR_LLM_API_KEY (never hardcoded).
    """
    from vector_os_nano.vcli.backends import create_backend

    api_key = os.environ.get("VECTOR_LLM_API_KEY")
    if not api_key:
        pytest.skip("live-LLM smoke needs VECTOR_LLM_API_KEY")
    backend = create_backend(
        provider=os.environ.get("VECTOR_LLM_PROVIDER", "openrouter"),
        api_key=api_key,
        model=os.environ.get("VECTOR_LLM_MODEL", "google/gemini-2.5-flash"),
    )
    decomposer = GoalDecomposer(
        backend,
        strategies=_ARM_STRATEGIES,
        verify_functions=_ARM_VERIFY_FNS,
        fallback_verify="True",
        has_base=False,
    )
    tree = decomposer.decompose(
        "grab everything on the table, one by one", "tabletop scene with six objects"
    )
    # A loop construct emerged somewhere in the plan.
    assert any(sg.foreach is not None for sg in tree.sub_goals)
    # No base/go2 strategy leaked into an arm-only world.
    all_strategies = {sg.strategy for sg in tree.sub_goals if sg.strategy}
    for sg in tree.sub_goals:
        if sg.foreach is not None:
            all_strategies |= {t.strategy for t in sg.foreach.body if t.strategy}
    assert all_strategies <= _ARM_STRATEGIES
