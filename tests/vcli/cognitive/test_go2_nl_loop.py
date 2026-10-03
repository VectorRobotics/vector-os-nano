# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E-2 — NL decompose + visible verified loop for the GO2 scenario (Stage E).

The Go2 counterpart to the arm's S4-3 acceptance. It proves the SAME closed loop
(NL -> decompose -> execute -> verify -> render) runs against the SECOND
embodiment (the Go2 quadruped, ``has_base``) with NO kernel changes, end-to-end
through the REAL kernel pieces:

  NL -> GoalDecomposer (mock backend, GO2-ONLY validation)
     -> GoalExecutor (real topo-sort)
     -> GoalVerifier (real AST sandbox)
     over the MERGED PlaygroundWorld(GO2_ROOM) base sim-oracle verify namespace
     against a DETERMINISTIC stub base that advances position / heading as the
     base primitives (walk_forward / turn / scan_360) run, so each per-step verify
     (at_position(...) / visited('kitchen') / facing(...)) flips False->True as the
     chain progresses.

The decomposed chain ("explore the room") is GO2-ONLY:

    walk_to_kitchen (walk_forward) -> scan (scan_360) -> explore (explore_skill)

ASSERTIONS (the E-2 bar):
  - decompose is GO2-ONLY: base primitives + go2 skills only — no arm skills, no
    hallucinated skills; arm/hallucinated strategies are rejected via
    validation_notes with the valid set surfaced (fail-loud replan feedback).
  - the run reaches verified-done (trace.success) through the base predicates.
  - the observation snapshot is JSON-safe and renders the verified loop.

Hermetic: mock backend, no live LLM, no MuJoCo. The stub base mirrors the base
sim-oracle's grounding contract (get_position / get_heading) exactly as the real
MuJoCoGo2 exposes them, so the SAME playground base predicates that run against
MuJoCo run here unchanged.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

import pytest

from vector_os_nano.playground.catalog import GO2_ROOM
from vector_os_nano.playground.world import PlaygroundWorld
from vector_os_nano.vcli.cognitive.oracle_registry import evidence_names
from vector_os_nano.vcli.cognitive.blackboard import Blackboard
from vector_os_nano.vcli.cognitive.goal_decomposer import GoalDecomposer
from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.observation import render_run_snapshot, run_snapshot
from vector_os_nano.vcli.cognitive.strategy_selector import StrategyResult


# ---------------------------------------------------------------------------
# Deterministic stub base + agent (the base sim oracle's ground truth).
# ---------------------------------------------------------------------------
#
# The base predicates read DETERMINISTIC ground truth off the base via the same
# accessors the real MuJoCoGo2 exposes:
#   - base.get_position() -> [x, y, z]
#   - base.get_heading()  -> yaw radians
#   - base._connected     -> bool (predicates fail safe when False)
# This stub advances that ground truth as the base primitives run, so the same
# base predicates flip False->True step by step. No MuJoCo, fully deterministic.

# The go2 spawns at its REAL scene_room.xml pose (10, 3) in the central hallway
# facing +x (MuJoCoGo2._reset sets data.qpos[0:3] = [10, 3, 0.35]). The REAL
# kitchen box is (14, 0, 20, 5) (f_kitchen pos="17 2.5" size="3 2.5"); a 7 m
# forward walk lands the base at (17, 3) — inside the kitchen and within the
# 0.5 m at_position tolerance of (17, 3). The walk distance is the spawn-to-target
# x-gap (17 - 10).
_GO2_SPAWN = (10.0, 3.0, 0.3)
_KITCHEN_TARGET = (17.0, 3.0)
_WALK_DISTANCE = _KITCHEN_TARGET[0] - _GO2_SPAWN[0]


class _StubBase:
    """A deterministic stand-in for MuJoCoGo2's oracle surface.

    Holds ground-truth planar pose; walk_forward advances along the current
    heading, turn rotates in place, scan_360 is a pure observation sweep (no pose
    change). FK is not needed — the base predicates read position / heading
    directly, exactly as they do against the real base.
    """

    def __init__(self) -> None:
        # Spawn at the REAL go2 scene pose (10, 3) in the hallway, facing +x.
        self._position: list[float] = list(_GO2_SPAWN)
        self._heading: float = 0.0
        self._connected: bool = True

    # --- grounding-contract reads (what the base predicates call) ---

    def get_position(self) -> list[float]:
        if not self._connected:
            raise RuntimeError("_StubBase: not connected")
        return list(self._position)

    def get_heading(self) -> float:
        if not self._connected:
            raise RuntimeError("_StubBase: not connected")
        return self._heading

    # --- mutations driven by the base primitives ---

    def walk(self, distance: float) -> None:
        self._position[0] += distance * math.cos(self._heading)
        self._position[1] += distance * math.sin(self._heading)

    def rotate(self, angle_rad: float) -> None:
        self._heading = math.atan2(
            math.sin(self._heading + angle_rad),
            math.cos(self._heading + angle_rad),
        )


class _StubAgent:
    def __init__(self) -> None:
        # The base is reached via getattr(agent, "_base", None) — the same
        # accessor the kernel + base predicates use.
        self._base = _StubBase()


# ---------------------------------------------------------------------------
# Primitives + go2 skills that advance the stub base. Keyed by the SAME strategy
# names the decomposer validates + emits, so the DirectSelector routes each
# concrete step straight to its primitive.
# ---------------------------------------------------------------------------


def _make_primitives(agent: _StubAgent) -> dict[str, Any]:
    base: _StubBase = agent._base

    def walk_forward(distance: float | None = None, **_: Any) -> dict[str, Any]:
        d = float(distance) if distance is not None else 0.0
        base.walk(d)
        return {"walked": d}

    def turn(angle: float | None = None, **_: Any) -> dict[str, Any]:
        # The decompose params carry degrees (positive=left), matching the robot
        # vocab's turn help; convert to radians for the stub.
        deg = float(angle) if angle is not None else 0.0
        base.rotate(math.radians(deg))
        return {"turned_deg": deg}

    def scan_360(**_: Any) -> dict[str, Any]:
        # Pure observation sweep — no pose change.
        return {"scanned": True}

    def explore(**_: Any) -> dict[str, Any]:
        # The go2 explore skill: a deterministic stand-in that sweeps in place.
        return {"explored": True}

    # Keyed by the strategy names the decomposer emits: base primitives keep their
    # bare names; go2 skills carry the ``_skill`` suffix.
    return {
        "walk_forward": walk_forward,
        "turn": turn,
        "scan_360": scan_360,
        "explore_skill": explore,
    }


class _DirectSelector:
    """Route the sub_goal's explicit strategy straight to a primitive call."""

    def select(self, sub_goal: Any) -> StrategyResult:
        return StrategyResult(
            "primitive", sub_goal.strategy, dict(sub_goal.strategy_params)
        )


# ---------------------------------------------------------------------------
# Mock LLM backend — returns the fixed go2 plan as JSON.
# ---------------------------------------------------------------------------


class _MockBackend:
    def __init__(self, response: str) -> None:
        self._response = response

    def call(self, messages, tools, system, max_tokens, on_text=None):  # noqa: ANN001
        resp = self._response

        class _R:
            text = resp

        return _R()


# The GO2-ONLY world vocab the decomposer validates against. Single-sourced here
# as the base primitives + go2 skills (no arm): an arm strategy or a hallucinated
# skill must be rejected. The verify functions are the base sim-oracle predicates.
_GO2_STRATEGIES = frozenset(
    {"walk_forward", "turn", "scan_360", "explore_skill", "look_skill"}
)
_GO2_VERIFY_FNS = frozenset({"at_position", "facing", "visited"})


def _explore_room_plan() -> dict[str, Any]:
    """walk_to_kitchen (walk_forward) -> scan (scan_360) -> explore (explore_skill)."""
    return {
        "goal": "explore the room",
        "sub_goals": [
            {
                "name": "walk_to_kitchen",
                "description": "walk forward into the kitchen",
                "verify": "visited('kitchen') and at_position(17.0, 3.0)",
                "strategy": "walk_forward",
                "depends_on": [],
                "strategy_params": {"distance": _WALK_DISTANCE, "speed": 0.3},
            },
            {
                "name": "scan",
                "description": "rotate 360 degrees to observe the kitchen",
                "verify": "facing(0.0)",
                "strategy": "scan_360",
                "depends_on": ["walk_to_kitchen"],
                "strategy_params": {},
            },
            {
                "name": "explore",
                "description": "autonomously explore the kitchen",
                "verify": "visited('kitchen')",
                "strategy": "explore_skill",
                "depends_on": ["scan"],
                "strategy_params": {},
            },
        ],
        "context_snapshot": "Go2 in the hallway (10, 3) facing +x; kitchen lies ahead.",
    }


def _decompose_explore_room(plan: dict[str, Any]):
    """Run the REAL decomposer (go2-only vocab) over the mock plan."""
    decomposer = GoalDecomposer(
        _MockBackend(json.dumps(plan)),
        strategies=_GO2_STRATEGIES,
        verify_functions=_GO2_VERIFY_FNS,
        fallback_verify="True",
        has_base=True,  # go2: base primitives in vocab; arm skills are not
    )
    return decomposer.decompose("explore the room", "go2 room scene")


def _executor(agent: _StubAgent) -> GoalExecutor:
    """Wire a GoalExecutor over the MERGED PlaygroundWorld(GO2_ROOM) namespace."""
    world = PlaygroundWorld(GO2_ROOM)
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
# Decompose: GO2-ONLY (base primitives + go2 skills, no arm, no hallucination).
# ---------------------------------------------------------------------------


def test_explore_room_decomposes_go2_only() -> None:
    tree = _decompose_explore_room(_explore_room_plan())

    assert [sg.name for sg in tree.sub_goals] == ["walk_to_kitchen", "scan", "explore"]
    # A clean go2-only plan generates no validator complaints.
    assert tree.validation_notes == ()

    # Every strategy is a base primitive or a go2 skill — no arm skill leaked in.
    strategies = {sg.strategy for sg in tree.sub_goals if sg.strategy}
    assert strategies <= _GO2_STRATEGIES
    assert "walk_forward" in strategies and "scan_360" in strategies
    for banned in ("pick_skill", "place_skill", "detect_all_skill", "grasp_skill"):
        assert banned not in strategies


def test_explore_room_rejects_arm_and_hallucinated_skills() -> None:
    plan = _explore_room_plan()
    # Hallucinate an arm strategy on one step and a bogus skill on another — neither
    # is in the go2 world vocab.
    plan["sub_goals"][0]["strategy"] = "pick_skill"  # arm — not in go2 world
    plan["sub_goals"][2]["strategy"] = "teleport_skill"  # bogus

    tree = _decompose_explore_room(plan)

    # Both invalid strategies are cleared (not silently routed).
    assert tree.sub_goals[0].strategy == ""
    assert tree.sub_goals[2].strategy == ""
    # Fail-loud feedback names both offenders + surfaces the valid go2 set.
    notes = "\n".join(tree.validation_notes)
    assert "pick_skill" in notes and "not valid" in notes
    assert "teleport_skill" in notes
    assert "walk_forward" in notes  # valid go2 set surfaced for the replan


# ---------------------------------------------------------------------------
# End-to-end: real executor + verifier over the merged base sim-oracle namespace.
# ---------------------------------------------------------------------------


def test_explore_room_runs_end_to_end_verified() -> None:
    agent = _StubAgent()

    # Before the run the base is at the origin: NOT in the kitchen, NOT at target.
    world = PlaygroundWorld(GO2_ROOM)
    ns = world.build_verify_namespace(agent)
    assert ns["visited"]("kitchen") is False
    assert ns["at_position"](*_KITCHEN_TARGET) is False

    tree = _decompose_explore_room(_explore_room_plan())
    executor = _executor(agent)

    trace = executor.execute(tree)

    # The whole go2 chain reached verified-done through the base predicates.
    assert trace.success is True
    assert [s.sub_goal_name for s in trace.steps] == [
        "walk_to_kitchen",
        "scan",
        "explore",
    ]
    # Every per-step verify passed DETERMINISTICALLY (no visual override).
    for s in trace.steps:
        assert s.success is True, f"step {s.sub_goal_name} failed: {s.error}"
        assert s.verify_result is True
        assert s.visual_override is False

    # The sim-oracle ground truth advanced: the base ended inside the kitchen at
    # the walk target, still facing +x after the 360 scan.
    assert ns["visited"]("kitchen") is True
    assert ns["at_position"](*_KITCHEN_TARGET) is True
    assert ns["facing"](0.0) is True


def test_explore_room_oracle_starts_false_each_step_flips() -> None:
    # The base predicates must be FALSE at the start and FLIP only once the
    # corresponding step has run — proving verify is grounded in real sim state,
    # not a constant.
    agent = _StubAgent()
    ns = PlaygroundWorld(GO2_ROOM).build_verify_namespace(agent)

    # Origin: not in kitchen.
    assert ns["visited"]("kitchen") is False

    tree = _decompose_explore_room(_explore_room_plan())
    executor = _executor(agent)
    trace = executor.execute(tree)

    assert trace.success is True
    # After the run the base is grounded in the kitchen.
    assert ns["visited"]("kitchen") is True
    # And it never strayed into the guest bedroom (a disjoint named room across
    # the house at y=10..14).
    assert ns["visited"]("guest_bedroom") is False


# ---------------------------------------------------------------------------
# The observation snapshot is JSON-safe and renders the verified loop.
# ---------------------------------------------------------------------------


def test_explore_room_snapshot_json_safe_and_renders() -> None:
    agent = _StubAgent()
    tree = _decompose_explore_room(_explore_room_plan())
    executor = _executor(agent)

    trace = executor.execute(tree)
    # Grade against the playground's tagged GT oracles (what verify_oracle_names
    # returns in this world). Each step's verify is a real base predicate.
    oracle_names = evidence_names(PlaygroundWorld(GO2_ROOM).build_verify_namespace(agent))
    snapshot = run_snapshot(trace, oracle_names)

    # Round-trips through json.dumps (deterministic, no exotic objects leaked).
    encoded = json.dumps(snapshot)
    assert isinstance(encoded, str)

    # The snapshot reflects the go2 chain.
    step_names = [sv["sub_goal_name"] for sv in snapshot["steps"]]
    assert step_names == ["walk_to_kitchen", "scan", "explore"]
    # Every exported step shows success + verify_result AND grades GROUNDED (the
    # verifies are real base predicates over the live sim-oracle ground truth).
    assert all(sv["success"] and sv["verify_result"] for sv in snapshot["steps"])
    assert all(sv["evidence"] == "GROUNDED" for sv in snapshot["steps"])
    assert snapshot["success"] is True and snapshot["verified"] is True

    # The verified loop renders with the spine's own evidence markers (GROUNDED),
    # not a success&verify_result "PASS" that would mislabel a merely-RAN step.
    text = render_run_snapshot(snapshot)
    assert "Goal: explore the room" in text
    # Three grounded step lines plus the overall outcome line, all GROUNDED.
    assert text.count("[GROUNDED]") == 4
    assert "Outcome: [GROUNDED] 3/3 steps grounded" in text
    assert "visited('kitchen')" in text  # a base predicate is surfaced
    assert "[FAILED]" not in text


# ---------------------------------------------------------------------------
# Optional live-LLM smoke (deselected from the canonical gate).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("VECTOR_LIVE_LLM") != "1",
    reason="live-LLM smoke: set VECTOR_LIVE_LLM=1 to run (deselected by default)",
)
def test_explore_room_live_llm_smoke() -> None:  # pragma: no cover - opt-in
    """Smoke: a live backend decomposes a go2 navigation goal GO2-ONLY.

    Opt-in only. Asserts on STRUCTURE (no arm/hallucinated strategy leaks into the
    go2 world), never on exact wording. Skipped by default so the canonical suite
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
        strategies=_GO2_STRATEGIES,
        verify_functions=_GO2_VERIFY_FNS,
        fallback_verify="True",
        has_base=True,
    )
    tree = decomposer.decompose(
        "explore the room and go to the kitchen", "go2 room scene with named rooms"
    )
    # No arm/hallucinated strategy leaked into the go2-only world.
    strategies = {sg.strategy for sg in tree.sub_goals if sg.strategy}
    assert strategies <= _GO2_STRATEGIES
