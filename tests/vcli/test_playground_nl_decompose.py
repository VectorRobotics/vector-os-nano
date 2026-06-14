# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Playground (INC7) — NL DECOMPOSITION drives the tabletop chain.

INC6 proved the verified loop on a *hand-authored* static GoalTree. INC7 closes
the north-star gap: when the ``tabletop`` playground scenario is active, the
kernel ``GoalDecomposer`` must turn a natural-language goal ("pick up the mug")
into an ARM-ONLY plan, validated against the playground/arm vocabulary that is
single-sourced from the skill registry, and run it green through the REAL
GoalExecutor + GoalVerifier over the merged playground verify namespace.

Two deterministic cases (mock backend — NO live LLM):

  (a) A plausible ARM plan (home -> detect -> pick -> place) validates (every
      strategy is a real arm ``<name>_skill``) and runs to verified-done, each
      sub_goal carrying a playground verify predicate that flips False -> True as
      the stub arm skills advance the shared sim oracle.

  (b) A go2/base / hallucinated plan (scan_360 / explore_skill / look_skill) is
      REJECTED — the offending strategies are dropped/flagged and
      GoalTree.validation_notes carries the fail-loud "not valid; valid
      strategies: ..." feedback. The surviving plan is arm-only.

The decompose VOCAB and the verify namespace are built exactly the way the
engine builds them for an active PlaygroundWorld over an arm agent:
``VectorEngine._build_registry_vocab_kwargs`` (single-sourced from the registry
schemas + the engine's verify namespace, ``has_base=False``) and
``PlaygroundWorld.build_verify_namespace`` (the playground sim-oracle predicates).

Hermetic: no MuJoCo, no network. The arm + gripper are deterministic stubs
sharing the oracle surface the predicates read; the arm skills are stubs that
mutate that oracle. One ``@pytest.mark.live_llm`` smoke is SKIPPED unless
``VECTOR_LIVE_LLM=1`` so the canonical gate never calls a real LLM.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any

import pytest

from vector_os_nano.playground import PlaygroundWorld
from vector_os_nano.playground.verify.arm_predicates import _HOME_JOINTS
from vector_os_nano.vcli.cognitive.goal_decomposer import GoalDecomposer
from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.strategy_selector import StrategySelector
from vector_os_nano.vcli.engine import VectorEngine
from vector_os_nano.vcli.intent_router import IntentRouter
from vector_os_nano.vcli.tools.base import CategorizedToolRegistry

# Base locomotion primitives an arm-only world must NEVER be taught.
_BASE_PRIMITIVES = frozenset({"walk_forward", "turn", "scan_360"})

# A region the mug ends up resting in once placed (x_min, y_min, x_max, y_max).
_REGION = (0.0, 0.0, 0.5, 0.5)
_EE = [0.20, 0.10, 0.25]  # end-effector xyz used by holding_object()
_NOT_HOME_JOINTS: tuple[float, ...] = tuple(j + 0.5 for j in _HOME_JOINTS)


# ---------------------------------------------------------------------------
# Deterministic stub arm + gripper — the mutable sim oracle the chain advances.
# Same oracle surface the playground predicates read (test_playground_chain).
# ---------------------------------------------------------------------------


class _StubArm:
    def __init__(self) -> None:
        self._joints: list[float] = list(_NOT_HOME_JOINTS)
        self._objects: dict[str, list[float]] = {
            "mug": [0.21, 0.10, 0.06],
            "banana": [0.40, 0.40, 0.06],
        }
        self._connected = True

    def get_joint_positions(self) -> list[float]:
        return list(self._joints)

    def get_object_positions(self) -> dict[str, list[float]]:
        return {k: list(v) for k, v in self._objects.items()}

    def fk(self, joint_positions: list[float]):
        return list(_EE), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    # --- mutations the arm skills drive ---
    def go_home(self) -> None:
        self._joints = list(_HOME_JOINTS)

    def lift(self, name: str) -> None:
        self._objects[name] = [_EE[0] + 0.01, _EE[1], _EE[2]]

    def place(self, name: str, xy: tuple[float, float]) -> None:
        self._objects[name] = [xy[0], xy[1], 0.06]


class _StubGripper:
    def __init__(self) -> None:
        self._holding = False

    def is_holding(self) -> bool:
        return self._holding

    def close(self) -> None:
        self._holding = True

    def open(self) -> None:
        self._holding = False


# ---------------------------------------------------------------------------
# Arm skill registry — mirrors the real arm SkillRegistry.to_schemas() shape.
# Each skill's execute() advances the shared oracle so the next playground
# verify predicate legitimately flips False -> True (real evidence, not a stub).
# ---------------------------------------------------------------------------

_ARM_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "pick",
        "description": "Pick up an object with the arm gripper.",
        "parameters": {"object_label": {"type": "string", "required": True}},
    },
    {
        "name": "place",
        "description": "Place the held object at a target pose.",
        "parameters": {"target": {"type": "string", "required": True}},
    },
    {"name": "scan", "description": "Sweep the camera across the workspace.", "parameters": {}},
    {
        "name": "detect",
        "description": "Detect objects matching a query.",
        "parameters": {"query": {"type": "string", "required": False}},
    },
    {"name": "describe", "description": "Describe the tabletop scene.", "parameters": {}},
    {"name": "home", "description": "Return the arm to its home pose.", "parameters": {}},
    {"name": "wave", "description": "Wave the arm.", "parameters": {}},
]

_ARM_SKILL_NAMES = frozenset(s["name"] for s in _ARM_SCHEMAS)


class _SkillResult:
    def __init__(self, success: bool, result_data: dict) -> None:
        self.success = success
        self.result_data = result_data
        self.error_message = ""


class _OracleSkill:
    """An arm skill whose execute() mutates the shared oracle (arm + gripper)."""

    def __init__(self, name: str, arm: _StubArm, gripper: _StubGripper) -> None:
        self.name = name
        self._arm = arm
        self._gripper = gripper

    def execute(self, params: dict, context: Any = None) -> _SkillResult:
        if self.name == "home":
            self._arm.go_home()
        elif self.name == "pick":
            self._gripper.close()
            self._arm.lift("mug")
        elif self.name == "place":
            self._arm.place("mug", (_REGION[0] + 0.1, _REGION[1] + 0.1))
            self._gripper.open()
        # detect / describe / scan / wave are pure reads — no oracle mutation.
        return _SkillResult(success=True, result_data={"ran": self.name})


class _ArmRegistry:
    """Duck-typed arm skill registry: list_skills/get/match/to_schemas."""

    def __init__(self, arm: _StubArm, gripper: _StubGripper) -> None:
        self._skills = {n: _OracleSkill(n, arm, gripper) for n in _ARM_SKILL_NAMES}

    def list_skills(self) -> list[str]:
        return sorted(self._skills)

    def get(self, name: str):
        return self._skills.get(name)

    def match(self, _description: str):
        return None  # explicit routing only — no alias matching in these tests

    def to_schemas(self) -> list[dict[str, Any]]:
        return [dict(s) for s in _ARM_SCHEMAS]


# ---------------------------------------------------------------------------
# Mock backend — returns a fixed JSON plan (records the system prompt it saw).
# ---------------------------------------------------------------------------


class _MockBackend:
    def __init__(self, response: str) -> None:
        self._response = response
        self.last_system: Any = None

    def call(self, messages, tools, system, max_tokens, on_text=None):
        self.last_system = system

        resp = self._response

        class _R:
            text = resp

        return _R()


# ---------------------------------------------------------------------------
# Shared wiring: build vocab + verify ns EXACTLY as the engine does for an
# active PlaygroundWorld over an arm agent (has_base=False, world wired in).
# ---------------------------------------------------------------------------


def _arm_agent(arm: _StubArm, gripper: _StubGripper) -> SimpleNamespace:
    # No ``_base`` attribute => the engine derives has_base=False (arm-only).
    return SimpleNamespace(_arm=arm, _gripper=gripper)


def _playground_vocab_kwargs(agent: Any, registry: _ArmRegistry) -> dict[str, Any]:
    """Derive the arm decompose vocab through the engine's production path.

    Wires PlaygroundWorld as the active world so the verify namespace (hence the
    verify-function allowlist single-sourced into the vocab) carries the
    playground sim-oracle predicates, then calls the engine's own
    ``_build_registry_vocab_kwargs`` with has_base=False.
    """
    engine = VectorEngine(
        backend=_MockBackend("{}"),
        registry=CategorizedToolRegistry(),
        system_prompt=[],
        intent_router=IntentRouter(),
    )
    engine._world = PlaygroundWorld()
    has_base = getattr(agent, "_base", None) is not None
    assert has_base is False  # the playground arm agent has no mobile base
    return engine._build_registry_vocab_kwargs(registry, agent, has_base=has_base)


def _decomposer(response: str, agent: Any, registry: _ArmRegistry) -> GoalDecomposer:
    return GoalDecomposer(
        _MockBackend(response),
        skill_registry=registry,
        has_base=False,
        **_playground_vocab_kwargs(agent, registry),
    )


def _executor(agent: Any, registry: _ArmRegistry) -> GoalExecutor:
    """Real executor + real verifier over the merged playground verify ns."""
    namespace = PlaygroundWorld().build_verify_namespace(agent)
    return GoalExecutor(
        strategy_selector=StrategySelector(skill_registry=registry, has_base=False),
        verifier=GoalVerifier(namespace),
        skill_registry=registry,
    )


# A plausible ARM plan for "pick up the mug" — every strategy is a real arm
# <name>_skill, every verify is a playground predicate.
_GOOD_PLAN = {
    "goal": "pick up the mug",
    "sub_goals": [
        {
            "name": "home_arm",
            "description": "move the arm to its home pose",
            "verify": "arm_at_home()",
            "strategy": "home_skill",
            "depends_on": [],
            "strategy_params": {},
        },
        {
            "name": "detect_mug",
            "description": "detect the mug on the table",
            "verify": "len(detect_objects('mug')) > 0",
            "strategy": "detect_skill",
            "depends_on": ["home_arm"],
            "strategy_params": {"query": "mug"},
        },
        {
            "name": "grasp_mug",
            "description": "pick up the mug",
            "verify": "holding_object()",
            "strategy": "pick_skill",
            "depends_on": ["detect_mug"],
            "strategy_params": {"object_label": "mug"},
        },
        {
            "name": "place_mug",
            "description": "place the mug in the target region",
            "verify": "placed_count((0.0, 0.0, 0.5, 0.5)) >= 1",
            "strategy": "place_skill",
            "depends_on": ["grasp_mug"],
            "strategy_params": {"target": "bin"},
        },
    ],
    "context_snapshot": "",
}

# A go2/base + hallucinated plan: scan_360 (base primitive), explore_skill and
# look_skill (go2-only skills) — none are arm skills.
_BAD_PLAN = {
    "goal": "pick up the mug",
    "sub_goals": [
        {
            "name": "spin",
            "description": "rotate to survey the room",
            "verify": "True",
            "strategy": "scan_360",  # base primitive — invalid for an arm
            "depends_on": [],
            "strategy_params": {},
        },
        {
            "name": "wander",
            "description": "explore the area",
            "verify": "True",
            "strategy": "explore_skill",  # go2-only — not a registered arm skill
            "depends_on": ["spin"],
            "strategy_params": {},
        },
        {
            "name": "peek",
            "description": "look around for the mug",
            "verify": "True",
            "strategy": "look_skill",  # go2-only — not a registered arm skill
            "depends_on": ["wander"],
            "strategy_params": {},
        },
        {
            "name": "grasp_mug",
            "description": "pick up the mug",
            "verify": "holding_object()",
            "strategy": "pick_skill",  # the one real arm skill
            "depends_on": ["peek"],
            "strategy_params": {"object_label": "mug"},
        },
    ],
    "context_snapshot": "",
}


# ---------------------------------------------------------------------------
# 0. The vocab reaching the decomposer is ARM-ONLY and single-sourced.
# ---------------------------------------------------------------------------


def test_playground_decompose_vocab_is_arm_only() -> None:
    arm, gripper = _StubArm(), _StubGripper()
    agent = _arm_agent(arm, gripper)
    registry = _ArmRegistry(arm, gripper)
    gd = _decomposer(json.dumps(_GOOD_PLAN), agent, registry)

    # Strategies are exactly the arm's <name>_skill set — no go2/base.
    assert gd.KNOWN_STRATEGIES == frozenset(f"{n}_skill" for n in _ARM_SKILL_NAMES)
    assert not (_BASE_PRIMITIVES & gd.KNOWN_STRATEGIES)

    prompt = gd._build_system_prompt()[0]["text"]
    for strat in ("pick_skill", "place_skill", "detect_skill", "home_skill"):
        assert strat in prompt
    # No go2 vocabulary leaked into the prompt.
    for absent in ("navigate_skill", "look_skill", "explore_skill", "scan_360", "去厨房", "nearest_room"):
        assert absent not in prompt
    # The playground verify oracle is the allowlist — not the go2 room functions.
    assert {"arm_at_home", "holding_object", "placed_count", "detect_objects"} <= gd.VERIFY_FUNCTIONS
    assert "nearest_room" not in gd.VERIFY_FUNCTIONS


# ---------------------------------------------------------------------------
# (a) A plausible arm plan validates and runs to verified-done.
# ---------------------------------------------------------------------------


def test_good_arm_plan_decomposes_and_runs_verified_done() -> None:
    arm, gripper = _StubArm(), _StubGripper()
    agent = _arm_agent(arm, gripper)
    registry = _ArmRegistry(arm, gripper)

    tree = _decomposer(json.dumps(_GOOD_PLAN), agent, registry).decompose(
        "pick up the mug", "tabletop scene"
    )

    # Decomposition kept all four arm steps — no strategy was cleared.
    assert tree.validation_notes == ()
    assert [sg.name for sg in tree.sub_goals] == [
        "home_arm",
        "detect_mug",
        "grasp_mug",
        "place_mug",
    ]
    # Every step's strategy is a real arm <name>_skill — arm-only, no go2/base.
    for sg in tree.sub_goals:
        assert sg.strategy.endswith("_skill")
        assert sg.strategy[: -len("_skill")] in _ARM_SKILL_NAMES
        assert sg.strategy not in _BASE_PRIMITIVES
    # Each sub_goal carries a playground verify predicate. The grasp step's
    # bare holding_object() is strengthened to its bound target (backlog #2b).
    assert [sg.verify for sg in tree.sub_goals] == [
        "arm_at_home()",
        "len(detect_objects('mug')) > 0",
        "holding_object('mug')",
        "placed_count((0.0, 0.0, 0.5, 0.5)) >= 1",
    ]

    # Run the decomposed tree through the real executor + verifier.
    trace = _executor(agent, registry).execute(tree)

    assert trace.success is True
    assert [s.sub_goal_name for s in trace.steps] == [
        "home_arm",
        "detect_mug",
        "grasp_mug",
        "place_mug",
    ]
    for s in trace.steps:
        assert s.success is True
        assert s.verify_result is True  # deterministic predicate, real evidence
        assert s.visual_override is False


# ---------------------------------------------------------------------------
# (b) A go2/hallucinated plan is rejected with fail-loud feedback; the
#     surviving plan is arm-only.
# ---------------------------------------------------------------------------


def test_hallucinated_strategies_rejected_with_failloud_feedback() -> None:
    arm, gripper = _StubArm(), _StubGripper()
    agent = _arm_agent(arm, gripper)
    registry = _ArmRegistry(arm, gripper)

    tree = _decomposer(json.dumps(_BAD_PLAN), agent, registry).decompose(
        "pick up the mug", "tabletop scene"
    )

    # The decomposer keeps the sub_goal structure but CLEARS every invalid
    # strategy (scan_360 / explore_skill / look_skill) — none of them are arm
    # skills — while the one real arm skill (pick_skill) survives.
    by_name = {sg.name: sg for sg in tree.sub_goals}
    assert by_name["spin"].strategy == ""  # scan_360 cleared
    assert by_name["wander"].strategy == ""  # explore_skill cleared
    assert by_name["peek"].strategy == ""  # look_skill cleared
    assert by_name["grasp_mug"].strategy == "pick_skill"  # valid arm skill kept

    # No surviving strategy is a go2/base/hallucinated one — arm-only.
    surviving = {sg.strategy for sg in tree.sub_goals if sg.strategy}
    assert surviving == {"pick_skill"}
    assert not (surviving & _BASE_PRIMITIVES)
    for bad in ("scan_360", "explore_skill", "look_skill"):
        assert bad not in surviving

    # Fail-loud feedback names each offending strategy + the valid arm set, so
    # the next replan is steered off the hallucination.
    notes = "\n".join(tree.validation_notes)
    for bad in ("scan_360", "explore_skill", "look_skill"):
        assert any(bad in n and "not valid" in n for n in tree.validation_notes)
    assert "pick_skill" in notes  # valid set surfaced


# ---------------------------------------------------------------------------
# Live-LLM smoke — SKIPPED unless VECTOR_LIVE_LLM=1 (never in the canonical gate).
# ---------------------------------------------------------------------------


@pytest.mark.live_llm
@pytest.mark.skipif(
    os.environ.get("VECTOR_LIVE_LLM") != "1",
    reason="live LLM smoke; set VECTOR_LIVE_LLM=1 to run (never in the canonical gate)",
)
def test_live_llm_decomposes_pick_the_mug_arm_only() -> None:  # pragma: no cover
    """Smoke: a real backend decomposing 'pick up the mug' under the playground
    arm vocab produces an arm-only, registry-valid plan (no go2/base/hallucination).

    Deselected by default — the canonical suite never calls a live LLM. Requires
    a configured backend (e.g. via ~/.vector/config.yaml) and VECTOR_LIVE_LLM=1.
    """
    from vector_os_nano.vcli.backends import create_backend
    from vector_os_nano.vcli.config import load_config

    arm, gripper = _StubArm(), _StubGripper()
    agent = _arm_agent(arm, gripper)
    registry = _ArmRegistry(arm, gripper)

    # Build the backend the same way the CLI does (config-driven, no hardcoded key).
    cfg = load_config()
    provider = cfg.get("provider", "openrouter")
    api_key = cfg.get(f"{provider}_api_key") or cfg.get("api_key")
    if not api_key:
        pytest.skip(f"no api_key for provider {provider!r} in ~/.vector/config.yaml")
    backend = create_backend(
        provider=provider,
        api_key=api_key,
        model=cfg.get("model", "google/gemini-2.5-flash"),
        base_url=cfg.get("base_url"),
    )
    gd = GoalDecomposer(
        backend,
        skill_registry=registry,
        has_base=False,
        **_playground_vocab_kwargs(agent, registry),
    )
    tree = gd.decompose("pick up the mug", "tabletop scene with a mug")

    assert tree.sub_goals  # the planner produced at least one step
    for sg in tree.sub_goals:
        if sg.strategy:
            # Every chosen strategy must be a real arm skill — never go2/base.
            assert sg.strategy.endswith("_skill")
            assert sg.strategy[: -len("_skill")] in _ARM_SKILL_NAMES
            assert sg.strategy not in _BASE_PRIMITIVES
