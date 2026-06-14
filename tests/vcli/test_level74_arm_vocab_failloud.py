# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Level 74 — Phase D Stage 2b: world-scoped selector + fail-loud validation.

Stage 2a single-sourced the decompose vocabulary from the live skill registry,
so an arm-only world (has_base=False) is taught its OWN skills and never the GO2
base primitives. Stage 2b finishes the repair on the selector + validator side:

  - The StrategySelector is world-scoped: on a baseless world the GO2
    locomotion/observation keyword ladder and base-primitive resolution are
    skipped, so a 'walk'/'navigate' description never routes to a base primitive
    that would call _require_base() and raise on an arm.
  - Resolution is fail-loud: an explicit '<X>_skill' whose X is not a registered
    skill resolves to a CLEAR, named error (with the valid set) instead of a
    phantom skill or the opaque 'unmatched' fallback.
  - Validation is fail-loud with feedback: the decomposer collects dropped /
    invalid-strategy notes on GoalTree.validation_notes, and the harness injects
    them into the next replan's decompose context so the LLM stops repeating the
    hallucination.

Hermetic: no real LLM, no mujoco. Real GoalDecomposer + StrategySelector +
GoalExecutor + GoalVerifier + VGGHarness over a stub arm skill registry.
"""
from __future__ import annotations

import json
from typing import Any

from vector_os_nano.vcli.cognitive.goal_decomposer import GoalDecomposer
from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.strategy_selector import StrategySelector
from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal
from vector_os_nano.vcli.cognitive.vgg_harness import HarnessConfig, VGGHarness
from vector_os_nano.vcli.cognitive.vocab_from_registry import build_decompose_vocab


# ---------------------------------------------------------------------------
# Arm-like skill registry (no mobile base) + mock backend
# ---------------------------------------------------------------------------

# Mirrors skill_registry.to_schemas(): an arm with manipulation/perception skills
# and NO locomotion. 'look'/'navigate'/'walk'/'turn' are deliberately absent.
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
    {"name": "home", "description": "Return the arm to its home pose.", "parameters": {}},
]

_ARM_VERIFY_SIGS = {
    "gripper_holding": "gripper_holding() -> bool  # True if gripper holds an object",
    "arm_at_home": "arm_at_home() -> bool  # True if arm is at home pose",
}

_ARM_SKILL_NAMES = frozenset(s["name"] for s in _ARM_SCHEMAS)
_BASE_PRIMITIVES = frozenset({"walk_forward", "turn", "scan_360"})


class _SkillResult:
    def __init__(self, success: bool, result_data: dict, error: str = "") -> None:
        self.success = success
        self.result_data = result_data
        self.error_message = error


class _StubSkill:
    def __init__(self, name: str) -> None:
        self.name = name

    def execute(self, params: dict, context: Any = None) -> _SkillResult:
        return _SkillResult(success=True, result_data={"ran": self.name})


class _ArmRegistry:
    """Duck-typed arm skill registry: list_skills/get/match/to_schemas."""

    def __init__(self) -> None:
        self._skills = {n: _StubSkill(n) for n in _ARM_SKILL_NAMES}

    def list_skills(self) -> list[str]:
        return sorted(self._skills)

    def get(self, name: str) -> _StubSkill | None:
        return self._skills.get(name)

    def match(self, _description: str) -> None:
        # No alias matching in these tests — keep routing explicit.
        return None

    def to_schemas(self) -> list[dict[str, Any]]:
        return [dict(s) for s in _ARM_SCHEMAS]


class _MockBackend:
    """Records the last system prompt; returns a fixed JSON response."""

    def __init__(self, response: str = "{}") -> None:
        self._response = response
        self.last_system: Any = None

    def call(self, messages, tools, system, max_tokens, on_text=None):
        self.last_system = system

        class _R:
            text = self._response

        return _R()


def _arm_vocab_kwargs() -> dict[str, Any]:
    vocab = build_decompose_vocab(_ARM_SCHEMAS, _ARM_VERIFY_SIGS, has_base=False)
    return vocab.as_kwargs()


def _arm_decomposer(response: str = "{}") -> GoalDecomposer:
    return GoalDecomposer(
        _MockBackend(response),
        skill_registry=_ArmRegistry(),
        has_base=False,
        **_arm_vocab_kwargs(),
    )


def _arm_selector() -> StrategySelector:
    return StrategySelector(skill_registry=_ArmRegistry(), has_base=False)


def _arm_executor(selector: StrategySelector | None = None) -> GoalExecutor:
    registry = _ArmRegistry()
    return GoalExecutor(
        strategy_selector=selector or StrategySelector(skill_registry=registry, has_base=False),
        verifier=GoalVerifier({}),
        skill_registry=registry,
    )


# ---------------------------------------------------------------------------
# 1. Arm prompt teaches arm skills only — never the GO2 vocabulary
# ---------------------------------------------------------------------------


def test_arm_prompt_is_arm_only() -> None:
    prompt = _arm_decomposer()._build_system_prompt()[0]["text"]
    # Arm skills present.
    for strat in ("pick_skill", "place_skill", "scan_skill", "detect_skill", "home_skill"):
        assert strat in prompt
    # GO2 vocabulary gone — no navigate/look/explore/scan_360 strategies, no
    # '去厨房' example.
    assert "navigate_skill" not in prompt
    assert "look_skill" not in prompt
    assert "explore_skill" not in prompt
    assert "scan_360" not in prompt
    assert "去厨房" not in prompt
    assert "nearest_room" not in prompt


# ---------------------------------------------------------------------------
# 2. No base primitives in the arm's KNOWN_STRATEGIES / vocab strategies
# ---------------------------------------------------------------------------


def test_no_base_primitives_when_baseless() -> None:
    vocab = build_decompose_vocab(_ARM_SCHEMAS, _ARM_VERIFY_SIGS, has_base=False)
    assert not (_BASE_PRIMITIVES & vocab.strategies)

    gd = _arm_decomposer()
    assert not (_BASE_PRIMITIVES & gd.KNOWN_STRATEGIES)
    # The arm's strategies are exactly its <name>_skill set.
    assert gd.KNOWN_STRATEGIES == frozenset(f"{n}_skill" for n in _ARM_SKILL_NAMES)


# ---------------------------------------------------------------------------
# 3. Unknown explicit strategy fails loud (clear error naming skill + valid set)
# ---------------------------------------------------------------------------


def test_unknown_strategy_fail_loud() -> None:
    executor = _arm_executor()
    # 'look_skill' is NOT a registered arm skill (look is GO2-only). Build the
    # tree directly so the selector — not the decomposer — sees it explicitly.
    tree = GoalTree(
        goal="observe the workspace",
        sub_goals=(
            SubGoal(
                name="observe",
                description="look at the workspace",
                verify="True",
                strategy="look_skill",
            ),
        ),
    )
    trace = executor.execute(tree)

    assert trace.success is False
    step = trace.steps[0]
    assert step.success is False
    # Clear, named error — names the offending skill and the valid set, NOT
    # the opaque 'unmatched'.
    assert "look_skill" in step.error
    assert "is not a skill in this world" in step.error
    assert "unmatched" not in step.error
    # Valid set names a real arm skill.
    assert "pick" in step.error


def test_unknown_strategy_resolves_to_invalid_not_phantom_skill() -> None:
    sel = _arm_selector()
    result = sel._resolve_explicit("look_skill", {})
    assert result.executor_type == "invalid"
    assert result.name == "look_skill"
    assert "pick" in result.params["valid_strategies"]
    # A REAL arm skill still resolves to a skill, byte-for-byte.
    ok = sel._resolve_explicit("pick_skill", {"object_label": "mug"})
    assert ok.executor_type == "skill"
    assert ok.name == "pick"
    assert ok.params == {"object_label": "mug"}


# ---------------------------------------------------------------------------
# 4. Validation feedback flows into the replan context
# ---------------------------------------------------------------------------


def test_validation_feedback_in_replan() -> None:
    # A decompose that uses an invalid strategy 'look_skill' — the decomposer
    # clears it and records a validation note.
    plan = {
        "goal": "observe and pick",
        "sub_goals": [
            {
                "name": "observe",
                "description": "look at the bench",
                "verify": "True",
                "strategy": "look_skill",  # invalid for the arm
                "depends_on": [],
                "strategy_params": {},
            },
            {
                "name": "grab",
                "description": "pick the mug",
                "verify": "True",
                "strategy": "pick_skill",  # valid
                "depends_on": ["observe"],
                "strategy_params": {"object_label": "mug"},
            },
        ],
    }
    tree = _arm_decomposer(json.dumps(plan)).decompose("observe and pick", "world")
    # The invalid strategy was cleared AND surfaced as a validation note.
    assert any("look_skill" in n and "not valid" in n for n in tree.validation_notes)

    # The harness's replan-context builder must inject those notes.
    harness = VGGHarness(decomposer=object(), executor=object())
    enriched = []

    class _RecordingDecomposer:
        def decompose(self, _task: str, world_context: str) -> GoalTree:
            enriched.append(world_context)
            return GoalTree(goal="t", sub_goals=())

    harness._decomposer = _RecordingDecomposer()
    harness._decompose_with_context(
        "observe and pick", "fresh-context", [], tree.validation_notes
    )

    ctx = enriched[0]
    assert "Do NOT use these invalid strategies again" in ctx
    assert "look_skill" in ctx
    assert "valid strategies are" in ctx


def test_replan_feedback_threaded_through_run() -> None:
    """End-to-end: a failing first plan with an invalid strategy makes the
    SECOND decompose see the 'Do NOT use ...' feedback (harness wiring)."""
    bad_plan = GoalTree(
        goal="task",
        sub_goals=(
            SubGoal(name="s0", description="d", verify="False", strategy="noop"),
        ),
        validation_notes=("strategy 'look_skill' is not valid; valid strategies: pick_skill",),
    )
    ok_plan = GoalTree(
        goal="task",
        sub_goals=(SubGoal(name="s0", description="d", verify="True", strategy="noop"),),
    )

    seen: list[str] = []
    plans = iter((bad_plan, ok_plan))

    class _Decomposer:
        def decompose(self, _task: str, world_context: str) -> GoalTree:
            seen.append(world_context)
            return next(plans)

    # A real executor over the arm registry: first plan's 'noop' is invalid and
    # verify=False, so the step fails -> pipeline retry -> second decompose.
    executor = _arm_executor()

    harness = VGGHarness(
        decomposer=_Decomposer(),
        executor=executor,
        config=HarnessConfig(max_step_retries=0, max_pipeline_retries=1),
    )
    harness.run("task", "world")

    assert len(seen) == 2
    # First decompose had no prior notes; the second carries the feedback.
    assert "Do NOT use these invalid strategies again" not in seen[0]
    assert "Do NOT use these invalid strategies again" in seen[1]
    assert "look_skill" in seen[1]


# ---------------------------------------------------------------------------
# 5. Baseless selector skips the GO2 keyword ladder
# ---------------------------------------------------------------------------


def test_baseless_selector_skips_go2_ladder() -> None:
    sel = _arm_selector()
    # A 'walk'/'navigate' DESCRIPTION (no explicit strategy) must NOT route to a
    # base primitive — the GO2 ladder is skipped on a baseless world. With no
    # alias match, it falls through to the opaque fallback (the executor then
    # fails loud), but crucially never to walk_forward/navigate/turn.
    for desc in ("walk forward two metres", "navigate to the kitchen", "turn right"):
        sg = SubGoal(name="step", description=desc, verify="True")
        result = sel.select(sg)
        assert result.executor_type not in ("primitive",)
        assert result.name not in _BASE_PRIMITIVES
        assert result.name not in ("navigate", "look")


def test_go2_selector_still_routes_ladder() -> None:
    """Regression guard: has_base=True keeps the GO2 ladder byte-identical."""
    sel = StrategySelector(has_base=True)  # default, no registry
    nav = sel.select(SubGoal(name="reach", description="navigate to kitchen", verify="True"))
    assert (nav.executor_type, nav.name) == ("skill", "navigate")
    walk = sel.select(SubGoal(name="go", description="walk forward", verify="True"))
    assert (walk.executor_type, walk.name) == ("primitive", "walk_forward")
    turn = sel.select(SubGoal(name="rot", description="turn left", verify="True"))
    assert (turn.executor_type, turn.name) == ("primitive", "turn")
