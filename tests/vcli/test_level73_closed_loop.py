# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Level 73 — Phase D Stage 1b: observations FLOW (consume side).

Stage 1a made each successful step's structured output land on a run-scoped
Blackboard. Stage 1b makes a *later* step consume an *earlier* step's output:

  A. Data binding — a sub-goal's ``strategy_params`` may reference a prior step's
     captured output via ``${step.output.key...}``; the executor resolves the
     reference against the blackboard (pure dict/list traversal, no eval) BEFORE
     running the strategy, so the reference reaches the executed skill.
  B. world_context refresh — when a ``context_provider`` is supplied, the harness
     rebuilds the world context on each (re)decompose so replans see fresh state.

Hermetic: no real LLM, no mujoco. Real GoalExecutor + StrategySelector +
GoalVerifier + Blackboard; a stub skill registry and a mock decomposer.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from vector_os_nano.vcli.cognitive.blackboard import Blackboard
from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.strategy_selector import StrategySelector
from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal
from vector_os_nano.vcli.cognitive.vgg_harness import HarnessConfig, VGGHarness


# ---------------------------------------------------------------------------
# Stub skills + registry (duck-typed; the executor only needs the 3 attrs)
# ---------------------------------------------------------------------------


class _SkillResult:
    """Minimal stand-in for a skill ExecutionResult."""

    def __init__(self, success: bool, result_data: dict, error: str = "") -> None:
        self.success = success
        self.result_data = result_data
        self.error_message = error


class _StubSkill:
    """A skill that returns a fixed result_data and records the params it got."""

    def __init__(self, result_data: dict) -> None:
        self._result_data = result_data
        self.received_params: dict | None = None

    def execute(self, params: dict, context: Any = None) -> _SkillResult:
        # Capture exactly what the executor passed in (post resolution).
        self.received_params = dict(params)
        return _SkillResult(success=True, result_data=dict(self._result_data))


class _StubRegistry:
    """A name -> skill map exposing .get (the only method the executor calls)."""

    def __init__(self, skills: dict[str, _StubSkill]) -> None:
        self._skills = skills

    def get(self, name: str) -> _StubSkill | None:
        return self._skills.get(name)

    def match(self, _description: str) -> None:  # never used in these tests
        return None


def _make_executor(registry: _StubRegistry, verifier: GoalVerifier) -> GoalExecutor:
    return GoalExecutor(
        strategy_selector=StrategySelector(),
        verifier=verifier,
        skill_registry=registry,
    )


def _decomposer_returning(tree: GoalTree) -> MagicMock:
    dec = MagicMock()
    dec.decompose.return_value = tree
    return dec


# ---------------------------------------------------------------------------
# A. Data binding — a later step consumes an earlier step's output
# ---------------------------------------------------------------------------


def test_data_binding() -> None:
    """Step 2's ``${detect_all.output.objects.0.id}`` resolves to step 1's output."""
    detect_skill = _StubSkill({"objects": [{"id": "obj_0"}]})
    pick_skill = _StubSkill({"picked": True})
    registry = _StubRegistry({"detect_all": detect_skill, "pick_obj": pick_skill})

    # Verifier namespace is trivial — both steps verify True so capture happens.
    verifier = GoalVerifier({})
    executor = _make_executor(registry, verifier)

    tree = GoalTree(
        goal="detect then pick",
        sub_goals=(
            SubGoal(
                name="detect_all",
                description="detect every object",
                verify="True",
                strategy="detect_all",  # routes to skill 'detect_all'
            ),
            SubGoal(
                name="pick_obj",
                description="pick the detected object",
                verify="True",
                depends_on=("detect_all",),
                strategy="pick_obj",
                strategy_params={"object_id": "${detect_all.output.objects.0.id}"},
            ),
        ),
    )

    harness = VGGHarness(
        decomposer=_decomposer_returning(tree),
        executor=executor,
        config=HarnessConfig(max_step_retries=0, max_pipeline_retries=0),
    )
    trace = harness.run("detect then pick", "world", goal_tree=tree)

    assert trace.success is True
    # The reference RESOLVED from the blackboard before the skill ran.
    assert pick_skill.received_params is not None
    assert pick_skill.received_params.get("object_id") == "obj_0"
    # And the capture path wrote what the consume path read.
    bb = executor.blackboard
    assert bb is not None
    assert bb.get("detect_all")["output"]["objects"][0]["id"] == "obj_0"


def test_data_binding_unknown_ref_passes_through() -> None:
    """An unresolvable reference is left as-is (no eval, no crash)."""
    pick_skill = _StubSkill({"picked": True})
    registry = _StubRegistry({"pick_obj": pick_skill})
    executor = _make_executor(registry, GoalVerifier({}))

    bb = Blackboard()
    executor.blackboard = bb  # attach directly (no prior step captured anything)

    sub_goal = SubGoal(
        name="pick_obj",
        description="pick",
        verify="True",
        strategy="pick_obj",
        strategy_params={"object_id": "${missing.output.id}"},
    )
    step = executor._execute_sub_goal(sub_goal)

    assert step.success is True
    # Passthrough: the raw ${...} text survives unchanged (no eval, no None).
    assert pick_skill.received_params == {"object_id": "${missing.output.id}"}


def test_data_binding_injection_payload_is_inert() -> None:
    """A dunder/injection-looking reference is a harmless passthrough."""
    pick_skill = _StubSkill({"picked": True})
    registry = _StubRegistry({"pick_obj": pick_skill})
    executor = _make_executor(registry, GoalVerifier({}))
    executor.blackboard = Blackboard()

    payload = "${__import__('os').system('echo pwned')}"
    sub_goal = SubGoal(
        name="pick_obj",
        description="pick",
        verify="True",
        strategy="pick_obj",
        strategy_params={"object_id": payload},
    )
    step = executor._execute_sub_goal(sub_goal)

    assert step.success is True
    assert pick_skill.received_params == {"object_id": payload}  # never evaluated


# ---------------------------------------------------------------------------
# Verify returns the raw value AND it lands on the StepRecord
# ---------------------------------------------------------------------------


def test_verify_returns_value() -> None:
    """GoalVerifier.evaluate returns (bool, raw); StepRecord carries it through."""
    # evaluate() surfaces the raw value, not just the bool.
    verifier = GoalVerifier({"count_objects": lambda: 3})
    ok, raw = verifier.evaluate("count_objects()")
    assert ok is True
    assert raw == 3

    # And the executor records output + verify_value on result_data.
    detect_skill = _StubSkill({"objects": [{"id": "obj_0"}, {"id": "obj_1"}]})
    registry = _StubRegistry({"detect_all": detect_skill})
    executor = _make_executor(registry, verifier)
    executor.blackboard = Blackboard()

    sub_goal = SubGoal(
        name="detect_all",
        description="detect",
        verify="count_objects() == 3",
        strategy="detect_all",
    )
    step = executor._execute_sub_goal(sub_goal)

    assert step.success is True
    assert step.result_data["verify_value"] is True
    assert step.result_data["output"]["objects"][0]["id"] == "obj_0"


# ---------------------------------------------------------------------------
# B. world_context refresh — provider is re-invoked on every (re)decompose
# ---------------------------------------------------------------------------


def test_world_context_refreshed() -> None:
    """context_provider is called fresh for the initial decompose AND each re-plan."""
    fail_tree = GoalTree(
        goal="task",
        sub_goals=(SubGoal(name="step_0", description="s0", verify="False", strategy="noop"),),
    )

    # Provider returns a different string on every call (a tick counter).
    counter = {"n": 0}

    def provider() -> str:
        counter["n"] += 1
        return f"world-state-{counter['n']}"

    seen_contexts: list[str] = []

    decomposer = MagicMock()

    def _decompose(_task: str, world_context: str) -> GoalTree:
        seen_contexts.append(world_context)
        return fail_tree

    decomposer.decompose.side_effect = _decompose

    # An executor whose only step always fails, forcing a pipeline re-decompose.
    executor = MagicMock()
    executor._stats = None
    executor._topological_sort.return_value = list(fail_tree.sub_goals)
    fail_step = MagicMock()
    fail_step.success = False
    fail_step.sub_goal_name = "step_0"
    fail_step.strategy = "noop"
    fail_step.error = "boom"
    fail_step.duration_sec = 0.0
    executor._execute_sub_goal.return_value = fail_step

    harness = VGGHarness(
        decomposer=decomposer,
        executor=executor,
        config=HarnessConfig(max_step_retries=0, max_pipeline_retries=1),
    )
    harness.run("task", "static-fallback", context_provider=provider)

    # Two decompose passes (initial + 1 re-plan) → provider invoked twice.
    assert decomposer.decompose.call_count == 2
    assert counter["n"] == 2
    # Each decompose saw the FRESH provider output, not the static fallback.
    assert seen_contexts[0].startswith("world-state-1")
    assert seen_contexts[1].startswith("world-state-2")
    assert all("static-fallback" not in c for c in seen_contexts)


def test_world_context_static_fallback_when_no_provider() -> None:
    """Without a provider, the static world_context arg is used (current behavior)."""
    tree = GoalTree(
        goal="task",
        sub_goals=(SubGoal(name="step_0", description="s0", verify="True", strategy="noop"),),
    )
    seen: list[str] = []

    decomposer = MagicMock()

    def _decompose(_task: str, world_context: str) -> GoalTree:
        seen.append(world_context)
        return tree

    decomposer.decompose.side_effect = _decompose

    executor = MagicMock()
    executor._stats = None
    executor._topological_sort.return_value = list(tree.sub_goals)
    ok_step = MagicMock()
    ok_step.success = True
    ok_step.sub_goal_name = "step_0"
    ok_step.strategy = "noop"
    ok_step.error = ""
    ok_step.duration_sec = 0.0
    executor._execute_sub_goal.return_value = ok_step

    harness = VGGHarness(
        decomposer=decomposer,
        executor=executor,
        config=HarnessConfig(max_step_retries=0, max_pipeline_retries=0),
    )
    harness.run("task", "static-context")

    assert seen == ["static-context"]
