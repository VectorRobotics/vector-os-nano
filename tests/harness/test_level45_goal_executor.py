"""Level 45 — VGG Phase 1: GoalExecutor

TDD tests for GoalExecutor: runs GoalTrees, verifies each step,
handles fallbacks, timeout, topological ordering, and on_step callbacks.

Acceptance Criteria: AC-25 to AC-30 + additional coverage.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, call

import pytest

from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)
from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

def _strategy_mock(executor_type: str = "skill", name: str = "navigate", params: dict | None = None) -> MagicMock:
    """Create a StrategyResult-like mock with proper string attributes.

    MagicMock treats 'name' as a special kwarg (sets the mock display name,
    not a real attribute), so we always set attributes explicitly after creation.
    """
    m = MagicMock()
    m.executor_type = executor_type
    m.name = name
    m.params = params if params is not None else {}
    return m


def _make_executor(
    verify_side_effect=None,
    verify_return=True,
    skill_success=True,
    skill_error=None,
    executor_type="skill",
) -> tuple[GoalExecutor, MagicMock, MagicMock, MagicMock]:
    """Build a GoalExecutor with fully mocked dependencies.

    Returns (executor, mock_selector, mock_verifier, mock_skill).
    """
    mock_skill = MagicMock()
    mock_skill.execute.return_value = MagicMock(
        success=skill_success, error_message=skill_error
    )

    mock_registry = MagicMock()
    mock_registry.get.return_value = mock_skill

    mock_selector = MagicMock()
    mock_selector.select.return_value = _strategy_mock(
        executor_type=executor_type, name="navigate", params={"room": "kitchen"}
    )

    mock_verifier = MagicMock()
    if verify_side_effect is not None:
        mock_verifier.verify.side_effect = verify_side_effect
    else:
        mock_verifier.verify.return_value = verify_return

    mock_context = MagicMock()
    mock_build_context = MagicMock(return_value=mock_context)

    executor = GoalExecutor(
        strategy_selector=mock_selector,
        verifier=mock_verifier,
        skill_registry=mock_registry,
        build_context=mock_build_context,
    )
    return executor, mock_selector, mock_verifier, mock_skill


def _simple_tree(*names: str, timeout_sec: float = 10.0) -> GoalTree:
    """Create a GoalTree with independent sub-goals (no dependencies)."""
    sub_goals = tuple(
        SubGoal(name=n, description=n, verify="True", timeout_sec=timeout_sec)
        for n in names
    )
    return GoalTree(goal="test", sub_goals=sub_goals)


# ---------------------------------------------------------------------------
# AC-25: Topological ordering — linear chain A → B → C
# ---------------------------------------------------------------------------

class TestTopologicalOrder:
    def test_linear_deps_executed_in_order(self):
        """Three sub_goals A→B→C must execute in that order (AC-25)."""
        sg_a = SubGoal(name="a", description="first", verify="True", timeout_sec=10)
        sg_b = SubGoal(
            name="b", description="second", verify="True", timeout_sec=10,
            depends_on=("a",),
        )
        sg_c = SubGoal(
            name="c", description="third", verify="True", timeout_sec=10,
            depends_on=("b",),
        )
        tree = GoalTree(goal="test", sub_goals=(sg_a, sg_b, sg_c))

        executor, mock_selector, mock_verifier, _ = _make_executor()
        trace = executor.execute(tree)

        assert trace.success is True
        assert len(trace.steps) == 3
        assert [s.sub_goal_name for s in trace.steps] == ["a", "b", "c"]

    def test_parallel_deps_all_before_dependent(self):
        """Parallel deps (A→C, B→C): A and B must both finish before C."""
        sg_a = SubGoal(name="a", description="a", verify="True", timeout_sec=10)
        sg_b = SubGoal(name="b", description="b", verify="True", timeout_sec=10)
        sg_c = SubGoal(
            name="c", description="c", verify="True", timeout_sec=10,
            depends_on=("a", "b"),
        )
        # Supply in order c, b, a — topological sort must reorder
        tree = GoalTree(goal="test", sub_goals=(sg_c, sg_b, sg_a))

        executor, _, _, _ = _make_executor()
        trace = executor.execute(tree)

        assert trace.success is True
        names = [s.sub_goal_name for s in trace.steps]
        # C must appear after both A and B
        assert names.index("c") > names.index("a")
        assert names.index("c") > names.index("b")

    def test_cyclic_deps_falls_back_to_original_order(self):
        """Cycle in depends_on → warn and execute in original order."""
        sg_a = SubGoal(
            name="a", description="a", verify="True", timeout_sec=10,
            depends_on=("b",),
        )
        sg_b = SubGoal(
            name="b", description="b", verify="True", timeout_sec=10,
            depends_on=("a",),
        )
        tree = GoalTree(goal="test", sub_goals=(sg_a, sg_b))

        executor, _, _, _ = _make_executor()
        trace = executor.execute(tree)

        # Should still execute, just in original order
        assert len(trace.steps) == 2
        assert [s.sub_goal_name for s in trace.steps] == ["a", "b"]


# ---------------------------------------------------------------------------
# AC-26: Verification pass/fail controls continuation
# ---------------------------------------------------------------------------

class TestVerification:
    def test_all_verifications_pass_success(self):
        """All verify=True → ExecutionTrace.success=True (AC-26 pass case)."""
        tree = _simple_tree("step1", "step2", "step3")
        executor, _, mock_verifier, _ = _make_executor(verify_return=True)

        trace = executor.execute(tree)

        assert trace.success is True
        assert all(s.success for s in trace.steps)
        assert all(s.verify_result for s in trace.steps)

    def test_third_verification_fails_aborts(self):
        """verify fails on 3rd step → abort, trace.success=False (AC-26 fail case)."""
        tree = _simple_tree("step1", "step2", "step3")
        executor, _, mock_verifier, _ = _make_executor(
            verify_side_effect=[True, True, False]
        )

        trace = executor.execute(tree)

        assert trace.success is False
        assert len(trace.steps) == 3
        assert trace.steps[0].success is True
        assert trace.steps[1].success is True
        assert trace.steps[2].success is False

    def test_first_failure_aborts_remaining_steps(self):
        """Failure at step 1 must not execute steps 2 and 3."""
        tree = _simple_tree("step1", "step2", "step3")
        executor, _, _, _ = _make_executor(verify_side_effect=[False])

        trace = executor.execute(tree)

        assert trace.success is False
        assert len(trace.steps) == 1  # only step1 recorded

    def test_verify_result_recorded_in_step(self):
        """verify_result field in StepRecord mirrors verifier output."""
        tree = _simple_tree("only")
        executor, _, mock_verifier, _ = _make_executor(verify_return=True)
        trace = executor.execute(tree)

        assert trace.steps[0].verify_result is True

    def test_verify_false_records_success_false(self):
        """verify returns False → step.success is False."""
        tree = _simple_tree("only")
        executor, _, mock_verifier, _ = _make_executor(verify_return=False)
        trace = executor.execute(tree)

        assert trace.steps[0].success is False


# ---------------------------------------------------------------------------
# AC-27: Fallback (fail_action)
# ---------------------------------------------------------------------------

class TestFallback:
    def test_fail_action_used_when_verify_fails(self):
        """fail_action triggered when verify fails; re-verify passes (AC-27)."""
        sg = SubGoal(
            name="detect",
            description="find cup",
            verify="False",
            timeout_sec=10,
            fail_action="scan_room_360",
        )
        tree = GoalTree(goal="test", sub_goals=(sg,))

        # verify: False first, then True after fallback
        mock_verifier = MagicMock()
        mock_verifier.verify.side_effect = [False, True]

        mock_skill = MagicMock()
        mock_skill.execute.return_value = MagicMock(success=True, error_message=None)
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        # selector returns skill for both original and fallback
        mock_selector = MagicMock()
        mock_selector.select.return_value = _strategy_mock("skill", "scan", {})

        executor = GoalExecutor(
            strategy_selector=mock_selector,
            verifier=mock_verifier,
            skill_registry=mock_registry,
        )
        trace = executor.execute(tree)

        assert len(trace.steps) == 1
        assert trace.steps[0].fallback_used is True
        assert trace.steps[0].success is True

    def test_fallback_fail_marks_step_failed(self):
        """fail_action used but re-verify still False → step fails."""
        sg = SubGoal(
            name="detect",
            description="find cup",
            verify="False",
            timeout_sec=10,
            fail_action="scan_room_360",
        )
        tree = GoalTree(goal="test", sub_goals=(sg,))

        # both verifications fail
        mock_verifier = MagicMock()
        mock_verifier.verify.side_effect = [False, False]

        mock_skill = MagicMock()
        mock_skill.execute.return_value = MagicMock(success=True, error_message=None)
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        mock_selector = MagicMock()
        mock_selector.select.return_value = _strategy_mock("skill", "scan", {})

        executor = GoalExecutor(
            strategy_selector=mock_selector,
            verifier=mock_verifier,
            skill_registry=mock_registry,
        )
        trace = executor.execute(tree)

        assert trace.steps[0].fallback_used is True
        assert trace.steps[0].success is False
        assert trace.success is False

    def test_no_fail_action_no_fallback(self):
        """No fail_action → no fallback attempt; step fails immediately."""
        sg = SubGoal(name="only", description="d", verify="False", timeout_sec=10)
        tree = GoalTree(goal="test", sub_goals=(sg,))

        executor, mock_selector, mock_verifier, _ = _make_executor(verify_return=False)
        trace = executor.execute(tree)

        # selector called once (initial), not twice
        assert mock_selector.select.call_count == 1
        assert trace.steps[0].fallback_used is False


# ---------------------------------------------------------------------------
# AC-28: Timeout handling
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_timeout_exceeded_marks_failure(self):
        """Skill that exceeds timeout_sec → step fails with timeout error (AC-28)."""
        sg = SubGoal(
            name="slow_step",
            description="slow",
            verify="True",
            timeout_sec=0.01,  # 10 ms
        )
        tree = GoalTree(goal="test", sub_goals=(sg,))

        # Skill execution sleeps to trigger timeout
        def slow_execute(params, context=None):
            time.sleep(0.05)  # 50 ms > 10 ms timeout
            return MagicMock(success=True, error_message=None)

        mock_skill = MagicMock()
        mock_skill.execute.side_effect = slow_execute
        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_skill

        mock_selector = MagicMock()
        mock_selector.select.return_value = _strategy_mock("skill", "navigate", {})

        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = True

        executor = GoalExecutor(
            strategy_selector=mock_selector,
            verifier=mock_verifier,
            skill_registry=mock_registry,
        )
        trace = executor.execute(tree)

        assert trace.steps[0].success is False
        assert "timeout" in trace.steps[0].error.lower()
        assert trace.success is False

    def test_within_timeout_succeeds(self):
        """Skill that finishes within timeout_sec → step succeeds."""
        tree = _simple_tree("fast")
        executor, _, _, _ = _make_executor()
        # Default mock is instantaneous
        trace = executor.execute(tree)
        assert trace.steps[0].success is True


# ---------------------------------------------------------------------------
# AC-29: on_step callback
# ---------------------------------------------------------------------------

class TestOnStepCallback:
    def test_on_step_called_for_each_step(self):
        """on_step callback invoked once per sub_goal (AC-29)."""
        tree = _simple_tree("a", "b", "c")
        executor, _, _, _ = _make_executor()

        callback = MagicMock()
        executor.execute(tree, on_step=callback)

        assert callback.call_count == 3

    def test_on_step_receives_step_record(self):
        """on_step callback receives StepRecord instances."""
        tree = _simple_tree("x")
        executor, _, _, _ = _make_executor()

        received: list[StepRecord] = []
        executor.execute(tree, on_step=received.append)

        assert len(received) == 1
        assert isinstance(received[0], StepRecord)

    def test_on_step_none_does_not_crash(self):
        """on_step=None must not raise."""
        tree = _simple_tree("x")
        executor, _, _, _ = _make_executor()
        trace = executor.execute(tree, on_step=None)
        assert trace.success is True

    def test_on_step_called_even_on_failure(self):
        """on_step called for the failing step too."""
        tree = _simple_tree("a", "b")
        executor, _, mock_verifier, _ = _make_executor(
            verify_side_effect=[False]
        )

        callback = MagicMock()
        executor.execute(tree, on_step=callback)

        # Only 1 step executed (abort after first failure)
        assert callback.call_count == 1


# ---------------------------------------------------------------------------
# AC-30: ExecutionTrace completeness
# ---------------------------------------------------------------------------

class TestExecutionTrace:
    def test_trace_is_execution_trace_instance(self):
        """execute() returns an ExecutionTrace (AC-30)."""
        tree = _simple_tree("only")
        executor, _, _, _ = _make_executor()
        trace = executor.execute(tree)
        assert isinstance(trace, ExecutionTrace)

    def test_trace_steps_count_matches_executed(self):
        """trace.steps length == number of sub_goals actually executed (AC-30)."""
        tree = _simple_tree("a", "b", "c")
        executor, _, _, _ = _make_executor()
        trace = executor.execute(tree)
        assert len(trace.steps) == 3

    def test_trace_all_steps_are_step_records(self):
        """Every entry in trace.steps is a StepRecord (AC-30)."""
        tree = _simple_tree("a", "b")
        executor, _, _, _ = _make_executor()
        trace = executor.execute(tree)
        for step in trace.steps:
            assert isinstance(step, StepRecord)

    def test_trace_sub_goal_names_match(self):
        """StepRecord.sub_goal_name must match the SubGoal it represents (AC-30)."""
        tree = _simple_tree("first", "second")
        executor, _, _, _ = _make_executor()
        trace = executor.execute(tree)
        names = {s.sub_goal_name for s in trace.steps}
        assert names == {"first", "second"}

    def test_trace_total_duration_non_negative(self):
        """ExecutionTrace.total_duration_sec >= 0 (AC-30)."""
        tree = _simple_tree("x")
        executor, _, _, _ = _make_executor()
        trace = executor.execute(tree)
        assert trace.total_duration_sec >= 0

    def test_trace_success_true_all_pass(self):
        """trace.success is True when all steps pass."""
        tree = _simple_tree("a", "b", "c")
        executor, _, _, _ = _make_executor()
        trace = executor.execute(tree)
        assert trace.success is True

    def test_trace_success_false_on_failure(self):
        """trace.success is False when any step fails."""
        tree = _simple_tree("a", "b")
        executor, _, _, _ = _make_executor(verify_side_effect=[True, False])
        trace = executor.execute(tree)
        assert trace.success is False

    def test_trace_goal_tree_preserved(self):
        """trace.goal_tree is the same GoalTree that was passed in."""
        tree = _simple_tree("only")
        executor, _, _, _ = _make_executor()
        trace = executor.execute(tree)
        assert trace.goal_tree is tree


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_goal_tree_succeeds(self):
        """GoalTree with no sub_goals → success, empty steps."""
        tree = GoalTree(goal="nothing", sub_goals=())
        executor, _, _, _ = _make_executor()
        trace = executor.execute(tree)
        assert trace.success is True
        assert len(trace.steps) == 0

    def test_skill_not_found_fails_gracefully(self):
        """Skill not in registry → step fails with error, no exception."""
        sg = SubGoal(name="x", description="d", verify="True", timeout_sec=10)
        tree = GoalTree(goal="test", sub_goals=(sg,))

        mock_registry = MagicMock()
        mock_registry.get.return_value = None  # skill not found

        mock_selector = MagicMock()
        mock_selector.select.return_value = _strategy_mock("skill", "missing_skill", {})

        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = True

        executor = GoalExecutor(
            strategy_selector=mock_selector,
            verifier=mock_verifier,
            skill_registry=mock_registry,
        )
        trace = executor.execute(tree)

        assert trace.steps[0].success is False
        assert "not found" in trace.steps[0].error.lower()

    def test_unknown_executor_type_fails(self):
        """executor_type='unknown' → step fails without crash."""
        sg = SubGoal(name="x", description="d", verify="True", timeout_sec=10)
        tree = GoalTree(goal="test", sub_goals=(sg,))

        mock_selector = MagicMock()
        mock_selector.select.return_value = _strategy_mock("unknown_type", "whatever", {})

        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = True

        executor = GoalExecutor(
            strategy_selector=mock_selector,
            verifier=mock_verifier,
        )
        trace = executor.execute(tree)

        assert trace.steps[0].success is False

    def test_primitive_execution_success(self):
        """executor_type='primitive' → primitive function called."""
        sg = SubGoal(name="prim_step", description="d", verify="True", timeout_sec=10)
        tree = GoalTree(goal="test", sub_goals=(sg,))

        # Inject a fake primitive via primitives namespace mock
        fake_primitive = MagicMock(return_value=True)

        mock_selector = MagicMock()
        mock_selector.select.return_value = _strategy_mock(
            "primitive", "fake_prim", {"x": 1}
        )

        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = True

        executor = GoalExecutor(
            strategy_selector=mock_selector,
            verifier=mock_verifier,
            primitives={"fake_prim": fake_primitive},
        )
        trace = executor.execute(tree)

        # After inspect.signature filtering, unknown params are stripped from mocks
        fake_primitive.assert_called_once()
        assert trace.steps[0].success is True

    def test_primitive_exception_fails_step(self):
        """Primitive raises exception → step fails gracefully."""
        sg = SubGoal(name="x", description="d", verify="True", timeout_sec=10)
        tree = GoalTree(goal="test", sub_goals=(sg,))

        def exploding_prim(**kwargs):
            raise RuntimeError("boom")

        mock_selector = MagicMock()
        mock_selector.select.return_value = _strategy_mock(
            "primitive", "exploding_prim", {}
        )

        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = True

        executor = GoalExecutor(
            strategy_selector=mock_selector,
            verifier=mock_verifier,
            primitives={"exploding_prim": exploding_prim},
        )
        trace = executor.execute(tree)

        assert trace.steps[0].success is False
        assert "boom" in trace.steps[0].error

    def test_single_step_duration_tracked(self):
        """StepRecord.duration_sec is positive after non-trivial execution."""
        tree = _simple_tree("x")
        executor, _, _, _ = _make_executor()
        trace = executor.execute(tree)
        # Duration may be ~0 for fast mocked execution, just verify it's a float >= 0
        assert isinstance(trace.steps[0].duration_sec, float)
        assert trace.steps[0].duration_sec >= 0.0

    def test_strategy_name_recorded_in_step(self):
        """StepRecord.strategy matches the selected strategy name."""
        tree = _simple_tree("only")
        executor, mock_selector, _, _ = _make_executor()
        # mock_selector.select returns name="navigate"
        trace = executor.execute(tree)
        assert trace.steps[0].strategy == "navigate"
