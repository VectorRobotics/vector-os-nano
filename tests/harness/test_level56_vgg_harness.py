"""Level 56 — VGGHarness feedback loop

TDD tests for VGGHarness: 3-layer retry/recovery around GoalDecomposer +
GoalExecutor.

Layer 1 — step-level strategy retry
Layer 2 — tree continues past failed steps, FailureRecord created
Layer 3 — pipeline-level re-plan with failure context injected

Acceptance Criteria: ~20 tests covering all three layers, config, callbacks,
edge cases, and integration.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)
from vector_os_nano.vcli.cognitive.vgg_harness import (
    FailureRecord,
    HarnessConfig,
    VGGHarness,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tree(n_steps: int = 1) -> GoalTree:
    sgs = tuple(
        SubGoal(
            name=f"step_{i}",
            description=f"step {i}",
            verify="True",
        )
        for i in range(n_steps)
    )
    return GoalTree(goal="test task", sub_goals=sgs)


def _make_step_record(name: str = "step_0", success: bool = True) -> StepRecord:
    return StepRecord(
        sub_goal_name=name,
        strategy="test_strategy",
        success=success,
        verify_result=success,
        duration_sec=0.01,
        error="" if success else "mock_error",
    )


def _make_trace(tree: GoalTree, success: bool = True) -> ExecutionTrace:
    steps = tuple(
        StepRecord(
            sub_goal_name=sg.name,
            strategy="test",
            success=success,
            verify_result=success,
            duration_sec=0.01,
        )
        for sg in tree.sub_goals
    )
    return ExecutionTrace(
        goal_tree=tree,
        steps=steps,
        success=success,
        total_duration_sec=0.1,
    )


def _make_harness(
    decomposer: Any | None = None,
    executor: Any | None = None,
    config: HarnessConfig | None = None,
    on_step: Any = None,
    on_replan: Any = None,
) -> VGGHarness:
    """Build a VGGHarness with mocked dependencies if not provided."""
    if decomposer is None:
        decomposer = MagicMock()
        tree = _make_tree(1)
        decomposer.decompose.return_value = tree
    if executor is None:
        executor = MagicMock()
        executor._stats = None
        tree = _make_tree(1)
        executor._topological_sort.return_value = list(tree.sub_goals)
        good_step = _make_step_record("step_0", success=True)
        executor._execute_sub_goal.return_value = good_step
    return VGGHarness(
        decomposer=decomposer,
        executor=executor,
        config=config or HarnessConfig(),
        on_step=on_step,
        on_replan=on_replan,
    )


# ---------------------------------------------------------------------------
# Layer 1: Step-level retry
# ---------------------------------------------------------------------------


class TestStepLevelRetry:
    """Layer 1: VGGHarness retries a failing step with alternative strategy."""

    def test_step_retry_succeeds_on_second_attempt(self) -> None:
        """Step fails once, succeeds on retry — harness returns success trace."""
        tree = _make_tree(1)

        fail_step = _make_step_record("step_0", success=False)
        success_step = _make_step_record("step_0", success=True)

        executor = MagicMock()
        executor._stats = None
        executor._topological_sort.return_value = list(tree.sub_goals)
        # First call fails, second call succeeds
        executor._execute_sub_goal.side_effect = [fail_step, success_step]

        decomposer = MagicMock()
        decomposer.decompose.return_value = tree

        harness = VGGHarness(
            decomposer=decomposer,
            executor=executor,
            config=HarnessConfig(max_step_retries=1, max_redecompose=0, max_pipeline_retries=0),
        )
        trace = harness.run("test task", "world ctx")

        assert trace.success is True
        assert executor._execute_sub_goal.call_count == 2

    def test_step_retry_exhausted(self) -> None:
        """Step fails on all retry attempts → step marked failed, trace fails."""
        tree = _make_tree(1)
        fail_step = _make_step_record("step_0", success=False)

        executor = MagicMock()
        executor._stats = None
        executor._topological_sort.return_value = list(tree.sub_goals)
        # Always fail
        executor._execute_sub_goal.return_value = fail_step

        decomposer = MagicMock()
        decomposer.decompose.return_value = tree

        harness = VGGHarness(
            decomposer=decomposer,
            executor=executor,
            config=HarnessConfig(max_step_retries=2, max_redecompose=0, max_pipeline_retries=0),
        )
        trace = harness.run("test task", "world ctx")

        assert trace.success is False
        # 1 initial attempt + 2 retries = 3 calls
        assert executor._execute_sub_goal.call_count == 3

    def test_step_retry_max_config_zero_means_no_retry(self) -> None:
        """max_step_retries=0 means exactly one attempt, no retries."""
        tree = _make_tree(1)
        fail_step = _make_step_record("step_0", success=False)

        executor = MagicMock()
        executor._stats = None
        executor._topological_sort.return_value = list(tree.sub_goals)
        executor._execute_sub_goal.return_value = fail_step

        decomposer = MagicMock()
        decomposer.decompose.return_value = tree

        harness = VGGHarness(
            decomposer=decomposer,
            executor=executor,
            config=HarnessConfig(max_step_retries=0, max_redecompose=0, max_pipeline_retries=0),
        )
        trace = harness.run("test task", "world ctx")

        assert trace.success is False
        assert executor._execute_sub_goal.call_count == 1

    def test_step_retry_passes_cleared_strategy_on_retry(self) -> None:
        """On retry the SubGoal passed to _execute_sub_goal has strategy=''."""
        tree = _make_tree(1)
        original_sg = tree.sub_goals[0]
        # Give original a strategy name so we can detect the cleared version
        sg_with_strategy = SubGoal(
            name="step_0",
            description="step 0",
            verify="True",
            strategy="primary_strategy",
        )
        tree_with_strategy = GoalTree(goal="test task", sub_goals=(sg_with_strategy,))

        fail_step = _make_step_record("step_0", success=False)
        success_step = _make_step_record("step_0", success=True)

        executor = MagicMock()
        executor._stats = None
        executor._topological_sort.return_value = [sg_with_strategy]
        executor._execute_sub_goal.side_effect = [fail_step, success_step]

        decomposer = MagicMock()
        decomposer.decompose.return_value = tree_with_strategy

        harness = VGGHarness(
            decomposer=decomposer,
            executor=executor,
            config=HarnessConfig(max_step_retries=1, max_redecompose=0, max_pipeline_retries=0),
        )
        harness.run("test task", "world ctx")

        calls = executor._execute_sub_goal.call_args_list
        # First call uses original sub_goal (strategy="primary_strategy")
        assert calls[0][0][0].strategy == "primary_strategy"
        # Second call uses cleared strategy
        assert calls[1][0][0].strategy == ""


# ---------------------------------------------------------------------------
# Layer 2: Tree continues after step failure
# ---------------------------------------------------------------------------


class TestTreeContinuesAfterStepFailure:
    """Layer 2: A failed step does not abort remaining steps."""

    def test_tree_continues_after_step_failure(self) -> None:
        """3-step tree: step_0 fails, steps 1+2 still execute."""
        tree = _make_tree(3)
        step0_fail = _make_step_record("step_0", success=False)
        step1_ok = _make_step_record("step_1", success=True)
        step2_ok = _make_step_record("step_2", success=True)

        executor = MagicMock()
        executor._stats = None
        executor._topological_sort.return_value = list(tree.sub_goals)
        # step_0 always fails (max_step_retries calls), then steps 1+2 succeed
        executor._execute_sub_goal.side_effect = [
            step0_fail,  # attempt 0 for step_0
            step1_ok,
            step2_ok,
        ]

        decomposer = MagicMock()
        decomposer.decompose.return_value = tree

        harness = VGGHarness(
            decomposer=decomposer,
            executor=executor,
            config=HarnessConfig(max_step_retries=0, max_redecompose=0, max_pipeline_retries=0),
        )
        trace = harness.run("test task", "world ctx")

        assert executor._execute_sub_goal.call_count == 3
        assert trace.success is False  # overall fail because step_0 failed

    def test_failure_recorded_for_feedback(self) -> None:
        """FailureRecord is appended to failures list when a step fails."""
        tree = _make_tree(1)
        fail_step = _make_step_record("step_0", success=False)
        fail_step = StepRecord(
            sub_goal_name="step_0",
            strategy="bad_strategy",
            success=False,
            verify_result=False,
            duration_sec=0.01,
            error="something went wrong",
        )

        executor = MagicMock()
        executor._stats = None
        executor._topological_sort.return_value = list(tree.sub_goals)
        executor._execute_sub_goal.return_value = fail_step

        decomposer = MagicMock()
        decomposer.decompose.return_value = tree

        harness = VGGHarness(
            decomposer=decomposer,
            executor=executor,
            config=HarnessConfig(max_step_retries=0, max_redecompose=0, max_pipeline_retries=0),
        )
        trace = harness.run("test task", "world ctx")

        # Trace itself is a failure
        assert trace.success is False
        # Internal failures list is not directly exposed, but we can verify
        # via pipeline retry path by checking decompose gets failure context
        # when max_pipeline_retries > 0.

    def test_failure_record_fields(self) -> None:
        """FailureRecord frozen dataclass has expected fields."""
        rec = FailureRecord(
            sub_goal_name="step_0",
            strategy_tried="navigate",
            error="timeout",
            step_index=0,
        )
        assert rec.sub_goal_name == "step_0"
        assert rec.strategy_tried == "navigate"
        assert rec.error == "timeout"
        assert rec.step_index == 0

    def test_all_steps_run_regardless_of_failure(self) -> None:
        """With 2-step tree where both fail, both are still attempted."""
        tree = _make_tree(2)
        fail_step_0 = _make_step_record("step_0", success=False)
        fail_step_1 = _make_step_record("step_1", success=False)

        executor = MagicMock()
        executor._stats = None
        executor._topological_sort.return_value = list(tree.sub_goals)
        executor._execute_sub_goal.side_effect = [fail_step_0, fail_step_1]

        decomposer = MagicMock()
        decomposer.decompose.return_value = tree

        harness = VGGHarness(
            decomposer=decomposer,
            executor=executor,
            config=HarnessConfig(max_step_retries=0, max_redecompose=0, max_pipeline_retries=0),
        )
        trace = harness.run("test task", "world ctx")

        assert executor._execute_sub_goal.call_count == 2
        assert trace.success is False


# ---------------------------------------------------------------------------
# Layer 3: Pipeline retry
# ---------------------------------------------------------------------------


class TestPipelineRetry:
    """Layer 3: On full-tree failure, harness re-decomposes with failure context."""

    def test_pipeline_retry_redecomposes(self) -> None:
        """First tree fails → decomposer called again → second tree succeeds."""
        tree1 = _make_tree(1)
        tree2 = _make_tree(1)
        fail_step = _make_step_record("step_0", success=False)
        success_step = _make_step_record("step_0", success=True)

        decomposer = MagicMock()
        decomposer.decompose.side_effect = [tree1, tree2]

        executor = MagicMock()
        executor._stats = None
        executor._topological_sort.side_effect = [
            list(tree1.sub_goals),
            list(tree2.sub_goals),
        ]
        executor._execute_sub_goal.side_effect = [fail_step, success_step]

        harness = VGGHarness(
            decomposer=decomposer,
            executor=executor,
            config=HarnessConfig(max_step_retries=0, max_redecompose=0, max_pipeline_retries=1),
        )
        trace = harness.run("test task", "world ctx")

        assert trace.success is True
        assert decomposer.decompose.call_count == 2

    def test_pipeline_retry_max_config_zero_means_no_replan(self) -> None:
        """max_pipeline_retries=0 means decomposer is only called once."""
        tree = _make_tree(1)
        fail_step = _make_step_record("step_0", success=False)

        decomposer = MagicMock()
        decomposer.decompose.return_value = tree

        executor = MagicMock()
        executor._stats = None
        executor._topological_sort.return_value = list(tree.sub_goals)
        executor._execute_sub_goal.return_value = fail_step

        harness = VGGHarness(
            decomposer=decomposer,
            executor=executor,
            config=HarnessConfig(max_step_retries=0, max_redecompose=0, max_pipeline_retries=0),
        )
        trace = harness.run("test task", "world ctx")

        assert trace.success is False
        assert decomposer.decompose.call_count == 1

    def test_failure_context_injected_into_redecompose(self) -> None:
        """Re-decompose receives enriched context containing previous failure info."""
        tree1 = _make_tree(1)
        tree2 = _make_tree(1)

        fail_step = StepRecord(
            sub_goal_name="step_0",
            strategy="bad_strategy",
            success=False,
            verify_result=False,
            duration_sec=0.01,
            error="robot_fell",
        )
        success_step = _make_step_record("step_0", success=True)

        decomposer = MagicMock()
        decomposer.decompose.side_effect = [tree1, tree2]

        executor = MagicMock()
        executor._stats = None
        executor._topological_sort.side_effect = [
            list(tree1.sub_goals),
            list(tree2.sub_goals),
        ]
        executor._execute_sub_goal.side_effect = [fail_step, success_step]

        harness = VGGHarness(
            decomposer=decomposer,
            executor=executor,
            config=HarnessConfig(max_step_retries=0, max_redecompose=0, max_pipeline_retries=1),
        )
        harness.run("test task", "base_world_context")

        # Second decompose call must include failure context in the world string
        second_call_context = decomposer.decompose.call_args_list[1][0][1]
        assert "Previous failures" in second_call_context
        assert "bad_strategy" in second_call_context
        assert "robot_fell" in second_call_context

    def test_pipeline_retry_only_last_5_failures_in_context(self) -> None:
        """Only the last 5 FailureRecords are injected, not all of them."""
        # Build a 6-step tree where all steps fail
        tree1 = _make_tree(6)
        tree2 = _make_tree(1)

        fail_steps = [
            StepRecord(
                sub_goal_name=f"step_{i}",
                strategy=f"strategy_{i}",
                success=False,
                verify_result=False,
                duration_sec=0.01,
                error=f"error_{i}",
            )
            for i in range(6)
        ]
        success_step = _make_step_record("step_0", success=True)

        decomposer = MagicMock()
        decomposer.decompose.side_effect = [tree1, tree2]

        executor = MagicMock()
        executor._stats = None
        executor._topological_sort.side_effect = [
            list(tree1.sub_goals),
            list(tree2.sub_goals),
        ]
        executor._execute_sub_goal.side_effect = fail_steps + [success_step]

        harness = VGGHarness(
            decomposer=decomposer,
            executor=executor,
            config=HarnessConfig(max_step_retries=0, max_redecompose=0, max_pipeline_retries=1),
        )
        harness.run("test task", "base ctx")

        second_call_context = decomposer.decompose.call_args_list[1][0][1]
        # step_0..step_4 are the first 5; step_5 should be included as it's one of last 5
        # Actually last 5 of 6 failures = steps 1..5
        assert "strategy_0" not in second_call_context
        assert "strategy_1" in second_call_context


# ---------------------------------------------------------------------------
# Full integration
# ---------------------------------------------------------------------------


class TestHarnessIntegration:
    """Full integration tests — happy path, best-trace, callbacks."""

    def test_harness_success_no_retry(self) -> None:
        """Happy path: all steps succeed on first try, no retry machinery invoked."""
        tree = _make_tree(2)
        step0 = _make_step_record("step_0", success=True)
        step1 = _make_step_record("step_1", success=True)

        executor = MagicMock()
        executor._stats = None
        executor._topological_sort.return_value = list(tree.sub_goals)
        executor._execute_sub_goal.side_effect = [step0, step1]

        decomposer = MagicMock()
        decomposer.decompose.return_value = tree

        harness = VGGHarness(
            decomposer=decomposer,
            executor=executor,
            config=HarnessConfig(max_step_retries=1, max_redecompose=1, max_pipeline_retries=1),
        )
        trace = harness.run("test task", "world ctx")

        assert trace.success is True
        assert executor._execute_sub_goal.call_count == 2
        assert decomposer.decompose.call_count == 1

    def test_harness_returns_best_trace(self) -> None:
        """Returns successful trace when pipeline retries succeed on second attempt."""
        tree1 = _make_tree(1)
        tree2 = _make_tree(1)
        fail_step = _make_step_record("step_0", success=False)
        success_step = _make_step_record("step_0", success=True)

        decomposer = MagicMock()
        decomposer.decompose.side_effect = [tree1, tree2]

        executor = MagicMock()
        executor._stats = None
        executor._topological_sort.side_effect = [
            list(tree1.sub_goals),
            list(tree2.sub_goals),
        ]
        executor._execute_sub_goal.side_effect = [fail_step, success_step]

        harness = VGGHarness(
            decomposer=decomposer,
            executor=executor,
            config=HarnessConfig(max_step_retries=0, max_redecompose=0, max_pipeline_retries=1),
        )
        trace = harness.run("test task", "world ctx")

        assert trace.success is True

    def test_harness_on_step_callback(self) -> None:
        """on_step callback fires for each executed step."""
        tree = _make_tree(3)
        steps = [_make_step_record(f"step_{i}", success=True) for i in range(3)]

        executor = MagicMock()
        executor._stats = None
        executor._topological_sort.return_value = list(tree.sub_goals)
        executor._execute_sub_goal.side_effect = steps

        decomposer = MagicMock()
        decomposer.decompose.return_value = tree

        fired: list[StepRecord] = []
        harness = VGGHarness(
            decomposer=decomposer,
            executor=executor,
            config=HarnessConfig(max_step_retries=0, max_redecompose=0, max_pipeline_retries=0),
            on_step=fired.append,
        )
        harness.run("test task", "world ctx")

        assert len(fired) == 3
        assert fired[0].sub_goal_name == "step_0"
        assert fired[2].sub_goal_name == "step_2"

    def test_harness_on_replan_callback(self) -> None:
        """on_replan callback fires when pipeline retries."""
        tree1 = _make_tree(1)
        tree2 = _make_tree(1)
        fail_step = _make_step_record("step_0", success=False)
        success_step = _make_step_record("step_0", success=True)

        decomposer = MagicMock()
        decomposer.decompose.side_effect = [tree1, tree2]

        executor = MagicMock()
        executor._stats = None
        executor._topological_sort.side_effect = [
            list(tree1.sub_goals),
            list(tree2.sub_goals),
        ]
        executor._execute_sub_goal.side_effect = [fail_step, success_step]

        replan_msgs: list[str] = []
        harness = VGGHarness(
            decomposer=decomposer,
            executor=executor,
            config=HarnessConfig(max_step_retries=0, max_redecompose=0, max_pipeline_retries=1),
            on_replan=replan_msgs.append,
        )
        harness.run("test task", "world ctx")

        assert len(replan_msgs) == 1
        assert "Re-planning" in replan_msgs[0]

    def test_harness_on_step_exception_does_not_abort(self) -> None:
        """on_step callback that raises must not abort execution."""
        tree = _make_tree(2)
        step0 = _make_step_record("step_0", success=True)
        step1 = _make_step_record("step_1", success=True)

        executor = MagicMock()
        executor._stats = None
        executor._topological_sort.return_value = list(tree.sub_goals)
        executor._execute_sub_goal.side_effect = [step0, step1]

        decomposer = MagicMock()
        decomposer.decompose.return_value = tree

        def bad_callback(step: StepRecord) -> None:
            raise RuntimeError("callback blew up")

        harness = VGGHarness(
            decomposer=decomposer,
            executor=executor,
            config=HarnessConfig(max_step_retries=0, max_redecompose=0, max_pipeline_retries=0),
            on_step=bad_callback,
        )
        trace = harness.run("test task", "world ctx")

        assert trace.success is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestHarnessEdgeCases:
    """Edge cases: pre-decomposed tree, decompose failure, config defaults."""

    def test_harness_with_provided_goal_tree_skips_decomposition(self) -> None:
        """Providing a GoalTree as goal_tree= skips first decomposition call."""
        tree = _make_tree(1)
        success_step = _make_step_record("step_0", success=True)

        executor = MagicMock()
        executor._stats = None
        executor._topological_sort.return_value = list(tree.sub_goals)
        executor._execute_sub_goal.return_value = success_step

        decomposer = MagicMock()
        decomposer.decompose.return_value = tree

        harness = VGGHarness(
            decomposer=decomposer,
            executor=executor,
            config=HarnessConfig(max_step_retries=0, max_redecompose=0, max_pipeline_retries=0),
        )
        trace = harness.run("test task", "world ctx", goal_tree=tree)

        assert trace.success is True
        # decompose was never called because we provided the tree
        decomposer.decompose.assert_not_called()

    def test_harness_decompose_fails_returns_empty_trace(self) -> None:
        """If decomposer raises, harness returns a failed trace with no steps."""
        decomposer = MagicMock()
        decomposer.decompose.side_effect = RuntimeError("LLM unavailable")

        executor = MagicMock()
        executor._stats = None

        harness = VGGHarness(
            decomposer=decomposer,
            executor=executor,
            config=HarnessConfig(max_step_retries=0, max_redecompose=0, max_pipeline_retries=0),
        )
        trace = harness.run("test task", "world ctx")

        assert trace.success is False
        assert len(trace.steps) == 0

    def test_harness_decompose_returns_none_returns_empty_trace(self) -> None:
        """If decomposer returns None, harness returns a failed trace."""
        decomposer = MagicMock()
        decomposer.decompose.return_value = None

        executor = MagicMock()
        executor._stats = None

        harness = VGGHarness(
            decomposer=decomposer,
            executor=executor,
            config=HarnessConfig(max_step_retries=0, max_redecompose=0, max_pipeline_retries=0),
        )
        trace = harness.run("test task", "world ctx")

        assert trace.success is False

    def test_harness_config_defaults(self) -> None:
        """Default HarnessConfig has reasonable positive values."""
        cfg = HarnessConfig()
        assert cfg.max_step_retries >= 1
        assert cfg.max_redecompose >= 0
        assert cfg.max_pipeline_retries >= 1

    def test_harness_config_frozen(self) -> None:
        """HarnessConfig is immutable (frozen dataclass)."""
        cfg = HarnessConfig(max_step_retries=3)
        with pytest.raises((AttributeError, TypeError)):
            cfg.max_step_retries = 99  # type: ignore[misc]

    def test_harness_empty_tree_returns_success(self) -> None:
        """An empty GoalTree (no sub_goals) results in a successful trace."""
        empty_tree = GoalTree(goal="nothing to do", sub_goals=())

        executor = MagicMock()
        executor._stats = None
        executor._topological_sort.return_value = []

        decomposer = MagicMock()
        decomposer.decompose.return_value = empty_tree

        harness = VGGHarness(
            decomposer=decomposer,
            executor=executor,
            config=HarnessConfig(max_step_retries=0, max_redecompose=0, max_pipeline_retries=0),
        )
        trace = harness.run("nothing to do", "world ctx")

        assert trace.success is True
        assert len(trace.steps) == 0

    def test_harness_stats_recording_called_per_step(self) -> None:
        """Executor._stats.record is invoked for each step."""
        tree = _make_tree(2)
        step0 = _make_step_record("step_0", success=True)
        step1 = _make_step_record("step_1", success=True)

        mock_stats = MagicMock()

        executor = MagicMock()
        executor._stats = mock_stats
        executor._topological_sort.return_value = list(tree.sub_goals)
        executor._execute_sub_goal.side_effect = [step0, step1]

        decomposer = MagicMock()
        decomposer.decompose.return_value = tree

        harness = VGGHarness(
            decomposer=decomposer,
            executor=executor,
            config=HarnessConfig(max_step_retries=0, max_redecompose=0, max_pipeline_retries=0),
        )
        harness.run("test task", "world ctx")

        assert mock_stats.record.call_count == 2
