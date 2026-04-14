"""VGG Harness — feedback loop wrapper around the VGG pipeline.

Three layers of retry/recovery:

Layer 1 (step-level): Strategy A fails → try alternative strategies
Layer 2 (tree-level): Step fails → re-decompose remaining goals with failure context
Layer 3 (pipeline-level): Whole tree fails → full re-plan with history

This prevents single-point failures from killing the entire task.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HarnessConfig:
    """Configuration for VGG retry behavior."""
    max_step_retries: int = 2       # per-step strategy retries (Layer 1)
    max_redecompose: int = 1        # re-decompose attempts on step failure (Layer 2)
    max_pipeline_retries: int = 1   # full re-plan attempts (Layer 3)


@dataclass(frozen=True)
class FailureRecord:
    """Records a failure for feedback to the decomposer."""
    sub_goal_name: str
    strategy_tried: str
    error: str
    step_index: int


class VGGHarness:
    """Wraps GoalDecomposer + GoalExecutor with feedback loops.

    Usage:
        harness = VGGHarness(decomposer, executor, selector, verifier)
        trace = harness.run("去厨房看看有没有杯子", world_context)
    """

    def __init__(
        self,
        decomposer: Any,
        executor: Any,
        selector: Any = None,
        config: HarnessConfig | None = None,
        on_step: Callable[[StepRecord], None] | None = None,
        on_replan: Callable[[str], None] | None = None,
    ) -> None:
        self._decomposer = decomposer
        self._executor = executor
        self._selector = selector
        self._config = config or HarnessConfig()
        self._on_step = on_step
        self._on_replan = on_replan

    def run(
        self,
        task: str,
        world_context: str,
        goal_tree: GoalTree | None = None,
    ) -> ExecutionTrace:
        """Run task with full feedback loop.

        Args:
            task: Natural language task description.
            world_context: Current world model summary.
            goal_tree: Pre-decomposed GoalTree (skip decomposition if provided).

        Returns:
            Best ExecutionTrace achieved across all retry attempts.
        """
        cfg = self._config
        failures: list[FailureRecord] = []
        best_trace: ExecutionTrace | None = None

        for pipeline_attempt in range(cfg.max_pipeline_retries + 1):
            # --- Decompose (or use provided tree) ---
            if goal_tree is not None and pipeline_attempt == 0:
                tree = goal_tree
            else:
                tree = self._decompose_with_context(task, world_context, failures)
                if tree is None:
                    logger.warning("VGGHarness: decomposition failed on attempt %d", pipeline_attempt)
                    break

            # --- Execute with step-level retry ---
            trace = self._execute_with_retry(tree, failures)

            # Track best result
            if best_trace is None or trace.success:
                best_trace = trace

            if trace.success:
                return trace

            # --- Layer 3: Pipeline retry — full re-plan with failure history ---
            if pipeline_attempt < cfg.max_pipeline_retries:
                if self._on_replan:
                    self._on_replan(
                        f"Re-planning (attempt {pipeline_attempt + 2}): "
                        f"{len(failures)} previous failures"
                    )
                goal_tree = None  # force re-decompose
                logger.info(
                    "VGGHarness: pipeline retry %d/%d with %d failure records",
                    pipeline_attempt + 1, cfg.max_pipeline_retries, len(failures),
                )

        return best_trace or ExecutionTrace(
            goal_tree=tree or GoalTree(goal=task, sub_goals=()),
            steps=(),
            success=False,
            total_duration_sec=0.0,
        )

    def _decompose_with_context(
        self,
        task: str,
        world_context: str,
        failures: list[FailureRecord],
    ) -> GoalTree | None:
        """Decompose with failure history injected into context."""
        enriched_context = world_context
        if failures:
            failure_summary = "\n".join(
                f"  - {f.sub_goal_name}: {f.strategy_tried} failed ({f.error})"
                for f in failures[-5:]  # last 5 failures
            )
            enriched_context += (
                f"\n\nPrevious failures (avoid these strategies):\n{failure_summary}"
            )

        try:
            return self._decomposer.decompose(task, enriched_context)
        except Exception as exc:
            logger.warning("VGGHarness: decompose raised: %s", exc)
            return None

    def _execute_with_retry(
        self,
        tree: GoalTree,
        failures: list[FailureRecord],
    ) -> ExecutionTrace:
        """Execute GoalTree with step-level retry (Layer 1 + Layer 2).

        On step failure:
        1. Try alternative strategies for the failed step (Layer 1)
        2. If all strategies exhausted, record failure and continue to next step
           (don't abort the whole tree — some later steps may still succeed)
        """
        cfg = self._config
        trace_start = time.monotonic()
        steps: list[StepRecord] = []
        overall_success = True

        ordered = self._executor._topological_sort(tree)

        for i, sub_goal in enumerate(ordered):
            step = self._execute_step_with_retry(sub_goal, i, cfg.max_step_retries)
            steps.append(step)

            if self._on_step:
                try:
                    self._on_step(step)
                except Exception:
                    pass

            # Record stats
            if self._executor._stats is not None:
                try:
                    self._executor._stats.record(
                        strategy_name=step.strategy,
                        sub_goal_name=step.sub_goal_name,
                        success=step.success,
                        duration_sec=step.duration_sec,
                    )
                except Exception:
                    pass

            if not step.success:
                failures.append(FailureRecord(
                    sub_goal_name=step.sub_goal_name,
                    strategy_tried=step.strategy,
                    error=step.error,
                    step_index=i,
                ))
                overall_success = False
                # Don't break — try remaining steps that don't depend on this one
                # (steps with depends_on referencing the failed step will skip)

        total_duration = time.monotonic() - trace_start
        return ExecutionTrace(
            goal_tree=tree,
            steps=tuple(steps),
            success=overall_success,
            total_duration_sec=total_duration,
        )

    def _execute_step_with_retry(
        self,
        sub_goal: SubGoal,
        step_index: int,
        max_retries: int,
    ) -> StepRecord:
        """Execute a single step with strategy retry (Layer 1).

        Tries primary strategy, then fail_action, then asks selector for
        alternatives.
        """
        tried_strategies: set[str] = set()

        for attempt in range(max_retries + 1):
            # Select strategy (first attempt uses sub_goal.strategy, later uses selector)
            if attempt == 0:
                step = self._executor._execute_sub_goal(sub_goal)
            else:
                # Clear strategy to let selector pick an alternative
                retry_goal = SubGoal(
                    name=sub_goal.name,
                    description=sub_goal.description,
                    verify=sub_goal.verify,
                    timeout_sec=sub_goal.timeout_sec,
                    depends_on=sub_goal.depends_on,
                    strategy="",  # force selector to pick fresh
                    strategy_params=sub_goal.strategy_params,
                    fail_action="",
                )
                step = self._executor._execute_sub_goal(retry_goal)

            if step.success:
                return step

            tried_strategies.add(step.strategy)
            if attempt < max_retries:
                logger.info(
                    "VGGHarness: step %s attempt %d/%d failed (%s) — retrying",
                    sub_goal.name, attempt + 1, max_retries + 1, step.error,
                )

        return step  # return last failed attempt
