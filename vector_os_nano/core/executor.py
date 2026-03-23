"""Deterministic task execution engine for Vector OS Nano SDK.

The TaskExecutor runs a TaskPlan step-by-step in dependency order.
It checks preconditions, calls skills, updates the world model, and
checks postconditions — with no LLM calls. Failures are immediate.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

from vector_os_nano.core.types import ExecutionResult, StepTrace, TaskPlan, TaskStep

logger = logging.getLogger(__name__)


class TaskExecutor:
    """Deterministic task execution engine.

    Executes a TaskPlan step-by-step in topological dependency order.
    On precondition failure, skill failure, or postcondition failure,
    execution stops immediately and returns a failed ExecutionResult.
    """

    def execute(
        self,
        plan: TaskPlan,
        skill_registry: Any,  # SkillRegistry — avoids circular import concern
        context: Any,         # SkillContext
        on_step: Any = None,  # Optional callback(skill_name, step_idx, total) — before step
        on_step_done: Any = None,  # Optional callback(skill_name, success, duration, params) — after step
    ) -> ExecutionResult:
        """Execute task plan step by step.

        For each step in topological order:
        1. Check preconditions against world model
        2. Look up skill in registry
        3. Call skill.execute(params, context)
        4. Update world model with skill effects
        5. Check postconditions
        6. On any failure, return immediately with the failed step

        Args:
            plan: TaskPlan with steps and dependency info.
            skill_registry: SkillRegistry instance.
            context: SkillContext (arm, gripper, world_model, etc.)

        Returns:
            ExecutionResult with trace of all attempted steps.
        """
        if not plan.steps:
            return ExecutionResult(
                success=True,
                status="completed",
                steps_completed=0,
                steps_total=0,
                trace=[],
            )

        steps_total = len(plan.steps)
        ordered_steps = self._topological_sort(list(plan.steps))
        trace: list[StepTrace] = []
        steps_completed = 0

        for step_idx, step in enumerate(ordered_steps):
            step_start = time.monotonic()

            # --- 0. Notify caller ---
            if on_step is not None:
                try:
                    on_step(step.skill_name, step_idx, steps_total, step.parameters)
                except Exception:
                    pass

            # --- 1. Look up skill ---
            skill = skill_registry.get(step.skill_name)
            if skill is None:
                reason = f"Skill not found: {step.skill_name!r}"
                logger.error("[Executor] %s", reason)
                trace.append(StepTrace(
                    step_id=step.step_id,
                    skill_name=step.skill_name,
                    status="skill_not_found",
                    duration_sec=time.monotonic() - step_start,
                    error=reason,
                ))
                return ExecutionResult(
                    success=False,
                    status="failed",
                    steps_completed=steps_completed,
                    steps_total=steps_total,
                    failed_step=step,
                    failure_reason=reason,
                    trace=trace,
                )

            # --- 2. Check preconditions ---
            world_model = context.world_model
            for pred in step.preconditions:
                if not world_model.check_predicate(pred):
                    reason = f"Precondition failed: {pred!r} in step {step.step_id!r}"
                    logger.warning("[Executor] %s", reason)
                    trace.append(StepTrace(
                        step_id=step.step_id,
                        skill_name=step.skill_name,
                        status="precondition_failed",
                        duration_sec=time.monotonic() - step_start,
                        error=reason,
                    ))
                    return ExecutionResult(
                        success=False,
                        status="failed",
                        steps_completed=steps_completed,
                        steps_total=steps_total,
                        failed_step=step,
                        failure_reason=reason,
                        trace=trace,
                    )

            # --- 3. Execute skill ---
            try:
                skill_result = skill.execute(step.parameters, context)
            except Exception as exc:
                reason = f"Skill {step.skill_name!r} raised exception: {exc}"
                logger.error("[Executor] %s", reason, exc_info=True)
                trace.append(StepTrace(
                    step_id=step.step_id,
                    skill_name=step.skill_name,
                    status="execution_failed",
                    duration_sec=time.monotonic() - step_start,
                    error=reason,
                ))
                return ExecutionResult(
                    success=False,
                    status="failed",
                    steps_completed=steps_completed,
                    steps_total=steps_total,
                    failed_step=step,
                    failure_reason=reason,
                    trace=trace,
                )

            duration = time.monotonic() - step_start

            if not skill_result.success:
                reason = (
                    skill_result.error_message
                    or f"Skill {step.skill_name!r} returned failure"
                )
                logger.warning("[Executor] Step %s failed: %s", step.step_id, reason)
                trace.append(StepTrace(
                    step_id=step.step_id,
                    skill_name=step.skill_name,
                    status="execution_failed",
                    duration_sec=duration,
                    error=reason,
                ))
                return ExecutionResult(
                    success=False,
                    status="failed",
                    steps_completed=steps_completed,
                    steps_total=steps_total,
                    failed_step=step,
                    failure_reason=reason,
                    trace=trace,
                )

            # --- 4. Apply world model effects ---
            world_model.apply_skill_effects(step.skill_name, step.parameters, skill_result)

            # --- 5. Check postconditions ---
            # Use SKILL-defined postconditions (authoritative), not LLM-generated ones.
            # LLM may invent predicates like "scan_complete" that don't exist.
            postconds = skill.postconditions if hasattr(skill, 'postconditions') else step.postconditions
            for pred in postconds:
                if not world_model.check_predicate(pred):
                    reason = f"Postcondition failed: {pred!r} in step {step.step_id!r}"
                    logger.warning("[Executor] %s", reason)
                    trace.append(StepTrace(
                        step_id=step.step_id,
                        skill_name=step.skill_name,
                        status="postcondition_failed",
                        duration_sec=duration,
                        error=reason,
                    ))
                    return ExecutionResult(
                        success=False,
                        status="failed",
                        steps_completed=steps_completed,
                        steps_total=steps_total,
                        failed_step=step,
                        failure_reason=reason,
                        trace=trace,
                    )

            trace.append(StepTrace(
                step_id=step.step_id,
                skill_name=step.skill_name,
                status="success",
                duration_sec=duration,
            ))
            steps_completed += 1
            logger.info(
                "[Executor] Step %s (%s) OK in %.3fs",
                step.step_id, step.skill_name, duration,
            )
            if on_step_done is not None:
                try:
                    on_step_done(step.skill_name, True, duration, step.parameters)
                except Exception:
                    pass

        return ExecutionResult(
            success=True,
            status="completed",
            steps_completed=steps_completed,
            steps_total=steps_total,
            trace=trace,
        )

    def _topological_sort(self, steps: list[TaskStep]) -> list[TaskStep]:
        """Sort steps by dependency order using Kahn's algorithm.

        Steps that have no dependencies come first; later steps follow
        after all their dependencies have been scheduled.

        Args:
            steps: list of TaskStep objects with depends_on references.

        Returns:
            Ordered list of TaskStep objects.

        Raises:
            ValueError: if a circular dependency is detected.
        """
        # Build lookup and in-degree map
        by_id: dict[str, TaskStep] = {s.step_id: s for s in steps}
        in_degree: dict[str, int] = defaultdict(int)
        dependents: dict[str, list[str]] = defaultdict(list)

        for step in steps:
            in_degree.setdefault(step.step_id, 0)
            for dep in step.depends_on:
                in_degree[step.step_id] += 1
                dependents[dep].append(step.step_id)

        # Start with steps that have no dependencies
        queue: list[str] = [s for s in by_id if in_degree[s] == 0]
        ordered: list[TaskStep] = []

        while queue:
            current_id = queue.pop(0)
            ordered.append(by_id[current_id])
            for dep_id in dependents[current_id]:
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    queue.append(dep_id)

        if len(ordered) != len(steps):
            raise ValueError(
                "Circular dependency detected in task plan. "
                f"Could not schedule: {set(by_id) - {s.step_id for s in ordered}}"
            )

        return ordered
