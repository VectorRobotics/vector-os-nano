"""GoalExecutor — executes GoalTrees by verifying each step and handling fallbacks.

Execution flow per sub_goal:
1. Select strategy via StrategySelector
2. Execute the selected strategy (skill or primitive)
3. Check elapsed time against timeout_sec
4. Verify success condition via GoalVerifier
5. On failure: attempt fail_action fallback, then re-verify
6. Record StepRecord; abort remaining goals on failure
"""
from __future__ import annotations

import inspect
import logging
import time
from collections import deque
from typing import Any, Callable

from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)

logger = logging.getLogger(__name__)


class GoalExecutor:
    """Executes a GoalTree, verifying each sub-goal and handling fallbacks."""

    def __init__(
        self,
        strategy_selector: Any,
        verifier: Any,
        skill_registry: Any = None,
        primitives: Any = None,
        build_context: Callable | None = None,
        stats: Any = None,
        visual_verifier_agent: Any = None,
    ) -> None:
        """Initialise the executor.

        Args:
            strategy_selector: StrategySelector — has .select(sub_goal) → StrategyResult.
            verifier: GoalVerifier — has .verify(expression) → bool.
            skill_registry: Optional SkillRegistry — has .get(name) → skill | None.
            primitives: Optional dict mapping primitive name → callable,
                        or any namespace with primitive functions.
            build_context: Optional callable that builds a SkillContext for skill execution.
            stats: Optional StrategyStats — records per-step outcomes for data-driven
                   strategy selection.  Auto-saved after each full execution.
            visual_verifier_agent: Optional agent ref for VLM-based visual verification
                                   fallback when primary verify fails on perception steps.
        """
        self._selector = strategy_selector
        self._verifier = verifier
        self._skill_registry = skill_registry
        self._primitives = primitives
        self._build_context = build_context
        self._stats = stats
        self._visual_verifier_agent = visual_verifier_agent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        goal_tree: GoalTree,
        on_step: Callable[[StepRecord], None] | None = None,
    ) -> ExecutionTrace:
        """Execute the goal tree step by step.

        Steps:
        1. Topological sort sub_goals by depends_on.
        2. For each sub_goal:
           a. Select strategy
           b. Execute strategy (with timeout tracking)
           c. Verify success condition
           d. On failure: try fail_action, re-verify
           e. Record StepRecord; fire on_step callback
        3. Abort on first failure; return ExecutionTrace.

        Args:
            goal_tree: The GoalTree to execute.
            on_step: Optional callback invoked after each sub_goal completes.

        Returns:
            ExecutionTrace capturing all steps and overall outcome.
        """
        trace_start = time.monotonic()
        ordered = self._topological_sort(goal_tree)
        steps: list[StepRecord] = []
        overall_success = True

        for sub_goal in ordered:
            # --- Abort check ---
            try:
                from vector_os_nano.vcli.cognitive.abort import is_abort_requested
                if is_abort_requested():
                    abort_step = StepRecord(
                        sub_goal_name=sub_goal.name,
                        strategy="",
                        success=False,
                        verify_result=False,
                        duration_sec=0.0,
                        error="aborted",
                        fallback_used=False,
                    )
                    steps.append(abort_step)
                    if on_step is not None:
                        try:
                            on_step(abort_step)
                        except Exception:
                            pass
                    overall_success = False
                    break
            except ImportError:
                pass

            step = self._execute_sub_goal(sub_goal)
            steps.append(step)

            # Record to stats if available
            if self._stats is not None:
                try:
                    self._stats.record(
                        strategy_name=step.strategy,
                        sub_goal_name=step.sub_goal_name,
                        success=step.success,
                        duration_sec=step.duration_sec,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GoalExecutor: stats.record raised: %s", exc)

            if on_step is not None:
                try:
                    on_step(step)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GoalExecutor: on_step callback raised: %s", exc)

            if not step.success:
                overall_success = False
                break  # abort remaining

        # Auto-save stats after full execution
        if self._stats is not None:
            try:
                self._stats.save()
            except Exception as exc:  # noqa: BLE001
                logger.warning("GoalExecutor: stats.save raised: %s", exc)

        total_duration = time.monotonic() - trace_start
        return ExecutionTrace(
            goal_tree=goal_tree,
            steps=tuple(steps),
            success=overall_success,
            total_duration_sec=total_duration,
        )

    # ------------------------------------------------------------------
    # Topological sort (Kahn's algorithm — BFS)
    # ------------------------------------------------------------------

    def _topological_sort(self, goal_tree: GoalTree) -> list[SubGoal]:
        """Return sub_goals in dependency order.

        Falls back to original order if a cycle is detected.
        """
        sub_goals = list(goal_tree.sub_goals)
        if not sub_goals:
            return sub_goals

        name_to_sg: dict[str, SubGoal] = {sg.name: sg for sg in sub_goals}

        # Build in-degree map and adjacency list
        in_degree: dict[str, int] = {sg.name: 0 for sg in sub_goals}
        adjacency: dict[str, list[str]] = {sg.name: [] for sg in sub_goals}

        for sg in sub_goals:
            for dep in sg.depends_on:
                if dep in name_to_sg:
                    in_degree[sg.name] += 1
                    adjacency[dep].append(sg.name)
                # Ignore deps referencing unknown names

        # BFS from nodes with in_degree == 0
        queue: deque[str] = deque(
            name for name, deg in in_degree.items() if deg == 0
        )
        # Preserve original relative order for determinism
        order_index = {sg.name: i for i, sg in enumerate(sub_goals)}
        sorted_names: list[str] = []

        while queue:
            # Pick next in original order among available nodes
            current = min(queue, key=lambda n: order_index[n])
            queue.remove(current)
            sorted_names.append(current)
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_names) != len(sub_goals):
            logger.warning(
                "GoalExecutor: cycle detected in sub_goal dependencies — "
                "executing in original order"
            )
            return sub_goals

        return [name_to_sg[name] for name in sorted_names]

    # ------------------------------------------------------------------
    # Sub-goal execution
    # ------------------------------------------------------------------

    def _execute_sub_goal(self, sub_goal: SubGoal) -> StepRecord:
        """Execute a single sub_goal and return its StepRecord."""
        step_start = time.monotonic()

        # Select strategy
        try:
            result = self._selector.select(sub_goal)
        except Exception as exc:  # noqa: BLE001
            logger.warning("GoalExecutor: selector.select raised: %s", exc)
            return self._make_step(sub_goal, result=None, success=False,
                                   verify_result=False, error=str(exc),
                                   start=step_start, fallback_used=False)

        strategy_name = self._extract_name(result)

        # Execute strategy
        exec_success, exec_error = self._execute_strategy(result)
        elapsed = time.monotonic() - step_start

        # Check timeout (takes priority over execution result)
        if elapsed > sub_goal.timeout_sec:
            error_msg = (
                f"timeout after {elapsed:.3f}s (limit {sub_goal.timeout_sec}s)"
            )
            logger.warning("GoalExecutor: %s — %s", sub_goal.name, error_msg)
            return StepRecord(
                sub_goal_name=sub_goal.name,
                strategy=strategy_name,
                success=False,
                verify_result=False,
                duration_sec=elapsed,
                error=error_msg,
                fallback_used=False,
            )

        # If execution itself failed (skill not found, unknown type, etc.),
        # mark the step failed immediately — no point verifying.
        if not exec_success:
            logger.warning(
                "GoalExecutor: execution failed for %s: %s", sub_goal.name, exec_error
            )
            return StepRecord(
                sub_goal_name=sub_goal.name,
                strategy=strategy_name,
                success=False,
                verify_result=False,
                duration_sec=time.monotonic() - step_start,
                error=exec_error,
                fallback_used=False,
            )

        # Verify
        verify_result = self._verifier.verify(sub_goal.verify)

        if verify_result:
            # Success path
            return StepRecord(
                sub_goal_name=sub_goal.name,
                strategy=strategy_name,
                success=True,
                verify_result=True,
                duration_sec=time.monotonic() - step_start,
                error="",
                fallback_used=False,
            )

        # --- Phase 3: Visual verification fallback ---
        if self._visual_verifier_agent is not None:
            try:
                from vector_os_nano.vcli.cognitive.visual_verifier import should_verify, verify_visual
                if should_verify(
                    sub_goal_name=sub_goal.name,
                    sub_goal_description=sub_goal.description,
                    strategy=strategy_name,
                    verify_expr=sub_goal.verify,
                    verify_result=verify_result,
                ):
                    vv_result = verify_visual(
                        agent=self._visual_verifier_agent,
                        sub_goal_description=sub_goal.description,
                        verify_expr=sub_goal.verify,
                    )
                    if vv_result.triggered and vv_result.success:
                        logger.info(
                            "GoalExecutor: visual verification overrode failed verify for %s",
                            sub_goal.name,
                        )
                        return StepRecord(
                            sub_goal_name=sub_goal.name,
                            strategy=strategy_name,
                            success=True,
                            verify_result=True,  # overridden by visual
                            duration_sec=time.monotonic() - step_start,
                            error="",
                            fallback_used=False,
                        )
            except Exception as exc:  # noqa: BLE001
                logger.debug("GoalExecutor: visual verification failed: %s", exc)

        # Verification failed — try fail_action if present
        if sub_goal.fail_action:
            fallback_sg = SubGoal(
                name=sub_goal.fail_action,
                description=f"fallback for {sub_goal.name}",
                verify=sub_goal.verify,
                timeout_sec=sub_goal.timeout_sec,
            )
            try:
                fallback_result = self._selector.select(fallback_sg)
                self._execute_strategy(fallback_result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("GoalExecutor: fallback strategy raised: %s", exc)

            # Re-verify after fallback
            verify_result_after = self._verifier.verify(sub_goal.verify)
            return StepRecord(
                sub_goal_name=sub_goal.name,
                strategy=strategy_name,
                success=verify_result_after,
                verify_result=verify_result_after,
                duration_sec=time.monotonic() - step_start,
                error="" if verify_result_after else "failed after fallback",
                fallback_used=True,
            )

        # No fallback, verification failed
        return StepRecord(
            sub_goal_name=sub_goal.name,
            strategy=strategy_name,
            success=False,
            verify_result=False,
            duration_sec=time.monotonic() - step_start,
            error="verification failed",
            fallback_used=False,
        )

    # ------------------------------------------------------------------
    # Strategy execution dispatchers
    # ------------------------------------------------------------------

    def _extract_name(self, result: Any) -> str:
        """Extract the strategy name from a StrategyResult safely.

        MagicMock treats 'name' as a special attribute, so we try the
        string representation and fall back to the mock's spec if needed.
        """
        raw = getattr(result, "name", "")
        if isinstance(raw, str):
            return raw
        # Non-string (e.g. MagicMock in tests) — try repr-based extraction
        # or return empty string as a safe fallback
        return ""

    def _execute_strategy(self, result: Any) -> tuple[bool, str]:
        """Dispatch to skill or primitive execution.

        Returns:
            (success: bool, error_message: str)
        """
        executor_type = getattr(result, "executor_type", "")
        name = self._extract_name(result)
        raw_params = getattr(result, "params", {})
        params = raw_params if isinstance(raw_params, dict) else {}

        if executor_type == "skill":
            return self._execute_skill(name, params)
        if executor_type == "primitive":
            return self._execute_primitive(name, params)
        # Unknown executor type
        error = f"No strategy for: {name} (executor_type={executor_type!r})"
        logger.warning("GoalExecutor: %s", error)
        return False, error

    def _execute_skill(self, name: str, params: dict) -> tuple[bool, str]:
        """Locate and execute a skill from the registry.

        Returns:
            (success: bool, error_message: str)
        """
        if self._skill_registry is None:
            return False, f"Skill not found: {name} (no registry)"

        skill = None
        try:
            skill = self._skill_registry.get(name)
        except Exception as exc:  # noqa: BLE001
            return False, f"Registry error for {name}: {exc}"

        if skill is None:
            return False, f"Skill not found: {name}"

        context = None
        if self._build_context is not None:
            try:
                context = self._build_context()
            except Exception as exc:  # noqa: BLE001
                logger.warning("GoalExecutor: build_context raised: %s", exc)

        try:
            skill_result = skill.execute(params, context)
            success = bool(getattr(skill_result, "success", False))
            error = getattr(skill_result, "error_message", "") or ""

            # --- Async skill wait: explore returns immediately, wait for completion ---
            result_data = getattr(skill_result, "result_data", {}) or {}
            if result_data.get("status") == "exploration_started":
                success, error = self._wait_for_async_skill(name)

            return success, error
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    def _wait_for_async_skill(self, name: str) -> tuple[bool, str]:
        """Block until an async skill (e.g. explore) completes or abort fires."""
        try:
            from vector_os_nano.vcli.cognitive.abort import is_abort_requested, wait_or_abort
        except ImportError:
            return True, ""

        logger.info("GoalExecutor: waiting for async skill %r to complete", name)
        for _ in range(300):  # max 10 min (300 * 2s)
            if is_abort_requested():
                return False, "aborted"
            # Check if exploration finished
            try:
                from vector_os_nano.skills.go2.explore import is_exploring
                if not is_exploring():
                    return True, ""
            except ImportError:
                return True, ""
            wait_or_abort(2.0)
            if is_abort_requested():
                return False, "aborted"
        return False, f"async skill {name} timed out (10 min)"

    def _execute_primitive(self, name: str, params: dict) -> tuple[bool, str]:
        """Locate and call a primitive function.

        Primitive sources (checked in order):
        1. self._primitives (dict or namespace)
        2. vcli.primitives sub-modules (locomotion, navigation, perception, world)

        Return value semantics:
        - bool → (value, "")
        - other non-None → (True, "")
        - Exception → (False, str(exc))

        Returns:
            (success: bool, error_message: str)
        """
        fn = self._resolve_primitive(name)
        if fn is None:
            return False, f"Primitive not found: {name}"

        try:
            sig = inspect.signature(fn)
            accepted = set(sig.parameters.keys())
            filtered = {k: v for k, v in params.items() if k in accepted}
            retval = fn(**filtered)
            if isinstance(retval, bool):
                return retval, ""
            return True, ""
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    def _resolve_primitive(self, name: str) -> Callable | None:
        """Find a primitive function by name.

        Checks self._primitives first, then imports from vcli.primitives modules.
        """
        if not isinstance(name, str) or not name:
            return None

        # 1. Check injected primitives namespace (dict or object with attr)
        if self._primitives is not None:
            if isinstance(self._primitives, dict):
                fn = self._primitives.get(name)
                if fn is not None:
                    return fn
            else:
                fn = getattr(self._primitives, name, None)
                if fn is not None:
                    return fn

        # 2. Try importing from vcli.primitives sub-modules
        _PRIMITIVE_MODULES = (
            "vector_os_nano.vcli.primitives.locomotion",
            "vector_os_nano.vcli.primitives.navigation",
            "vector_os_nano.vcli.primitives.perception",
            "vector_os_nano.vcli.primitives.world",
        )
        import importlib
        for module_path in _PRIMITIVE_MODULES:
            try:
                mod = importlib.import_module(module_path)
                fn = getattr(mod, name, None)
                if fn is not None:
                    return fn
            except ImportError:
                continue

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_step(
        self,
        sub_goal: SubGoal,
        result: Any,
        success: bool,
        verify_result: bool,
        error: str,
        start: float,
        fallback_used: bool,
    ) -> StepRecord:
        """Convenience factory for StepRecord."""
        strategy_name = self._extract_name(result) if result is not None else ""
        return StepRecord(
            sub_goal_name=sub_goal.name,
            strategy=strategy_name,
            success=success,
            verify_result=verify_result,
            duration_sec=time.monotonic() - start,
            error=error,
            fallback_used=fallback_used,
        )
