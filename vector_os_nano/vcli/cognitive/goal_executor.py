# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""GoalExecutor — executes GoalTrees by verifying each step and handling fallbacks.

Execution flow per sub_goal:
1. Select strategy via StrategySelector
2. Execute the selected strategy (skill or primitive)
3. Check elapsed time against timeout_sec
4. Verify success condition via GoalVerifier
5. On failure: attempt fail_action fallback, then re-verify
6. Record StepRecord; a failure poisons its (transitive) dependents with a
   skipped record (failure_class="dep_skipped") while independent goals
   still run — the same contract the harness applies (#11, campaign #4)
"""
from __future__ import annotations

import inspect
import logging
import time
from collections import deque
from typing import Any, Callable

from vector_os_nano.vcli.cognitive.trace_store import step_evidence_ok
from vector_os_nano.vcli.cognitive.trace_store import (
    _NO_EVIDENCE as _NO_EVIDENCE_VERIFIES,
)
from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    ForEachSpec,
    GoalTree,
    StepRecord,
    SubGoal,
    classify_exec_failure,
)

logger = logging.getLogger(__name__)


def blocking_dependency(sub_goal: SubGoal, failed_names: set[str]) -> str:
    """Name of the first failed/skipped step this sub_goal depends on, or "".

    Transitivity needs no graph walk: a skipped step's own name is added to
    ``failed_names`` by the caller, so a chain a→b→c poisons c through b.
    Single source for BOTH execution paths (GoalExecutor.execute and
    VGGHarness._execute_with_retry) — one tree, one failure semantics (#11).
    """
    for dep in sub_goal.depends_on:
        if dep in failed_names:
            return dep
    return ""


def skipped_step_record(sub_goal: SubGoal, blocked_by: str) -> StepRecord:
    """A StepRecord for a step that never ran: its dependency failed.

    Executing it anyway would act from an unestablished world state (pick
    after a failed navigate — the hardware-dangerous case). failure_class
    "dep_skipped" is a first-class member of FAILURE_CLASSES so the replan
    LLM can branch on it.
    """
    return StepRecord(
        sub_goal_name=sub_goal.name,
        strategy=sub_goal.strategy,
        success=False,
        verify_result=False,
        duration_sec=0.0,
        error=f"skipped: dependency '{blocked_by}' failed upstream",
        fallback_used=False,
        failure_class="dep_skipped",
    )


def _make_step_output(exec_output: Any) -> Callable[..., Any]:
    """Build the per-evaluation ``step_output(path='')`` verify function.

    Bound to the CURRENT step's own structured output (backlog #3, Rule 4):
    a verify expression can consume what THIS step observed — e.g. detect's
    alias-aware ``objects`` list — instead of re-querying a separate oracle
    that false-passes when the requested target is absent.

    Pure dict/list traversal on dotted *path* segments (mirrors the
    Blackboard's resolution style); any miss returns ``None`` (fail-safe —
    never raises into the sandbox). ``path=''`` returns the whole output.
    """

    def step_output(path: str = "") -> Any:
        if not path:
            return exec_output
        cur: Any = exec_output
        for part in str(path).split("."):
            if isinstance(cur, dict):
                cur = cur.get(part)
            elif isinstance(cur, list) and part.lstrip("-").isdigit():
                idx = int(part)
                cur = cur[idx] if -len(cur) <= idx < len(cur) else None
            else:
                return None
        return cur

    return step_output


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
        code_executor: Any = None,
        tool_dispatcher: Any = None,
        capability_registry: Any = None,
        blackboard: Any = None,
        is_robot: bool = False,
        evidence_exempt: "frozenset[str]" = frozenset(),
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
            code_executor: Optional CodeExecutor — runs ``code`` (code-as-policy)
                           sub-goals in an AST sandbox. None disables the ``code`` branch.
            tool_dispatcher: Optional ToolDispatcher — runs ``tool`` sub-goals through
                             the permission gate + per-world allowlist. None disables
                             the ``tool`` branch. Both default None so the robot path is
                             byte-identical when no code/tool sub-goal is produced.
            capability_registry: Optional CapabilityRegistry — routes ``capability``
                             sub-goals to a named routable capability (Phase C). None
                             disables the ``capability`` branch.
            blackboard: Optional Blackboard (Stage 1a) — run-scoped store that
                             captures each successful step's structured output under
                             the sub-goal name. None disables capture; the executor's
                             behavior is otherwise byte-identical.
            is_robot: True when the active world is a robot world. The SINGLE source
                             of truth for the per-step learning-tier reward gate
                             (W1.1): every bandit record site routes through
                             ``_record_strategy_stats``, which gates the recorded
                             ``success`` on ``step_evidence_ok(..., is_robot)``. For a
                             robot world the gate is unconditionally True, so the reward
                             collapses to ``step.success`` (robot learning unchanged);
                             for dev/playground it additionally requires deterministic
                             evidence. Defaulted False so existing constructions are
                             byte-identical.
        """
        self._selector = strategy_selector
        self._verifier = verifier
        self._skill_registry = skill_registry
        self._primitives = primitives
        self._build_context = build_context
        self._stats = stats
        self._visual_verifier_agent = visual_verifier_agent
        self._code_executor = code_executor
        self._tool_dispatcher = tool_dispatcher
        self._capability_registry = capability_registry
        self._blackboard = blackboard
        self._is_robot = bool(is_robot)
        # Invariant II: world-declared per-step evidence exemptions — the old
        # world-level is_robot bypass is gone from the gate; this bounded set
        # is what keeps legitimately-postcondition-free skills learnable.
        self._evidence_exempt = frozenset(evidence_exempt)

    # ------------------------------------------------------------------
    # Strategy-stats reward gate (W1.1) — single chokepoint
    # ------------------------------------------------------------------

    def _record_strategy_stats(self, step: StepRecord, sub_goal: SubGoal) -> None:
        # W1.1: gate the per-step bandit reward on deterministic evidence. The reward is
        # step.success AND step_evidence_ok(...): robot world collapses to step.success
        # (learning unchanged); dev/playground additionally requires real evidence so a
        # sentinel/visual_override 'success' cannot train the bandit. Single chokepoint
        # for ALL record sites (execute fallback, foreach, harness).
        if self._stats is None:
            return
        try:
            self._stats.record(
                strategy_name=step.strategy,
                sub_goal_name=step.sub_goal_name,
                success=step.success and step_evidence_ok(
                    step, sub_goal, self._is_robot,
                    exempt_strategies=self._evidence_exempt,
                ),
                duration_sec=step.duration_sec,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("GoalExecutor: stats.record raised: %s", exc)

    # ------------------------------------------------------------------
    # Blackboard accessor (Stage 1b)
    # ------------------------------------------------------------------

    @property
    def blackboard(self) -> Any:
        """The run-scoped Blackboard, or ``None`` when capture is disabled."""
        return self._blackboard

    @blackboard.setter
    def blackboard(self, value: Any) -> None:
        """Attach a fresh Blackboard for the current run (Stage 1b).

        The VGGHarness sets this at the start of each ``run()`` so capture and
        data-binding are scoped to a single execution. ``None`` disables both.
        """
        self._blackboard = value

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
        3. A failed step poisons its (transitive) dependents: they get a
           skipped StepRecord (failure_class="dep_skipped") and never run;
           independent steps still execute. Same contract as the harness.

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
        failed_names: set[str] = set()

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

            # #11 — dependency-failure skip (shared contract with the harness).
            blocked_by = blocking_dependency(sub_goal, failed_names)
            if blocked_by:
                skip = skipped_step_record(sub_goal, blocked_by)
                steps.append(skip)
                failed_names.add(sub_goal.name)  # poison transitively
                overall_success = False
                if on_step is not None:
                    try:
                        on_step(skip)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "GoalExecutor: on_step callback raised: %s", exc)
                continue

            # Stage 4 (S4-2): a foreach node is not a leaf step — it expands at
            # runtime into N concrete children (the body instantiated once per
            # item of the producing step's list). Execute the expansion in order;
            # each child carries its own verify and is recorded/captured normally.
            if getattr(sub_goal, "foreach", None) is not None:
                expanded = self._execute_foreach(sub_goal, on_step)
                steps.extend(expanded)
                if any(not s.success for s in expanded):
                    overall_success = False
                    failed_names.add(sub_goal.name)
                continue

            step = self._execute_sub_goal(sub_goal)
            steps.append(step)

            # Record to stats if available (W1.1: evidence-gated via the chokepoint).
            self._record_strategy_stats(step, sub_goal)

            if on_step is not None:
                try:
                    on_step(step)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("GoalExecutor: on_step callback raised: %s", exc)

            if not step.success:
                overall_success = False
                failed_names.add(sub_goal.name)

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
        return self._topological_sort_list(list(goal_tree.sub_goals))

    def _topological_sort_list(self, sub_goals_in: list[SubGoal]) -> list[SubGoal]:
        """Order a flat list of SubGoals by their intra-list ``depends_on``.

        Kahn's algorithm with original-order tie-breaking (deterministic). Deps
        referencing names outside *sub_goals_in* are ignored, and a cycle falls
        back to original order — identical semantics to the GoalTree variant, just
        reusable for the foreach body (whose templates also carry ``depends_on``).
        """
        sub_goals = list(sub_goals_in)
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

        # W1.4 — world producing-step primitive: if the sub-goal's explicit strategy
        # names a world-injected producer (build_step_primitives), dispatch it
        # DIRECTLY as a ``primitive`` (preserving the sub-goal's real strategy_params
        # for the ${...} blackboard binding) instead of letting the StrategySelector
        # route the ``*_skill`` name to ``skill``/``invalid`` — neither of which would
        # run the wired producer. Only producer-only ``*_skill`` keys match, so a real
        # registered skill is never intercepted (see _world_primitive_strategy).
        _prim_key = self._world_primitive_strategy(sub_goal)
        if _prim_key is not None:
            from vector_os_nano.vcli.cognitive.strategy_selector import StrategyResult
            result: Any = StrategyResult(
                "primitive", _prim_key, dict(sub_goal.strategy_params)
            )
        else:
            # Select strategy
            try:
                result = self._selector.select(sub_goal)
            except Exception as exc:  # noqa: BLE001
                logger.warning("GoalExecutor: selector.select raised: %s", exc)
                # The selector raised before any strategy ran — an unclassified
                # execution failure (W2.4).
                return self._make_step(sub_goal, result=None, success=False,
                                       verify_result=False, error=str(exc),
                                       start=step_start, fallback_used=False,
                                       failure_class="exec_error")

        strategy_name = self._extract_name(result)

        # Data binding (Stage 1b): resolve ``${step.key}`` references in the
        # selected strategy's params against the run blackboard BEFORE executing,
        # so a later step can consume an earlier step's captured output. Pure
        # dict/list traversal — never eval (see Blackboard.resolve). StrategyResult
        # is frozen, so build a resolved copy rather than mutating in place.
        result = self._resolve_params(result)

        # Invariant I (design review 2026-06-12 #1) — pre-execution baseline:
        # evaluate a NON-TRIVIAL predicate BEFORE the strategy runs, so an
        # initial-state PASS is visible instead of indistinguishable from
        # earned success. Best-effort and fail-safe (errors -> False).
        pre_satisfied = self._pre_satisfied(sub_goal.verify)

        # Execute strategy — captures the step's structured output (Stage 1a).
        exec_success, exec_error, exec_output = self._execute_strategy(result)
        elapsed = time.monotonic() - step_start

        # Check timeout (takes priority over execution result).
        # R2-2: floor the effective limit at the skill's declared real-time duration
        # so a live-viewer pick/place (20s+) is never falsely marked timeout just
        # because the LLM emitted a small timeout_sec (e.g. 15s). The floor is
        # opt-in — a skill without typical_duration_sec is unaffected.
        effective_timeout = self._effective_timeout(sub_goal.timeout_sec, strategy_name)
        timeout_msg = ""
        if elapsed > effective_timeout:
            timeout_msg = (
                f"timeout after {elapsed:.3f}s "
                f"(limit {effective_timeout:.1f}s"
                + (f"; plan said {sub_goal.timeout_sec:.1f}s"
                   if effective_timeout != sub_goal.timeout_sec else "")
                + ")"
            )
            logger.warning("GoalExecutor: %s — %s", sub_goal.name, timeout_msg)
            # #17 (campaign #4): a slow-but-COMPLETED action still runs its
            # verify below — verified means an honest PASS with a timing
            # warning, not a replayed motion. Only an execution that also
            # FAILED short-circuits here — and it carries the UNDERLYING exec
            # error (R11: a bare 'timeout' masked the real moved_short/wall
            # diagnosis and misled the replan AND the human).
            if not exec_success:
                return StepRecord(
                    sub_goal_name=sub_goal.name,
                    strategy=strategy_name,
                    success=False,
                    verify_result=False,
                    duration_sec=elapsed,
                    error=(f"{timeout_msg}; exec: {exec_error}"
                           if exec_error else timeout_msg),
                    fallback_used=False,
                    result_data={"output": exec_output, "verify_value": None},
                    failure_class="timeout",  # W2.4: post-hoc step-timeout path
                    pre_satisfied=pre_satisfied,
                )

        # If execution itself failed (skill not found, unknown type, etc.),
        # mark the step failed immediately — no point verifying.
        if not exec_success:
            logger.warning(
                "GoalExecutor: execution failed for %s: %s", sub_goal.name, exec_error
            )
            # W2.4: classify deterministically from already-available signals —
            # the resolved strategy's executor_type and the step's machine-readable
            # ``diagnosis`` (e.g. 'ik_unreachable'). NO new model call; the raw
            # error string is NOT widened (rule 10 — failure_class is a bounded enum).
            failure_class = classify_exec_failure(
                executor_type=getattr(result, "executor_type", ""),
                diagnosis=str(exec_output.get("diagnosis", ""))
                if isinstance(exec_output, dict)
                else "",
            )
            return StepRecord(
                sub_goal_name=sub_goal.name,
                strategy=strategy_name,
                success=False,
                verify_result=False,
                duration_sec=time.monotonic() - step_start,
                error=exec_error,
                fallback_used=False,
                result_data={"output": exec_output, "verify_value": None},
                failure_class=failure_class,
                pre_satisfied=pre_satisfied,
            )

        # Verify — yields (bool, raw value) from the same sandbox. The step's
        # own structured output is injected as step_output() (backlog #3).
        verify_result, verify_value = self._verify_and_value(
            sub_goal.verify, exec_output
        )

        if verify_result:
            # Success path
            result_data = {"output": exec_output, "verify_value": verify_value}
            if timeout_msg:
                # #17 — verified despite the budget: pass, but say so.
                result_data["timing_warning"] = timeout_msg
            self._capture(sub_goal.name, result_data)
            return StepRecord(
                sub_goal_name=sub_goal.name,
                strategy=strategy_name,
                success=True,
                verify_result=True,
                duration_sec=time.monotonic() - step_start,
                error="",
                fallback_used=False,
                result_data=result_data,
                pre_satisfied=pre_satisfied,
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
                        vo_result_data = {"output": exec_output, "verify_value": verify_value}
                        self._capture(sub_goal.name, vo_result_data)
                        return StepRecord(
                            sub_goal_name=sub_goal.name,
                            strategy=strategy_name,
                            success=True,
                            verify_result=True,  # overridden by visual
                            duration_sec=time.monotonic() - step_start,
                            error="",
                            fallback_used=False,
                            visual_override=True,  # not deterministic evidence
                            result_data=vo_result_data,
                            pre_satisfied=pre_satisfied,
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
            fallback_output: dict = {}
            try:
                fallback_result = self._selector.select(fallback_sg)
                _, _, fallback_output = self._execute_strategy(fallback_result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("GoalExecutor: fallback strategy raised: %s", exc)

            # Re-verify after fallback — against the freshest output in hand.
            verify_result_after, verify_value_after = self._verify_and_value(
                sub_goal.verify, fallback_output or exec_output
            )
            # Prefer the fallback's output; fall back to the primary attempt's.
            fb_result_data = {
                "output": fallback_output or exec_output,
                "verify_value": verify_value_after,
            }
            if verify_result_after:
                self._capture(sub_goal.name, fb_result_data)
            if verify_result_after and timeout_msg:
                fb_result_data["timing_warning"] = timeout_msg
            return StepRecord(
                sub_goal_name=sub_goal.name,
                strategy=strategy_name,
                success=verify_result_after,
                verify_result=verify_result_after,
                duration_sec=time.monotonic() - step_start,
                error="" if verify_result_after else (
                    f"{timeout_msg}; failed after fallback" if timeout_msg
                    else "failed after fallback"),
                fallback_used=True,
                result_data=fb_result_data,
                # W2.4: executed without error but verify is still False after the
                # fallback -> a verify-miss. "" on the success branch; a step that
                # ALSO blew its budget classifies as timeout (#17).
                failure_class="" if verify_result_after else (
                    "timeout" if timeout_msg else "verify_fail"),
                pre_satisfied=pre_satisfied,
            )

        # No fallback, verification failed
        return StepRecord(
            sub_goal_name=sub_goal.name,
            strategy=strategy_name,
            success=False,
            verify_result=False,
            duration_sec=time.monotonic() - step_start,
            error=(f"{timeout_msg}; verification failed" if timeout_msg
                   else "verification failed"),
            fallback_used=False,
            result_data={"output": exec_output, "verify_value": verify_value},
            # W2.4: executed OK but verify was False; a step that ALSO blew
            # its budget classifies as timeout (#17).
            failure_class="timeout" if timeout_msg else "verify_fail",
            pre_satisfied=pre_satisfied,
        )

    # ------------------------------------------------------------------
    # FOREACH expansion (Stage 4, S4-2)
    # ------------------------------------------------------------------

    def _resolve_foreach_items(self, spec: ForEachSpec) -> list[Any]:
        """Read the producing step's iterable from the blackboard (pure traversal).

        The list is reached through the SAME ``${step.path}`` convention every
        other step reference uses — resolved by the Blackboard's pure dict/list
        traversal, never ``eval``. ``result_data`` is captured wrapped under an
        ``"output"`` key (``{"output": <strategy output>, "verify_value": ...}``),
        so the producing list is most naturally addressed as
        ``<source_step>.output.<source_path>``; we also try the bare
        ``<source_step>.<source_path>`` for a producer that stored the list at the
        top level. The first form that resolves to a list wins.

        Returns the resolved list, or ``None`` when the path does NOT resolve
        (no blackboard, missing producer/key, or resolution raised). Batch 2
        #10 (design review): "resolved to an empty list" (an honest zero —
        nothing to iterate) and "never resolved" (a plan/producer contract
        miss the replan must hear about) are DIFFERENT outcomes; the old code
        collapsed both to zero silent iterations and let the tree PASS.
        """
        if self._blackboard is None:
            return None
        path = spec.source_path.strip(".")
        candidates = (
            f"${{{spec.source_step}.output.{path}}}",
            f"${{{spec.source_step}.{path}}}",
        )
        for ref in candidates:
            try:
                resolved = self._blackboard.resolve(ref)
            except Exception as exc:  # noqa: BLE001
                logger.warning("GoalExecutor: foreach resolve raised: %s", exc)
                return None
            if isinstance(resolved, list):
                return list(resolved)
        logger.warning(
            "GoalExecutor: foreach source %r.%r did not resolve to a list",
            spec.source_step,
            spec.source_path,
        )
        return None

    def _foreach_producer_keys(self, source_step: str) -> "list[str]":
        """The producer step's actually-available top-level keys (for the
        loud unresolved-foreach error — fed to the replan so the next plan
        can bind the REAL path). Best-effort; never raises."""
        try:
            data = self._blackboard.get(source_step) if self._blackboard else None
            if not isinstance(data, dict):
                return []
            keys = set(data.keys())
            out = data.get("output")
            if isinstance(out, dict):
                keys |= {f"output.{k}" for k in out.keys()}
            return sorted(keys)
        except Exception:  # noqa: BLE001
            return []

    def _execute_foreach(
        self,
        sub_goal: SubGoal,
        on_step: Callable[[StepRecord], None] | None = None,
    ) -> list[StepRecord]:
        """Expand + execute a foreach node, returning each child's StepRecord.

        Reads the producing step's list from the blackboard, then for each item
        instantiates the body templates once and executes them IN ORDER, each with
        its own per-step verify. The iteration variable is bound by storing the
        item on the run blackboard under ``spec.var`` BEFORE each child executes,
        so a body reference like ``${obj.name}`` resolves through the existing pure
        dict/list path traversal (never string eval/format). Child outputs are
        captured to the blackboard as usual.

        Stops at the first failed child (mirrors the leaf abort semantics). An
        empty producing list yields zero children — not an error.
        """
        spec = sub_goal.foreach
        records: list[StepRecord] = []
        if spec is None:  # defensive — caller only dispatches real foreach nodes
            return records
        if not spec.body:
            return records

        items = self._resolve_foreach_items(spec)
        if items is None:
            # Unresolved producer path — a loud, replannable failure (#10),
            # never a silent zero-iteration PASS.
            keys = self._foreach_producer_keys(spec.source_step)
            err = (
                f"foreach source '{spec.source_step}.{spec.source_path}' did "
                f"not resolve to a list; producer's available keys: "
                f"{keys or '<none — producer step missing or not captured>'}"
            )
            failed = StepRecord(
                sub_goal_name=sub_goal.name,
                strategy="foreach",
                success=False,
                verify_result=False,
                duration_sec=0.0,
                error=err,
                fallback_used=False,
                failure_class="exec_error",
            )
            records.append(failed)
            if on_step is not None:
                try:
                    on_step(failed)
                except Exception:  # noqa: BLE001
                    pass
            return records
        # The body runs once per item, but template-to-template ``depends_on`` must
        # still order the body (e.g. place_obj depends_on pick_obj). Order the body
        # ONCE by its intra-body deps; list order is the deterministic tie-break, so
        # an already-ordered body is byte-unchanged. Without this, a body emitted
        # out of dependency order would run a consumer before its producer.
        ordered_body = self._topological_sort_list(list(spec.body))
        for index, item in enumerate(items):
            # --- Abort check (mirror the leaf loop) ---
            try:
                from vector_os_nano.vcli.cognitive.abort import is_abort_requested
                if is_abort_requested():
                    abort_step = StepRecord(
                        sub_goal_name=f"{sub_goal.name}[{index}]",
                        strategy="",
                        success=False,
                        verify_result=False,
                        duration_sec=0.0,
                        error="aborted",
                        fallback_used=False,
                    )
                    records.append(abort_step)
                    if on_step is not None:
                        try:
                            on_step(abort_step)
                        except Exception:  # noqa: BLE001
                            pass
                    break
            except ImportError:
                pass

            # Bind the current item under the iteration var so body references
            # (``${var.field}``) resolve via pure blackboard traversal. The item
            # must be a dict for the blackboard to hold it; a non-dict item is
            # wrapped under ``"value"`` so ``${var.value}`` is still reachable.
            self._bind_iteration_var(spec.var, item)

            stop = False
            for template in ordered_body:
                child = self._instantiate_body_template(template, sub_goal, index)
                step = self._execute_sub_goal(child)
                records.append(step)

                # W1.1: ``child`` (the per-iteration body template) is the predicate
                # carrier — it holds the verify — so the evidence gate must read it,
                # NOT the parent foreach sub_goal.
                self._record_strategy_stats(step, child)

                if on_step is not None:
                    try:
                        on_step(step)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("GoalExecutor: on_step callback raised: %s", exc)

                if not step.success:
                    stop = True
                    break
            if stop:
                break

        return records

    def _bind_iteration_var(self, var: str, item: Any) -> None:
        """Store the current foreach item on the blackboard under *var*.

        Fail-soft: a missing blackboard simply means body references stay
        unresolved (passthrough) rather than aborting. A non-dict item is wrapped
        under ``"value"`` so the blackboard (which only holds dicts) can store it
        and ``${var.value}`` remains reachable.
        """
        if self._blackboard is None:
            return
        payload = item if isinstance(item, dict) else {"value": item}
        try:
            self._blackboard.put(var, payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("GoalExecutor: foreach var bind raised: %s", exc)

    def _instantiate_body_template(
        self,
        template: SubGoal,
        owner: SubGoal,
        index: int,
    ) -> SubGoal:
        """Return a per-item concrete child from a body *template*.

        The child gets a unique, traceable name (``<owner>[<index>].<template>``)
        and drops intra-body ``depends_on`` (the body runs sequentially in list
        order under the executor, so positional order is the dependency).

        The iteration var has ALREADY been bound on the blackboard by the caller,
        so both the child's ``verify`` expression and its ``strategy_params`` are
        materialised here by the Blackboard's PURE ``${var.field}`` traversal —
        plain dict/list/string substitution, never eval/format. Resolving the
        verify string yields concrete literal data (e.g. ``picked('mug')``) that
        the AST-sandboxed GoalVerifier then evaluates; strategy_params resolution
        is idempotent with the executor's pre-exec ``_resolve_params`` pass.
        """
        import dataclasses

        verify = template.verify
        params = template.strategy_params
        bb = self._blackboard
        if bb is not None:
            try:
                verify = bb.resolve(template.verify)
                if not isinstance(verify, str):
                    verify = template.verify  # a stray exact-ref returned non-str
            except Exception as exc:  # noqa: BLE001
                logger.warning("GoalExecutor: foreach verify bind raised: %s", exc)
                verify = template.verify
            try:
                params = bb.resolve(dict(template.strategy_params))
            except Exception as exc:  # noqa: BLE001
                logger.warning("GoalExecutor: foreach params bind raised: %s", exc)
                params = template.strategy_params

        return dataclasses.replace(
            template,
            name=f"{owner.name}[{index}].{template.name}",
            depends_on=(),
            verify=verify,
            strategy_params=params,
            foreach=None,
        )

    # ------------------------------------------------------------------
    # Observation capture (Stage 1a)
    # ------------------------------------------------------------------

    def _pre_satisfied(self, verify: str) -> bool:
        """The predicate's value BEFORE execution (invariant I).

        Trivial sentinels ('', 'True' — trace_store's no-evidence set) are
        never flagged: pre-eval on them is meaningless noise. Expressions
        that cannot evaluate without the step's own output (step_output())
        fail closed to False — the baseline never blocks execution.
        """
        expr = (verify or "").strip()
        if expr in _NO_EVIDENCE_VERIFIES:
            return False
        try:
            ok, _ = self._verify_and_value(expr, None)
        except Exception:  # noqa: BLE001 — best-effort baseline
            return False
        return bool(ok)

    def _verify_and_value(
        self, expression: str, exec_output: Any = None
    ) -> tuple[bool, Any]:
        """Return ``(verify_bool, verify_value)`` for *expression*.

        Prefers the verifier's :meth:`evaluate` (which surfaces the raw value);
        falls back to :meth:`verify` (value = None) for any verifier — including
        test mocks — that only exposes ``verify``. The boolean result is always
        identical to what ``verify`` alone would have returned.

        ``exec_output`` (backlog #3) is the CURRENT step's own structured
        output. When given, a kernel ``step_output(path='')`` function is
        injected per-evaluation so the verify can consume what THIS step
        observed (Rule 4 — e.g. detect verifying its own alias-aware result)
        instead of re-querying a separate oracle. Old-signature verifiers
        (and mocks) keep working: a TypeError falls back to the bare call.
        """
        extra_ns = (
            {"step_output": _make_step_output(exec_output)}
            if exec_output is not None
            else None
        )
        evaluate = getattr(self._verifier, "evaluate", None)
        if callable(evaluate):
            try:
                if extra_ns is not None:
                    try:
                        outcome = evaluate(expression, extra_ns=extra_ns)
                    except TypeError:
                        outcome = evaluate(expression)
                else:
                    outcome = evaluate(expression)
                if isinstance(outcome, tuple) and len(outcome) == 2:
                    return bool(outcome[0]), outcome[1]
            except Exception as exc:  # noqa: BLE001
                logger.debug("GoalExecutor: verifier.evaluate raised: %s", exc)
        return bool(self._verifier.verify(expression)), None

    def _resolve_params(self, result: Any) -> Any:
        """Return *result* with its ``params`` resolved against the blackboard.

        Data binding (Stage 1b). No-op when no blackboard is attached, when the
        result carries no dict ``params``, or when resolution raises — the path
        stays byte-identical so non-binding runs are unaffected. Resolution is
        pure dict/list/str traversal (Blackboard.resolve), never ``eval``.

        StrategyResult is a frozen dataclass; a resolved copy is built via
        :func:`dataclasses.replace` so the original is never mutated. A non-frozen
        result (e.g. a test MagicMock) is returned unchanged.
        """
        if self._blackboard is None:
            return result
        raw_params = getattr(result, "params", None)
        if not isinstance(raw_params, dict) or not raw_params:
            return result
        try:
            resolved = self._blackboard.resolve(dict(raw_params))
        except Exception as exc:  # noqa: BLE001
            logger.warning("GoalExecutor: blackboard.resolve raised: %s", exc)
            return result
        if resolved == raw_params:
            return result  # nothing referenced — avoid an unnecessary copy
        try:
            import dataclasses
            if dataclasses.is_dataclass(result) and not isinstance(result, type):
                return dataclasses.replace(result, params=resolved)
        except Exception as exc:  # noqa: BLE001
            logger.debug("GoalExecutor: could not rebuild result with resolved params: %s", exc)
        return result

    def _capture(self, step_name: str, result_data: dict) -> None:
        """Store a successful step's result_data on the run blackboard, if any.

        Fail-soft: a missing blackboard or a misbehaving ``put`` never aborts
        execution (capture is a side channel, not a control-flow dependency).
        """
        if self._blackboard is None:
            return
        try:
            self._blackboard.put(step_name, result_data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("GoalExecutor: blackboard.put raised: %s", exc)

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

    def _execute_strategy(self, result: Any) -> tuple[bool, str, dict]:
        """Dispatch to skill or primitive execution.

        Returns:
            (success: bool, error_message: str, output: dict)

        ``output`` is the step's captured structured output (Stage 1a). Each
        branch surfaces its native structured payload (skill_result.result_data,
        CapabilityResult.output, etc.); non-dict payloads are wrapped under a
        ``"value"`` key so the contract is always a dict.
        """
        executor_type = getattr(result, "executor_type", "")
        name = self._extract_name(result)
        raw_params = getattr(result, "params", {})
        params = raw_params if isinstance(raw_params, dict) else {}

        if executor_type == "skill":
            return self._execute_skill(name, params)
        if executor_type == "primitive":
            return self._execute_primitive(name, params)
        if executor_type == "code":
            return self._execute_code(params)
        if executor_type == "tool":
            return self._execute_tool(params)
        if executor_type == "capability":
            return self._execute_capability(name, params)
        if executor_type == "answer":
            return self._execute_answer(params)
        if executor_type == "invalid":
            # Fail-loud (Stage 2b): the selector resolved an explicit strategy
            # that is NOT a skill in this world. Surface a clear, named error
            # including the valid set rather than the opaque 'unmatched' fallback.
            error = self._invalid_strategy_error(name, params)
            logger.warning("GoalExecutor: %s", error)
            return False, error, {}
        # Unknown executor type
        error = self._unmatched_strategy_error(name, executor_type)
        logger.warning("GoalExecutor: %s", error)
        return False, error, {}

    def _world_primitive_strategy(self, sub_goal: SubGoal) -> str | None:
        """Return the injected world-primitive key for *sub_goal*'s strategy, or None.

        W1.4 — a world (e.g. PlaygroundWorld) injects per-step PRODUCER primitives
        via ``GoalExecutor(primitives=...)``, keyed by the EXACT strategy name a plan
        emits for the producing step (``detect_objects_skill`` / ``locate_rooms_skill``
        and, for a foreach body, ``pick_skill`` / ``place_skill`` when the world
        provides them). The StrategySelector would route such a ``*_skill`` name to
        ``skill`` (its bare name) or, when the bare name is not a registered skill, to
        ``invalid`` — and the ``invalid`` route DROPS the sub-goal's real
        ``strategy_params`` (it carries diagnostics instead), so a ``${obj.name}``
        binding would be lost. Neither route consults ``self._primitives``, so the
        wired producer the tabletop/foreach tests exercise would never run live.

        Match the sub-goal's explicit strategy against the injected dict so the
        producer is dispatched DIRECTLY (preserving ``strategy_params`` for the
        blackboard ``${...}`` binding). Only a ``dict`` primitives namespace and
        producer-only ``*_skill`` keys are eligible, so a real registered skill is
        never shadowed. Returns the matching key, or None (normal selection runs).
        """
        prims = self._primitives
        if not isinstance(prims, dict) or not prims:
            return None
        strategy = getattr(sub_goal, "strategy", "")
        if not isinstance(strategy, str) or not strategy.endswith("_skill"):
            return None
        return strategy if strategy in prims else None

    def _valid_strategy_set(self) -> list[str] | None:
        """Return the sorted registered skill names, or None if undeterminable."""
        registry = self._skill_registry
        if registry is None:
            return None
        lister = getattr(registry, "list_skills", None)
        if not callable(lister):
            return None
        try:
            return sorted(str(n) for n in lister())
        except Exception:  # noqa: BLE001
            return None

    def _invalid_strategy_error(self, name: str, params: dict) -> str:
        """Build a clear error for an explicit strategy that is not a skill."""
        strategy = str(params.get("strategy", name)) if isinstance(params, dict) else name
        valid = None
        if isinstance(params, dict):
            valid = params.get("valid_strategies")
        if not valid:
            valid = self._valid_strategy_set()
        if valid:
            return (
                f"strategy {strategy!r} is not a skill in this world "
                f"(valid: {sorted(valid)})"
            )
        return f"strategy {strategy!r} is not a skill in this world"

    def _unmatched_strategy_error(self, name: str, executor_type: str) -> str:
        """Build a clear error for an unroutable/unmatched strategy result."""
        valid = self._valid_strategy_set()
        if valid:
            return (
                f"no strategy matched for {name!r} "
                f"(executor_type={executor_type!r}; valid: {valid})"
            )
        return f"no strategy matched for {name!r} (executor_type={executor_type!r})"

    def _execute_code(self, params: dict) -> tuple[bool, str, dict]:
        """Execute an AST-sandboxed code-as-policy snippet.

        Requires ``params['code']``. The CodeExecutor's AST validator rejects
        imports (except ``math``) and dunder access; any name not in its
        restricted builtins (e.g. ``open``) fails at runtime. Either way a
        violation yields ``success=False``.

        Returns:
            (success: bool, error_message: str, output: dict)
        """
        if self._code_executor is None:
            return False, "code branch requires a CodeExecutor (none configured)", {}
        code = params.get("code")
        if not isinstance(code, str) or not code.strip():
            return False, 'code branch requires a non-empty params["code"]', {}
        result = self._code_executor.execute(code)
        output = self._coerce_output(getattr(result, "return_value", None))
        return bool(getattr(result, "success", False)), getattr(result, "error", "") or "", output

    def _execute_tool(self, params: dict) -> tuple[bool, str, dict]:
        """Dispatch a kernel tool through the permission-gated ToolDispatcher.

        Requires ``params['tool']`` (the tool name) and optional ``params['args']``
        (a dict of tool arguments). The dispatcher enforces a per-world allowlist
        plus the shared PermissionContext (bash deny-list, file_write overwrite
        guard, deny/always-allow rules).

        Returns:
            (success: bool, error_message: str, output: dict)
        """
        if self._tool_dispatcher is None:
            return False, "tool branch requires a ToolDispatcher (none configured)", {}
        tool_name = params.get("tool")
        if not isinstance(tool_name, str) or not tool_name:
            return False, 'tool branch requires params["tool"]', {}
        args = params.get("args", {})
        if not isinstance(args, dict):
            args = {}
        success, error = self._tool_dispatcher.dispatch(tool_name, args)
        # The dispatcher returns (success, error) only; expose the tool name as a
        # minimal structured output so downstream steps can reference the step ran.
        return success, error, {"tool": tool_name}

    def _execute_answer(self, params: dict) -> tuple[bool, str, dict]:
        """Return a pure-conversation answer step's text as structured output.

        Stage 5 (S5.2). An answer-only step carries no robot evidence by design —
        it is a degenerate, side-effect-free leaf that simply surfaces the
        answer text so the harness/CLI can render it. The text is read from
        ``params['answer']`` (the decomposer's answer plan) and exposed under the
        captured-output ``"text"`` key (mirroring the chat capability's contract),
        so a downstream step or the run snapshot can reference it via
        ``${step.output.text}``.

        This branch performs NO I/O and NO model call (the answer is already
        decided by the time the plan is built); it never decides "done" — the
        sub-goal's ``verify`` predicate does. Combined with the ``answer_only``
        marker the evidence gate reads, this keeps the answer path cheap and the
        moat intact (an action step with a sentinel verify is NOT treated as
        answer-only and still fails the gate).

        Returns:
            (success: bool, error_message: str, output: dict)
        """
        text = params.get("answer", "") if isinstance(params, dict) else ""
        if not isinstance(text, str):
            text = str(text)
        return True, "", {"text": text}

    def _execute_capability(self, name: str, params: dict) -> tuple[bool, str, dict]:
        """Route a sub-goal to a named routable capability (Phase C).

        ``params`` is the capability input payload. A read-only capability is
        invoked directly; a side-effecting one fails closed until C.3 wires the
        permission gate. The capability never decides success — the sub-goal's
        ``verify`` predicate (checked separately by the caller) does.

        Returns:
            (success: bool, error_message: str, output: dict)
        """
        if self._capability_registry is None:
            return False, "capability branch requires a CapabilityRegistry (none configured)", {}
        cap = self._capability_registry.get(name)
        if cap is None:
            return False, f"unknown capability: {name}", {}
        if getattr(cap, "side_effecting", False):
            # Side-effecting capabilities (e.g. a VLA policy) must route through a
            # permission gate — wired in Phase C.3. Fail closed until then.
            return False, (
                f"side-effecting capability '{name}' requires a permission gate "
                "(deferred to Phase C.3)"
            ), {}
        payload = params if isinstance(params, dict) else {}
        try:
            from vector_os_nano.vcli.cognitive.capabilities import validate_input
            err = validate_input(getattr(cap, "input_schema", {}) or {}, payload)
            if err is not None:
                return False, f"capability '{name}' input invalid: {err}", {}
        except Exception:  # noqa: BLE001
            pass
        context = None
        if self._build_context is not None:
            try:
                context = self._build_context()
            except Exception as exc:  # noqa: BLE001
                logger.debug("GoalExecutor: build_context for capability raised: %s", exc)
        try:
            result = cap.invoke(payload, context)
        except Exception as exc:  # noqa: BLE001
            return False, f"capability error: {exc}", {}
        output = self._coerce_output(getattr(result, "output", {}))
        return bool(getattr(result, "success", False)), getattr(result, "error", "") or "", output

    def _execute_skill(self, name: str, params: dict) -> tuple[bool, str, dict]:
        """Locate and execute a skill from the registry.

        Returns:
            (success: bool, error_message: str, output: dict)
        """
        if self._skill_registry is None:
            return False, f"Skill not found: {name} (no registry)", {}

        skill = None
        try:
            skill = self._skill_registry.get(name)
        except Exception as exc:  # noqa: BLE001
            return False, f"Registry error for {name}: {exc}", {}

        if skill is None:
            return False, f"Skill not found: {name}", {}

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

            return success, error, self._coerce_output(result_data)
        except Exception as exc:  # noqa: BLE001
            return False, str(exc), {}

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

    def _execute_primitive(self, name: str, params: dict) -> tuple[bool, str, dict]:
        """Locate and call a primitive function.

        Primitive sources (checked in order):
        1. self._primitives (dict or namespace)
        2. vcli.primitives sub-modules (locomotion, navigation, perception, world)

        Return value semantics:
        - bool → (value, "", {})
        - other non-None → (True, "", coerced output dict)
        - Exception → (False, str(exc), {})

        Returns:
            (success: bool, error_message: str, output: dict)
        """
        fn = self._resolve_primitive(name)
        if fn is None:
            return False, f"Primitive not found: {name}", {}

        try:
            sig = inspect.signature(fn)
            has_var_kw = any(
                p.kind is inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            accepted = set(sig.parameters.keys())
            unknown = (
                []
                if has_var_kw  # **kwargs accepts anything by contract
                else sorted(k for k in params.keys() if k not in accepted)
            )
            if unknown:
                # Batch 3 #12: silently dropping params hid plan/contract
                # drift (a typo'd or mis-named param became a default-value
                # action that still PASSed). Loud, with the real signature.
                return (
                    False,
                    f"primitive '{name}' got unknown param(s) {unknown}; "
                    f"accepted: {sorted(accepted)}",
                    {},
                )
            retval = fn(**params)
            if isinstance(retval, bool):
                return retval, "", {}
            return True, "", self._coerce_output(retval)
        except Exception as exc:  # noqa: BLE001
            return False, str(exc), {}

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

    def _effective_timeout(self, plan_timeout: float, strategy_name: str) -> float:
        """Return max(plan_timeout, skill.typical_duration_sec or 0).

        R2-2 floor: slow motor skills declare a typical real-time duration; the
        executor floors the step's effective timeout at that value so a completed
        motor action is not falsely failed under a live MuJoCo viewer even when the
        LLM underestimates timeout_sec. A skill without typical_duration_sec (or no
        registry) leaves plan_timeout unchanged — the floor is fully opt-in.

        Args:
            plan_timeout: the sub_goal.timeout_sec from the decomposed plan.
            strategy_name: the resolved skill name (already _skill-suffix-stripped,
                           as returned by _extract_name).
        """
        if not strategy_name or self._skill_registry is None:
            return plan_timeout
        try:
            skill = self._skill_registry.get(strategy_name)
        except Exception:  # noqa: BLE001
            return plan_timeout
        if skill is None:
            return plan_timeout
        floor = getattr(skill, "typical_duration_sec", 0.0)
        if not isinstance(floor, (int, float)):
            return plan_timeout
        return max(plan_timeout, float(floor))

    def _make_step(
        self,
        sub_goal: SubGoal,
        result: Any,
        success: bool,
        verify_result: bool,
        error: str,
        start: float,
        fallback_used: bool,
        result_data: dict | None = None,
        failure_class: str = "",
    ) -> StepRecord:
        """Convenience factory for StepRecord.

        ``failure_class`` (W2.4) is the deterministic typed failure class for a
        FAILED step (one of FAILURE_CLASSES; "" for success). Defaulted "" so
        existing callers are unaffected.
        """
        strategy_name = self._extract_name(result) if result is not None else ""
        return StepRecord(
            sub_goal_name=sub_goal.name,
            strategy=strategy_name,
            success=success,
            verify_result=verify_result,
            duration_sec=time.monotonic() - start,
            error=error,
            fallback_used=fallback_used,
            result_data=result_data or {},
            failure_class=failure_class,
        )

    @staticmethod
    def _coerce_output(value: Any) -> dict:
        """Coerce a strategy return value into a structured-output dict.

        A dict passes through unchanged; any other non-None value is wrapped under
        a ``"value"`` key so the captured output contract is always a dict. None /
        empty yields an empty dict.
        """
        if isinstance(value, dict):
            return value
        if value is None:
            return {}
        return {"value": value}
