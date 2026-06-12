# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

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
from dataclasses import dataclass
from typing import Any, Callable

from vector_os_nano.vcli.cognitive.goal_executor import (
    blocking_dependency,
    skipped_step_record,
)
from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HarnessConfig:
    """Configuration for VGG retry behavior.

    Layer 2 (mid-tree re-decompose) intentionally does NOT exist: Layer-1
    strategy retry + Layer-3 whole-tree replan cover it, and rewriting a
    half-executed tree is high-complexity/low-yield (design review 2026-06-12
    §5). The dead ``max_redecompose`` knob that promised it was deleted —
    do not reintroduce.
    """
    max_step_retries: int = 2       # per-step strategy retries (Layer 1)
    max_pipeline_retries: int = 1   # full re-plan attempts (Layer 3)
    # Stage 4 (S4-4) — observation-driven mid-tree replan. The maximum number of
    # times the harness may re-decompose a SUCCEEDING run because a step's
    # result_data showed a live observation that diverged from the plan's
    # assumptions (e.g. detect found a different object set). Bounded so a
    # detector that always reports divergence cannot loop forever. Only consulted
    # when an ``observation_divergence`` detector is wired into the harness; with
    # no detector the run path is byte-identical. Additive + LAST + defaulted so
    # existing HarnessConfig(...) constructions are unaffected.
    max_obs_replan: int = 1


@dataclass(frozen=True)
class FailureRecord:
    """Records a failure for feedback to the decomposer."""
    sub_goal_name: str
    strategy_tried: str
    error: str
    step_index: int
    # W2.4 — the DETERMINISTIC typed failure class for this failed step (one of
    # types.FAILURE_CLASSES; "" when the producing StepRecord carried none). It
    # is threaded into the re-decompose context so the LLM re-plan can branch on
    # the failure CLASS (e.g. timeout -> a shorter/faster strategy; ik_fail -> an
    # alternate grasp pose) instead of parsing the opaque ``error`` string.
    # SECURITY (rule 10): a bounded enum string only — no raw exception detail or
    # paths. Additive + LAST + defaulted "" so existing constructions are
    # byte-unaffected (rule 6).
    failure_class: str = ""


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
        observation_divergence: Callable[[ExecutionTrace], str | None] | None = None,
    ) -> None:
        """Construct the harness.

        Args:
            observation_divergence: Optional detector for observation-driven
                mid-tree replan (Stage 4, S4-4). Given the just-completed
                ExecutionTrace, it returns a non-empty reason string when a step's
                live observation (its ``result_data``) diverged from what the plan
                assumed — e.g. a detect step found a different object set, or a
                post-pick re-detect changed the remaining set. Returning ``None`` /
                "" means no divergence. When this is ``None`` (the default) the
                observation-replan loop is entirely skipped and ``run()`` is
                byte-identical to before. The detector is a PURE read over the
                trace/result_data — it never executes model output.
        """
        self._decomposer = decomposer
        self._executor = executor
        self._selector = selector
        self._config = config or HarnessConfig()
        self._on_step = on_step
        self._on_replan = on_replan
        self._obs_divergence = observation_divergence

    def run(
        self,
        task: str,
        world_context: str,
        goal_tree: GoalTree | None = None,
        context_provider: Callable[[], str] | None = None,
    ) -> ExecutionTrace:
        """Run task with full feedback loop.

        Args:
            task: Natural language task description.
            world_context: Current world model summary (static fallback).
            goal_tree: Pre-decomposed GoalTree (skip decomposition if provided).
            context_provider: Optional callable that (re)builds the world context
                on demand (Stage 1b). When given, it is called to produce a FRESH
                world context for the initial decompose and before every
                re-decompose, so replans see current state. When ``None`` the
                static *world_context* arg is used (current behavior).

        Returns:
            Best ExecutionTrace achieved across all retry attempts.
        """
        cfg = self._config
        failures: list[FailureRecord] = []
        best_trace: ExecutionTrace | None = None
        tree: GoalTree | None = None
        last_trace: ExecutionTrace | None = None  # #17: completed-step source
        obs_replans_used = 0  # S4-4: bounded observation-driven re-decomposes

        # Data binding (Stage 1b): one fresh Blackboard scoped to this run. The
        # executor captures each successful step's output here and resolves later
        # steps' ``${step.key}`` params against it. Fail-soft: if the executor
        # has no blackboard attribute (e.g. a bare mock) capture is simply off.
        self._attach_blackboard()

        for pipeline_attempt in range(cfg.max_pipeline_retries + 1):
            # --- Abort check ---
            try:
                from vector_os_nano.vcli.cognitive.abort import is_abort_requested
                if is_abort_requested():
                    break
            except ImportError:
                pass

            # --- Decompose (or use provided tree) ---
            if goal_tree is not None and pipeline_attempt == 0:
                tree = goal_tree
            else:
                fresh_context = self._current_context(world_context, context_provider)
                # Validator feedback (Stage 2b): carry the PRIOR attempt's dropped/
                # invalid-strategy notes into this decompose so the next plan stops
                # repeating the hallucination. ``tree`` still holds the previous
                # attempt's GoalTree here (None on the very first attempt).
                prior_notes = tuple(getattr(tree, "validation_notes", ()) or ())
                prev_tree = tree
                tree = self._decompose_with_context(
                    task, fresh_context, failures, prior_notes,
                    completed=self._completed_step_lines(last_trace),
                )
                if tree is None:
                    logger.warning("VGGHarness: decomposition failed on attempt %d", pipeline_attempt)
                    break
                # Owner finding (c): a replanned step reusing a strategy must
                # never DEGRADE to empty params (the '走到sofa' replan lost
                # {"label": "sofa"} and produced an opaque bad_params error).
                tree = self._inherit_replan_params(tree, prev_tree)

            # --- Execute with step-level retry ---
            trace = self._execute_with_retry(tree, failures)
            last_trace = trace

            # Track best result
            if best_trace is None or trace.success:
                best_trace = trace

            if trace.success:
                # --- S4-4: observation-driven mid-tree replan ---
                # Even a fully-verified run may have observed live state that the
                # plan did not assume (e.g. detect found a different object set, or
                # a post-pick re-detect changed the remaining set). When a divergence
                # detector is wired AND replan budget remains, re-decompose against
                # the CURRENT world_context/Blackboard (not the stale T=0 context)
                # and re-execute. This is its OWN bounded loop, independent of the
                # pipeline-retry counter (max_obs_replan), so a detector that always
                # reports divergence cannot loop forever.
                while True:
                    reason = self._obs_divergence_reason(trace, obs_replans_used, cfg)
                    if not reason:
                        break
                    obs_replans_used += 1
                    new_tree = self._obs_replan_decompose(
                        task, world_context, context_provider, failures, tree, reason
                    )
                    if new_tree is None:
                        return trace  # re-decompose failed — keep the verified run
                    tree = self._inherit_replan_params(new_tree, tree)
                    trace = self._execute_with_retry(tree, failures)
                    if trace.success:
                        best_trace = trace
                    else:
                        # The post-divergence plan did not verify; fall back to
                        # Layer 3 pipeline retry with the accumulated failures.
                        break
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

    def _attach_blackboard(self) -> None:
        """Attach a fresh run-scoped Blackboard to the executor (Stage 1b).

        Fail-soft: a missing import or an executor that does not accept a
        ``blackboard`` (e.g. a bare test double) leaves capture disabled and
        never aborts the run.
        """
        try:
            from vector_os_nano.vcli.cognitive.blackboard import Blackboard
            self._executor.blackboard = Blackboard()
        except Exception as exc:  # noqa: BLE001
            logger.debug("VGGHarness: could not attach blackboard: %s", exc)

    @staticmethod
    def _current_context(
        world_context: str,
        context_provider: Callable[[], str] | None,
    ) -> str:
        """Return a fresh world context (Stage 1b).

        Calls *context_provider* when supplied so each (re)decompose sees current
        state; on any failure or when no provider is given, falls back to the
        static *world_context* (current behavior).
        """
        if context_provider is None:
            return world_context
        try:
            fresh = context_provider()
        except Exception as exc:  # noqa: BLE001
            logger.warning("VGGHarness: context_provider raised: %s", exc)
            return world_context
        return fresh if isinstance(fresh, str) else world_context

    def _obs_divergence_reason(
        self,
        trace: ExecutionTrace,
        obs_replans_used: int,
        cfg: HarnessConfig,
    ) -> str | None:
        """Return the divergence reason for a verified run, or ``None`` (S4-4).

        Consulted ONCE per replan decision (the reason is then threaded into the
        re-decompose so the detector is never invoked twice for one decision).
        Returns a non-empty reason only when a divergence detector is wired, the
        per-run observation-replan budget (``cfg.max_obs_replan``) is not yet
        spent, and the detector reports a non-empty reason for *trace*. With no
        detector this is always ``None``, so the run path is unchanged. The
        detector is consulted defensively — any exception is treated as
        "no divergence" so a misbehaving detector can never abort a verified run.
        """
        if self._obs_divergence is None:
            return None
        if obs_replans_used >= max(0, cfg.max_obs_replan):
            return None
        try:
            reason = self._obs_divergence(trace)
        except Exception as exc:  # noqa: BLE001
            logger.warning("VGGHarness: observation_divergence raised: %s", exc)
            return None
        return reason if reason else None

    @staticmethod
    def _inherit_replan_params(
        new_tree: "GoalTree", prev_tree: "GoalTree | None"
    ) -> "GoalTree":
        """Carry prior param bindings into a replanned tree (owner finding (c)).

        #18 (campaign #4) — TARGET-AWARE matching: a sub-goal in *new_tree*
        with EMPTY ``strategy_params`` inherits by ``(strategy, sub_goal
        name)`` first; the strategy-level fallback applies ONLY when the
        prior tree bound that strategy in exactly ONE step (nothing to guess
        over). Two prior navigates with different targets + a replanned step
        matching neither name stays EMPTY — it fails bad_params and the
        replan hears it, instead of the old 'latest wins' silently sending
        BOTH navigates to the same target. Non-empty new params always win —
        the LLM may deliberately re-bind. Deterministic, kernel-side;
        identity when nothing applies.
        """
        if prev_tree is None:
            return new_tree

        def _norm(strategy: str) -> str:
            # 'navigate_to' and 'navigate_to_skill' are the same route — the
            # LLM uses both forms across replans.
            return strategy[:-6] if strategy.endswith("_skill") else strategy

        by_key: dict[tuple[str, str], dict] = {}
        strategy_bindings: dict[str, list[dict]] = {}
        for sg in prev_tree.sub_goals:
            if sg.strategy and sg.strategy_params:
                s = _norm(sg.strategy)
                by_key[(s, sg.name)] = sg.strategy_params
                strategy_bindings.setdefault(s, []).append(sg.strategy_params)
        if not strategy_bindings:
            return new_tree

        import dataclasses

        changed = False
        merged = []
        for sg in new_tree.sub_goals:
            if not sg.strategy_params:
                s = _norm(sg.strategy)
                params = by_key.get((s, sg.name))
                if params is None and len(strategy_bindings.get(s, ())) == 1:
                    params = strategy_bindings[s][0]  # unambiguous fallback
                if params is not None:
                    sg = dataclasses.replace(sg, strategy_params=dict(params))
                    changed = True
            merged.append(sg)
        if not changed:
            return new_tree
        return dataclasses.replace(new_tree, sub_goals=tuple(merged))

    def _obs_replan_decompose(
        self,
        task: str,
        world_context: str,
        context_provider: Callable[[], str] | None,
        failures: list[FailureRecord],
        prior_tree: GoalTree | None,
        reason: str,
    ) -> GoalTree | None:
        """Re-decompose after an observed divergence on a verified run (S4-4).

        Builds a FRESH world context via *context_provider* (so the new plan sees
        CURRENT state, not the stale T=0 context), threads the detector's *reason*
        + the prior plan's validation notes into the decompose context, fires the
        ``on_replan`` callback, and returns the new GoalTree (or ``None`` on
        decompose failure). The run-scoped Blackboard is intentionally NOT reset:
        the prior run's captured observations (e.g. the latest detect set) remain
        addressable so a foreach in the new plan iterates the CURRENT set.
        """
        if self._on_replan:
            try:
                self._on_replan(
                    "Re-planning on observed divergence: "
                    f"{reason or 'live observation changed'}"
                )
            except Exception:  # noqa: BLE001
                pass

        fresh_context = self._current_context(world_context, context_provider)
        if reason:
            fresh_context += (
                "\n\nObservation diverged from the previous plan (re-plan around the "
                f"CURRENT observed state):\n  - {reason}"
            )
        prior_notes = tuple(getattr(prior_tree, "validation_notes", ()) or ())
        logger.info(
            "VGGHarness: observation-driven replan (reason=%r)", reason or "(none)"
        )
        return self._decompose_with_context(task, fresh_context, failures, prior_notes)

    def _completed_step_lines(
        self, trace: ExecutionTrace | None
    ) -> tuple[str, ...]:
        """One line per VERIFIED step of the previous attempt (#17).

        The re-decompose used to see only failures, so it re-planned (and the
        robot REPLAYED) motions that had already succeeded. Skipped /
        visual-override steps don't count — only deterministic passes.
        """
        if trace is None:
            return ()
        return tuple(
            f"{s.sub_goal_name} (strategy: {s.strategy or '?'}, verified)"
            for s in trace.steps
            if s.success and s.verify_result and not s.visual_override
        )

    def _decompose_with_context(
        self,
        task: str,
        world_context: str,
        failures: list[FailureRecord],
        prior_validation_notes: tuple[str, ...] = (),
        completed: tuple[str, ...] = (),
    ) -> GoalTree | None:
        """Decompose with failure history + prior validator feedback in context.

        *prior_validation_notes* are the previous attempt's
        ``GoalTree.validation_notes`` (Stage 2b) — dropped/invalid-strategy
        messages. Injecting them tells the next decompose which strategies were
        invalid so it stops hallucinating them.
        """
        enriched_context = world_context
        if completed:
            done_block = "\n".join(f"  - {line}" for line in completed)
            enriched_context += (
                "\n\nAlready COMPLETED and verified in the previous attempt "
                "(do NOT plan these again — their outputs are still available "
                "via ${step.path} bindings; plan only the remainder):\n"
                f"{done_block}"
            )
        if failures:
            failure_summary = "\n".join(
                self._format_failure_line(f) for f in failures[-5:]  # last 5 failures
            )
            enriched_context += (
                f"\n\nPrevious failures (avoid these strategies):\n{failure_summary}"
            )
            # W2.4: a small DETERMINISTIC hint per typed failure class present in
            # the recent failures, so the LLM re-plan can ADAPT to the class
            # (timeout -> shorter/faster; ik_fail -> alternate pose; ...). The
            # value is the LLM reasoning over the typed signal — no brittle
            # deterministic strategy switch in the selector.
            hint_block = self._failure_class_hints(failures[-5:])
            if hint_block:
                enriched_context += f"\n{hint_block}"

        if prior_validation_notes:
            invalid = self._invalid_strategies_from_notes(prior_validation_notes)
            notes_block = "\n".join(f"  - {n}" for n in prior_validation_notes)
            enriched_context += (
                "\n\nValidator rejected part of the previous plan:\n"
                f"{notes_block}"
            )
            if invalid:
                enriched_context += (
                    "\nDo NOT use these invalid strategies again: "
                    f"{', '.join(invalid)}; valid strategies are listed in "
                    "KNOWN_STRATEGIES above."
                )

        try:
            return self._decomposer.decompose(task, enriched_context)
        except Exception as exc:
            logger.warning("VGGHarness: decompose raised: %s", exc)
            return None

    # W2.4 — typed-failure-class context surfacing
    # ------------------------------------------------------------------

    # A short, deterministic adaptation hint per failure class. The LLM does the
    # actual adapting; this only names the class + the kind of change that helps,
    # so the re-plan reads a typed signal instead of guessing from the error
    # string. Bounded enum keys only — no per-failure free text here.
    _FAILURE_CLASS_HINTS: dict[str, str] = {
        "timeout": (
            "timeout: the step exceeded its time budget — prefer a shorter/faster "
            "strategy or a larger timeout_sec."
        ),
        "ik_fail": (
            "ik_fail: a target was unreachable — try an alternate grasp/approach "
            "pose, reposition, or a different object."
        ),
        "verify_fail": (
            "verify_fail: the step ran but its check did not pass — try a different "
            "strategy or add a precondition step."
        ),
        "tool_error": (
            "tool_error: a kernel tool was denied or failed — use an allowed "
            "tool/strategy instead."
        ),
        "exec_error": (
            "exec_error: the strategy raised or could not run — pick a valid, "
            "registered strategy."
        ),
    }

    @staticmethod
    def _format_failure_line(f: "FailureRecord") -> str:
        """Render one failure as a human-readable context line (W2.4).

        Prefixes the typed ``[failure_class]`` tag (a bounded enum string) so the
        re-decompose sees the CLASS, not just the opaque ``error``. The error text
        itself is the StepRecord's existing ``error`` field — this change does NOT
        widen what that captures (rule 10).
        """
        cls = getattr(f, "failure_class", "") or ""
        tag = f" [{cls}]" if cls else ""
        return f"  - {f.sub_goal_name}: {f.strategy_tried} failed{tag} ({f.error})"

    @classmethod
    def _failure_class_hints(cls, failures: list["FailureRecord"]) -> str:
        """Build a deterministic per-class hint block for the recent failures.

        One line per DISTINCT failure class present (in first-seen order), drawn
        from the bounded ``_FAILURE_CLASS_HINTS`` table. Returns "" when no recent
        failure carries a typed class, so the path is byte-identical for the
        untyped/abort case.
        """
        seen: list[str] = []
        for f in failures:
            c = getattr(f, "failure_class", "") or ""
            if c and c in cls._FAILURE_CLASS_HINTS and c not in seen:
                seen.append(c)
        if not seen:
            return ""
        lines = "\n".join(f"  - {cls._FAILURE_CLASS_HINTS[c]}" for c in seen)
        return f"Failure classes observed (adapt the plan to each):\n{lines}"

    @staticmethod
    def _invalid_strategies_from_notes(notes: tuple[str, ...]) -> list[str]:
        """Extract the invalid strategy names quoted in validator notes.

        Notes have the shape ``strategy 'look_skill' is not valid; ...``. The
        first single-quoted token of such a note is the offending strategy.
        Best-effort and side-effect-free; returns a de-duplicated, ordered list.
        """
        import re

        found: list[str] = []
        for note in notes:
            if "is not valid" not in note:
                continue
            m = re.search(r"'([^']+)'", note)
            if m and m.group(1) not in found:
                found.append(m.group(1))
        return found

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
        failed_names: set[str] = set()

        ordered = self._executor._topological_sort(tree)

        for i, sub_goal in enumerate(ordered):
            # --- Abort check ---
            try:
                from vector_os_nano.vcli.cognitive.abort import is_abort_requested
                if is_abort_requested():
                    overall_success = False
                    break
            except ImportError:
                pass

            # #11 — dependency-failure skip (the contract the old comment here
            # PROMISED without implementing). Shared helper with
            # GoalExecutor.execute: one tree, one failure semantics.
            blocked_by = blocking_dependency(sub_goal, failed_names)
            if blocked_by:
                skip = skipped_step_record(sub_goal, blocked_by)
                steps.append(skip)
                failed_names.add(sub_goal.name)  # poison transitively
                overall_success = False
                if self._on_step:
                    try:
                        self._on_step(skip)
                    except Exception:
                        pass
                failures.append(FailureRecord(
                    sub_goal_name=skip.sub_goal_name,
                    strategy_tried=skip.strategy,
                    error=skip.error,
                    step_index=i,
                    failure_class="dep_skipped",
                ))
                continue

            # Stage 4 (S4-2): a foreach node expands at runtime into N children.
            # Delegate the whole expansion to the executor (it reads the producing
            # step's list off the run blackboard and binds the iteration var per
            # item). Layer-1 strategy retry does not apply to the loop node itself;
            # each expanded child still carries its own verify.
            if getattr(sub_goal, "foreach", None) is not None:
                expanded = self._executor._execute_foreach(sub_goal, self._on_step)
                steps.extend(expanded)
                # NOTE (H-1a): _execute_foreach already records each child's
                # StrategyStats internally. Do NOT re-record here — doing so
                # double-counts every foreach child whenever a live stats object
                # is attached. We only need to harvest failures for replan context.
                for child in expanded:
                    if not child.success:
                        failures.append(FailureRecord(
                            sub_goal_name=child.sub_goal_name,
                            strategy_tried=child.strategy,
                            error=child.error,
                            step_index=i,
                            failure_class=getattr(child, "failure_class", ""),
                        ))
                        overall_success = False
                        failed_names.add(sub_goal.name)  # poison dependents
                continue

            step = self._execute_step_with_retry(sub_goal, i, cfg.max_step_retries)
            steps.append(step)

            if self._on_step:
                try:
                    self._on_step(step)
                except Exception:
                    pass

            # Record stats (W1.1: evidence-gated via the executor's single chokepoint,
            # which owns is_robot — the harness no longer carries the gate).
            self._executor._record_strategy_stats(step, sub_goal)

            if not step.success:
                failures.append(FailureRecord(
                    sub_goal_name=step.sub_goal_name,
                    strategy_tried=step.strategy,
                    error=step.error,
                    step_index=i,
                    failure_class=getattr(step, "failure_class", ""),
                ))
                overall_success = False
                failed_names.add(sub_goal.name)
                # Don't break — independent steps still run; steps whose
                # depends_on references this one get a skipped record above.

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
                # Retry: normally clear the explicit strategy so the selector can
                # pick a *different* alternative (in case the original strategy is
                # what failed). But only do so when the cleared, empty-strategy
                # selector path can actually re-derive a usable strategy. On a
                # world with no fallback route for this sub-goal (e.g. a baseless
                # arm world: GO2 keyword ladder gated off + no registry alias
                # match) clearing resolves to the opaque ``fallback``/``unmatched``
                # (or ``invalid``) result, which would DESTROY a valid explicit
                # strategy and mask the real failure (rule 8: fail loud with the
                # real cause, never a silent wrong fallback). In that case keep the
                # original explicit strategy so the step routes to the real skill
                # and surfaces the honest failure. World-agnostic: the decision is
                # driven solely by what the selector resolves, not the embodiment.
                retry_goal = SubGoal(
                    name=sub_goal.name,
                    description=sub_goal.description,
                    verify=sub_goal.verify,
                    timeout_sec=sub_goal.timeout_sec,
                    depends_on=sub_goal.depends_on,
                    strategy=self._retry_strategy(sub_goal),
                    strategy_params=sub_goal.strategy_params,
                    fail_action="",
                    # Preserve the fail-loud hallucination marker (rule 8): a step
                    # whose explicit strategy was an unknown hallucination must keep
                    # surfacing the clear ``invalid`` error on every retry, never
                    # silently re-route the cleared (empty) strategy to a phantom
                    # skill / ``unmatched`` fallback. Empty for normal steps, so the
                    # retry path is byte-identical for them.
                    cleared_strategy=sub_goal.cleared_strategy,
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

    def _retry_strategy(self, sub_goal: SubGoal) -> str:
        """Choose the strategy string for a retry attempt.

        Returns ``""`` (clear → let the selector pick a fresh alternative) only
        when the empty-strategy selector path can actually re-derive a usable
        strategy for this sub-goal. If clearing would resolve to a
        non-actionable result — the opaque ``fallback``/``unmatched`` or a
        fail-loud ``invalid`` (e.g. a baseless arm world where the GO2 keyword
        ladder is gated off and the registry alias match misses) — the original
        explicit strategy is KEPT instead, so the step routes to the real skill
        and surfaces the honest failure rather than throwing the valid strategy
        away (rule 8). No explicit strategy to begin with → nothing to preserve,
        so the path is byte-identical (``""``).

        Probe uses the harness's selector when available, falling back to the
        executor's selector; if neither exposes ``select`` the historical clear
        behaviour is preserved.
        """
        original = sub_goal.strategy
        if not original:
            return ""  # no explicit strategy — byte-identical to old clear path

        selector = self._selector or getattr(self._executor, "_selector", None)
        select = getattr(selector, "select", None)
        if not callable(select):
            return ""  # cannot probe — preserve historical behaviour

        cleared = SubGoal(
            name=sub_goal.name,
            description=sub_goal.description,
            verify=sub_goal.verify,
            timeout_sec=sub_goal.timeout_sec,
            depends_on=sub_goal.depends_on,
            strategy="",
            strategy_params=sub_goal.strategy_params,
            fail_action="",
        )
        try:
            probe = select(cleared)
        except Exception:  # noqa: BLE001 — probe is best-effort; never break retry
            return ""
        # A cleared strategy that resolves to a real, actionable executor is a
        # genuine alternative → clear. Otherwise keep the explicit strategy.
        if getattr(probe, "executor_type", None) in {"fallback", "invalid"}:
            return original
        return ""
