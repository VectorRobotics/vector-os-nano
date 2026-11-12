# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""native_runner — the step runner: dispatch, verify, grade, record (``NativeStepRunner``).

The model issues the tool calls; the runner gates and dispatches each one, evaluates the
model's ``verify(expr)`` with the live GoalVerifier, grades actor causation and appends
the StepRecords the spine reads. It never decides what to do next.

StepRecord granularity (review fix 2): EXACTLY ONE StepRecord per
(action-chain -> verify) pair. Intermediate bare skill calls between two
``verify`` calls are NOT each a checked sub-goal and are NOT sentinel-verified
(that would re-open the no-op/teleport hole). The ``verify`` tool HANDLER is what
appends the StepRecord, binding the just-run skill calls as the producing action
and the model's expr as the SubGoal.verify.

Capture/grade timing (review fix 3):
the actor-causation baseline is captured immediately BEFORE the first skill call
of a step; the grade reads a FRESH post-capture immediately AFTER evaluating that
step's verify expr. A re-verify of the same predicate (an attempt group, WP07) that
retried or only re-checked the latest attempt is graded against that attempt's
baseline; after any other action it is graded on its own step. Only the actor's own
tools count as commanded motion (the bound world's ``actor_epoch()``).
"""
from __future__ import annotations

import contextlib
import logging
import re
import string
from dataclasses import replace
from typing import Any, Callable

from vector_os_nano.vcli.cognitive import actor_causation
from vector_os_nano.vcli.cognitive.evidence_classifier import classify_verify_expr
from vector_os_nano.vcli.cognitive.trace_store import attempt_key, graded_indices, turn_succeeded
from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)
from vector_os_nano.vcli.permissions import PermissionContext
from vector_os_nano.vcli.tool_execution import execute_resolved_tool, resolve_permission
from vector_os_nano.vcli.tools.base import ToolContext, ToolResult

logger = logging.getLogger(__name__)

# The sim lifecycle tools: they change WHICH body the turn drives. They count as an
# action (the turn is owned; nothing re-runs them) but prove nothing themselves, so
# they never open a graded step; after one succeeds the runner rebinds the turn to
# the new body (tools, verifier, oracle names, causation baseline).
LIFECYCLE_TOOLS: frozenset[str] = frozenset({"start_simulation", "stop_simulation"})


# ---------------------------------------------------------------------------
# NativeStepRunner — the ONLY stateful piece (capture -> dispatch -> verify -> grade)
# ---------------------------------------------------------------------------


def sim_session(world: Any, agent: Any) -> Any:
    """*agent*'s live sim session, asked of the bound *world* (None without one)."""
    return world.session(agent) if world is not None else None


def actor_state(world: Any, agent: Any) -> Any:
    """The causation baseline of *agent*: its drivers' motion counters, the true pose of
    its sim session (none outside a sim: every robot channel then fails closed)."""
    return actor_causation.capture(agent, sim_session(world, agent))


class NativeStepRunner:
    """Assemble an ExecutionTrace from a native tool-use turn loop.

    Holds the live GoalVerifier namespace + oracle names/tags + the motor tool-set,
    and the per-step accumulator (the action chain + its capture baseline). The
    MODEL issues the tool calls; this runner only dispatches, verifies, grades, and
    records — never decides what to do next.

    WP07 bookkeeping (no planning):
    - attempt groups: a verify is keyed by its normalized AST (+ an optional model
      ``goal_id``); a re-verify of the same predicate supersedes the earlier row (the
      spine grades the latest only — ``trace_store.graded_indices``) and is graded
      against the group's baseline only when it retried or re-checked the latest
      attempt (``_group_baseline``), else on its own step;
    - failed attempts and recovery (charter 3.5): a side-effecting tool that reports an
      error with no commanded motion (``actor_causation.commanded_nothing``) was REFUSED
      — not an action, kept on its step's row; one that commanded motion is a FAILED
      ATTEMPT, recorded at once as its own FAILED row, so a later action's row is
      neither failed nor backed by it;
    - auto post-verify: an actuating tool whose ``verify_hint`` binds from its call
      params is checked right after it runs; that check closes the step unless the
      model verifies first (then the model's predicate is the step's — but a FAILED
      auto-verify of another predicate stays its own row);
    - probe refusal: a model verify with no action since its last verify is refused
      unevaluated (no GT bit leak);
    - every dispatch runs inside the bound world's ``actor_epoch()``, so only the
      actor's own commanded motion counts toward causation.
    """

    def __init__(
        self,
        agent: Any,
        verifier: Any,
        oracle_names: frozenset[str],
        motor_tools: dict[str, Any],
        tool_context: ToolContext,
        ask_permission: Callable[[str, dict[str, Any]], str] | None = None,
        oracles: dict[str, Any] | None = None,
        world: Any = None,
    ) -> None:
        self._agent = agent
        self._world = world  # the bound world: the body's sim session and actor epoch
        self._verifier = verifier
        self._oracle_names = oracle_names
        # {name: OracleTag}: the channel / transition the causation grader reads.
        self._oracles = oracles if oracles is not None else {}
        self._motor_tools = motor_tools
        self._ctx = tool_context
        # Resolves an "ask" permission (the REPL's prompt). None denies: the
        # headless default — only --no-permission (PermissionContext) allows.
        self._ask = ask_permission
        # The actor state when the turn (or its current body) started: a turn-window
        # oracle (kept_holding) is caused only by a transition since then.
        self._turn_baseline: Any = actor_state(world, agent)

        # Per-step accumulator: the producing action chain (skill names) + the
        # capture baseline taken BEFORE the first skill of the current step.
        self._chain: list[str] = []
        self._baseline: Any = None
        self._step_open: bool = False
        self._step_acted: bool = False  # a side-effecting tool ran this step
        # Every tool of the step's chain is read-only or a declared housekeeping
        # action (stop): the row may stay un-GROUNDED when another row proved the turn
        # (trace_store.turn_reject_reason), and a re-verify after it only re-checks.
        self._housekeeping: bool = True
        self._last_ok: bool = True      # the chain's last tool did not report an error
        self._acting: set[str] = set()  # the step's side-effecting tools
        self._refused_calls: list[str] = []  # "<tool>: <message>" refused in this step
        self._last_result: ToolResult | None = None
        # The auto post-verify of the last actuating tool, until the step closes.
        self._pending: dict[str, Any] | None = None
        # Probe refusal: a model verify needs a side-effecting action since the last.
        self._acted_since_verify: bool = True
        # attempt group -> (the baseline its LATEST attempt was graded against, that
        # attempt's side-effecting tools). See ``_group_baseline``.
        self._group_baselines: dict[tuple[str, str], tuple[Any, frozenset[str]]] = {}
        # Manipulation intents of the dispatched skills ("grasp" / "place").
        self.intents: set[str] = set()

        # Side-effecting tool calls over the whole turn: nonzero means the turn
        # acted, so no other producer may re-run it (no double execution).
        self.actions_dispatched: int = 0
        # Calls the permission gate refused ("<tool>: <why>"): the user sees them, and
        # no other producer re-attempts what was just refused.
        self.refused: list[str] = []

        # The assembled trace pieces (one SubGoal + one StepRecord per verify pair).
        self._sub_goals: list[SubGoal] = []
        self._steps: list[StepRecord] = []
        self._step_idx: int = 0

    @property
    def has_unverified_action(self) -> bool:
        """True iff a SIDE-EFFECTING tool ran for the current step and nothing proved
        it yet — neither a model verify nor an auto post-verify (D23: weak models skip
        verify on easy tasks). A read (file_read, detect, where_am_i) changes nothing,
        so it is never nudged: the model answers from it.
        """
        return self._step_acted and self._pending is None

    # ------------------------------------------------------------------
    # Skill dispatch (capture baseline before the first skill of a step)
    # ------------------------------------------------------------------

    def dispatch_skill(self, name: str, params: dict[str, Any]) -> ToolResult:
        """Gate, then run a motor/skill tool; open a step if needed.

        Every dispatch goes through the permission gate (``resolve_permission``):
        an intrinsic deny (the bash deny-list, a
        protected path) never runs; an ``ask`` runs only on an explicit yes from the
        resolver (None denies — the headless default). A denied call is not an action.

        The actor-causation baseline is captured immediately BEFORE the FIRST skill
        call of the current step (review fix 3), then frozen — advancing the robot
        afterwards cannot mutate it (ActorBaseline is a value snapshot).
        """
        tool = self._motor_tools.get(name)
        if tool is None:
            return ToolResult(content=f"Unknown tool '{name}'.", is_error=True)
        decision = resolve_permission(
            tool, params, self._ctx, self._ctx.permissions or PermissionContext(), self._ask,
        )
        if not decision.allowed:
            why = decision.reason if decision.kind == "deny" else (
                "not allowed (permission was not granted)" if decision.kind == "ask_deny"
                else f"permission check failed: {decision.reason}"
            )
            self.refused.append(f"{name}: {why}")
            return ToolResult(content=f"Permission denied for {name}: {why}. The tool did NOT run.",
                              is_error=True)
        if name in LIFECYCLE_TOOLS:
            # The body is about to change: a chain on the old one is never verified now.
            self.close_chain()
            self.actions_dispatched += 1
            return execute_resolved_tool(
                tool, params, self._ctx, error_prefix=f"Tool '{name}' raised",
                on_error=lambda _n, exc: logger.debug("native_loop: %r raised: %s", name, exc),
            )
        try:
            read_only = bool(tool.is_read_only(params))
        except Exception:  # noqa: BLE001 — no/broken declaration: fail closed (it acts)
            read_only = False
        if read_only:
            self._open_step(actor_state(self._world, self._agent))
            self._chain.append(name)
            result = self._run(tool, name, params)
            self._last_ok, self._last_result = not result.is_error, result
            return result
        self.actions_dispatched += 1
        self._note_intents(tool)
        hint = self._bound_hint(tool, params)
        before = self._check(hint) if hint else None  # F->T: the hint at the baseline
        pre = actor_state(self._world, self._agent)
        result = self._run(tool, name, params)
        # A call the owner stopped mid-run stays on the open step: the abort row records
        # it (``record_abort``); only another error is refused or a failed attempt.
        abort = getattr(self._ctx, "abort", None)
        error = result.is_error and not (abort is not None and abort.is_set())
        if error and actor_causation.commanded_nothing(
                self._agent, pre, actor_state(self._world, self._agent)):
            # Charter 3.5: REFUSED — an error with no commanded motion is not an action.
            # The model reads its message; the row of the step it is in records the try.
            self._open_step(pre)
            self._refused_calls.append(f"{name}: {str(result.content).strip().splitlines()[0][:200]}"
                                       if str(result.content).strip() else name)
            return result
        self._flush_pending()  # an auto-verified step closes before the next action
        if error and self._step_acted:
            # An earlier action of this step succeeded unverified: its own unproven row,
            # never folded into the failed attempt's (which a proven goal may excuse).
            self._record_open("not verified", success=self._last_ok)
        self._open_step(pre)
        self._chain.append(name)
        if getattr(tool, "housekeeping", False) is not True:
            self._housekeeping = False
        self._step_acted = self._acted_since_verify = True
        self._acting.add(name)
        self._last_ok, self._last_result = not result.is_error, result
        if error:
            # Charter 3.5: a FAILED ATTEMPT (it commanded motion) closes its step at once,
            # under its bound hint, so it neither fails nor backs a later action's row;
            # what follows is graded on a fresh baseline, never an earlier attempt's.
            self._record(hint or "", False, actor_causation.ActorCaused.NOT_GRADED, ("failed attempt",),
                         "", self._result_data(), note="failed attempt", failed=True)
            self._group_baselines.clear()
            return result
        if hint and self._last_ok:
            ok = self._check(hint)
            caused, channels = self._grade(hint)
            # F->T for THIS action — unless it re-attempts a group whose earlier attempt
            # made it true: then the group baseline (in _grade) is the judge.
            if before and self._group_baseline(("native", attempt_key(hint))) is None:
                caused, channels = actor_causation.ActorCaused.UNCAUSED, ("already true before the action",)
            # The auto check belongs to THIS tool: keep its own result for the record
            # (a later read in the same step must not stand in for it).
            self._pending = {"expr": hint, "ok": ok, "caused": caused, "channels": channels,
                             "data": self._result_data(), "chain": len(self._chain)}
            result = ToolResult(
                content=f"{result.content}\nauto-verify {hint} -> {verify_word(ok)}",
                is_error=result.is_error, metadata=result.metadata,
            )
        return result

    def _open_step(self, baseline: Any) -> None:
        """Open a step if none is: *baseline* (captured before its first tool) is frozen
        as its causation baseline."""
        if not self._step_open:
            self._baseline, self._step_open = baseline, True

    def _run(self, tool: Any, name: str, params: dict[str, Any]) -> ToolResult:
        """Run *tool* inside the bound world's actor epoch (only its motion is the actor's)."""
        with self._world.actor_epoch() if self._world is not None else contextlib.nullcontext():
            return execute_resolved_tool(
                tool, params, self._ctx, error_prefix=f"Skill '{name}' raised",
                on_error=lambda _n, exc: logger.debug("native_loop: skill %r raised: %s", name, exc),
            )

    def _bound_hint(self, tool: Any, params: dict[str, Any]) -> str | None:
        """The tool's ``verify_hint`` bound from its call params, if it binds to a GT
        evidence predicate (``at_position({x}, {y})`` -> ``at_position(2.0, 3.0)``)."""
        hint = str(getattr(tool, "verify_hint", "") or "").strip()
        if not hint or hint == "True":
            return None
        try:
            bound = hint.format_map({k: repr(v) for k, v in (params or {}).items()})
        except (KeyError, IndexError, ValueError):
            return None  # a placeholder the call did not fill
        if re.search(r"<\w+>", bound) or classify_verify_expr(bound, self._oracle_names) != "GROUNDED":
            return None
        # An object the world manifest cannot resolve binds nothing (no evidence), never a guess.
        world = getattr(sim_session(self._world, self._agent), "world", None)
        used = {f for _, f, _, _ in string.Formatter().parse(hint) if f}
        if world is not None and any(isinstance(params[f], str) and world.resolve(params[f]) is None for f in used):
            return None
        return bound

    def _note_intents(self, tool: Any) -> None:
        """A dispatched skill whose hint names a gripper oracle is a manipulation:
        a rising one (holding) a grasp, a falling / level one (release, placement) a place.
        A skill whose spec releases what is held (``effects['held_object'] = None``:
        gripper_open, handover) is a place even without a hint."""
        effects = getattr(tool, "effects", None)
        if isinstance(effects, dict) and "held_object" in effects and effects["held_object"] is None:
            self.intents.add("place")
        for _name, _call, channel, want in actor_causation.robot_calls(
            str(getattr(tool, "verify_hint", "") or ""), self._oracles
        ):
            if channel == "gripper":
                self.intents.add("grasp" if want == "rise" else "place")

    def _check(self, expr: str) -> bool:
        try:
            return bool(self._verifier.verify(expr))
        except Exception as exc:  # noqa: BLE001
            logger.debug("native_loop: verify(%r) raised: %s", expr, exc)
            return False

    # ------------------------------------------------------------------
    # The verify-tool HANDLER — appends EXACTLY ONE StepRecord per pair
    # ------------------------------------------------------------------

    def handle_verify(self, expr: str, goal_id: str = "") -> ToolResult:
        """Evaluate *expr* via the live GoalVerifier, grade causation, record ONE step.

        The (action-chain -> verify) pair becomes ONE StepRecord whose SubGoal.verify
        is the model's expr and whose strategy is the producing action chain. A verify
        with no action since the model's last one is refused unevaluated. The loop
        NEVER computes verified — the spine does, from this trace.
        """
        expr = (expr or "").strip()
        if not self._acted_since_verify:
            return ToolResult(
                content=(f"verify({expr}) refused: no action ran since your last verify. "
                         "Act first; a verify alone cannot change the world."),
                is_error=True,
            )
        self._acted_since_verify = False
        auto = self._pending
        if auto is not None and not auto["ok"] and attempt_key(auto["expr"]) != attempt_key(expr):
            # The model may ADD a verify, never replace a FAILED auto-verify of a
            # different predicate: that FAIL is its own graded row.
            self._flush_pending()
            auto = None
        self._pending = None  # the model's own predicate is this step's verify
        verify_result = self._check(expr)
        # Why the predicate could not be evaluated (its exception's type and message,
        # or the sandbox's refusal), so the model can correct it. Never a value it read.
        why = "" if verify_result else str(getattr(self._verifier, "last_error", "") or "")
        actor_caused, channels = self._grade(expr, goal_id)
        data = self._result_data()
        if auto is not None:
            data["auto_verify"] = {"expr": auto["expr"], "result": auto["ok"]}
        self._record(expr, verify_result, actor_caused, channels, goal_id, data)
        return ToolResult(
            content=(
                f"verify({expr}) -> {verify_word(verify_result)} "
                f"(result={verify_result}, actor={actor_caused.value})"
                + (f"; it could not be evaluated: {why}" if why else "")
            )
        )

    def rebind(self, agent: Any, verifier: Any, oracles: dict[str, Any], motor_tools: dict[str, Any],
               world: Any) -> None:
        """A lifecycle tool switched the body: later steps run, verify and grade on *agent*
        in *world* (an attempt baseline of the old body never grades the new one)."""
        self._agent, self._verifier, self._world = agent, verifier, world
        self._oracles, self._oracle_names = oracles, frozenset(oracles)
        self._motor_tools = motor_tools
        self._ctx.agent = agent
        self._group_baselines.clear()
        self._turn_baseline = actor_state(world, agent)

    def _record(
        self, expr: str, verify_result: bool, actor_caused: Any, channels: tuple[str, ...],
        goal_id: str, data: dict[str, Any], note: str = "verify", failed: bool | None = None,
    ) -> None:
        strategy = self._chain[-1] if self._chain else ""
        step_name = f"native_step_{self._step_idx}"
        self._step_idx += 1
        self._sub_goals.append(SubGoal(
            name=step_name,
            description=f"native: {' -> '.join(self._chain) or '(no action)'} | {note} {expr}",
            verify=expr,
            strategy=strategy,
        ))
        # The step "ran" (success) iff its LAST tool did not report an error (a failed
        # side-effecting tool is its own row: dispatch_skill); a pure check (no tool)
        # records success=True so the gate keys on GROUNDED/RAN.
        if failed is None:
            failed = bool(self._chain) and not self._last_ok
        key = (goal_id or "native", attempt_key(expr))
        if self._baseline is not None and not self._rechecks(key):  # a new attempt of the group
            self._group_baselines[key] = (self._group_baseline(key) or self._baseline,
                                          frozenset(self._acting))
        self._steps.append(StepRecord(
            sub_goal_name=step_name,
            strategy=strategy,
            success=not failed,
            verify_result=verify_result,
            duration_sec=0.0,
            error=str(data.get("content", ""))[:300] if failed else "",
            result_data=self._with_refused(data),
            actor_caused=actor_caused,
            attempt_of=goal_id or "native",
            channels=channels,
            housekeeping=self._housekeeping,
            failed_attempt=note == "failed attempt",
        ))
        self._close_step()

    def _result_data(self) -> dict[str, Any]:
        """The producing tool's own result (content, is_error, data) for the record."""
        res = self._last_result
        if res is None or not self._chain:
            return {}
        out: dict[str, Any] = {"tool": self._chain[-1], "content": str(res.content)[:2000],
                               "is_error": bool(res.is_error)}
        if isinstance(getattr(res, "metadata", None), dict) and res.metadata:
            out["output"] = res.metadata
        return out

    def _with_refused(self, data: dict[str, Any]) -> dict[str, Any]:
        """*data* plus the calls refused in this step (charter 3.5: tried, not an action)."""
        return {**data, "refused": list(self._refused_calls)} if self._refused_calls else data

    def _flush_pending(self) -> None:
        """Record a pending auto post-verify as its step's verify (the model did not
        verify the action itself)."""
        auto, self._pending = self._pending, None
        if auto is not None:
            del self._chain[auto["chain"]:]  # reads after the action are not its producer
            # The hinted tool succeeded (else no auto-verify), and a failed one before it
            # closed its own step: the row ran.
            self._record(auto["expr"], auto["ok"], auto["caused"], ("auto-verify", *auto["channels"]),
                         "", auto["data"], note="auto-verify", failed=False)

    def _close_step(self) -> None:
        # Reset the action chain + baseline for the next pair. NEVER carry a baseline
        # across a verify (else the next step's causation folds in).
        self._chain = []
        self._baseline = None
        self._step_open = False
        self._step_acted = False
        self._housekeeping = True
        self._last_ok = True
        self._acting = set()
        self._refused_calls = []
        self._last_result = None

    def _rechecks(self, key: tuple[str, str]) -> bool:
        """This step only re-checks the group's latest attempt: its chain is all
        read-only or declared housekeeping (a stop), which changes no task state."""
        return key in self._group_baselines and self._housekeeping and bool(self._chain)

    def _group_baseline(self, key: tuple[str, str]) -> Any:
        """The baseline a re-verify of the attempt group *key* is graded against, or
        None (graded on its own step).

        A RETRY (this step re-ran one of the latest attempt's side-effecting tools) or
        a RE-CHECK (``_rechecks``) is graded against the baseline the latest attempt
        was, so a FAIL -> navigate -> PASS recovery counts the whole motion. After any
        other action — a different one, a no-op — the re-verify is graded on its own
        step and never inherits an earlier attempt's motion; that attempt then becomes
        the group's latest, so a later stop cannot launder it either.
        """
        entry = self._group_baselines.get(key)
        if entry is None:
            return None
        return entry[0] if entry[1] & self._acting or self._rechecks(key) else None

    def record_abort(self, reason: str) -> None:
        """Close the turn with ONE failed step: it was stopped (or its LLM died).

        The unverified action chain (if any) is its producing strategy, so an
        interrupted action is on the record — never silently dropped — and grades
        FAILED (success=False), never GROUNDED. Its auto post-verify is dropped: the
        aborted row is that action's record.
        """
        self._pending = None
        self._record_open(reason, success=False)

    def close_chain(self) -> None:
        """At turn end (or before a lifecycle tool switches the body): an auto-verified
        step is recorded; an open chain the model never verified is on the record as an
        unproven step (grades RAN at best)."""
        self._flush_pending()
        if self._step_open:
            self._record_open("not verified", success=self._last_ok)

    def _record_open(self, reason: str, *, success: bool) -> None:
        step_name = f"native_step_{self._step_idx}"
        self._step_idx += 1
        strategy = self._chain[-1] if self._chain else ""
        self._sub_goals.append(SubGoal(
            name=step_name,
            description=f"native: {' -> '.join(self._chain) or '(no action)'} | {reason}",
            verify="",
            strategy=strategy,
        ))
        self._steps.append(StepRecord(
            sub_goal_name=step_name, strategy=strategy, success=success,
            verify_result=False, duration_sec=0.0, error=reason,
            result_data=self._with_refused(self._result_data()),
            housekeeping=self._housekeeping,
        ))
        self._close_step()

    def _grade(self, expr: str, goal_id: str = "") -> tuple[Any, tuple[str, ...]]:
        """Grade actor-causation for the just-verified step.

        ROBOT predicate (a call to a base / arm / gripper / perception oracle): a fresh
        post-capture vs the step's baseline -> CAUSED / UNCAUSED, per channel. A retry
        or a re-check of the attempt group's latest attempt is graded against that
        attempt's baseline instead (``_group_baseline``). No baseline (no skill ran) grades
        UNCAUSED (grade fail-closes).

        NON-ROBOT predicate (a dev / state oracle): no displacement to attribute, so
        grounding is tied to whether the ACTOR ACTED this step — no side-effecting
        action -> UNCAUSED (the verify reads ambient state), else NOT_GRADED.

        Fail-safe to NOT_GRADED on any unexpected error (never raises).
        """
        key = (goal_id or "native", attempt_key(expr))
        try:
            if not actor_causation.is_robot_predicate(expr, self._oracles):
                if not self._step_acted:
                    return actor_causation.ActorCaused.UNCAUSED, ("no action this step",)
                return actor_causation.ActorCaused.NOT_GRADED, ()
            post = actor_state(self._world, self._agent)
            baseline = self._group_baseline(key) or self._baseline
            # a gripper target resolves through the world manifest
            world = getattr(sim_session(self._world, self._agent), "world", None)
            return actor_causation.grade_channels(baseline, post, expr, self._oracles, self._turn_baseline, world)
        except Exception as exc:  # noqa: BLE001
            logger.debug("native_loop: grade(%r) raised: %s", expr, exc)
            return actor_causation.ActorCaused.NOT_GRADED, ()

    # ------------------------------------------------------------------
    # Trace assembly
    # ------------------------------------------------------------------

    def build_trace(
        self, goal: str, *, llm_error: str = "", model: str = "", cancelled: bool = False,
        contract: tuple[str, ...] = (), contract_start: tuple[bool, ...] = (),
        contract_end: tuple[bool, ...] = (), scene: str = "", world: str = "",
    ) -> ExecutionTrace:
        """Assemble the ExecutionTrace from the recorded (chain -> verify) steps.

        ``success`` is the spine's structural success flag (``trace_store.turn_succeeded``:
        a step was recorded and every GRADED step — the latest attempt of each sub-goal —
        succeeded, a failed attempt on the way to a proven goal aside); the moat is
        GROUNDED/RAN, not this flag. A superseded attempt is marked for the audit row. An
        empty trace (no verify ever called) is success=False, so the gate fails closed.
        """
        goal_tree = GoalTree(goal=goal, sub_goals=tuple(self._sub_goals))
        probe = ExecutionTrace(goal_tree=goal_tree, steps=tuple(self._steps), success=False,
                               total_duration_sec=0.0)
        graded = set(graded_indices(probe))
        steps = tuple(s if i in graded else replace(s, superseded=True) for i, s in enumerate(self._steps))
        trace = ExecutionTrace(
            goal_tree=goal_tree,
            steps=steps,
            success=False,
            total_duration_sec=0.0,
            actions_dispatched=self.actions_dispatched,
            llm_error=llm_error,
            model=model,
            refused=tuple(self.refused),
            cancelled=cancelled,
            contract=tuple(contract),
            contract_start=tuple(contract_start),
            contract_end=tuple(contract_end),
            intents=tuple(sorted(self.intents)),
            scene=scene,
            world=world,
        )
        return replace(trace, success=turn_succeeded(trace, self._oracle_names))


def verify_word(ok: bool) -> str:
    return "PASS" if ok else "FAIL"
