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
tools count as commanded motion (``ACTOR_EPOCH``).
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
from vector_os_nano.vcli.cognitive.trace_store import attempt_key, graded_indices
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
    - a step FAILED when any side-effecting tool of its chain reported an error, not
      only its last tool;
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
        self._acting_failed: bool = False  # one of them reported an error
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
        if not read_only:
            self._flush_pending()  # an auto-verified step closes before the next action
        if not self._step_open:
            # First skill of a fresh step -> capture the causation baseline NOW.
            self._baseline = actor_state(self._world, self._agent)
            self._step_open = True
        self._chain.append(name)
        hint = None
        if not read_only and getattr(tool, "housekeeping", False) is not True:
            self._housekeeping = False
        if not read_only:
            self.actions_dispatched += 1
            self._step_acted = True
            self._acted_since_verify = True
            self._acting.add(name)
            hint = self._bound_hint(tool, params)
            self._note_intents(tool)
        before = self._check(hint) if hint else None  # F->T: the hint at the baseline
        with self._world.actor_epoch() if self._world is not None else contextlib.nullcontext():
            result = execute_resolved_tool(
                tool, params, self._ctx, error_prefix=f"Skill '{name}' raised",
                on_error=lambda _n, exc: logger.debug("native_loop: skill %r raised: %s", name, exc),
            )
        self._last_ok = not result.is_error
        self._acting_failed |= result.is_error and not read_only
        self._last_result = result
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
               world: Any = None) -> None:
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
        # The step "ran" (success) iff no side-effecting tool of its chain and not its
        # LAST tool reported an error; a pure check (no tool) records success=True so
        # the gate keys on GROUNDED/RAN. A failed skill followed by a read and a true
        # verify is FAILED, never GROUNDED: the skill's own failure is kept, with its
        # message, in error / result_data (rule 4).
        if failed is None:
            failed = self._acting_failed or (bool(self._chain) and not self._last_ok)
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
            result_data=data,
            actor_caused=actor_caused,
            attempt_of=goal_id or "native",
            channels=channels,
            housekeeping=self._housekeeping,
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

    def _flush_pending(self) -> None:
        """Record a pending auto post-verify as its step's verify (the model did not
        verify the action itself)."""
        auto, self._pending = self._pending, None
        if auto is not None:
            del self._chain[auto["chain"]:]  # reads after the action are not its producer
            # The hinted tool succeeded (else no auto-verify), but an earlier actuating
            # tool of the same step may not have: the row is FAILED then.
            self._record(auto["expr"], auto["ok"], auto["caused"], ("auto-verify", *auto["channels"]),
                         "", auto["data"], note="auto-verify", failed=self._acting_failed)

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
        self._acting_failed = False
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
            self._record_open("not verified", success=self._last_ok and not self._acting_failed)

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
            verify_result=False, duration_sec=0.0, error=reason, result_data=self._result_data(),
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

        ``success`` is True iff at least one step was recorded AND every GRADED step
        (the latest attempt of each sub-goal) succeeded — the structural success flag
        the verdict reads (the moat is GROUNDED/RAN, not this flag). A superseded
        attempt is marked for the audit row. An empty trace (no verify ever called) is
        success=False, so the verdict gate fails closed.
        """
        goal_tree = GoalTree(goal=goal, sub_goals=tuple(self._sub_goals))
        probe = ExecutionTrace(goal_tree=goal_tree, steps=tuple(self._steps), success=False,
                               total_duration_sec=0.0)
        graded = set(graded_indices(probe))
        steps = tuple(s if i in graded else replace(s, superseded=True) for i, s in enumerate(self._steps))
        success = bool(steps) and all(steps[i].success for i in graded)
        return ExecutionTrace(
            goal_tree=goal_tree,
            steps=steps,
            success=success,
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


def verify_word(ok: bool) -> str:
    return "PASS" if ok else "FAIL"
