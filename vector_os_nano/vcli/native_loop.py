# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""native_loop — a frontier-model NATIVE TOOL-USE turn producer (strangler-fig core).

Campaign #13, M1 FIRST STEP. This is the producer half of the strangle move: the
MODEL drives a native tool-use ReAct loop (skills-as-tools + a synthetic
``verify(expr)`` + ``finish``), and the loop assembles an ``ExecutionTrace`` the
EXISTING honest verify spine consumes BYTE-FOR-BYTE UNCHANGED. It does NOT compute
``verified`` itself — it hands the trace to ``VerdictReport.from_trace`` /
``evidence_passed`` (rule 5: verify is the moat, single-sourced).

REPLACE-NOT-REBUILD, enforced BY CONSTRUCTION (review fix 5):

    This module imports ONLY the honest spine — ``actor_causation``,
    ``trace_store`` (the verdict + oracle names live one hop further), the
    GoalVerifier namespace builder (via the engine, passed in), and
    ``SkillWrapperTool``. The runner's ONLY logic is: capture baseline -> dispatch the skill via
    SkillWrapperTool -> evaluate verify via the live GoalVerifier -> grade via
    ``actor_causation.grade`` -> append ONE StepRecord. The MODEL owns
    decompose / route / replan — there is NO replan / iteration / "landed-short"
    bookkeeping here (that would be re-growing the planner — the tell).

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

Turn contract (WP07): the goal compiles to GT predicates (``task_contract``),
printed as "Task contract: ..." before the first model call and evaluated at turn
start and turn end; the spine requires each False -> True.
"""
from __future__ import annotations

import contextlib
import logging
import math
import os
import re
import string
import threading
import time
from dataclasses import replace
from typing import Any, Callable

from vector_os_nano.vcli.backends.types import LLMResponse
from vector_os_nano.vcli.cognitive import abort as _abort
from vector_os_nano.vcli.cognitive import actor_causation
from vector_os_nano.vcli.cognitive.evidence_classifier import classify_verify_expr
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.object_goal import task_contract, unbound_reason
from vector_os_nano.vcli.cognitive.trace_store import attempt_key, graded_indices, verify_oracles
from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)
from vector_os_nano.vcli.native_tools import (
    FINISH_TOOL,
    VERIFY_TOOL,
    _agent_base,
    _build_motor_tools,
    _native_tool_schemas,
)
from vector_os_nano.vcli.permissions import PermissionContext
from vector_os_nano.vcli.prompt import user_prompt_files
from vector_os_nano.vcli.tool_execution import execute_resolved_tool, resolve_permission
from vector_os_nano.vcli.tools.base import ToolContext, ToolResult

logger = logging.getLogger(__name__)

# A hard cap on native round-trips so a misbehaving model / script can never spin
# forever. Mirrors the engine's _max_turns spirit; the loop also stops on finish.
_MAX_NATIVE_TURNS = 24

# How many times the runner re-prompts a model that tries to finish/stop with an
# action it never verified (D23). Bounded so a model that stubbornly refuses to
# verify still terminates — the unverified action then grades honestly (empty/RAN),
# never a false green. The verify stays MODEL-authored; the runner never invents it.
_MAX_VERIFY_NUDGES = 2

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


# ---------------------------------------------------------------------------
# run_turn_native — the producer (drives the model, hands the spine a trace)
# ---------------------------------------------------------------------------


def _progress_args(inp: Any) -> str:
    """A short '(k=v, …)' summary of a tool call's input for progress display."""
    if not isinstance(inp, dict) or not inp:
        return ""
    parts = []
    for k, v in inp.items():
        parts.append(f"{k}={str(v)[:18]}")
        if len(parts) >= 2:
            break
    return "(" + ", ".join(parts) + ")"


def _model_id(backend: Any) -> str:
    return str(getattr(backend, "_model", "") or getattr(backend, "model", "") or type(backend).__name__)


def _fallback_backend(backend: Any) -> Any:
    """The opt-in OpenRouter fallback (VECTOR_LLM_FALLBACK=openrouter + a key), or None."""
    if os.environ.get("VECTOR_LLM_FALLBACK", "").strip().lower() != "openrouter":
        return None
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key or getattr(backend, "_is_llm_fallback", False):
        return None
    from vector_os_nano.vcli.backends import create_backend

    fallback = create_backend(
        "openrouter", key, os.environ.get("VECTOR_LLM_FALLBACK_MODEL", "anthropic/claude-sonnet-4-6")
    )
    fallback._is_llm_fallback = True
    return fallback


def _call_llm(backend: Any, kwargs: dict[str, Any]) -> tuple[Any, LLMResponse]:
    """One model round-trip. The backends already retry timeouts / 429 / 5xx with a
    bounded backoff; if that still fails, try the opt-in fallback provider once.
    Returns the backend that answered (the turn keeps using it)."""
    try:
        return backend, backend.call(**kwargs)
    except Exception as exc:  # noqa: BLE001
        fallback = _fallback_backend(backend)
        if fallback is None:
            raise
        logger.warning("native_loop: LLM %s failed (%s); falling back to OpenRouter %s",
                       _model_id(backend), exc, _model_id(fallback))
        return fallback, fallback.call(**kwargs)


def _base_xy(base: Any) -> tuple[float, float] | None:
    try:
        pos = base.get_position()
        return float(pos[0]), float(pos[1])
    except Exception:  # noqa: BLE001 — no base / no pose: no distance in the line
        return None


def _progress_line(name: str, params: dict[str, Any], base: Any, start: Any, elapsed: float) -> str:
    """'navigate 7s · 2.3 m to go' / 'walk 4s · moved 1.2 m' — one live progress line."""
    line = f"{name} {elapsed:.0f}s"
    now = _base_xy(base) if base is not None else None
    if now is None:
        return line
    try:
        return f"{line} · {math.hypot(float(params['x']) - now[0], float(params['y']) - now[1]):.1f} m to go"
    except (KeyError, TypeError, ValueError):
        pass
    if start is not None:
        return f"{line} · moved {math.hypot(now[0] - start[0], now[1] - start[1]):.1f} m"
    return line


def run_turn_native(
    engine: Any,
    user_message: str,
    *,
    agent: Any | None = None,
    session: Any = None,
    app_state: dict[str, Any] | None = None,
    max_turns: int = _MAX_NATIVE_TURNS,
    on_progress: Callable[[str], None] | None = None,
    ask_permission: Callable[[str, dict[str, Any]], str] | None = None,
    cancel: Any = None,
) -> ExecutionTrace:
    """Run ONE user turn through the native tool-use loop; return an ExecutionTrace.

    The producer half of the strangle. The MODEL (via ``engine._backend``) is given
    a NARROW tool-set — the motor skills + the synthetic ``verify``/``finish`` — and
    drives a ReAct loop: it calls action skills, then ``verify(expr)``, re-issuing a
    tool call after a False verify tool_result (the loop does NOT replan). When the
    model finishes (``finish`` or end_turn with no tool calls), the runner assembles
    the trace from the recorded (action-chain -> verify) pairs and returns it.

    Safety (WP05): every tool call goes through the permission gate
    (``ask_permission`` resolves an ``ask``; None denies). ``cancel`` is this run's
    ``CancelToken`` (a fresh one bound to this thread when None); it is checked
    before every model round-trip and every tool call, and a cancelled turn ends with
    ONE recorded ``aborted`` step and ``trace.cancelled`` set (the caller owns it: a
    stopped turn never falls through to another producer). A model failure never
    raises out of here once an action ran — the partial trace comes back with
    ``llm_error`` set and a failed step closing it, so no caller re-runs a turn that
    already moved the robot and an interrupted plan never verifies.

    The caller (``cli.run_one_turn``) feeds the returned trace to the EXISTING
    ``VerdictReport.from_trace(trace, verify_oracle_names(agent, engine))`` — this
    function NEVER computes ``verified``.

    The sim session (if any) is told where the turn starts and ends: its GT oracles
    scope ``kept_holding`` / ``localized`` to the turn, and ``VECTOR_GT_DUMP`` gets a
    ground-truth line at both ends for the independent grader (scripts/gt_grade.py).
    """
    agent = agent if agent is not None else getattr(engine, "_agent", None)
    # The body the turn drives now and its world: a lifecycle tool rebinds both.
    body = [agent, getattr(engine, "_world", None)]
    _mark_turn(body, "start", user_message)
    try:
        return _run_turn_native(
            engine, user_message, body=body, session=session, app_state=app_state,
            max_turns=max_turns, on_progress=on_progress,
            ask_permission=ask_permission, cancel=cancel,
        )
    finally:
        _mark_turn(body, "end", user_message)


def _mark_turn(body: list[Any], phase: str, goal: str) -> None:
    """Tell the body's sim session (asked of its world) where the turn starts / ends."""
    if body[1] is not None:
        body[1].mark_turn(body[0], phase, goal)


def _bind_body(engine: Any, agent: Any) -> tuple[Any, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Everything a turn derives from its body: (verifier, oracles, tools, schemas).

    The verifier + oracles ({name: OracleTag}) are single-sourced from the SAME
    namespace the spine reads (rule 3); fail closed on any error (an empty oracle set
    classifies every predicate RAN). The motor tool-set is the world's skills + the
    registry's tools; the synthetic verify/finish are handled in-loop, never dispatched.
    """
    builder = getattr(engine, "_build_verifier_namespace", None)
    namespace = builder(agent) if builder is not None else {}
    oracles = verify_oracles(agent, engine)
    motor_tools = _build_motor_tools(agent, engine)
    schemas = _native_tool_schemas(motor_tools, namespace, frozenset(oracles))
    return GoalVerifier(namespace), oracles, motor_tools, schemas


def _run_turn_native(
    engine: Any,
    user_message: str,
    *,
    body: list[Any],
    session: Any,
    app_state: dict[str, Any] | None,
    max_turns: int,
    on_progress: Callable[[str], None] | None,
    ask_permission: Callable[[str, dict[str, Any]], str] | None = None,
    cancel: Any = None,
) -> ExecutionTrace:
    token = cancel if cancel is not None else _abort.new_run()
    agent = body[0]
    verifier, oracles, motor_tools, tool_schemas = _bind_body(engine, agent)
    oracle_names = frozenset(oracles)
    ctx = _build_tool_context(agent, session, app_state, engine, token)
    runner = NativeStepRunner(agent, verifier, oracle_names, motor_tools, ctx, ask_permission, oracles, body[1])

    def _rebind() -> None:
        """A lifecycle tool succeeded: the next round-trip sees the new body's tools,
        verifier and oracle names, and the sim's turn marks move to it."""
        nonlocal oracle_names, tool_schemas
        app = ctx.app_state
        new = app.get("agent") if app is not None and "agent" in app else getattr(engine, "_agent", None)
        world = getattr(engine, "_world", None)
        verifier, oracles, motor_tools, tool_schemas = _bind_body(engine, new)
        oracle_names = frozenset(oracles)
        runner.rebind(new, verifier, oracles, motor_tools, world)
        if new is not body[0]:
            _mark_turn(body, "end", user_message)
            body[:] = [new, world]
            _mark_turn(body, "start", user_message)
        _compile_contract()  # the goal against the new body (the spine grades that one)

    turn: dict[str, Any] = {"contract": (), "start": (), "world": "", "note": ""}

    def _compile_contract() -> None:
        """The turn-level task contract (WP07): GT end-state predicates the goal
        compiles to against the body's oracles, its object / surface bound through the
        body's world manifest; read now (each must be False) and again at turn end
        (each must be True)."""
        world = str(getattr(getattr(sim_session(body[1], body[0]), "world", None), "id", "") or "")
        contract = task_contract(user_message, oracle_names, world)
        # Fail closed: the verifier maps an oracle error to False, so ask for ``not``
        # — an error at the start then reads as "already satisfied", never as "not yet".
        turn.update(contract=contract, world=world,
                    start=tuple(not runner._check(f"not ({c})") for c in contract))
        # The same lines reach the model (a system block, I4c): it verifies the bound
        # object and surface itself, or asks the user when a slot did not bind.
        lines = [f"Task contract: {' and '.join(contract)}"] if contract else []
        lines += [f"Unbound: {c}: {why}" for c in contract if (why := unbound_reason(c, user_message, world))]
        turn["note"] = "\n".join(lines)
        for line in lines:
            _emit(line)

    backend = getattr(engine, "_backend", None)
    if backend is None:
        # No backend wired -> no native turns -> empty trace (verdict fails closed).
        return runner.build_trace(user_message)
    model = _model_id(backend)
    _append_user(session, user_message)

    # Progress feedback (D9 #2 perceived latency): the user's "load 好几秒" is the
    # opaque wait during the synchronous LLM round-trips. Stream the model's text
    # TAIL while it thinks (on_text deltas) and emit each tool call as it dispatches,
    # so the spinner shows live activity instead of a frozen line. Pure UX — no
    # effect on the trace, the verify spine, or routing. Fail-safe: never raises.
    def _emit(msg: str) -> None:
        if on_progress is None:
            return
        try:
            on_progress(msg)
        except Exception:  # noqa: BLE001 — progress is best-effort, never fatal
            pass

    _compile_contract()
    _narration: list[str] = []

    def _on_text(chunk: str) -> None:
        _narration.append(chunk)
        tail = "".join(_narration)[-72:].replace("\n", " ").strip()
        if tail:
            _emit(tail)

    def _dispatch(name: str, params: dict[str, Any]) -> ToolResult:
        """Run one tool; while it runs, a 1 Hz progress line (elapsed + distance)."""
        if on_progress is None:
            return runner.dispatch_skill(name, params)
        done = threading.Event()
        base = _agent_base(ctx)
        start = _base_xy(base) if base is not None else None
        t0 = time.monotonic()

        def _tick() -> None:
            while not done.wait(1.0):
                _emit(_progress_line(name, params, base, start, time.monotonic() - t0))

        threading.Thread(target=_tick, name="native-progress", daemon=True).start()
        try:
            return runner.dispatch_skill(name, params)
        finally:
            done.set()

    llm_error = ""
    turns = 0
    verify_nudges = 0  # D23: re-prompts spent forcing a verify before a finish/stop
    try:
        while turns < max_turns:
            if token.cancelled:
                runner.record_abort("aborted")
                break
            messages = _to_messages(session, tool_schemas)
            _narration.clear()  # fresh "thinking" tail per round-trip
            try:
                backend, response = _call_llm(backend, dict(
                    messages=messages,
                    tools=tool_schemas,
                    # Rebuilt per round-trip: the [Robot State] block follows the body.
                    system=_native_system_prompt(engine, oracle_names, body[0])
                    + ([{"type": "text", "text": turn["note"]}] if turn["note"] else []),
                    max_tokens=getattr(engine, "_max_tokens", 4096),
                    on_text=_on_text if on_progress is not None else None,
                ))
            except Exception as exc:  # noqa: BLE001 — infrastructure, not the agent
                llm_error = f"{type(exc).__name__}: {exc}"
                logger.warning("native_loop: LLM unavailable: %s", llm_error)
                if runner._step_open or runner._steps or runner._pending:
                    # The plan was cut short: an earlier verified step never makes
                    # the interrupted turn a success.
                    runner.record_abort(f"LLM unavailable: {llm_error}")
                break
            model = _model_id(backend)
            if session is not None and hasattr(session, "add_usage") and getattr(response, "usage", None):
                session.add_usage(response.usage)
            tool_calls = list(response.tool_calls or [])
            _append_assistant(session, response, tool_calls)

            if not tool_calls:
                # D23: the model stopped (narrated, no tools). If it ran an action but
                # never verified it, force a model-authored verify before accepting the
                # stop (up to a bounded number of nudges). Never invent the predicate;
                # the model must choose it.
                if runner.has_unverified_action and verify_nudges < _MAX_VERIFY_NUDGES:
                    verify_nudges += 1
                    _emit("re-prompting: verify before finishing")
                    _append_user(
                        session,
                        "You ran an action but did NOT verify it. You MUST call "
                        "verify(<predicate>) to PROVE the goal was achieved before you "
                        "stop. Pick the deterministic predicate that measures the goal "
                        "and call verify now.",
                    )
                    continue
                break  # end_turn, no tools — conversation complete

            finished = False
            result_dicts: list[dict[str, Any]] = []
            for tc in tool_calls:
                if token.cancelled:
                    result_dicts.append(_tool_result_dict(tc.id, "Stopped: not run.", is_error=True))
                    continue
                if getattr(tc, "parse_error", ""):  # never dispatched with defaults
                    result_dicts.append(_tool_result_dict(
                        tc.id, f"{tc.name} was not run: its {tc.parse_error}. Call it again with "
                        "valid JSON arguments.", is_error=True))
                    continue
                if tc.name == FINISH_TOOL:
                    # D23: refuse a finish that rides on an unverified action — re-prompt
                    # for a model-authored verify first (bounded).
                    if runner.has_unverified_action and verify_nudges < _MAX_VERIFY_NUDGES:
                        verify_nudges += 1
                        _emit("verify required before finish")
                        result_dicts.append(_tool_result_dict(
                            tc.id,
                            "Cannot finish yet: you ran an action but did not verify it. "
                            "Call verify(<predicate>) to PROVE the goal FIRST, then finish.",
                            is_error=True,
                        ))
                        continue
                    _emit("finishing")
                    finished = True
                    result_dicts.append(_tool_result_dict(tc.id, "Task finished."))
                    continue
                if tc.name == VERIFY_TOOL:
                    expr = str((tc.input or {}).get("expr", ""))
                    _emit(f"verify {expr[:56]}")
                    res = runner.handle_verify(expr, str((tc.input or {}).get("goal_id", "") or ""))
                else:
                    _emit(f"{tc.name} {_progress_args(tc.input)}".strip())
                    res = _dispatch(tc.name, dict(tc.input or {}))
                    if tc.name in LIFECYCLE_TOOLS and not res.is_error:
                        _rebind()
                result_dicts.append(_tool_result_dict(tc.id, res.content, res.is_error))

            _append_tool_results(session, result_dicts)
            turns += 1
            if finished:
                break
        else:  # the round-trip cap ran out before the model finished: say so, on the record
            runner.record_abort("aborted" if token.cancelled else f"turn cap reached ({max_turns} model round-trips)")
    except Exception as exc:  # noqa: BLE001 — a producer bug after an action: own the turn
        if not runner.actions_dispatched:
            raise
        logger.warning("native_loop: turn failed after %d action(s): %s", runner.actions_dispatched, exc)
        runner.record_abort(f"producer error: {type(exc).__name__}: {exc}")
    # An auto-verified action is recorded; a chain the model never verified (a read it
    # answered from, or an action once the nudges are spent) is on the record as an
    # unproven step, never silently dropped.
    runner.close_chain()
    return runner.build_trace(
        user_message, llm_error=llm_error, model=model, cancelled=token.cancelled,
        contract=turn["contract"], contract_start=turn["start"],
        contract_end=tuple(runner._check(c) for c in turn["contract"]), scene=body[1].sim_id(body[0]) if body[1] is not None else "",
        world=turn["world"],
    )


# ---------------------------------------------------------------------------
# Helpers — verifier / tools / system prompt / session glue (no planner imports)
# ---------------------------------------------------------------------------


# The KERNEL loop rules: how any turn acts and proves, in every world. How a body
# does a task (find -> approach -> grasp -> carry -> place) is the world's persona
# (RobotWorld.persona_blocks, composed from the manifests), never written here.
_KERNEL_RULES = (
    "You act through tools. For each goal: call the action that achieves it, then call "
    "verify(expr) with the deterministic predicate that MEASURES that goal, then finish. "
    "verify reads the real world (ground truth you cannot write), so it is the only proof: "
    "your own words, a tool's message, a file or a shell command never are. A physical "
    "action is done only by the robot skill for it; never stand in for one by writing a "
    "file or running a command. If no tool can do it, say so and stop.\n"
    "- The action that should achieve a goal is the last tool call before its verify; a "
    "task with several goals verifies each after its own action.\n"
    "- A FAIL means the goal is not met: recover (repeat the action or take a better one), "
    "then verify the SAME predicate again. Never finish while a goal's latest verify is FAIL; if you cannot "
    "recover, say why and stop.\n"
    "- A read-only tool (status, memory, perception) changes nothing: answer from it, no "
    "verify needed.\n"
    "- start_simulation / stop_simulation need no verify (they return only once the sim "
    "is ready or gone). After one succeeds, your tools and verify predicates belong to "
    "the new body from your next call on.\n"
)


def _native_system_prompt(engine: Any, oracle_names: frozenset[str], agent: Any = None) -> list[dict[str, Any]]:
    """The native system prompt: kernel loop rules + the world's persona + world state.

    The verify vocab is the SAME live oracle set ``verify_oracle_names`` reads; each
    predicate's signature and meaning is in the verify tool's description. The world persona (``engine._world.persona_blocks(agent)``) carries the
    body's task policy; the user's own prompt files follow it (``user_prompt_files``); the ``[Robot State]`` block is the actor's belief right now
    (pose, room, gripper, localized objects, nav/explore backend), rebuilt for every
    round-trip. Nothing here names a ground-truth object.
    """
    names = ", ".join(sorted(oracle_names)) if oracle_names else "(none)"
    text = _KERNEL_RULES + f"Available verify predicate oracles: {names}."
    blocks = [{"type": "text", "text": text}]
    world = getattr(engine, "_world", None)
    if world is not None:
        try:
            blocks += [{"type": "text", "text": t.strip()} for t in world.persona_blocks(agent) if t]
        except Exception as exc:  # noqa: BLE001 — the kernel rules still stand
            logger.warning("native_loop: world persona unavailable: %s", exc)
    blocks += [{"type": "text", "text": t} for t in user_prompt_files()]
    if agent is not None:
        from vector_os_nano.vcli.robot_context import RobotContextProvider

        blocks.append(RobotContextProvider.for_agent(agent).get_context_block())
    return blocks


def _build_tool_context(
    agent: Any, session: Any, app_state: dict[str, Any] | None, engine: Any, cancel: Any = None
) -> ToolContext:
    """Build the ToolContext skills run under; ``abort`` is this run's CancelToken."""
    from pathlib import Path

    return ToolContext(
        agent=agent,
        cwd=Path.cwd(),
        session=session,
        permissions=getattr(engine, "_permissions", None),
        abort=cancel if cancel is not None else _abort.new_run(),
        app_state=app_state,
    )


# -- session glue (duck-typed; works with the real Session + None) -----------


def _append_user(session: Any, text: str) -> None:
    if session is not None and hasattr(session, "append_user"):
        session.append_user(text)


def _append_assistant(session: Any, response: LLMResponse, tool_calls: list[Any]) -> None:
    if session is None or not hasattr(session, "append_assistant"):
        return
    tool_use_dicts = (
        [{"id": tc.id, "name": tc.name, "input": tc.input, "type": "tool_use"} for tc in tool_calls]
        if tool_calls
        else None
    )
    session.append_assistant(response.text, tool_use_dicts)


def _append_tool_results(session: Any, result_dicts: list[dict[str, Any]]) -> None:
    if session is not None and hasattr(session, "append_tool_results"):
        session.append_tool_results(result_dicts)


def _to_messages(session: Any, tool_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if session is not None and hasattr(session, "to_messages"):
        # A legacy turn's tool_use (bash, file_read...) absent from this list goes as text.
        return session.to_messages(tool_names={t.get("name") for t in tool_schemas})
    return []


def _tool_result_dict(tool_use_id: str, content: str, is_error: bool = False) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": is_error,
    }
