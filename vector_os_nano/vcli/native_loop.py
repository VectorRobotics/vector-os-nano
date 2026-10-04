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
    ``SkillWrapperTool``. It MUST NOT import ``goal_decomposer`` / ``goal_executor``
    / ``strategy_selector`` / ``vgg_harness``. ``tests/unit/vcli/test_native_loop
    _import_firewall.py`` FAILS if any of those appear in this module's imports.
    The runner's ONLY logic is: capture baseline -> dispatch the skill via
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

Capture/grade timing (review fix 3), mirroring ``GoalExecutor._execute_sub_goal``:
the actor-causation baseline is captured immediately BEFORE the first skill call
of a step; the grade reads a FRESH post-capture immediately AFTER evaluating that
step's verify expr, NEVER spanning a later step's actions (else the go2 daemon's
between-turn motion folds into the causation grade).
"""
from __future__ import annotations

import inspect
import json
import logging
import math
import os
import threading
import time
from typing import Any, Callable

from vector_os_nano.vcli.backends.types import LLMResponse
from vector_os_nano.vcli.cognitive import abort as _abort
from vector_os_nano.vcli.cognitive import actor_causation
from vector_os_nano.vcli.cognitive.trace_store import verify_oracle_names
from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)
from vector_os_nano.vcli.permissions import PermissionContext
from vector_os_nano.vcli.tool_execution import execute_resolved_tool, resolve_permission
from vector_os_nano.vcli.tools.base import PermissionResult, ToolContext, ToolResult
from vector_os_nano.vcli.tools.skill_wrapper import wrap_skills

logger = logging.getLogger(__name__)

# Synthetic tool names the runner OWNS (never wrapped from a skill).
VERIFY_TOOL = "verify"
FINISH_TOOL = "finish"

# A hard cap on native round-trips so a misbehaving model / script can never spin
# forever. Mirrors the engine's _max_turns spirit; the loop also stops on finish.
_MAX_NATIVE_TURNS = 24

# How many times the runner re-prompts a model that tries to finish/stop with an
# action it never verified (D23). Bounded so a model that stubbornly refuses to
# verify still terminates — the unverified action then grades honestly (empty/RAN),
# never a false green. The verify stays MODEL-authored; the runner never invents it.
_MAX_VERIFY_NUDGES = 2

# The registry category whose tools are the kernel's domain-general ACTION surface
# (file_read/file_write/file_edit/bash/glob/grep — see tools.__init__._TOOL_CATEGORIES
# and worlds.dev.DEV_TOOL_ALLOWLIST, which is this category). The native loop offers
# these as motor tools alongside the world's skills, so the dev world (no robot agent,
# hence no wrapped skills) can still ACT — and the robot world gains the same code
# tools. World-agnostic BY CONSTRUCTION: native asks the ENGINE'S registry for its
# registered action tools; there is NO "if dev" branch.
_CODE_TOOL_CATEGORY = "code"
_SIM_TOOL_CATEGORY = "sim"
_ROBOT_TOOL_CATEGORY = "robot"

# The sim lifecycle tools: they change WHICH body the turn drives. They count as an
# action (the turn is owned; nothing re-runs them) but prove nothing themselves, so
# they never open a graded step; after one succeeds the runner rebinds the turn to
# the new body (tools, verifier, oracle names, causation baseline).
LIFECYCLE_TOOLS: frozenset[str] = frozenset({"start_simulation", "stop_simulation"})

# The MUTATING code tools — the ones whose effect is a file/shell WRITE the model can
# author at will (so it can "accomplish" a task by writing a marker file or running a
# shell command), NEVER a read-only diagnostic. In a ROBOT world these must NOT be an
# action path to a PHYSICAL goal: the fakeable-grasp defect was deepseek satisfying
# "抓前面的东西" by ``file_write('grabbed.txt')`` then ``verify(file_exists('grabbed.txt'))``
# (D17). A physical robot task is accomplished by a robot SKILL and proven by a GT
# oracle the actor cannot author — never by touching a file. So a robot world drops
# these from the loop's ACTION toolset (Prong 1) while KEEPING the read-only diagnostics
# (file_read / glob / grep) the persona uses to inspect code. The DEV world is untouched
# — it has no robot agent, so the gate below never fires there. Prong 2 (the goal-
# authenticity gate in the frozen spine) is the un-fakeable backstop if a model ever
# reaches a file oracle another way; this prong removes the easy path by construction.
_MUTATING_CODE_TOOLS: frozenset[str] = frozenset({"file_write", "file_edit", "bash"})

# at_position tolerance (metres) — single-sourced for the system-prompt vocab from
# the go2 oracle so the model's verify expr and the verifier agree. Read live with
# a safe fallback so this module never hard-depends on the oracle's private const.
def _at_position_tol() -> float:
    try:
        from vector_os_nano.vcli.worlds.sim_oracles import _AT_POSITION_TOL_M
        return float(_AT_POSITION_TOL_M)
    except Exception:  # noqa: BLE001
        return 0.5


# D9 #1 — the avoidance NAVIGATION route.
_NAV_TIMEOUT_S = 60.0


def _agent_base(context: ToolContext) -> Any:
    """The connected mobile base from the tool context's agent (the WIRED accessor).

    The ``walk`` skill, the verify oracles, and actor-causation all reach the live base
    via ``agent._base`` — the standalone ``vcli/primitives`` layer is NOT wired in the
    product (``init_primitives`` is never called; its ``_ctx`` is None). So native must
    use ``agent._base`` too, never ``primitives.navigation/locomotion``.
    """
    agent = getattr(context, "agent", None)
    return getattr(agent, "_base", None)


class _NativeBaseNavigateTool:
    """Coordinate navigation through the nav-stack AVOIDANCE route (D9 #1).

    ``execute`` calls the connected base's ``navigate_to(x, y)``: the in-process
    planner (vgraph + pure-pursuit follower) routes around the scene's obstacles and
    streams velocity while holding skill control, so its commands are counted by
    actor-causation and an ``at_position`` verify can grade CAUSED. This is the
    obstacle avoidance the open-loop ``walk`` lacks. Duck-typed like a ``Tool`` so
    ``dispatch_skill`` runs it uniformly.
    """

    name = "navigate"
    description = (
        "Go to a world coordinate (x, y) using the navigation PLANNER, which AVOIDS "
        "obstacles (lidar + local planner). Use this to REACH a place/coordinate. After "
        "it returns, call verify(at_position(x, y))."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "x": {"type": "number", "description": "Target x (world frame, metres)."},
            "y": {"type": "number", "description": "Target y (world frame, metres)."},
        },
        "required": ["x", "y"],
    }

    def check_permissions(self, params: dict[str, Any], context: ToolContext) -> PermissionResult:
        # It drives the base: confirm unless the body is a live in-process sim.
        from vector_os_nano.sim.session import is_sim_agent

        return PermissionResult("allow" if is_sim_agent(getattr(context, "agent", None)) else "ask")

    def is_read_only(self, params: dict[str, Any]) -> bool:
        return False

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            x = float(params["x"])
            y = float(params["y"])
        except (KeyError, TypeError, ValueError):
            return ToolResult(content="navigate requires numeric x and y.", is_error=True)

        base = _agent_base(context)
        navigate_to = getattr(base, "navigate_to", None)
        if not callable(navigate_to):
            return ToolResult(
                content="navigate: the connected base has no navigate_to (no nav stack).",
                is_error=True,
            )

        res = None
        # Pass a timeout only to a base that takes one: retrying on ANY TypeError
        # would re-run a navigation that raised one mid-way.
        try:
            params_ = inspect.signature(navigate_to).parameters.values()
            takes_timeout = any(p.name == "timeout" or p.kind is p.VAR_KEYWORD for p in params_)
        except (TypeError, ValueError):
            takes_timeout = False
        try:
            res = navigate_to(x, y, timeout=_NAV_TIMEOUT_S) if takes_timeout else navigate_to(x, y)
            ok = bool(res)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(content=f"navigate({x}, {y}) failed: {exc}", is_error=True)

        pos = None
        getter = getattr(base, "get_position", None)
        if callable(getter):
            try:
                pos = getter()
            except Exception as exc:  # noqa: BLE001
                logger.debug("native_loop: navigate get_position failed: %s", exc)
        where = f"({pos[0]:.2f}, {pos[1]:.2f})" if pos else "unknown"
        reason = getattr(res, "reason", None)  # the in-process planner's NavResult (Rule 4)
        if ok:
            status = "arrived"
            if reason is not None and res.get("snapped"):
                status += f" at {tuple(res['target'])}, the nearest free point (the target is inside an obstacle)"
        elif getattr(getattr(context, "abort", None), "is_set", lambda: False)():
            status = "was STOPPED before arrival"
        elif reason is not None:
            status = f"did NOT arrive ({reason})"
        else:
            status = "did NOT confirm arrival (FAR graph unavailable or timeout)"
        return ToolResult(
            content=f"navigate: {status}; now at {where} (target ({x}, {y})). "
            "Call verify(at_position) to confirm."
        )


class _NativeDetectTool:
    """Route a NAMED/COLOUR query to the learned grounding-dino DETECTOR capability.

    R4 — composes the two North-Star axes on ONE turn: this surfaces the registered
    ``DetectorCapability`` (the learned SECOND model family, grounding-dino-tiny) into
    the native loop so a bare-cli detect command reaches it on whatever camera-bearing
    embodiment is connected — go2+arm OR g1 (the humanoid, camera but NO arm). It does
    NOT re-implement detection: ``execute`` pulls the SAME capability instance the world
    registered into the engine's ``CapabilityRegistry`` (agent-bound for the live
    perception) and invokes it — single-sourced, no second model load (rule 3).

    Read-only PERCEPTION: the detector only localizes, it does not act on the world.
    The capability never self-certifies: the suggested verify is the GT ``in_view``
    oracle (segmentation of the robot's own camera, which the detector never sees), and
    it grades on the base channel, so a detect without a base move grades RAN — the
    honest grade for a perceive-only route. ``localized`` (the scene graph vs GT) holds
    once look/explore have put the object on the map.

    Duck-typed like a ``Tool`` so ``dispatch_skill`` runs it uniformly. Only surfaced
    when a ``detect`` capability is registered AND the agent has a camera (see
    ``_build_motor_tools``) — sensorless/dev paths never see it.
    """

    name = "detect"
    description = (
        "Localize a named or coloured object in the robot's CAMERA view using the "
        "learned open-vocabulary detector (grounding-dino). Pass the target as 'query' "
        "(e.g. 'red object', 'a red stool', '红色的东西'). Returns bounding box(es) + "
        "label(s) + score(s). This is PERCEPTION only (read-only). After it returns, "
        "call the verify expression the tool result tells you to use to confirm the "
        "model localized the object (the exact verify depends on the embodiment)."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language target to localize (named or coloured).",
            },
        },
        "required": ["query"],
    }

    def __init__(self, capability: Any) -> None:
        # The SAME registered DetectorCapability the world bound to the agent's live
        # perception — never a fresh instance, never a second model load.
        self._capability = capability
        self.name = str(getattr(capability, "name", "") or "detect")

    def check_permissions(self, params: dict[str, Any], context: ToolContext) -> PermissionResult:
        return PermissionResult("allow")  # perception only

    def is_read_only(self, params: dict[str, Any]) -> bool:
        return True

    @staticmethod
    def _verify_hint(agent: Any, query: str) -> str:
        """The GT verify the model should make after detect (it still authors it)."""
        return f"verify(in_view({query!r}))"

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        query = str((params or {}).get("query", "")).strip()
        if not query:
            return ToolResult(content="detect requires a non-empty 'query'.", is_error=True)
        hint = self._verify_hint(getattr(context, "agent", None), query)
        try:
            result = self._capability.invoke({"query": query}, context)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(content=f"detect({query!r}) raised: {exc}", is_error=True)

        output = dict(getattr(result, "output", {}) or {})
        boxes = output.get("boxes") or []
        labels = output.get("labels") or []
        scores = output.get("scores") or []

        # MOAT DISCIPLINE (R6 audit): the detector's OWN output is never stashed on
        # the agent for a verify to read back (the D61 self-certifying tautology); it
        # flows only to the human-readable tool output below.
        if not getattr(result, "success", False) and not boxes:
            err = getattr(result, "error", "") or "no detections"
            return ToolResult(
                content=(
                    f"detect({query!r}): the detector RAN but localized nothing ({err}). "
                    f"{hint} checks whether it is in the camera view at all."
                )
            )
        first = (
            f"label={labels[0]!r} box={[round(float(v), 1) for v in boxes[0]]} "
            f"score={float(scores[0]):.2f}"
            if boxes and labels and scores
            else "(box)"
        )
        return ToolResult(
            content=(
                f"detect({query!r}): grounding-dino localized {len(boxes)} object(s); "
                f"top {first}. Call {hint} to confirm."
            )
        )


class _NativeCapabilityTool:
    """Any registered Capability as a native tool (S6): the model routes to it by name.

    A ``side_effecting`` capability (a VLA, a planner that moves the robot) goes
    through the permission gate as ``ask``; a read-only one is allowed. The
    capability never self-certifies — its output is tool text, the verify is the moat.
    """

    def __init__(self, capability: Any) -> None:
        self._capability = capability
        self.name = str(capability.name)
        self.description = str(
            getattr(capability, "description", "")
            or f"Routable {getattr(capability, 'kind', 'model')} capability '{self.name}'."
        )
        self.input_schema = dict(
            getattr(capability, "input_schema", None) or {"type": "object", "properties": {}}
        )
        self._side_effecting = bool(getattr(capability, "side_effecting", True))

    def check_permissions(self, params: dict[str, Any], context: ToolContext) -> PermissionResult:
        return PermissionResult("ask" if self._side_effecting else "allow")

    def is_read_only(self, params: dict[str, Any]) -> bool:
        return not self._side_effecting

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        result = self._capability.invoke(dict(params or {}), context)
        if not getattr(result, "success", False):
            return ToolResult(
                content=f"{self.name} failed: {getattr(result, 'error', '') or 'no output'}",
                is_error=True,
            )
        output = json.dumps(getattr(result, "output", {}) or {}, ensure_ascii=False, default=str)
        return ToolResult(content=f"{self.name}: {output[:2000]}")


# ---------------------------------------------------------------------------
# Synthetic tool schemas (the verify/finish/motor tool-set the model is offered)
# ---------------------------------------------------------------------------


def _verify_tool_schema(oracle_names: frozenset[str]) -> dict[str, Any]:
    """Anthropic-shaped schema for the synthetic ``verify(expr)`` tool.

    The description single-sources the registry-derived verify vocab (the live
    oracle names + the ``at_position`` tol semantics) so the model's verify expr is
    grounded in the SAME namespace ``verify_oracle_names`` reads (review fix 6).
    """
    names = ", ".join(sorted(oracle_names)) if oracle_names else "(none)"
    tol = _at_position_tol()
    desc = (
        "Check a deterministic post-condition predicate against real world state, "
        "then bind it as this step's verification. Call this AFTER the action skill(s) "
        "that should have achieved the goal. The predicate is evaluated by an "
        "independent verifier over real sensor/sim ground truth — it cannot be faked. "
        f"Available predicate oracles: {names}. "
        f"at_position(x, y, tol={tol}) is True when the robot's planar position is "
        f"within tol metres of (x, y) (tol defaults to {tol}). "
        "Pass the FULL predicate as 'expr', e.g. at_position(2.0, 0.0)."
    )
    return {
        "name": VERIFY_TOOL,
        "description": desc,
        "input_schema": {
            "type": "object",
            "properties": {
                "expr": {
                    "type": "string",
                    "description": "The deterministic verify predicate to evaluate.",
                }
            },
            "required": ["expr"],
        },
    }


def _finish_tool_schema() -> dict[str, Any]:
    return {
        "name": FINISH_TOOL,
        "description": (
            "Signal the task is complete. Call this once every step has been "
            "executed and verified."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }


# ---------------------------------------------------------------------------
# NativeStepRunner — the ONLY stateful piece (capture -> dispatch -> verify -> grade)
# ---------------------------------------------------------------------------


class NativeStepRunner:
    """Assemble an ExecutionTrace from a native tool-use turn loop.

    Holds the live GoalVerifier namespace + oracle names + the motor tool-set, and
    the per-step accumulator (the action chain + its capture baseline). The MODEL
    issues the tool calls; this runner only dispatches, verifies, grades, and
    records — never decides what to do next.
    """

    def __init__(
        self,
        agent: Any,
        verifier: Any,
        oracle_names: frozenset[str],
        motor_tools: dict[str, Any],
        tool_context: ToolContext,
        ask_permission: Callable[[str, dict[str, Any]], str] | None = None,
    ) -> None:
        self._agent = agent
        self._verifier = verifier
        self._oracle_names = oracle_names
        self._motor_tools = motor_tools
        self._ctx = tool_context
        # Resolves an "ask" permission (the REPL's prompt). None denies: the
        # headless default — only --no-permission (PermissionContext) allows.
        self._ask = ask_permission

        # Per-step accumulator: the producing action chain (skill names) + the
        # capture baseline taken BEFORE the first skill of the current step.
        self._chain: list[str] = []
        self._baseline: Any = None
        self._step_open: bool = False
        self._step_acted: bool = False  # a side-effecting tool ran this step
        self._last_ok: bool = True      # the chain's last tool did not report an error

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
        """True iff a SIDE-EFFECTING tool ran for the current step but no verify has
        closed it — i.e. the model is about to finish/stop WITHOUT proving the action
        achieved the goal (D23: weak models skip verify on easy tasks, yielding an
        empty, ungraded trace). Used to nudge a model-authored verify before accepting
        finish. A read (file_read, detect, where_am_i) changes nothing, so it is never
        nudged: the model answers from it.
        """
        return self._step_acted

    # ------------------------------------------------------------------
    # Skill dispatch (capture baseline before the first skill of a step)
    # ------------------------------------------------------------------

    def dispatch_skill(self, name: str, params: dict[str, Any]) -> ToolResult:
        """Gate, then run a motor/skill tool; open a step if needed.

        Every dispatch goes through the SAME permission seam the ReAct and VGG tool
        paths use (``resolve_permission``): an intrinsic deny (the bash deny-list, a
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
            swallow_check_errors=True, swallow_always_allow_errors=True,
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
            if self._step_open:
                self.record_unverified()
            self.actions_dispatched += 1
            return execute_resolved_tool(
                tool, params, self._ctx, error_prefix=f"Tool '{name}' raised",
                on_error=lambda _n, exc: logger.debug("native_loop: %r raised: %s", name, exc),
            )
        if not self._step_open:
            # First skill of a fresh step -> capture the causation baseline NOW.
            self._baseline = actor_causation.capture(self._agent)
            self._step_open = True
        self._chain.append(name)
        try:
            read_only = bool(tool.is_read_only(params))
        except Exception:  # noqa: BLE001 — no/broken declaration: fail closed (it acts)
            read_only = False
        if not read_only:
            self.actions_dispatched += 1
            self._step_acted = True
        result = execute_resolved_tool(
            tool, params, self._ctx, error_prefix=f"Skill '{name}' raised",
            on_error=lambda _n, exc: logger.debug("native_loop: skill %r raised: %s", name, exc),
        )
        self._last_ok = not result.is_error
        return result

    # ------------------------------------------------------------------
    # The verify-tool HANDLER — appends EXACTLY ONE StepRecord per pair
    # ------------------------------------------------------------------

    def handle_verify(self, expr: str) -> ToolResult:
        """Evaluate *expr* via the live GoalVerifier, grade causation, record ONE step.

        This is the heart of the granularity contract (review fix 2): the
        (action-chain -> verify) pair becomes ONE StepRecord whose SubGoal.verify is
        the model's expr and whose strategy is the producing action chain. The loop
        NEVER computes verified — the spine does, from this trace.
        """
        expr = (expr or "").strip()
        # Evaluate the predicate via the SAME GoalVerifier sandbox the spine uses.
        try:
            verify_result = bool(self._verifier.verify(expr))
        except Exception as exc:  # noqa: BLE001
            logger.debug("native_loop: verify(%r) raised: %s", expr, exc)
            verify_result = False

        # Grade actor-causation with a FRESH post-capture, immediately after the
        # verify read (review fix 3) — only for a graded robot predicate, else
        # NOT_GRADED (legacy-equivalent). The baseline was captured before this
        # step's first skill; if no skill ran (a verify-only NO-OP step), baseline
        # is None -> grade fail-closed UNCAUSED for a robot predicate.
        actor_caused = self._grade(expr)

        strategy = self._chain[-1] if self._chain else ""
        step_name = f"native_step_{self._step_idx}"
        self._step_idx += 1

        sub_goal = SubGoal(
            name=step_name,
            description=f"native: {' -> '.join(self._chain) or '(no action)'} | verify {expr}",
            verify=expr,
            strategy=strategy,
        )
        # The step "ran" (success) iff its producing action — the LAST tool of the
        # chain — did not report an error; a pure check (no tool) records
        # success=True so the gate keys on GROUNDED/RAN (a NO-OP reaches the verdict
        # as RAN, not FAILED). A failed skill followed by a true verify is FAILED,
        # never GROUNDED: the skill's own failure is not discarded.
        step = StepRecord(
            sub_goal_name=step_name,
            strategy=strategy,
            success=self._last_ok if self._chain else True,
            verify_result=verify_result,
            duration_sec=0.0,
            error="" if (self._last_ok or not self._chain) else f"{strategy} reported an error",
            actor_caused=actor_caused,
        )
        self._sub_goals.append(sub_goal)
        self._steps.append(step)
        self._close_step()

        return ToolResult(
            content=(
                f"verify({expr}) -> {verify_word(verify_result)} "
                f"(result={verify_result}, actor={actor_caused.value})"
            )
        )

    def rebind(self, agent: Any, verifier: Any, oracle_names: frozenset[str], motor_tools: dict[str, Any]) -> None:
        """A lifecycle tool switched the body: later steps run, verify and grade on *agent*."""
        self._agent, self._verifier, self._oracle_names = agent, verifier, oracle_names
        self._motor_tools = motor_tools
        self._ctx.agent = agent

    def _close_step(self) -> None:
        # Reset the action chain + baseline for the next pair. NEVER carry a baseline
        # across a verify (else the next step's causation folds in).
        self._chain = []
        self._baseline = None
        self._step_open = False
        self._step_acted = False
        self._last_ok = True

    def record_abort(self, reason: str) -> None:
        """Close the turn with ONE failed step: it was stopped (or its LLM died).

        The unverified action chain (if any) is its producing strategy, so an
        interrupted action is on the record — never silently dropped — and grades
        FAILED (success=False), never GROUNDED.
        """
        self._record_open(reason, success=False)

    def record_unverified(self) -> None:
        """Record the open chain as a step the model never verified (grades RAN at best)."""
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
        ))
        self._close_step()

    def _grade(self, expr: str) -> "actor_causation.ActorCaused":
        """Grade actor-causation for the just-verified step (R2b + step-9 semantics).

        ROBOT predicate (base/arm/gripper, graded live in the oracle set): a fresh
        post-capture vs the step's entry baseline -> CAUSED / UNCAUSED (mirrors
        ``GoalExecutor._grade_actor_causation``). A robot-predicate step whose
        baseline is None (no skill ran) grades UNCAUSED (grade fail-closes).

        NON-ROBOT predicate (dev / state oracle — ``file_exists`` / ``path_contains``
        / a ``get_position()`` compare): actor-causation has no displacement metric,
        so step-9 ties grounding to whether the ACTOR ACTED this step. If NO action
        skill was dispatched (``self._chain`` empty), the verify reads pre-existing /
        ambient state the actor did not cause -> UNCAUSED (the spine R2b downgrade in
        ``classify_step_evidence`` then flips an otherwise-GROUNDED predicate to RAN),
        closing the verify-only dev/state NO-OP hole (2026-06-19 review). If ≥1 action
        skill ran -> NOT_GRADED (legacy-equivalent; the action plausibly produced the
        state — e.g. file_write -> path_contains stays GROUNDED). Goal AUTHENTICITY of
        the verify (does the constant match the task goal; an action + a trivial-but-
        true compare like ``len(get_position())==3``) remains the deepest residual,
        deferred — it needs the real task goal, not a structural check.

        Fail-safe to NOT_GRADED on any unexpected error (never raises).
        """
        try:
            if not actor_causation.is_robot_predicate(expr, self._oracle_names):
                # No side-effecting action this step (nothing, or only read-only
                # tools like file_read) -> the actor caused nothing.
                if not self._step_acted:
                    return actor_causation.ActorCaused.UNCAUSED
                return actor_causation.ActorCaused.NOT_GRADED
            post = actor_causation.capture(self._agent)
            return actor_causation.grade(self._baseline, post, expr, self._oracle_names)
        except Exception as exc:  # noqa: BLE001
            logger.debug("native_loop: grade(%r) raised: %s", expr, exc)
            return actor_causation.ActorCaused.NOT_GRADED

    # ------------------------------------------------------------------
    # Trace assembly
    # ------------------------------------------------------------------

    def build_trace(
        self, goal: str, *, llm_error: str = "", model: str = "", cancelled: bool = False
    ) -> ExecutionTrace:
        """Assemble the ExecutionTrace from the recorded (chain -> verify) steps.

        ``success`` is True iff at least one step was recorded AND none failed —
        the structural success flag the verdict reads (the moat is GROUNDED/RAN, not
        this flag). An empty trace (no verify ever called) is success=False, so the
        verdict gate fails closed (no checked step -> not verified).
        """
        steps = tuple(self._steps)
        success = bool(steps) and all(s.success for s in steps)
        goal_tree = GoalTree(goal=goal, sub_goals=tuple(self._sub_goals))
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
    ground-truth line at both ends for the independent grader (tools/gt_grade.py).
    """
    from vector_os_nano.sim.session import mark_turn

    agent = agent if agent is not None else getattr(engine, "_vgg_agent", None)
    body = [agent]  # the body the turn drives now: a lifecycle tool rebinds it
    mark_turn(agent, "start", user_message)
    try:
        return _run_turn_native(
            engine, user_message, body=body, session=session, app_state=app_state,
            max_turns=max_turns, on_progress=on_progress,
            ask_permission=ask_permission, cancel=cancel,
        )
    finally:
        mark_turn(body[0], "end", user_message)


def _bind_body(engine: Any, agent: Any) -> tuple[Any, frozenset[str], dict[str, Any], list[dict[str, Any]]]:
    """Everything a turn derives from its body: (verifier, oracle names, tools, schemas).

    The verifier + oracle names are single-sourced from the SAME namespace the spine
    reads (rule 3); fail closed on any error (an empty oracle set classifies every
    predicate RAN). The motor tool-set is the world's skills + the registry's tools;
    the synthetic verify/finish are handled in-loop, never dispatched.
    """
    verifier = _build_verifier(engine, agent)
    oracle_names = verify_oracle_names(agent, engine)
    motor_tools = _build_motor_tools(agent, engine)
    return verifier, oracle_names, motor_tools, _native_tool_schemas(motor_tools, oracle_names)


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
    from vector_os_nano.sim.session import mark_turn

    token = cancel if cancel is not None else _abort.new_run()
    agent = body[0]
    verifier, oracle_names, motor_tools, tool_schemas = _bind_body(engine, agent)
    ctx = _build_tool_context(agent, session, app_state, engine, token)
    runner = NativeStepRunner(agent, verifier, oracle_names, motor_tools, ctx, ask_permission)

    def _rebind() -> None:
        """A lifecycle tool succeeded: the next round-trip sees the new body's tools,
        verifier and oracle names, and the sim's turn marks move to it."""
        nonlocal oracle_names, tool_schemas
        app = ctx.app_state
        new = app.get("agent") if app is not None and "agent" in app else getattr(engine, "_vgg_agent", None)
        verifier, oracle_names, motor_tools, tool_schemas = _bind_body(engine, new)
        runner.rebind(new, verifier, oracle_names, motor_tools)
        if new is not body[0]:
            mark_turn(body[0], "end", user_message)
            body[0] = new
            mark_turn(new, "start", user_message)

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
            messages = _to_messages(session)
            _narration.clear()  # fresh "thinking" tail per round-trip
            try:
                backend, response = _call_llm(backend, dict(
                    messages=messages,
                    tools=tool_schemas,
                    # Rebuilt per round-trip: the [Robot State] block follows the body.
                    system=_native_system_prompt(engine, oracle_names, body[0]),
                    max_tokens=getattr(engine, "_max_tokens", 4096),
                    on_text=_on_text if on_progress is not None else None,
                ))
            except Exception as exc:  # noqa: BLE001 — infrastructure, not the agent
                llm_error = f"{type(exc).__name__}: {exc}"
                logger.warning("native_loop: LLM unavailable: %s", llm_error)
                if runner._step_open or runner._steps:
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
                    res = runner.handle_verify(expr)
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
        else:
            if token.cancelled:
                runner.record_abort("aborted")
    except Exception as exc:  # noqa: BLE001 — a producer bug after an action: own the turn
        if not runner.actions_dispatched:
            raise
        logger.warning("native_loop: turn failed after %d action(s): %s", runner.actions_dispatched, exc)
        runner.record_abort(f"producer error: {type(exc).__name__}: {exc}")
    if runner._step_open:
        # The model never verified its last tool chain (a read it answered from, or
        # an action once the nudges are spent): on the record as an unproven step,
        # never silently dropped.
        runner.record_unverified()
    return runner.build_trace(
        user_message, llm_error=llm_error, model=model, cancelled=token.cancelled
    )


# ---------------------------------------------------------------------------
# Helpers — verifier / tools / system prompt / session glue (no planner imports)
# ---------------------------------------------------------------------------


def _build_verifier(engine: Any, agent: Any) -> Any:
    """Build a live GoalVerifier over the engine's verify namespace.

    Reuses the EXISTING namespace builder so the verify sandbox is byte-identical
    to the one the spine uses. Reuses the already-wired GoalVerifier on the engine's
    goal executor when present (same namespace), else constructs a fresh one.
    """
    from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier

    executor = getattr(engine, "_goal_executor", None)
    live = getattr(executor, "_verifier", None)
    if live is not None and hasattr(live, "verify"):
        return live
    builder = getattr(engine, "_build_verifier_namespace", None)
    ns = builder(agent) if builder is not None else {}
    return GoalVerifier(ns)


def _build_motor_tools(agent: Any, engine: Any) -> dict[str, Any]:
    """Assemble the loop's ACTION surface: the world's skills + the engine's code tools.

    Two sources, both world-agnostic by construction:

    1. ``wrap_skills(agent)`` — the world's robot skills (``walk`` etc.). ``navigate``
       is intentionally NOT surfaced as a step strategy: its cmd_vel is GATED OUT of
       the actor-causation counter, so offering it would false-FAIL an honest move. We
       drop it by construction (the acceptance pins per-step strategy == 'walk', never
       'navigate'). When ``agent`` is None (the dev world), this source is empty.
    2. The engine registry's ``code``-category tools (file_read/file_write/file_edit/
       bash/glob/grep) — the kernel's domain-general action surface, the SAME tools the
       legacy dev path dispatches via ``DEV_TOOL_ALLOWLIST`` + ``ToolDispatcher``. These
       are real ``Tool`` objects whose ``.execute(params, ctx)`` is the interface
       ``dispatch_skill`` already calls, so they slot in directly. This is what lets the
       dev world (no agent -> no skills) ACT, and adds the code tools to the robot world
       too. Native asks the ENGINE for its registered action tools — there is NO
       embodiment/"if dev" branch in this module (rule 7).

    The synthetic ``verify``/``finish`` names are loop-owned and never collide with a
    registry tool; a code tool can never shadow them. A wrapped skill that happens to
    share a code-tool name (none do today) would take precedence — skills are layered
    AFTER the code tools so the world's own skill wins if a clash ever arises.
    """
    tools: dict[str, Any] = {}
    # Source 2: the engine registry's code tools (present in every world). In a ROBOT
    # world (a connected robot ``agent``), the MUTATING code tools (file_write/file_edit/
    # bash) are DROPPED from the action surface — a physical robot goal must be achieved
    # by a robot SKILL and proven by a GT oracle, never by writing a marker file or
    # running a shell command (the D17 fakeable-grasp path). The READ-ONLY diagnostics
    # (file_read/glob/grep) are kept so the persona can still inspect code. The dev world
    # (no robot agent) keeps the full set unchanged.
    is_robot_world = agent is not None
    for name, code_tool in _registry_tools(engine, _CODE_TOOL_CATEGORY).items():
        if is_robot_world and name in _MUTATING_CODE_TOOLS:
            continue  # Prong 1: no file/shell WRITE as a path to a physical robot task
        tools[name] = code_tool
    # The sim lifecycle (start/stop_simulation) in every world: a dev-world turn can
    # bring a body up and use it in the same turn (the runner rebinds after it).
    tools.update(_registry_tools(engine, _SIM_TOOL_CATEGORY))
    # A robot world's registry tools that are not skill wrappers: scene_graph_query
    # (read-only memory), go_to_room. Skills come fresh from wrap_skills below.
    if is_robot_world:
        tools.update({n: t for n, t in _registry_tools(engine, _ROBOT_TOOL_CATEGORY).items()
                      if not getattr(t, "_is_skill_wrapper", False)})
    # Source 3 (D9 #1): the avoidance NAVIGATION route, only for a world with a mobile
    # base: the base's planner routes around obstacles, holding skill control, so the
    # move is counted for actor-causation. The arm/dev worlds (no base) do not get it.
    # The base gate is single-sourced (Rule 11) onto the capability resolver.
    from vector_os_nano.embodiments.capability_profile import resolve_capability_profile

    if resolve_capability_profile(agent).has_base:
        tools["navigate"] = _NativeBaseNavigateTool()
    # Source 4 (S6): EVERY capability the world registered into the engine's
    # CapabilityRegistry, as a tool the model routes to by name — the SAME instances
    # the world bound (agent-bound for live perception), never a second model load.
    # A detector keeps its verify-hint adapter (R4); any other capability is generic,
    # and a side-effecting one asks permission first. Layered before the world skills
    # so a learned route WINS a name clash (the bare-cli 'detect' is the MODEL route,
    # not the classical DetectSkill).
    cap_names: set[str] = set()
    for cap in _registered_capabilities(engine):
        if getattr(cap, "kind", "") == "chat":
            continue  # the model driving this loop IS the chat route: no nested LLM call
        tool = _NativeDetectTool(cap) if getattr(cap, "kind", "") == "detector" else _NativeCapabilityTool(cap)
        tools[tool.name] = tool
        cap_names.add(tool.name)
    # Source 1: the world's skills (robot worlds only; dev world has no agent). The
    # world's own ``navigate`` skill (door-chain walk, NOT lidar avoidance) stays
    # excluded; our coordinate ``navigate`` above is the planner/avoidance route.
    if agent is not None:
        try:
            for skill_tool in wrap_skills(agent):
                if skill_tool.name == "navigate":
                    continue  # world room-navigate is door-chain walk; ours is the avoidance route
                if skill_tool.name in cap_names:
                    continue  # a registered capability (Source 4) wins over a same-named skill
                tools[skill_tool.name] = skill_tool
        except Exception as exc:  # noqa: BLE001
            logger.debug("native_loop: wrap_skills failed: %s", exc)
    return tools


def should_attempt_native(user_input: str, *, agent: Any, engine: Any) -> bool:
    """REGISTRY-DRIVEN native-attempt hint (S5c) — the replacement for the keyword router.

    The native producer routes by the MODEL reading tool DESCRIPTIONS, not a keyword
    table — so the only PRE-gate question ("is it worth attempting native?") collapses
    to "does this world expose any actionable tool the model could dispatch?". That is
    derived SINGLE-SOURCE from ``_build_motor_tools`` (the EXACT toolset the native loop
    offers; Rule 3), never from ``IntentRouter._RULES`` / ``should_use_vgg`` keyword sets.

    FAIL-OPEN by design: when the world is actionable, attempt native and let the model
    decide routing — a goal it cannot route falls back to legacy on no-action (a wasted
    LLM call, NEVER a missed command). This makes the hint a SAFE SUPERSET of the keyword
    ``should_use_vgg`` **WITHIN AN ACTIONABLE WORLD** (proven in
    tests/vcli/test_native_routing_hint.py): for every input the keyword router routes to
    VGG, this also attempts — so S8 can rewire the gate sites off ``should_use_vgg`` onto
    this with no missed routing. The superset is SCOPED to an actionable world by
    construction: in a TOOLLESS world (``_build_motor_tools`` empty — a sensorless/dev path)
    this returns False even for an action-shaped command, which is CORRECT (nothing for the
    model to route to; the keyword router's VGG attempt would only fail there too). The
    trivial-input threshold MATCHES ``should_use_vgg`` exactly (``len(user_input) < 2`` on the
    RAW input — NOT stripped — so a 1-char-after-strip command like "去 " that ``should_use_vgg``
    accepts is not silently dropped; the earlier ``len(strip())`` was an off-by-strip MISS,
    D79). A toolset-build error fails OPEN to native. The REPL gates on this (WP14): every
    world now offers the sim lifecycle, so in practice every non-trivial turn attempts native.
    """
    if not user_input or len(user_input) < 2:
        return False
    try:
        tools = _build_motor_tools(agent, engine)
    except Exception:  # noqa: BLE001 — fail OPEN to native; never silently skip the redesign
        return True
    return len(tools) >= 1


def _registered_capabilities(engine: Any) -> list[Any]:
    """Every Capability in the ENGINE's CapabilityRegistry (``engine._capability_registry``).

    The engine owns the registry the world populated at init (the same one the
    GoalExecutor routes through), so the native loop surfaces the EXACT instances —
    never a fresh construction (rule 3). No registry / a lookup failure -> none.
    """
    registry = getattr(engine, "_capability_registry", None)
    if registry is None:
        return []
    try:
        return [c for c in (registry.get(n) for n in sorted(registry.names())) if c is not None]
    except Exception as exc:  # noqa: BLE001
        logger.debug("native_loop: capability discovery failed: %s", exc)
        return []


def _registry_tools(engine: Any, category: str) -> dict[str, Any]:
    """Return the engine registry's *category* Tool objects, keyed by name.

    Pulls the live, instantiated tools out of the engine's ``CategorizedToolRegistry``
    so the native loop offers the EXACT action surface the kernel registered (no
    duplicate construction, no hand-authored allowlist here — single-sourced from the
    registry, rule 3). Defensive: a registry without the categorized API, a missing
    category, or any lookup failure yields an empty set. Disabled categories are
    still surfaced on purpose: the loop's tool-set is independent of the chat-path
    category gating (the caller gates robot tools on a connected agent).
    """
    registry = getattr(engine, "_registry", None)
    if registry is None:
        return {}
    list_categories = getattr(registry, "list_categories", None)
    get = getattr(registry, "get", None)
    if list_categories is None or get is None:
        return {}  # a plain ToolRegistry (no categories) -> no code tools to surface
    tools: dict[str, Any] = {}
    try:
        names = list_categories().get(category, [])
        for name in names:
            code_tool = get(name)
            if code_tool is not None:
                tools[name] = code_tool
    except Exception as exc:  # noqa: BLE001
        logger.debug("native_loop: %s-tool discovery failed: %s", category, exc)
    return tools


def _native_tool_schemas(
    motor_tools: dict[str, Any], oracle_names: frozenset[str]
) -> list[dict[str, Any]]:
    """The Anthropic-shaped tool schemas the model is offered for a native turn."""
    schemas: list[dict[str, Any]] = []
    for name, tool in motor_tools.items():
        schemas.append(
            {
                "name": name,
                "description": getattr(tool, "description", name),
                "input_schema": getattr(tool, "input_schema", {"type": "object", "properties": {}}),
            }
        )
    schemas.append(_verify_tool_schema(oracle_names))
    schemas.append(_finish_tool_schema())
    return schemas


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
    "then verify again. Never finish while a goal's latest verify is FAIL; if you cannot "
    "recover, say why and stop.\n"
    "- A read-only tool (status, memory, perception) changes nothing: answer from it, no "
    "verify needed.\n"
    "- start_simulation / stop_simulation need no verify (they return only once the sim "
    "is ready or gone). After one succeeds, your tools and verify predicates belong to "
    "the new body from your next call on.\n"
)


def _native_system_prompt(engine: Any, oracle_names: frozenset[str], agent: Any = None) -> list[dict[str, Any]]:
    """The native system prompt: kernel loop rules + the world's persona + world state.

    The verify vocab is the SAME live oracle set ``verify_oracle_names`` reads (review
    fix 6). The world persona (``engine._world.persona_blocks(agent)``) carries the
    body's task policy; the ``[Robot State]`` block is the actor's belief right now
    (pose, room, gripper, localized objects, nav/explore backend), rebuilt for every
    round-trip. Nothing here names a ground-truth object.
    """
    names = ", ".join(sorted(oracle_names)) if oracle_names else "(none)"
    text = _KERNEL_RULES + f"Available verify predicate oracles: {names}."
    if "at_position" in oracle_names:
        text += f" at_position(x, y, tol={_at_position_tol()}) is True within tol metres of (x, y)."
    blocks = [{"type": "text", "text": text}]
    world = getattr(engine, "_world", None)
    if world is not None:
        try:
            blocks += [{"type": "text", "text": t.strip()} for t in world.persona_blocks(agent) if t]
        except Exception as exc:  # noqa: BLE001 — the kernel rules still stand
            logger.warning("native_loop: world persona unavailable: %s", exc)
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


def _to_messages(session: Any) -> list[dict[str, Any]]:
    if session is not None and hasattr(session, "to_messages"):
        return session.to_messages()
    return []


def _tool_result_dict(tool_use_id: str, content: str, is_error: bool = False) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": is_error,
    }
