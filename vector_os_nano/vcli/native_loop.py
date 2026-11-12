# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""native_loop — the turn loop of the ONE producer: the model, its prompt, the session.

The MODEL drives a native tool-use ReAct loop (its tools from ``native_tools``, the
synthetic ``verify(expr)`` + ``finish``); ``native_runner`` dispatches, verifies, grades
and records, and this loop hands the resulting ``ExecutionTrace`` to the honest verify
spine unchanged. It does NOT compute ``verified`` itself — the caller feeds the trace
to ``VerdictReport.from_trace`` / ``evidence_passed`` (rule 5: verify is the moat,
single-sourced).

REPLACE-NOT-REBUILD, enforced BY CONSTRUCTION (review fix 5):

    The producer (``native_loop`` / ``native_runner`` / ``native_tools``) imports ONLY
    the honest spine — ``actor_causation``, ``trace_store``, the GoalVerifier, the
    task contract — and the tool layer; the body's sim facts come from the bound
    world. The runner's ONLY logic is: capture baseline -> dispatch the tool -> evaluate
    verify via the live GoalVerifier -> grade via ``actor_causation`` -> append ONE
    StepRecord. The MODEL owns decompose / route / replan — there is NO replan /
    iteration / "landed-short" bookkeeping here (that would be re-growing the planner).

Turn contract (WP07): the goal compiles to GT predicates (``task_contract``),
printed as "Task contract: ..." before the first model call and evaluated at turn
start and turn end; the spine requires each False -> True.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Any, Callable

from vector_os_nano.vcli.backends.types import LLMResponse
from vector_os_nano.vcli.cognitive import abort as _abort
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.object_goal import task_contract, unbound_reason
from vector_os_nano.vcli.cognitive.trace_store import verify_oracles
from vector_os_nano.vcli.cognitive.types import ExecutionTrace
from vector_os_nano.vcli.native_runner import LIFECYCLE_TOOLS, NativeStepRunner, sim_session
from vector_os_nano.vcli.native_tools import (
    FINISH_TOOL,
    VERIFY_TOOL,
    _agent_base,
    _build_motor_tools,
    _native_tool_schemas,
)
from vector_os_nano.vcli.prompt import user_prompt_files
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
            # On absolute seconds (t0 + k): an emit's own cost never drifts the labels
            # (a 1 s sleep after each emit drifted '2s' -> '4s'); only a tick that ran
            # over a whole second skips, to the next whole one.
            k = 1
            while not done.wait(max(0.0, t0 + k - time.monotonic())):
                _emit(_progress_line(name, params, base, start, k))
                k = max(k + 1, int(time.monotonic() - t0) + 1)

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
# Helpers — system prompt / tool context / session glue (no planner imports)
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
