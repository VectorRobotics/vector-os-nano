# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The interactive REPL: the input loop, the turn worker, stop words and Ctrl-C.

A turn runs on a worker thread while the main thread keeps reading input: a stop
phrase or Ctrl-C stops the robot at once (no LLM call), any other line cancels the
running turn and runs next, and a permission prompt the turn raises is answered here.
"""
from __future__ import annotations

import json
import logging
import math
import os
import queue
import select
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PTStyle
from rich.markup import escape
from rich.prompt import Prompt

from vector_os_nano.vcli.cognitive import abort as _abort
from vector_os_nano.vcli.render import TEAL, console, print_banner, render_turn
from vector_os_nano.vcli.runtime import flush, run_turn
from vector_os_nano.vcli.slash import EXIT_COMMANDS, VectorCompleter, _handle_slash_command

logger = logging.getLogger(__name__)


# A stop reports the base speed at the first SIM time after the stop command, its
# peak from there to the second, and its speed at the second.
_STOP_SETTLE_SIM_S = (0.2, 1.0)


# After a stop, wait this long for the cancelled turn to unwind before re-prompting.
_UNWIND_S = 5.0


PT_STYLE = PTStyle.from_dict({
    # Completion menu
    "completion-menu": "bg:#0a0a1a",
    "completion-menu.completion": "bg:#0a0a1a #00b4b4",
    "completion-menu.completion.current": "bg:#00b4b4 #000000 bold",
    "completion-menu.meta.completion": "bg:#0a0a1a #555555",
    "completion-menu.meta.completion.current": "bg:#00b4b4 #000000",
    "completion-menu.multi-column-meta": "bg:#0a0a1a #555555",
    # Scrollbar
    "scrollbar.background": "bg:#0a0a1a",
    "scrollbar.button": "bg:#006666",
    # Bottom toolbar
    "bottom-toolbar": "bg:#0a0a1a #00b4b4",
    "bottom-toolbar.text": "bg:#0a0a1a #00b4b4",
    # Prompt
    "prompt": "bold #00b4b4",
})


def _enable_crash_log(session: Any) -> Any:
    """faulthandler -> ``<session>.crash.log`` next to the transcript (C-level crash traces)."""
    import faulthandler

    try:
        fh = open(Path(session._path).with_suffix(".crash.log"), "a")  # noqa: SIM115 — lives as long as the process
        faulthandler.enable(file=fh, all_threads=True)
        return fh
    except Exception as exc:  # noqa: BLE001
        logger.debug("crash log unavailable: %s", exc)
        return None


def _repl_turn(rt: Any, user_input: str, console: Any) -> None:
    """Run ONE REPL turn through ``run_turn`` and render it.

    The model's own last words are shown; then, for a turn that claimed something
    (acted, verified, was refused, stopped or lost its LLM), its steps and the honest
    verdict — single-sourced from the SAME gate the ``-p`` path uses
    (``VerdictReport.from_trace``; the loop NEVER computes ``verified``). A chat turn
    is the model's text alone.

    The turn runs on the REAL session, so a follow-up turn sees what this one did
    ("把绿色瓶子拿过来" then "再把它放到餐桌上": 'it' resolves). A sim started by
    the turn rebinds ``engine._agent``, so the verdict is graded on the body the
    turn ended on.
    """
    engine = rt.engine
    # Suppress ROS2/subprocess stderr noise during the turn; restore unconditionally.
    _saved_stderr = sys.stderr
    trace: Any = None
    try:
        try:
            sys.stderr = open(os.devnull, "w")
        except OSError:
            pass
        with console.status(f"[{TEAL}]native[/] working…", spinner="dots") as _status:
            # Live progress (D9 #2 perceived latency): stream the model's thinking
            # tail + each tool call into the spinner so the multi-second LLM wait
            # shows activity instead of a frozen line. Best-effort, never fatal.
            def _on_progress(msg: str) -> None:
                try:
                    if msg.startswith("Task contract:"):  # stays on screen (WP07)
                        console.print(f"  [{TEAL}]▸[/] {escape(msg)}")
                        return
                    _status.update(f"[{TEAL}]native[/] {msg}")
                except Exception:  # noqa: BLE001
                    pass

            def _ask(name: str, params: dict[str, Any]) -> str:
                _status.stop()  # a live spinner would draw over the prompt
                try:
                    return ask_permission(name, params)
                finally:
                    _status.start()
            trace, answer, turn = run_turn(rt, user_input, on_progress=_on_progress, ask_permission=_ask)
    finally:
        if sys.stderr is not _saved_stderr:
            try:
                sys.stderr.close()
            except Exception:  # noqa: BLE001
                pass
        sys.stderr = _saved_stderr

    if answer:
        rt.app_state["last_response"] = answer  # for /copy
    render_turn(console, engine, trace, answer, turn)


def _stop_report(agent: Any) -> str:
    """'base 0.03 m/s at 0.2 sim-s, peak 0.06 m/s to 1.0 sim-s, 0.00 m/s at 1.0 sim-s'.

    The measured proof: the command is zeroed at once and a legged body decelerates,
    so its horizontal speed is sampled (sim time when the body exposes it) from the
    first settle mark to the second — the peak, not just two samples.
    """
    base = getattr(agent, "_base", None)
    get_v = getattr(base, "get_velocity", None)
    if not callable(get_v):
        return ""
    sim_t = getattr(base, "get_sim_time", None)
    clock = sim_t if callable(sim_t) else time.monotonic
    first, last = _STOP_SETTLE_SIM_S
    try:
        t0, wall_end, at_first, peak = clock(), time.monotonic() + 4.0, None, 0.0
        while True:
            dt = clock() - t0
            v = get_v()
            speed = math.hypot(v[0], v[1])
            stalled = time.monotonic() >= wall_end  # the sim clock fell behind: say when
            if dt >= first or stalled:
                at_first = at_first or (speed, dt)
                peak = max(peak, speed)
            if dt >= last or stalled:
                break
            time.sleep(0.005)
        return (f"base {at_first[0]:.2f} m/s at {at_first[1]:.1f} sim-s, peak {peak:.2f} m/s "
                f"to {dt:.1f} sim-s, {speed:.2f} m/s at {dt:.1f} sim-s")
    except Exception:  # noqa: BLE001 — the report is display only
        return ""


def _p0_stop(engine: Any, text: str, app_state: dict[str, Any], session: Any = None) -> None:
    """Stop NOW: cancel every run + the stop skill, no LLM call (P0, < 100 ms)."""
    t0 = time.monotonic()
    engine._emergency_stop(text, session, app_state.get("agent"), app_state)
    ms = (time.monotonic() - t0) * 1000.0
    console.print(f"[yellow]■ stopped[/] [dim]({ms:.0f} ms, no LLM call)[/]")
    report = _stop_report(app_state.get("agent"))
    if report:
        console.print(f"[dim]  {report}[/]")


def _read_line(timeout: float) -> str | None:
    """A line typed on stdin within *timeout* s, else None (the busy loop's input)."""
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
    except (OSError, ValueError):  # stdin closed / not selectable
        time.sleep(timeout)
        return None
    return sys.stdin.readline() if ready else None


def _run_turn_busy(engine: Any, fn: Callable[[], None], app_state: dict[str, Any]) -> str | None:
    """Run one engine turn on a worker thread while this (main) thread keeps reading input.

    During the turn:
      * a stop phrase (the registered stop skill's aliases) or Ctrl-C stops the robot
        and cancels the running turn at once, with no LLM call; Ctrl-C never exits;
      * a permission prompt the turn raises is answered here (this thread owns stdin);
      * any OTHER line cancels the running turn (robot stopped) and is returned, so
        the caller runs it next: cancel-then-run.
    The turn is one stop scope (``abort.turn``): a stop that lands anywhere in it —
    native's LLM call included — also stops every run the turn would start later.
    Under mjpython with a live viewer (GLFW is main-thread-only) the turn runs inline.
    """
    from vector_os_nano.hardware.sim.viewer_mode import running_under_mjpython

    if engine._has_live_viewer() and running_under_mjpython():
        try:
            with _abort.turn():
                fn()
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            _p0_stop(engine, "interrupt", app_state)
        return None

    global _ASK_Q
    asks: "queue.Queue[tuple[str, dict[str, Any], queue.Queue[str]]]" = queue.Queue()
    done = threading.Event()

    def _target() -> None:
        try:
            with _abort.turn():
                fn()
        except Exception as exc:  # noqa: BLE001 — the turn's own handler already printed
            console.print(f"[red]Error:[/] {exc}")
        finally:
            done.set()

    _ASK_Q = asks
    threading.Thread(target=_target, name="vector-turn", daemon=True).start()
    next_line: str | None = None
    stopped_at: float | None = None
    interrupted = False
    reply: "queue.Queue[str] | None" = None
    try:
        while True:
            # EVERY step, the wait included, sits inside this try: a Ctrl-C lands
            # wherever this thread happens to be and must never escape the REPL.
            # The stop it triggers runs on the next pass (a Ctrl-C during that stop
            # just stops again — the stop is idempotent).
            try:
                if interrupted:
                    interrupted = False
                    console.print("\n[yellow]Interrupted.[/yellow]")
                    _p0_stop(engine, "interrupt", app_state)
                    stopped_at = stopped_at or time.monotonic()
                if done.wait(0.02):
                    break
                if stopped_at is not None and time.monotonic() - stopped_at > _UNWIND_S:
                    console.print("[dim]  the stopped task is still unwinding; back to the prompt[/]")
                    break
                try:
                    name, params, reply = asks.get_nowait()
                except queue.Empty:
                    reply = None
                else:
                    reply.put("n" if stopped_at is not None else ask_permission(name, params))
                    reply = None
                    continue
                line = (_read_line(0.03) or "").strip()
                if not line:
                    continue
                if engine.is_stop(line):
                    next_line = None  # a stop also drops a command queued behind the turn
                else:
                    console.print(f"[dim]  ↳ cancelling the running task for: {line}[/]")
                    next_line = line  # the latest command wins
                    if stopped_at is not None:
                        continue  # already stopped: just queue it
                _p0_stop(engine, line, app_state)
                stopped_at = stopped_at or time.monotonic()
            except KeyboardInterrupt:
                if reply is not None:
                    reply.put("n")
                    reply = None
                interrupted = True
    finally:
        _ASK_Q = None
        while True:  # nobody answers once the loop ends: deny what is still queued
            try:
                asks.get_nowait()[2].put("n")
            except queue.Empty:
                break
    return next_line


def is_slash_command(text: str) -> bool:
    return text.strip().startswith("/")


def is_exit_command(text: str) -> bool:
    return text.strip().lower() in EXIT_COMMANDS


# While a REPL turn runs on its worker thread the MAIN thread owns stdin: a worker's
# permission request is queued here and answered by the busy loop (_run_turn_busy).
_ASK_Q: "queue.Queue[tuple[str, dict[str, Any], queue.Queue[str]]] | None" = None


def ask_permission(tool_name: str, params: dict[str, Any]) -> str:
    if threading.current_thread() is not threading.main_thread():
        q = _ASK_Q
        if q is None or _abort.is_abort_requested():
            # Nobody can answer this prompt, or it comes from a stopped turn (a stale
            # worker must never prompt under the owner's next turn): fail closed.
            return "n"
        reply: "queue.Queue[str]" = queue.Queue(maxsize=1)
        q.put((tool_name, params, reply))
        return reply.get()
    console.print(f"\n[yellow bold]Permission required:[/yellow bold]")
    console.print(f"  Tool: [{TEAL}]{tool_name}[/]")
    params_str = json.dumps(params, indent=2, ensure_ascii=False)
    if len(params_str) > 200:
        params_str = params_str[:200] + "..."
    console.print(f"  Params: [dim]{params_str}[/dim]")
    return Prompt.ask("  Allow? [y/n/a=always]", choices=["y", "n", "a"], default="y")


def run_repl(rt: Any) -> None:
    """The bare ``vector-cli`` REPL over *rt*: read a line, run it, render it; until quit."""
    engine, session, registry, app_state = rt.engine, rt.session, rt.registry, rt.app_state

    print_banner(app_state["credentials"], rt.agent)
    console.print(f"[dim]Session: {session.session_id}[/dim]\n")

    # REPL setup
    history_dir = Path.home() / ".vector"
    history_dir.mkdir(parents=True, exist_ok=True)

    def _get_toolbar() -> HTML:
        agent_now = app_state.get("agent")
        current_model = app_state["credentials"].model
        parts: list[str] = [f"<b>V</b>"]
        arm_now = getattr(agent_now, "_arm", None) if agent_now else None
        base_now = getattr(agent_now, "_base", None) if agent_now else None
        if arm_now is not None:
            parts.append(f"arm:{getattr(arm_now, 'name', 'arm')}")
        if base_now is not None:
            parts.append(f"base:{getattr(base_now, 'name', 'base')}")
        parts.append(f"model:{current_model.split('/')[-1]}")
        parts.append(f"tools:{len(registry.list_tools())}")
        parts.append(f"msgs:{len(session._entries)}")
        return HTML(f' {" | ".join(parts)} ')

    pt_session: PromptSession = PromptSession(
        history=FileHistory(str(history_dir / "history")),
        completer=VectorCompleter(),
        complete_while_typing=True,
        style=PT_STYLE,
    )

    # Ctrl-C UX (R2-4): a ^C during a turn stops it and returns to the prompt (it
    # never exits); two consecutive ^C AT THE PROMPT exit cleanly.
    interrupt_pending = False
    app_state["crash_log"] = _enable_crash_log(session)

    # A line typed while a turn ran (cancel-then-run) is the next input.
    pending_input: list[str] = []

    try:
        while True:
            flush(rt)  # a segfault loses at most one turn
            # ---- Read input ----
            try:
                if pending_input:
                    raw = pending_input.pop(0)
                else:
                    raw = pt_session.prompt(
                        HTML(f'<style fg="{TEAL}" bold="true">vector&gt;</style> '),
                        bottom_toolbar=_get_toolbar,
                    )
            except EOFError:
                break
            except KeyboardInterrupt:
                if interrupt_pending:
                    break  # second consecutive Ctrl-C -> exit cleanly
                interrupt_pending = True
                console.print("\n[dim]Press Ctrl-C again or type quit to exit.[/dim]")
                continue

            interrupt_pending = False  # any successful input clears the pending exit
            user_input = raw.strip()
            if not user_input:
                continue

            # ---- Exit ----
            if is_exit_command(user_input):
                break

            # ---- Slash commands ----
            if is_slash_command(user_input):
                parts = user_input.split()
                if not _handle_slash_command(parts[0][1:], parts[1:], registry, session, app_state):
                    break
                continue

            # ---- ! shell passthrough ----
            if user_input.startswith("!"):
                cmd = user_input[1:].strip()
                if cmd:
                    try:
                        proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=30)
                        if proc.stdout:
                            console.print(proc.stdout, end="")
                        if proc.stderr:
                            console.print(f"[dim]{proc.stderr}[/]", end="")
                    except subprocess.TimeoutExpired:
                        console.print("[yellow]Command timed out (30s)[/]")
                    except Exception as exc:
                        console.print(f"[red]Error:[/] {exc}")
                continue

            # ---- Engine turn ----
            if engine._backend is None:
                console.print(f"[yellow]{escape(app_state['credentials'].instruction())}[/]")
                continue

            # P0 stop: the REGISTERED stop skill's own words, before the producer —
            # zero LLM calls, < 100 ms. Like any turn it runs on a worker thread while
            # this thread keeps reading input: a typed stop / Ctrl-C stops it mid-turn
            # (a Ctrl-C during the stop's speed report never exits the REPL), and
            # another command cancels it and runs next (cancel-then-run).
            if engine.is_stop(user_input):
                turn = lambda: _p0_stop(engine, user_input, app_state, session)  # noqa: E731
            else:
                turn = lambda: _repl_turn(rt, user_input, console)  # noqa: E731
            typed_next = _run_turn_busy(engine, turn, app_state)
            if typed_next:
                pending_input.append(typed_next)

    finally:
        session.save()
        console.print(f"[dim]Session saved: {session.session_id}[/dim]")
        # Persist scene graph if agent has one
        _agent_final = app_state.get("agent")
        if _agent_final is not None:
            _sm = getattr(_agent_final, "_spatial_memory", None)
            if _sm is not None and hasattr(_sm, "save") and hasattr(_sm, "stats"):
                try:
                    _sm.save()
                    _sm_stats = _sm.stats()
                    console.print(
                        f"[dim]Scene graph saved: "
                        f"{_sm_stats['rooms']} rooms, {_sm_stats['objects']} objects[/dim]"
                    )
                except Exception as _exc:
                    console.print(f"[yellow]Scene graph save failed: {_exc}[/yellow]")
