# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Vector CLI — interactive REPL entry point.

Ties together VectorEngine, Session, PermissionContext, and all tools
into an interactive agent loop with V's personality.

    python -m vector_os_nano.vcli.cli [options]
    # or via console_scripts: vector-cli [options]
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import select
import threading
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PTStyle

from vector_os_nano.vcli.backends import create_backend
from vector_os_nano.vcli.cognitive import abort as _abort
from vector_os_nano.vcli.engine import VectorEngine
from vector_os_nano.vcli.native_loop import LIFECYCLE_TOOLS
from vector_os_nano.vcli.session import (
    Session,
    create_session,
    get_latest_session,
    list_sessions,
    load_session,
)
from vector_os_nano.vcli.permissions import PermissionContext
from vector_os_nano.vcli.tools import CategorizedToolRegistry, ToolRegistry, discover_categorized_tools
from vector_os_nano.core.run_dir import run_path

logger = logging.getLogger(__name__)
console = Console()

VERSION = "0.1.0"

TEAL = "#00b4b4"

EXIT_COMMANDS: frozenset[str] = frozenset({"quit", "exit", "q"})

_LOGO_PATH = Path(__file__).resolve().parent.parent / "cli" / "logo_braille.txt"

# A stop reports the base speed at the first SIM time after the stop command, its
# peak from there to the second, and its speed at the second.
_STOP_SETTLE_SIM_S = (0.2, 1.0)
# After a stop, wait this long for the cancelled turn to unwind before re-prompting.
_UNWIND_S = 5.0

# Popular models on OpenRouter for /model completion
KNOWN_MODELS: list[str] = [
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-haiku-4-5",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4.1",
    "openai/o3-mini",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "deepseek/deepseek-chat-v3-0324",
    "meta-llama/llama-4-maverick",
]

# Slash command definitions: (name, description, has_args)
SLASH_COMMANDS: list[tuple[str, str, bool]] = [
    ("help", "Show all commands and shortcuts", False),
    ("login", "Set up API key (Anthropic or OpenRouter)", True),
    ("model", "Show or switch model  (/model <name>)", True),
    ("config", "Show saved configuration", False),
    ("tools", "List all registered tools", False),
    ("agent", "Show V's identity and capabilities", False),
    ("status", "Show hardware, tools, session info", False),
    ("usage", "Show token usage this session", False),
    ("copy", "Copy last response to clipboard", False),
    ("export", "Export session as markdown", False),
    ("compact", "Compress context window", False),
    ("clear", "Reset conversation", False),
    ("clear_memory", "Clear scene graph (forget all explored rooms/objects)", False),
    ("reset", "Reset robot pose (stand up after tip-over, sim only)", False),
    ("sessions", "List saved sessions", False),
    ("quit", "Exit", False),
]


# ---------------------------------------------------------------------------
# Custom completer — slash commands with descriptions + model picker
# ---------------------------------------------------------------------------


class VectorCompleter(Completer):
    """Context-aware completer for the vector-cli REPL.

    - Typing `/` shows all slash commands with descriptions
    - Typing `/model ` shows known model names
    - Typing `!` shows nothing (shell passthrough)
    """

    def get_completions(self, document: Document, complete_event: Any) -> Any:
        text = document.text_before_cursor

        # Only complete slash commands and exit keywords
        if not text.startswith("/") and not text.strip().lower() in ("q", "qu", "qui", "qui", "ex", "exi"):
            return

        word = document.get_word_before_cursor(WORD=True)

        # Slash commands
        if text.startswith("/"):
            parts = text.split(None, 1)
            cmd_part = parts[0]  # e.g. "/mod"

            if len(parts) == 1 and not text.endswith(" "):
                # Still typing the command name — filter slash commands
                prefix = cmd_part[1:]  # strip leading /
                for name, desc, _has_args in SLASH_COMMANDS:
                    if name.startswith(prefix):
                        yield Completion(
                            f"/{name}",
                            start_position=-len(cmd_part),
                            display=f"/{name}",
                            display_meta=desc,
                        )
            elif cmd_part == "/model" and len(parts) >= 1:
                # After "/model " — complete model names
                model_prefix = parts[1] if len(parts) > 1 else ""
                for m in KNOWN_MODELS:
                    if m.startswith(model_prefix):
                        yield Completion(
                            m,
                            start_position=-len(model_prefix),
                            display=m,
                        )

        # Exit commands
        elif word and not text.startswith("!"):
            lower_word = word.lower()
            for ec in EXIT_COMMANDS:
                if ec.startswith(lower_word) and ec != lower_word:
                    yield Completion(ec, start_position=-len(word))


# ---------------------------------------------------------------------------
# prompt_toolkit theme — teal completion menu, styled toolbar
# ---------------------------------------------------------------------------

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


# Braille dot-art V — embedded in panel titles
V_LABEL = f"[bold {TEAL}] ⠣⡠⠃ [/]"


# Last response storage for /copy
_last_response: str = ""


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vector-cli",
        description="Vector CLI — Agentic CLI for Vector OS Nano",
    )
    parser.add_argument("--sim", action="store_true", help="Start with MuJoCo arm simulation")
    parser.add_argument("--sim-go2", action="store_true", help="Start with Go2 quadruped simulation")
    from vector_os_nano.embodiments.registry import embodiment_ids
    parser.add_argument(
        "--embodiment", default=None, choices=embodiment_ids(), metavar="ID",
        help="Start with this embodiment's simulation (--sim = so101, --sim-go2 = go2). "
        f"One of: {', '.join(embodiment_ids())}",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Suppress the MuJoCo viewer window (default: window opens when --sim is active)",
    )
    parser.add_argument("--model", default=None, help="Model to use (overrides config; default reads ~/.vector/config.yaml)")
    parser.add_argument("--resume", nargs="?", const="latest", default=None, help="Resume session")
    parser.add_argument("--api-key", default=None, help="API key (or set ANTHROPIC_API_KEY / OPENROUTER_API_KEY)")
    parser.add_argument("--base-url", default=None, help="API base URL")
    parser.add_argument("--no-permission", action="store_true", help="Allow all tools without prompts")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    parser.add_argument("--system-prompt", default=None, help="Path to custom system prompt file")
    parser.add_argument(
        "-p", "--print",
        dest="print_prompt",
        default=None,
        metavar="TEXT",
        help=(
            "Run ONE turn for TEXT non-interactively and exit (no REPL). With "
            "--json, also emit one machine-checkable VECTOR_VERDICT line on stdout."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit exactly one verdict line 'VECTOR_VERDICT {<json>}' on stdout for "
            "the -p turn (Rich/banner routed to stderr). Exit 0=verified / "
            "2=ran-not-verified / 1=error|no-trace."
        ),
    )
    return parser.parse_args(argv)


def _flush_turn_state(session: Any, app_state: dict[str, Any]) -> None:
    """Persist the transcript and the scene graph NOW (called between REPL turns).

    A MuJoCo segfault kills the process without the exit-time saves, so this
    bounds the loss to the turn that crashed; ``vector-cli --resume`` restores
    the transcript and the next sim start reloads the scene graph.
    """
    try:
        session.save()
        sm = getattr(app_state.get("agent"), "_spatial_memory", None)
        if sm is not None and hasattr(sm, "save"):
            sm.save()
    except Exception as exc:  # noqa: BLE001 — persistence is best-effort, never a crash
        logger.debug("per-turn flush failed: %s", exc)


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


def _gait_backend(agent: Any) -> str:
    """The sim body's active locomotion backend ('mpc' / 'sinusoidal'), else ''."""
    return str(getattr(getattr(agent, "_base", None), "gait_backend", "") or "")


def _trace_acted(trace: Any) -> bool:
    """True iff the turn ran a side-effecting tool (``actions_dispatched``)."""
    return int(getattr(trace, "actions_dispatched", 0) or 0) > 0


def _chat_turn(trace: Any) -> bool:
    """A turn that made no claim: no action, no verify, no refusal, no stop, no LLM error.

    Its only steps (if any) are reads the model answered from, never verified.
    """
    return not (
        _trace_acted(trace)
        or any(sg.verify for sg in trace.goal_tree.sub_goals)
        or trace.refused or trace.cancelled or trace.llm_error
    )


def _goal_line(report: Any, why: str) -> str:
    """The verdict block's goal check in words (charter 3.5): the end state proven, why
    it is not met, or that the sentence was not machine-checked and what was proven."""
    if report.goal_check == "proven":
        return f"goal proven: {' and '.join(report.contract)}"
    if report.goal_check == "failed":
        return f"goal not met: {why}"
    proven = [s.verify for s in report.per_step if s.evidence == "GROUNDED" and not s.superseded]
    return f"goal not machine-checked (no task contract for this sentence); proven: {'; '.join(proven) or 'nothing'}"


def _lifecycle_lines(entries: list[dict[str, Any]]) -> list[str]:
    """The first line of every start/stop_simulation result among *entries* (in order)."""
    ids = {b.get("id") for e in entries if e.get("type") == "assistant"
           for b in (e.get("tool_use") or ()) if b.get("name") in LIFECYCLE_TOOLS}
    return [str(r.get("content") or "").splitlines()[0][:200] for e in entries
            if e.get("type") == "tool_result" for r in e.get("results") or ()
            if r.get("tool_use_id") in ids and r.get("content")]


def _repl_turn(
    engine: Any,
    user_input: str,
    session: Any,
    app_state: dict[str, Any],
    console: Any,
) -> None:
    """Run ONE REPL turn through the native producer and render it.

    The model's own last words are shown; then, for a turn that claimed something
    (acted, verified, was refused, stopped or lost its LLM), its steps and the honest
    verdict — single-sourced from the SAME gate the ``-p`` path uses
    (``VerdictReport.from_trace``; the loop NEVER computes ``verified``). A chat turn
    is the model's text alone.

    The turn runs on the REAL session, so a follow-up turn sees what this one did
    ("把绿色瓶子拿过来" then "再把它放到餐桌上": 'it' resolves). A sim started by
    the turn rebinds ``engine._vgg_agent``, so the verdict is graded on the body the
    turn ended on.
    """
    global _last_response
    from vector_os_nano.vcli.cognitive.trace_store import goal_check, verify_oracle_names
    from vector_os_nano.vcli.verdict import VerdictReport

    n0 = len(session._entries)

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
            trace = engine.run_turn_native(
                user_input, agent=getattr(engine, "_vgg_agent", None), session=session,
                app_state=app_state, on_progress=_on_progress, ask_permission=_ask,
            )
    finally:
        if sys.stderr is not _saved_stderr:
            try:
                sys.stderr.close()
            except Exception:  # noqa: BLE001
                pass
        sys.stderr = _saved_stderr

    turn = session._entries[n0:]
    # The model's own last words (the README summary, the test run's outcome): shown,
    # then the steps and the verdict, which are the evidence.
    answer = session.last_assistant_text(n0)
    if len(session._entries) > 50:
        session.compact(keep_recent=12)
    agent = getattr(engine, "_vgg_agent", None)  # the body the turn ended on
    sub_goals = list(getattr(trace.goal_tree, "sub_goals", ()) or ())
    steps = list(getattr(trace, "steps", ()) or ())
    console.print()
    for line in _lifecycle_lines(turn):
        console.print(f"  [{TEAL}]▸[/] [dim]{escape(line)}[/]")
    if answer:
        _last_response = answer
        console.print(escape(answer))
    if _chat_turn(trace):
        return
    for refusal in getattr(trace, "refused", ()) or ():
        console.print(f"  [{TEAL}]▸[/] [red]refused[/] {escape(refusal)}")
    for i, s in enumerate(steps):
        verify_expr = sub_goals[i].verify if i < len(sub_goals) else ""
        chain = (getattr(s, "strategy", "") or "").strip() or "(no action)"
        ok = bool(getattr(s, "verify_result", False))
        actor = getattr(getattr(s, "actor_caused", None), "value", "?")
        if not verify_expr and getattr(s, "error", ""):
            console.print(f"  [{TEAL}]▸[/] {chain} → [yellow]{escape(s.error)}[/]")
            continue
        mark = "[green]✓[/]" if ok else "[yellow]·[/]"
        sup = ", superseded" if getattr(s, "superseded", False) else ""
        console.print(
            f"  [{TEAL}]▸[/] {chain} → verify {verify_expr} {mark} [dim](actor={actor}{sup})[/]"
        )

    if steps or trace.refused or trace.cancelled or trace.llm_error:  # a sim start/stop alone claims nothing
        try:
            oracle_names = verify_oracle_names(agent, engine)
            report = VerdictReport.from_trace(trace, oracle_names, gait_backend=_gait_backend(agent))
            color = "green" if report.verified else "yellow"
            gait = f" gait={report.gait_backend}" if report.gait_backend else ""
            console.print(
                f"  [{TEAL}]verdict[/] {report.evidence} "
                f"[{color}]verified={report.verified}[/] "
                f"[dim]({report.n_grounded}/{report.n_steps} grounded){gait} goal={report.goal_check}[/]"
            )
            if report.reject_reason and not report.llm_unavailable:
                console.print(f"  [dim]not verified: {escape(report.reject_reason)}[/]")
            console.print(f"  [dim]{escape(_goal_line(report, goal_check(trace, oracle_names)[1]))}[/]")
            if report.llm_unavailable:
                console.print(f"  [yellow]{escape(report.error)}[/] [dim](infrastructure, not an agent failure)[/]")
        except Exception as exc:  # noqa: BLE001 — fail closed (display only)
            console.print(f"  [yellow]verdict unavailable:[/] {exc}")



# ---------------------------------------------------------------------------
# Input classification
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


_COMPACT_LOGO = f"""\
[bold {TEAL}]╲  ╱[/] [bold]ECTOR[/]
[bold {TEAL}] ╲╱[/]  [dim]OS Nano[/dim]"""


def _load_logo_lines() -> list[str]:
    """Load braille logo lines, or empty list if file missing."""
    try:
        return _LOGO_PATH.read_text(encoding="utf-8").rstrip().splitlines()
    except (FileNotFoundError, OSError):
        return []


def format_banner(model: str, agent: Any = None) -> str:
    """Return banner info text (testable, no side effects)."""
    lines = [
        f"Vector CLI v{VERSION}",
        f"Model: {model}",
    ]
    if agent is not None:
        arm = getattr(agent, "_arm", None)
        base = getattr(agent, "_base", None)
        if arm is not None:
            lines.append(f"Arm: {getattr(arm, 'name', type(arm).__name__)}")
        if base is not None:
            lines.append(f"Base: {getattr(base, 'name', type(base).__name__)}")
    lines.append("/help for commands, quit to exit")
    return "\n".join(lines)


def print_banner(model: str, provider: str, agent: Any = None) -> None:
    """Print startup banner with braille logo (auto-scales to terminal width)."""
    import shutil
    term_w = shutil.get_terminal_size().columns
    logo_lines = _load_logo_lines()
    max_logo_w = max((len(l) for l in logo_lines), default=0) if logo_lines else 0

    console.print()
    if logo_lines and term_w >= max_logo_w:
        for line in logo_lines:
            console.print(f"[bold {TEAL}]{line}[/]")
            time.sleep(0.08)
    elif logo_lines:
        # Truncate each line to fit terminal
        for line in logo_lines:
            console.print(f"[bold {TEAL}]{line[:term_w - 1]}[/]")
            time.sleep(0.08)
    else:
        console.print(f"[bold {TEAL}]Vector OS Nano[/]")

    console.print(f"[dim]{'':>{min(40, term_w - 10)}}v{VERSION}[/]")
    time.sleep(0.15)

    info_parts = [f"Model: {model}", f"Provider: {provider}"]
    if agent is not None:
        arm = getattr(agent, "_arm", None)
        base = getattr(agent, "_base", None)
        if arm is not None:
            info_parts.append(f"Arm: {getattr(arm, 'name', type(arm).__name__)}")
        if base is not None:
            info_parts.append(f"Base: {getattr(base, 'name', type(base).__name__)}")
    console.print(f"[dim]  {' | '.join(info_parts)}[/]")
    console.print(f"[dim]  Type / for commands, quit to exit[/]")
    console.print()


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


# ---------------------------------------------------------------------------
# Hardware init
# ---------------------------------------------------------------------------


def _requested_embodiment(args: argparse.Namespace) -> str | None:
    """--embodiment wins; --sim-go2 / --sim are aliases for go2 / so101."""
    return (
        getattr(args, "embodiment", None)
        or ("go2" if getattr(args, "sim_go2", False) else None)
        or ("so101" if getattr(args, "sim", False) else None)
    )


def _init_agent(args: argparse.Namespace) -> Any:
    """Start the requested SimSession (the one NL start_simulation uses), or None.

    Fails loud: a requested sim that does not come up exits 2 instead of
    silently degrading to the dev world.
    """
    embodiment = _requested_embodiment(args)
    if embodiment is None:
        return None
    from vector_os_nano.sim.session import SimSession, SimSpec, SimStartError

    console.print(f"[dim]  Starting {embodiment} simulation...[/dim]")
    try:
        # No vlm_api_key: --api-key is the chat LLM's credential (Anthropic /
        # DeepSeek) and must never be sent to the OpenRouter VLM.
        session = SimSession.start(
            SimSpec(embodiment=embodiment, gui=not getattr(args, "headless", False)),
        )
    except SimStartError as exc:
        console.print(f"[red]Simulation failed to start:[/] {exc}")
        sys.exit(2)
    console.print(f"[dim]  {embodiment} {session.summary()}[/dim]")
    return session


# ---------------------------------------------------------------------------
# Slash command handler
# ---------------------------------------------------------------------------


def _handle_slash_command(
    cmd: str,
    args_rest: list[str],
    registry: ToolRegistry,
    session: Session | None = None,
    app_state: dict[str, Any] | None = None,
) -> bool:
    """Handle /command. Returns True to continue REPL, False to exit."""

    if cmd == "help":
        console.print()
        tbl = Table(show_header=False, box=None, padding=(0, 2))
        tbl.add_column(style=TEAL, no_wrap=True)
        tbl.add_column(style="dim")
        for name, desc, _has_args in SLASH_COMMANDS:
            tbl.add_row(f"/{name}", desc)
        tbl.add_row("!<cmd>", "Run shell command directly (e.g. !ls -la)")
        tbl.add_row("quit", "Exit (also: exit, q, Ctrl-D)")
        console.print(tbl)
        console.print()
        console.print(f"[bold {TEAL}]Shortcuts:[/]")
        console.print("[dim]  /          show command menu (auto-complete)[/dim]")
        console.print("[dim]  Tab        accept completion[/dim]")
        console.print("[dim]  Ctrl+R     search history[/dim]")
        console.print("[dim]  Ctrl+C     cancel current turn[/dim]")
        console.print("[dim]  Ctrl+D     exit[/dim]")
        console.print()

    elif cmd in ("quit", "exit", "q"):
        return False

    elif cmd == "login":
        from vector_os_nano.vcli.config import load_config, save_config
        provider_choice = args_rest[0] if args_rest else None
        if provider_choice not in ("claude", "anthropic", "openrouter", None):
            console.print(f"[yellow]  Usage: /login claude | /login anthropic | /login openrouter[/]")
            return True

        if provider_choice is None:
            console.print(f"\n[bold {TEAL}]Authentication:[/]")
            console.print()
            console.print(f"  [{TEAL}]/login claude[/]      Log in with Claude subscription (opens browser)")
            console.print(f"  [{TEAL}]/login anthropic[/]   Enter Anthropic API key manually")
            console.print(f"  [{TEAL}]/login openrouter[/]  Enter OpenRouter key (multi-model)")
            console.print()
            console.print("[dim]  /login claude gives V its own rate limit pool, independent of Claude Code.[/dim]\n")
            return True

        config = load_config()

        if provider_choice == "claude":
            from vector_os_nano.vcli.oauth import login_oauth
            console.print(f"\n[bold {TEAL}]Claude subscription login[/]")
            console.print("[dim]  Opening browser for authentication...[/dim]\n")
            creds = login_oauth()
            if creds:
                console.print(f"[green]  Authenticated.[/] Token saved to ~/.vector/oauth_credentials.json")
                console.print(f"[dim]  Restart vector-cli to use your subscription.[/dim]\n")
            else:
                console.print("[red]  Authentication failed or timed out.[/]")
                console.print("[dim]  Make sure you have an active Claude subscription.[/dim]\n")

        elif provider_choice == "anthropic":
            console.print(f"\n[bold {TEAL}]Anthropic API key[/]")
            console.print("[dim]  Get your key at: https://console.anthropic.com/settings/keys[/dim]\n")
            key = Prompt.ask("  API key (sk-ant-...)")
            if key.strip():
                config["anthropic_api_key"] = key.strip()
                config["provider"] = "anthropic"
                save_config(config)
                console.print(f"[green]  Saved.[/] Restart vector-cli to apply.")
            else:
                console.print("[dim]  Cancelled.[/dim]")

        elif provider_choice == "openrouter":
            console.print(f"\n[bold {TEAL}]OpenRouter API key[/]")
            console.print("[dim]  Get your key at: https://openrouter.ai/keys[/dim]\n")
            key = Prompt.ask("  API key (sk-or-...)")
            if key.strip():
                config["openrouter_api_key"] = key.strip()
                config["provider"] = "openrouter"
                config["base_url"] = "https://openrouter.ai/api/v1"
                save_config(config)
                console.print(f"[green]  Saved.[/] Restart vector-cli to apply.")
            else:
                console.print("[dim]  Cancelled.[/dim]")

    elif cmd == "config":
        from vector_os_nano.vcli.config import load_config, load_claude_oauth, _CONFIG_PATH
        config = load_config()
        oauth = load_claude_oauth()
        console.print()
        tbl = Table(show_header=False, box=None, padding=(0, 2))
        tbl.add_column(style="dim", no_wrap=True)
        tbl.add_column()
        tbl.add_row("Config", f"[dim]{_CONFIG_PATH}[/dim]")
        # Claude Code OAuth
        if oauth:
            sub = oauth.get("subscriptionType", "?")
            tbl.add_row("Claude auth", f"[green]{sub} (auto-detected from Claude Code)[/]")
        else:
            tbl.add_row("Claude auth", "[dim]not found[/]")
        # API keys
        ak = config.get("anthropic_api_key", "")
        ok = config.get("openrouter_api_key", "")
        tbl.add_row("Anthropic key", f"[green]{ak[:8]}...{ak[-4:]}[/]" if len(ak) > 12 else "[dim]not set[/]")
        tbl.add_row("OpenRouter key", f"[green]{ok[:8]}...{ok[-4:]}[/]" if len(ok) > 12 else "[dim]not set[/]")
        # Active
        active_provider = (app_state or {}).get("provider", config.get("provider", "?"))
        active_model = (app_state or {}).get("model", config.get("model", "?"))
        tbl.add_row("Active", f"[{TEAL}]{active_provider} / {active_model}[/]")
        console.print(tbl)
        console.print()

    elif cmd == "tools":
        tool_names = registry.list_tools()
        if not tool_names:
            console.print("[dim]No tools registered.[/dim]")
        else:
            console.print()
            tbl = Table(show_header=True, header_style=f"bold {TEAL}", box=None, padding=(0, 2))
            tbl.add_column("Tool", no_wrap=True)
            tbl.add_column("Type", no_wrap=True)
            tbl.add_column("Description")
            for name in tool_names:
                t = registry.get(name)
                desc = getattr(t, "description", "") if t else ""
                ro = "read-only" if t and hasattr(t, "is_read_only") and t.is_read_only({}) else "write"
                tbl.add_row(f"[{TEAL}]{name}[/]", f"[dim]{ro}[/]", f"[dim]{desc}[/]")
            console.print(tbl)
            console.print()

    elif cmd == "agent":
        console.print()
        console.print(
            Panel(
                "V -- the AI core of Vector OS Nano.\n"
                "Built by Vector Robotics at CMU Robotics Institute.\n\n"
                "Capabilities:\n"
                "  Robot control    start sims, walk, explore, pick, place, navigate\n"
                "  Codebase work    read/write/edit files, run bash, search code\n"
                "  Perception       query world model, check hardware status\n"
                "  Web              fetch URLs for documentation and references\n\n"
                "V owns the hardware. Arms, grippers, quadrupeds, cameras -- V's body.\n"
                "V speaks your language. Safety is non-negotiable.",
                title=V_LABEL,
                title_align="left",
                border_style=TEAL,
                padding=(1, 2),
                width=min(console.width, 76),
            )
        )
        console.print()

    elif cmd == "sessions":
        sessions = list_sessions()
        if not sessions:
            console.print("[dim]No saved sessions.[/dim]")
        else:
            for s in sessions:
                console.print(f"  [{TEAL}]{s.session_id}[/]  {s.created_at}  ({s.message_count} msgs)")

    elif cmd == "usage":
        if session is not None:
            u = session.token_usage
            total = u.input_tokens + u.output_tokens
            console.print(f"  in={u.input_tokens:,}  out={u.output_tokens:,}  total={total:,}")
        else:
            console.print("[dim]No session.[/dim]")

    elif cmd == "compact":
        if session is not None:
            before, after = session.compact(keep_recent=8)
            console.print(f"[dim]  Compacted: {before} -> {after} entries (old context summarized)[/dim]")
        else:
            console.print("[dim]No session.[/dim]")

    elif cmd == "copy":
        if _last_response:
            try:
                subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=_last_response.encode(), check=True,
                )
                console.print(f"[dim]  Copied to clipboard ({len(_last_response)} chars)[/dim]")
            except FileNotFoundError:
                try:
                    subprocess.run(
                        ["xsel", "--clipboard", "--input"],
                        input=_last_response.encode(), check=True,
                    )
                    console.print(f"[dim]  Copied to clipboard ({len(_last_response)} chars)[/dim]")
                except FileNotFoundError:
                    console.print("[dim]  Install xclip or xsel for clipboard support[/dim]")
        else:
            console.print("[dim]  No response to copy.[/dim]")

    elif cmd == "export":
        if session is not None:
            export_dir = Path.home() / ".vector" / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            export_path = export_dir / f"{session.session_id}.md"
            lines: list[str] = [f"# Vector CLI Session\n\nSession: {session.session_id}\n"]
            for entry in session._entries:
                etype = entry.get("type")
                if etype == "user":
                    lines.append(f"\n**You:** {entry['content']}\n")
                elif etype == "assistant":
                    lines.append(f"\n**V:** {entry.get('text', '')}\n")
            export_path.write_text("\n".join(lines), encoding="utf-8")
            console.print(f"[dim]  Exported to {export_path}[/dim]")
        else:
            console.print("[dim]No session.[/dim]")

    elif cmd == "clear":
        if session is not None:
            session._entries.clear()
            console.print(f"[dim]  Conversation cleared.[/dim]")
        else:
            console.print("[dim]No session.[/dim]")

    elif cmd == "clear_memory":
        import os as _os
        _sg_path = _os.path.expanduser("~/.vector_os_nano/scene_graph.yaml")
        cleared = False

        # Clear in-memory scene graph if agent is running
        agent_obj = app_state.get("agent") if app_state else None
        if agent_obj is not None:
            sm = getattr(agent_obj, "_spatial_memory", None)
            if sm is not None and hasattr(sm, "clear"):
                sm.clear()  # in place: skills, oracles and the base keep this instance
                cleared = True

        # Always delete the persist file(s) (works even without agent); a running
        # sim's world-keyed file is deleted too.
        _live = getattr(getattr(agent_obj, "_spatial_memory", None), "_persist_path", None)
        for _p in {_sg_path, _live} - {None}:
            try:
                _os.remove(_p)
                cleared = True
            except FileNotFoundError:
                pass

        # Delete terrain map if present
        _terrain_path = _os.path.expanduser("~/.vector_os_nano/terrain_map.npz")
        try:
            _os.remove(_terrain_path)
            cleared = True
        except FileNotFoundError:
            pass

        if cleared:
            console.print(f"[dim]  Scene graph cleared. All rooms/objects forgotten.[/dim]")
        else:
            console.print(f"[dim]  No scene graph file found.[/dim]")

    elif cmd == "reset":
        import os as _os
        # Signal bridge to reset robot pose via file flag
        try:
            with open(run_path("vector_reset_pose"), "w") as _f:
                _f.write("1")
            console.print(f"[dim]  Reset signal sent. Robot will stand up at current position.[/dim]")
        except OSError as _exc:
            console.print(f"[yellow]  Failed to send reset: {_exc}[/]")

    elif cmd == "model":
        if not args_rest:
            current = app_state.get("model", "unknown") if app_state else "unknown"
            console.print(f"  [{TEAL}]{current}[/]")
            console.print(f"[dim]  /model <name> to switch. Tab for suggestions.[/dim]")
        else:
            new_model = args_rest[0]
            if app_state is None:
                console.print("[yellow]No app state.[/]")
            else:
                prov = app_state["provider"]
                api_key = app_state["api_key"]

                # Auto-detect provider from model name
                # "openai/gpt-4o", "google/gemini-*", "meta-llama/*" → openrouter
                # "claude-*" without prefix → anthropic (if current provider)
                if "/" in new_model and prov == "anthropic":
                    # Model has provider prefix → switch to OpenRouter
                    prov = "openrouter"
                    # Use OpenRouter API key (from config or env)
                    import os
                    or_key = os.environ.get("OPENROUTER_API_KEY", "")
                    if not or_key:
                        try:
                            from vector_os_nano.vcli.config import load_config as _lc
                            or_key = _lc().get("openrouter_api_key", "")
                        except Exception:
                            pass
                    if not or_key:
                        # Try user.yaml
                        try:
                            import yaml
                            cfg_path = os.path.join(os.path.dirname(os.path.dirname(
                                os.path.dirname(os.path.abspath(__file__))
                            )), "config", "user.yaml")
                            with open(cfg_path) as f:
                                cfg = yaml.safe_load(f)
                            or_key = cfg.get("llm", {}).get("api_key", "")
                        except Exception:
                            pass
                    if or_key:
                        api_key = or_key
                    else:
                        console.print("[yellow]  No OpenRouter API key found (set OPENROUTER_API_KEY)[/]")
                        return True
                elif prov == "anthropic" and "/" in new_model:
                    new_model = new_model.split("/", 1)[1]
                elif prov == "openrouter" and "/" not in new_model:
                    new_model = f"anthropic/{new_model}"

                new_backend = create_backend(
                    provider=prov,
                    api_key=api_key,
                    model=new_model,
                    base_url=app_state.get("base_url"),
                )
                app_state["engine"]._backend = new_backend
                app_state["model"] = new_model
                app_state["provider"] = prov
                console.print(f"  Switched to [{TEAL}]{new_model}[/] ({prov})")

    elif cmd == "status":
        agent = (app_state or {}).get("agent")
        arm = getattr(agent, "_arm", None) if agent else None
        base = getattr(agent, "_base", None) if agent else None
        perc = getattr(agent, "_perception", None) if agent else None
        current_model = (app_state or {}).get("model", "unknown")
        tool_count = len(registry.list_tools())
        msg_count = len(session._entries) if session else 0

        console.print()
        tbl = Table(show_header=False, box=None, padding=(0, 2))
        tbl.add_column(style="dim", no_wrap=True)
        tbl.add_column()
        tbl.add_row("Model", f"[{TEAL}]{current_model}[/]")
        tbl.add_row("Arm", f"[green]{getattr(arm, 'name', type(arm).__name__)}[/]" if arm else "[dim]none[/]")
        tbl.add_row("Base", f"[green]{getattr(base, 'name', type(base).__name__)}[/]" if base else "[dim]none[/]")
        tbl.add_row("Perception", f"[green]{getattr(perc, 'name', type(perc).__name__)}[/]" if perc else "[dim]none[/]")
        tbl.add_row("Tools", str(tool_count))
        tbl.add_row("Messages", str(msg_count))
        console.print(tbl)
        console.print()

    else:
        console.print(f"[yellow]  Unknown: /{cmd}[/]  (type / + Tab)")

    return True


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------


_ROOM_LABELS: dict[str, str] = {
    "living_room": "Living Room", "dining_room": "Dining Room",
    "kitchen": "Kitchen", "study": "Study",
    "master_bedroom": "Master Bedroom", "guest_bedroom": "Guest Bedroom",
    "bathroom": "Bathroom", "hallway": "Hallway",
}


def _setup_explore_events(console: Any) -> None:
    """Hook exploration background events into console output.

    Uses print() instead of console.print() because explore events fire
    from a background thread — Rich Console is not thread-safe. Lines go to the
    console's own stream: stderr under ``-p --json``, whose stdout carries only
    the verdict line.
    """
    try:
        from vector_os_nano.skills.go2.explore import set_event_callback
    except ImportError:
        return

    def _out(line: str) -> None:
        print(line, file=console.file, flush=True)

    def _on_explore_event(event_type: str, data: dict) -> None:
        if event_type == "started":
            total = data.get("total_rooms", 8)
            _out(f"  >> Exploration started ({total} rooms to discover)")

        elif event_type == "room_entered":
            room = data.get("room", "?")
            visited = data.get("visited", 0)
            total = data.get("total", 8)
            elapsed = data.get("elapsed_sec", 0)
            label = _ROOM_LABELS.get(room, room)
            bar = f"[{'#' * visited}{'.' * (total - visited)}]"
            _out(f"  >> {label} {bar} {visited}/{total} ({elapsed}s)")

        elif event_type == "status":
            elapsed = data.get("elapsed_sec", 0)
            rooms = data.get("rooms_found", 0)
            total = data.get("total", 8)
            room = data.get("current_room", "?")
            pos = data.get("position", [0, 0])
            _out(f"  .. {elapsed}s | {rooms}/{total} rooms | at {room} ({pos[0]}, {pos[1]})")

        elif event_type in ("completed", "complete"):
            rooms = data.get("rooms", [])
            reason = data.get("reason", "")
            elapsed = data.get("elapsed_sec", 0)
            if reason == "tare_finished":
                _out(f"  >> Exploration complete ({elapsed}s, TARE: all frontiers covered). {len(rooms)} rooms.")
            else:
                _out(f"  >> Exploration complete ({elapsed}s). {len(rooms)} rooms.")

        elif event_type == "stopped":
            reason = data.get("reason", "unknown")
            rooms = data.get("rooms", [])
            if reason == "cancelled":
                _out(f"  >> Exploration stopped. {len(rooms)} rooms so far.")
            elif reason == "robot_fell":
                _out("  >> Robot fell! Exploration aborted.")
            else:
                _out(f"  >> Exploration {reason} — {len(rooms)} rooms.")

    set_event_callback(_on_explore_event)


def _wants_window(args: argparse.Namespace) -> bool:
    """Return True when the user wants a visible MuJoCo viewer window.

    A window is wanted when --sim or --sim-go2 is active (NL-triggered sims are
    handled separately in sim_tool.py) and --headless is NOT set.
    """
    return bool(_requested_embodiment(args)) and not getattr(args, "headless", False)


def _maybe_reexec_under_mjpython(args: argparse.Namespace) -> None:
    """On macOS, re-exec the whole CLI under mjpython if a window is wanted.

    Gates (ALL must be true for re-exec to fire):
    1. sys.platform == 'darwin'                 — macOS only
    2. _wants_window(args)                      — --sim/--sim-go2 without --headless
    3. VECTOR_REEXEC != '1'                     — not already re-exec'd (loop guard)
    4. not running under pytest                 — never re-exec during tests
    5. mujoco.viewer._MJPYTHON is falsy         — not already under mjpython

    Falls back to a one-line warning (headless) when mjpython is missing.
    Does NOT call os.execv when any gate fails — safe for headless / CI / pytest.
    """
    if sys.platform != "darwin":
        return
    if not _wants_window(args):
        return
    if os.environ.get("VECTOR_REEXEC") == "1":
        return
    # Never re-exec under pytest (pytest sets this env or injects its own sys.argv)
    if "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST"):
        return

    # Check if already under mjpython
    try:
        import mujoco.viewer as _mj_viewer  # type: ignore[import]
        if getattr(_mj_viewer, "_MJPYTHON", None):
            return  # already under mjpython — nothing to do
    except Exception:
        pass  # mujoco not importable yet; proceed to re-exec attempt

    # Locate mjpython next to the running interpreter (the venv's bin/) — robust
    # regardless of where this file sits or how the CLI was launched. The old
    # parents[N]-from-__file__ computation was off by one and resolved to $HOME, so
    # mjpython was never found and the viewer silently fell back to headless.
    from vector_os_nano.vcli.tools.sim_tool import locate_mjpython
    mjpython: str | None = locate_mjpython()

    if not mjpython:
        print(
            "Warning: mjpython not found — running headless "
            "(install mujoco into .venv-nano to get a viewer window).",
            file=sys.stderr,
        )
        return

    # Re-exec the entire process under mjpython with the same argv.
    # VECTOR_REEXEC=1 prevents infinite loops.
    new_env = os.environ.copy()
    new_env["VECTOR_REEXEC"] = "1"
    os.execve(mjpython, [mjpython, "-m", "vector_os_nano.vcli.cli"] + sys.argv[1:], new_env)


# Loggers whose per-step INFO/WARNING lines are pure REPL noise on the non-verbose
# console (the rich step UI already surfaces every failure).  Quieted to ERROR on
# the non-verbose REPL; restored to NOTSET under --verbose.
_QUIET_LOGGERS: tuple[str, ...] = (
    "vector_os_nano.vcli.cognitive",   # step-failure WARNINGs duplicated by rich UI
    "vector_os_nano.skills",           # [PICK]/[SCAN] INFO lines
    "vector_os_nano.perception",       # perception pipeline INFO lines
    "vector_os_nano.hardware",         # [SIM DETECT] INFO/WARNING lines
)

# Back-compat alias used by the existing tests.
_COGNITIVE_LOGGER = _QUIET_LOGGERS[0]


def _setup_logging(verbose: bool) -> None:
    """Configure logging for the REPL entry path (CLI only).

    --verbose -> root DEBUG; all noisy sub-package loggers restored to NOTSET so
    they inherit the root level (full logging preserved).

    Non-verbose -> root WARNING; the sub-package loggers in _QUIET_LOGGERS are
    pinned to ERROR so their INFO/WARNING lines don't flood the console.  Every
    step failure is ALREADY surfaced in the rich step UI ("[FAIL] ..."); the
    duplicate log lines are pure noise.  The quieting is scoped to those specific
    package prefixes — NOT the root logger — so real ERRORs still surface and the
    engine/kernel loggers are unaffected.  Library code and the test suite never
    call this function; it is the CLI entry path only.
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        # Undo any prior non-verbose quieting so --verbose always restores full logging.
        for name in _QUIET_LOGGERS:
            logging.getLogger(name).setLevel(logging.NOTSET)
    else:
        logging.basicConfig(level=logging.WARNING)
        for name in _QUIET_LOGGERS:
            logging.getLogger(name).setLevel(logging.ERROR)


def _ensure_sigint_under_mjpython() -> None:
    """Restore Python's default Ctrl-C handling when running under mjpython.

    mjpython drives a Cocoa main loop on macOS and can leave SIGINT bound to its
    own handler, so a single Ctrl-C is swallowed instead of raising
    KeyboardInterrupt in the REPL (R2-4: the owner had to ^C^C then type quit).
    Re-installing the default int handler makes one Ctrl-C abort the running task
    and return to the prompt. No-op off mjpython; harmless if it cannot be set
    (e.g. not the main thread).
    """
    from vector_os_nano.hardware.sim.viewer_mode import running_under_mjpython
    if not running_under_mjpython():
        return
    import signal
    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
    except (ValueError, OSError):
        pass  # not the main thread / no SIGINT on this platform — leave as-is


def create_backend_with_fake_seam(
    *, provider: str, api_key: str, model: str, base_url: str | None
) -> Any:
    """Create the LLM backend, with a TEST-ONLY deterministic injection seam.

    Production is ``create_backend``. The ONLY override is gated on the env var
    ``VECTOR_FAKE_LLM_TOOLS=<json-path>``: a ``FakeToolScriptBackend``
    (``tests/harness/fake_backend.py``) replays a script of native tool_use turns in
    place of the network model. Every tool, verify, permission and verdict is still
    the product's own.
    """
    tools_path = os.environ.get("VECTOR_FAKE_LLM_TOOLS")
    if tools_path:
        # TEST-ONLY import (lazy, gated on the env var): production never reaches it.
        from tests.harness.fake_backend import FakeToolScriptBackend

        return FakeToolScriptBackend.from_json_file(tools_path)
    return create_backend(provider=provider, api_key=api_key, model=model, base_url=base_url)


@dataclass
class TurnContext:
    """Everything the REPL (and a non-interactive ``-p`` turn) needs to run."""

    args: Any
    api_key: str
    provider: str
    model: str
    base_url: str | None
    agent: Any
    registry: Any
    session: Any
    engine: "VectorEngine | None"
    app_state: dict[str, Any]


def _build_turn_context(args: Any) -> TurnContext:
    """Shared setup for BOTH the REPL and a non-interactive ``-p`` turn.

    Credentials -> agent (a sim when one was requested) -> tool registry -> session
    -> engine, bound to the agent and its world. The LLM backend is built through
    ``create_backend_with_fake_seam`` so a scripted test run drives the REAL
    cli.main. A bind failure raises; the caller reports it.
    """
    from vector_os_nano.vcli.config import resolve_credentials
    from vector_os_nano.vcli.worlds import resolve_world

    api_key, provider, model, base_url = resolve_credentials(
        cli_api_key=args.api_key,
        cli_base_url=args.base_url,
        cli_model=args.model,
    )
    if not api_key:
        # Each console.print is parsed independently — markup tags must balance
        # within a single call (rich does not span tags across print calls).
        console.print("[yellow]No API key configured.[/]")
        console.print("[dim]  /login claude     auto-detect Claude Code subscription[/dim]")
        console.print("[dim]  /login anthropic  enter Anthropic API key[/dim]")
        console.print("[dim]  /login openrouter enter OpenRouter key[/dim]\n")

    sim_session = _init_agent(args)
    agent = sim_session.agent if sim_session is not None else None

    # Tools (categorized registry for scalable tool management)
    registry: CategorizedToolRegistry = CategorizedToolRegistry()
    tools_list, cat_map = discover_categorized_tools()
    for t in tools_list:
        cat = "default"
        for c, names in cat_map.items():
            if t.name in names:
                cat = c
                break
        registry.register(t, category=cat)
    if agent is not None:
        from vector_os_nano.vcli.tools.skill_wrapper import wrap_skills
        for skill_tool in wrap_skills(agent):
            registry.register(skill_tool, category="robot")

    session: Session
    if args.resume is not None:
        if args.resume == "latest":
            loaded = get_latest_session()
            if loaded is None:
                console.print("[dim]No previous session, starting new.[/dim]")
                session = create_session(metadata={"model": model})
            else:
                session = loaded
                console.print(f"[dim]Resumed: {session.session_id}[/dim]")
        else:
            session = load_session(args.resume)
            console.print(f"[dim]Resumed: {session.session_id}[/dim]")
    else:
        session = create_session(metadata={"model": model})

    # Explore event streaming (the go2 explore skill's background progress lines).
    _setup_explore_events(console)

    # Backend + engine (deferred if no API key — /login can set it up).
    engine: VectorEngine | None = None
    if api_key:
        backend = create_backend_with_fake_seam(
            provider=provider, api_key=api_key, model=model, base_url=base_url
        )
        engine = VectorEngine(
            backend=backend, registry=registry,
            permissions=PermissionContext(no_permission=args.no_permission),
        )
        engine.bind(agent, resolve_world(agent))

    app_state: dict[str, Any] = {
        "agent": agent,
        "registry": registry,
        "engine": engine,
        "model": model,
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
        "scene_graph": getattr(agent, "_spatial_memory", None),
        "skill_registry": getattr(agent, "_skill_registry", None),
        "sim_session": sim_session,
    }
    return TurnContext(
        args=args, api_key=api_key, provider=provider, model=model, base_url=base_url,
        agent=agent, registry=registry, session=session, engine=engine, app_state=app_state,
    )


def run_one_turn(args: Any) -> int:
    """Run ONE turn for ``args.print_prompt`` non-interactively and return an exit code.

    The machine-checkable acceptance entrypoint. It reuses the REPL's setup
    (``_build_turn_context``), runs the turn SYNCHRONOUSLY through the native
    producer, prints the model's own words, then builds a frozen ``VerdictReport``
    from the EXISTING gate (never a second opinion). Under ``--json`` it emits
    exactly one stdout line ``VECTOR_VERDICT {<json>}`` (all Rich output is already
    routed to stderr by ``main``): the report plus ``acted``, whether a
    side-effecting tool ran.

    Exit codes: 0 = verified, or a chat turn (no action, no verify: nothing claimed);
    2 = ran, not verified; 1 = error / no trace.
    """
    from vector_os_nano.vcli.cognitive.trace_store import verify_oracle_names
    from vector_os_nano.vcli.verdict import EXIT_VERIFIED, VERDICT_SENTINEL, VerdictReport

    prompt = args.print_prompt or ""
    emit_json = bool(getattr(args, "json", False))

    def _emit(report: "VerdictReport", *, acted: bool = False, chat: bool = False) -> int:
        if emit_json:
            # The ONE machine line on real stdout (Rich/banner are on stderr).
            line = json.dumps({**report.to_dict(), "acted": acted}, ensure_ascii=False)
            print(f"{VERDICT_SENTINEL} {line}", flush=True)
        else:
            console.print(
                f"[{TEAL}]verdict[/] {report.evidence} verified={report.verified} "
                f"acted={acted} ({report.n_grounded}/{report.n_steps} grounded) "
                f"goal={report.goal_check}"
            )
        return EXIT_VERIFIED if chat else report.exit_code()

    try:
        ctx = _build_turn_context(args)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Setup error:[/] {exc}")
        return _emit(VerdictReport.no_trace(goal=prompt, error=str(exc)))

    engine = ctx.engine
    if engine is None:
        console.print("[yellow]No API key. Use /login to authenticate first.[/]")
        return _emit(VerdictReport.no_trace(goal=prompt, error="no engine (no API key)"))

    # P0 stop: the registered stop skill's words, no LLM call.
    if engine.is_stop(prompt):
        _p0_stop(engine, prompt, ctx.app_state, ctx.session)
        return _emit(VerdictReport.no_trace(goal=prompt, error="stopped (P0, no LLM call)"))

    n0 = len(ctx.session._entries)
    # ask_permission=None: headless denies every 'ask' (--no-permission allows).
    try:
        trace = engine.run_turn_native(
            prompt, agent=getattr(engine, "_vgg_agent", None), session=ctx.session,
            app_state=ctx.app_state,
        )
    except Exception as exc:  # noqa: BLE001
        return _emit(VerdictReport.no_trace(goal=prompt, error=f"native turn failed: {exc}"))
    answer = ctx.session.last_assistant_text(n0)
    if answer:
        console.print(escape(answer))
    if _chat_turn(trace):
        return _emit(VerdictReport.no_trace(goal=prompt), chat=True)

    # Build the verdict from the EXISTING gate, single-sourced from the SAME
    # namespace GoalVerifier uses. Fail CLOSED on any error reading the gate.
    try:
        agent = getattr(engine, "_vgg_agent", None)
        oracle_names = verify_oracle_names(agent, engine)
        report = VerdictReport.from_trace(trace, oracle_names, gait_backend=_gait_backend(agent))
    except Exception as exc:  # noqa: BLE001
        report = VerdictReport.no_trace(goal=prompt, error=f"verdict failed: {exc}")
    return _emit(report, acted=_trace_acted(trace))


def _provider_display(api_key: str, provider: str, base_url: str | None) -> str:
    """The banner's provider label (the auth source for a Claude subscription)."""
    from vector_os_nano.vcli.config import load_claude_oauth

    oauth = load_claude_oauth()
    if oauth and api_key == oauth.get("accessToken"):
        return f"Claude {oauth.get('subscriptionType', 'auth')}"
    if provider == "openrouter":
        return "OpenRouter"
    if base_url and "deepseek" in base_url:
        return "DeepSeek"
    if provider == "openai_compat":
        return "OpenAI-compatible"
    if base_url and "localhost" in base_url:
        return f"Local ({base_url})"
    return "Anthropic"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # --- macOS mjpython re-exec guard (must be before any credential/agent init) ---
    _maybe_reexec_under_mjpython(args)

    _setup_logging(args.verbose)
    _ensure_sigint_under_mjpython()

    # Non-interactive -p/--print: run ONE turn and exit with the verdict code.
    # Under --json route ALL Rich/banner output to stderr so stdout carries ONLY
    # the single VECTOR_VERDICT sentinel line.
    if getattr(args, "print_prompt", None) is not None:
        if getattr(args, "json", False):
            global console
            console = Console(stderr=True)
        _code = run_one_turn(args)
        # The verdict line is already flushed to stdout and the exit code is
        # decided. A live sim leaves a MuJoCo physics daemon thread + ROS2 stack
        # running whose interpreter-teardown can SIGABRT/segv AFTER the verdict,
        # corrupting the exit code. For a single-use turn there is nothing to
        # gracefully shut down, so exit HARD with the decided code (after flushing).
        # os._exit skips atexit: stop every session first (its nav pgids included),
        # the flag's and one the turn launched by NL alike.
        from vector_os_nano.sim.session import stop_all
        if stop_all() or _requested_embodiment(args):
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:  # noqa: BLE001
                pass
            os._exit(_code)
        sys.exit(_code)

    try:
        ctx = _build_turn_context(args)
    except Exception as exc:  # noqa: BLE001 — a body the engine cannot bind: fail loud
        console.print(f"[red]Setup error:[/] {exc}")
        sys.exit(2)
    engine, session, registry, app_state = ctx.engine, ctx.session, ctx.registry, ctx.app_state

    print_banner(ctx.model, _provider_display(ctx.api_key, ctx.provider, ctx.base_url), ctx.agent)
    console.print(f"[dim]Session: {session.session_id}[/dim]\n")

    # REPL setup
    history_dir = Path.home() / ".vector"
    history_dir.mkdir(parents=True, exist_ok=True)

    def _get_toolbar() -> HTML:
        agent_now = app_state.get("agent")
        current_model = app_state.get("model", "?")
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
            _flush_turn_state(session, app_state)  # a segfault loses at most one turn
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
            if engine is None:
                console.print(f"[yellow]No API key. Use /login to authenticate first.[/]")
                continue

            # P0 stop: the REGISTERED stop skill's own words, before the producer —
            # zero LLM calls, < 100 ms. Like any turn it runs on a worker thread while
            # this thread keeps reading input: a typed stop / Ctrl-C stops it mid-turn
            # (a Ctrl-C during the stop's speed report never exits the REPL), and
            # another command cancels it and runs next (cancel-then-run).
            if engine.is_stop(user_input):
                turn = lambda: _p0_stop(engine, user_input, app_state, session)  # noqa: E731
            else:
                turn = lambda: _repl_turn(engine, user_input, session, app_state, console)  # noqa: E731
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


if __name__ == "__main__":
    main()
