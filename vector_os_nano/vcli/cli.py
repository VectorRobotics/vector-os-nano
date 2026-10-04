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

import re

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PTStyle

from vector_os_nano.vcli.backends import create_backend
from vector_os_nano.vcli.cognitive import abort as _abort
from vector_os_nano.vcli.engine import VectorEngine, TurnResult
from vector_os_nano.vcli.native_loop import LIFECYCLE_TOOLS, should_attempt_native
from vector_os_nano.vcli.session import (
    Session,
    create_session,
    get_latest_session,
    list_sessions,
    load_session,
)
from vector_os_nano.vcli.permissions import PermissionContext
from vector_os_nano.vcli.prompt import build_system_prompt
from vector_os_nano.vcli.turn_status import TurnStatus
from vector_os_nano.vcli.tools import CategorizedToolRegistry, ToolRegistry, discover_all_tools, discover_categorized_tools
from vector_os_nano.core.run_dir import run_path

logger = logging.getLogger(__name__)
console = Console()

VERSION = "0.1.0"

TEAL = "#00b4b4"
DIM_TEAL = "#006666"

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
    ("scenario", "Show or enter a playground scenario  (/scenario <id>)", True),
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


# ---------------------------------------------------------------------------
# Response rendering — code block highlighting + path coloring
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
_PATH_RE = re.compile(r"(?<!\w)(/[\w./\-_]+\.\w+)")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")


def render_response(text: str, width: int = 80) -> Panel:
    """Render V's response with syntax-highlighted code blocks and paths."""
    parts = _CODE_BLOCK_RE.split(text)

    group = Text()
    i = 0
    while i < len(parts):
        if i + 2 < len(parts) and (i % 3) == 0:
            _append_highlighted_text(group, parts[i])
            i += 1
            lang = parts[i] or "text"
            code = parts[i + 1]
            i += 2
            group.append("\n")
            for line in code.splitlines():
                group.append(f"  {line}\n", style="#88c0d0")
        else:
            _append_highlighted_text(group, parts[i])
            i += 1

    return Panel(
        group,
        title=V_LABEL,
        title_align="left",
        border_style=TEAL,
        padding=(0, 1),
        width=width,
    )


def _strip_markdown(raw: str) -> str:
    """Strip markdown formatting that should not appear in terminal output."""
    # Bold: **text** or __text__
    raw = re.sub(r"\*\*(.+?)\*\*", r"\1", raw)
    raw = re.sub(r"__(.+?)__", r"\1", raw)
    # Italic: *text* (single)
    raw = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", raw)
    # Headers: # ## ###
    raw = re.sub(r"^#{1,3}\s+", "", raw, flags=re.MULTILINE)
    # Horizontal rules
    raw = re.sub(r"^---+\s*$", "", raw, flags=re.MULTILINE)
    return raw


def _append_highlighted_text(target: Text, raw: str) -> None:
    """Append text with file paths in teal and `inline code` highlighted."""
    raw = _strip_markdown(raw)
    last = 0
    # Merge path and inline code patterns, process in order
    for m in re.finditer(r"(?P<path>(?<!\w)/[\w./\-_]+\.\w+)|(?P<code>`[^`]+`)", raw):
        if m.start() > last:
            target.append(raw[last:m.start()])
        if m.group("path"):
            target.append(m.group("path"), style=f"bold {TEAL}")
        elif m.group("code"):
            # Strip backticks
            code_text = m.group("code")[1:-1]
            target.append(code_text, style="#88c0d0")
        last = m.end()
    if last < len(raw):
        target.append(raw[last:])


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
        "--scenario",
        default=None,
        metavar="ID",
        help=(
            "Start in a named playground scenario (e.g. 'tabletop'). When set it "
            "selects the playground world (and its verify predicates) instead of "
            "the agent-driven default. Unknown ids fail loud with the valid set."
        ),
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
    parser.add_argument(
        "--native-loop",
        action="store_true",
        default=None,
        help=(
            "Campaign #13 M1 (default OFF): run the -p turn through the frontier-model "
            "NATIVE TOOL-USE producer (engine.run_turn_native) instead of the legacy "
            "decompose plan. The verdict block is unchanged. Also enableable via "
            "VECTOR_NATIVE_LOOP=1."
        ),
    )
    parser.add_argument(
        "--native-first",
        action="store_true",
        default=None,
        help=(
            "STEP 5 native-attempt-then-fallback (default OFF): in the -p turn, ATTEMPT "
            "the native tool-use producer first; if it took NO action (could not route "
            "the goal) FALL BACK to the legacy decompose plan. A SEPARATE additive mode "
            "from --native-loop. Also enableable via VECTOR_NATIVE_FIRST=1."
        ),
    )
    return parser.parse_args(argv)


def _native_loop_enabled(args: Any) -> bool:
    """True iff the M1 native tool-use path is selected (flag OR env, default OFF).

    Single source for the strangler-fig flag: ``--native-loop`` on the CLI OR
    ``VECTOR_NATIVE_LOOP=1`` in the env. Default OFF — every existing path is
    byte-identical when neither is set.
    """
    if getattr(args, "native_loop", None):
        return True
    return os.environ.get("VECTOR_NATIVE_LOOP", "").strip() in ("1", "true", "True")


def _native_first_enabled(args: Any) -> bool:
    """True iff STEP 5 native-attempt-then-fallback is selected (flag OR env, default OFF).

    A SEPARATE, additive mode from ``_native_loop_enabled``: ``--native-first`` on the
    CLI OR ``VECTOR_NATIVE_FIRST=1`` in the env. Default OFF — every existing path is
    byte-identical when neither this nor ``--native-loop`` is set. This reader does NOT
    consult the native-loop flag/env, so the two modes stay independent.
    """
    if getattr(args, "native_first", None):
        return True
    return os.environ.get("VECTOR_NATIVE_FIRST", "").strip() in ("1", "true", "True")


def _print_native_enabled() -> bool:
    """PRINT-PATH CUTOVER (S5b): native is the DEFAULT producer on the ``-p`` path.

    Mirrors the owner-approved REPL cutover (``_repl_native_enabled``) on the
    NON-interactive acceptance entrypoint: ``run_one_turn`` ATTEMPTS the native
    tool-use producer first, then FALLS BACK to the legacy decompose+execute when
    native took NO action — so bare ``vector-cli -p`` + natural language exercises the
    redesign by default (CLAUDE.md North Star "Acceptance interface"), the same way
    the interactive REPL already does. Default ON; ``VECTOR_PRINT_NATIVE`` in
    {0, false, off, no} forces the pure-legacy ``-p`` path (byte-identical to the
    pre-cutover behavior) — a reversible escape hatch. The explicit ``--native-first``
    flag / ``VECTOR_NATIVE_FIRST`` stays an INDEPENDENT force-on reader (default OFF,
    unchanged) so its documented semantics are preserved; this is the additional
    default-ON knob, NOT a change to that flag's default.
    """
    return os.environ.get("VECTOR_PRINT_NATIVE", "").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


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


def _native_trace_acted(trace: Any) -> bool:
    """True iff the native producer OWNS the turn: it ran any side-effecting tool.

    ``trace.actions_dispatched`` counts every side-effecting dispatch — verified or
    not, interrupted or not — so a turn that moved the robot is never handed to the
    legacy planner to run again (no double execution). A recorded step with a
    producing ``strategy`` also counts: a verified chain, or a read the model
    answered from (the REPL prints that answer; the read never runs twice). Only a
    turn that called no tool (could not route the goal) may fall back. Pure +
    defensive: ``None`` -> False.
    """
    if int(getattr(trace, "actions_dispatched", 0) or 0) > 0:
        return True
    steps = getattr(trace, "steps", None) or ()
    return any((getattr(s, "strategy", "") or "").strip() for s in steps)


def _native_owns_turn(trace: Any) -> bool:
    """The native result is final: it acted, the permission gate refused what it
    tried (no second producer re-attempts it), the owner stopped the turn (a stop
    landing during native's LLM call must not hand the command to the legacy
    planner), or the LLM itself is unavailable (the legacy producer would only hit
    the same dead model)."""
    return trace is not None and (
        _native_trace_acted(trace)
        or bool(getattr(trace, "refused", ()))
        or bool(getattr(trace, "cancelled", False))
        or bool(getattr(trace, "llm_error", ""))
    )


def _repl_native_enabled() -> bool:
    """REPL CUTOVER (2026-06-19, owner-approved): native is the DEFAULT REPL turn path.

    The interactive ``vector-cli`` REPL attempts the frontier-model NATIVE TOOL-USE
    producer first (then falls back to the legacy planner) so the owner's ONLY
    acceptance interface — bare ``vector-cli`` + natural language — exercises the
    redesign (CLAUDE.md North Star -> "Acceptance interface"). Default ON.
    ``VECTOR_REPL_NATIVE`` in {0, false, off, no} forces the pure-legacy REPL
    (byte-identical to the pre-cutover turn path) — a reversible escape hatch.
    """
    return os.environ.get("VECTOR_REPL_NATIVE", "").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def _lifecycle_lines(entries: list[dict[str, Any]]) -> list[str]:
    """The first line of every start/stop_simulation result among *entries* (in order)."""
    ids ={b.get("id") for e in entries if e.get("type") == "assistant"
           for b in (e.get("tool_use") or ()) if b.get("name") in LIFECYCLE_TOOLS}
    return [str(r.get("content") or "").splitlines()[0][:200] for e in entries
            if e.get("type") == "tool_result" for r in e.get("results") or ()
            if r.get("tool_use_id") in ids and r.get("content")]


def _repl_attempt_native(
    engine: Any,
    user_input: str,
    session: Any,
    app_state: dict[str, Any],
    console: Any,
) -> bool:
    """Attempt the native tool-use producer for ONE REPL turn (the cutover path).

    Returns True iff the native producer OWNS the turn (it acted, was refused, was
    stopped, or the LLM is down): the honest verdict — single-sourced from the SAME
    gate the ``-p`` path uses (``VerdictReport.from_trace``; the loop NEVER computes
    ``verified``) — is rendered into the REPL conversation. Returns False iff native
    took NO action: its entries are removed from the session again and the caller
    falls THROUGH to the legacy routing (no half-action, no double side-effect).

    The turn runs on the REAL session, so a follow-up turn sees what this one did
    ("把绿色瓶子拿过来" then "再把它放到餐桌上": 'it' resolves). A sim started by
    the turn rebinds ``engine._vgg_agent``, so the verdict is graded on the body the
    turn ended on.
    """
    from vector_os_nano.vcli.cognitive.trace_store import verify_oracle_names
    from vector_os_nano.vcli.verdict import VerdictReport

    n0 = len(session._entries)

    # Suppress ROS2/subprocess stderr noise during the turn (the REPL does the same
    # around its legacy turn); restore unconditionally.
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
                    _status.update(f"[{TEAL}]native[/] {msg}")
                except Exception:  # noqa: BLE001
                    pass

            def _ask(name: str, params: dict[str, Any]) -> str:
                _status.stop()  # a live spinner would draw over the prompt
                try:
                    return ask_permission(name, params)
                finally:
                    _status.start()
            try:
                trace = engine.run_turn_native(
                    user_input, agent=getattr(engine, "_vgg_agent", None), session=session,
                    app_state=app_state, on_progress=_on_progress, ask_permission=_ask,
                )
            except Exception as exc:  # noqa: BLE001 — raised before any action: safe to fall back
                logger.debug("native attempt failed before acting: %s", exc)
                trace = None
    finally:
        if sys.stderr is not _saved_stderr:
            try:
                sys.stderr.close()
            except Exception:  # noqa: BLE001
                pass
        sys.stderr = _saved_stderr

    if not _native_owns_turn(trace):
        del session._entries[n0:]  # the fallback producer starts from a clean transcript
        return False  # native could not route -> caller falls back to legacy

    turn = session._entries[n0:]
    agent = getattr(engine, "_vgg_agent", None)  # the body the turn ended on
    sub_goals = list(getattr(trace.goal_tree, "sub_goals", ()) or ())
    steps = list(getattr(trace, "steps", ()) or ())
    # The model's own last words (the README summary, the test run's outcome): shown,
    # then the steps and the verdict, which are the evidence.
    answer = next((e.get("text") or "" for e in reversed(turn)
                   if e.get("type") == "assistant" and (e.get("text") or "").strip()), "").strip()
    console.print()
    for line in _lifecycle_lines(turn):
        console.print(f"  [{TEAL}]▸[/] [dim]{escape(line)}[/]")
    if answer:
        console.print(escape(answer))
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
        console.print(
            f"  [{TEAL}]▸[/] {chain} → verify {verify_expr} {mark} [dim](actor={actor})[/]"
        )

    # A turn that only started / stopped a sim made no claim: nothing to grade.
    if steps or trace.refused or trace.cancelled or trace.llm_error:
        try:
            oracle_names = verify_oracle_names(agent, engine)
            report = VerdictReport.from_trace(trace, oracle_names, gait_backend=_gait_backend(agent))
            color = "green" if report.verified else "yellow"
            gait = f" gait={report.gait_backend}" if report.gait_backend else ""
            console.print(
                f"  [{TEAL}]verdict[/] {report.evidence} "
                f"[{color}]verified={report.verified}[/] "
                f"[dim]({report.n_grounded}/{report.n_steps} grounded){gait}[/]"
            )
            if report.llm_unavailable:
                console.print(f"  [yellow]{escape(report.error)}[/] [dim](infrastructure, not an agent failure)[/]")
        except Exception as exc:  # noqa: BLE001 — fail closed (display only)
            console.print(f"  [yellow]verdict unavailable:[/] {exc}")

    if len(session._entries) > 50:  # the legacy path's bound, for the native transcript too
        session.compact(keep_recent=12)
    return True


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


def format_banner(model: str, agent: Any = None, scenario: str | None = None) -> str:
    """Return banner info text (testable, no side effects).

    When a playground ``scenario`` is active, its name is surfaced on its own
    line so a user can see at a glance which preset scene the session is in.
    """
    lines = [
        f"Vector CLI v{VERSION}",
        f"Model: {model}",
    ]
    if scenario:
        lines.append(f"Scenario: {scenario}")
    if agent is not None:
        arm = getattr(agent, "_arm", None)
        base = getattr(agent, "_base", None)
        if arm is not None:
            lines.append(f"Arm: {getattr(arm, 'name', type(arm).__name__)}")
        if base is not None:
            lines.append(f"Base: {getattr(base, 'name', type(base).__name__)}")
    lines.append("/help for commands, quit to exit")
    return "\n".join(lines)


def print_banner(
    model: str, provider: str, agent: Any = None, scenario: str | None = None
) -> None:
    """Print startup banner with braille logo (auto-scales to terminal width).

    When a playground ``scenario`` is active its name is shown in the info line.
    """
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
    if scenario:
        info_parts.append(f"Scenario: {scenario}")
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


def _resolve_active_world(args: argparse.Namespace, agent: Any) -> Any:
    """Select the active world, honouring an explicit playground ``--scenario``.

    Precedence:
      1. ``--scenario <id>`` selected -> the playground world for that scenario
         WINS. Loading the playground package (an explicit, user-requested track)
         registers its scenarios into the world registry via the lazy hook; we
         then resolve the named world. The kernel still never hard-imports the
         playground — this import only happens because the user asked for it.
      2. Otherwise -> ``resolve_world(agent)`` (exactly today's behaviour:
         a connected agent selects the robot world, else the default dev world).

    Fails loud on an unknown scenario id with the valid set — never a silent
    fallback to another world.
    """
    from vector_os_nano.vcli.worlds import resolve_world, resolve_world_named

    scenario = getattr(args, "scenario", None)
    if not scenario:
        return resolve_world(agent)

    # Explicit playground track: importing the package runs register_scenarios(),
    # wiring the scenario factories into the process-wide world registry.
    import vector_os_nano.playground  # noqa: F401  (side-effect: register scenarios)

    try:
        return resolve_world_named(scenario)
    except KeyError as exc:
        # Surface the fail-loud message (valid set included) to the user, then
        # re-raise so an unknown scenario id never silently degrades to a default.
        console.print(f"[red]Unknown scenario:[/red] {exc}")
        raise


def enter_scenario(scenario_id: str, app_state: dict[str, Any]) -> Any:
    """Switch the LIVE session into the playground ``scenario_id`` (mid-session).

    This is the conversational / ``/scenario`` counterpart to the ``--scenario``
    launch flag: it makes the playground reachable WITHOUT relaunching. Loading
    the playground package (an explicit, user-requested track) registers its
    scenarios into the world registry via the lazy hook; we then resolve the
    named world, swap it into ``app_state`` and re-init the engine's VGG layer so
    the playground verify predicates take effect on the next decompose.

    The kernel still never hard-imports the playground — this import only happens
    because the user asked for it (one-way dependency intact).

    Returns the resolved world. Fails loud (``KeyError`` with the valid set) on an
    unknown scenario id — never a silent fallback to another world.
    """
    from vector_os_nano.vcli.worlds import resolve_world_named

    # Explicit playground track: importing the package runs register_scenarios().
    import vector_os_nano.playground  # noqa: F401  (side-effect: register scenarios)

    world = resolve_world_named(scenario_id)  # KeyError -> fail loud, caller reports

    app_state["world"] = world
    app_state["scenario"] = getattr(world, "name", scenario_id)

    # Re-init VGG so the verifier namespace picks up the playground predicates.
    # Best-effort: a missing engine (no API key yet) or init failure must not
    # crash the REPL — the world swap still stands for the next engine init.
    engine = app_state.get("engine")
    if engine is not None:
        try:
            engine.init_vgg(
                agent=app_state.get("agent"),
                skill_registry=app_state.get("skill_registry"),
                on_vgg_step=app_state.get("vgg_step_callback"),
                on_vgg_step_view=app_state.get("vgg_step_view_callback"),
                world=world,
                tool_permission_resolver=app_state.get("tool_permission_resolver"),
                # Mirror the launch path: learning tier (persist_dir) is dev-world only.
                persist_dir=(Path.home() / ".vector") if not world.is_robot() else None,
            )
        except Exception as exc:  # noqa: BLE001 — display path, never crash REPL
            logger.warning("init_vgg after scenario switch failed: %s", exc)
    return world


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
            if sm is not None:
                persist_path = getattr(sm, "_persist_path", None) or _sg_path
                from vector_os_nano.core.scene_graph import SceneGraph
                new_sg = SceneGraph(persist_path=persist_path)
                agent_obj._spatial_memory = new_sg
                base = getattr(agent_obj, "_base", None)
                if base is not None and hasattr(base, "_scene_graph"):
                    base._scene_graph = new_sg
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

    elif cmd == "scenario":
        app = app_state or {}
        if not args_rest:
            # No id -> show the active scenario (or that none is active).
            active = app.get("scenario")
            if active:
                console.print(f"[{TEAL}]  Active scenario:[/] {active}")
            else:
                console.print("[dim]  No playground scenario active.[/]")
            console.print("[dim]  Switch with: /scenario <id>[/]")
        else:
            scenario_id = args_rest[0]
            try:
                world = enter_scenario(scenario_id, app)
            except KeyError as exc:
                # Fail loud with the valid set — never a silent fallback.
                console.print(f"[red]  Unknown scenario:[/red] {exc}")
            else:
                console.print(
                    f"[green]  Entered scenario:[/] {getattr(world, 'name', scenario_id)}"
                )

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

    By default this is byte-identical to ``create_backend`` — the production path
    is unchanged. The ONLY override is gated on the env var
    ``VECTOR_FAKE_LLM=<json-path>``: when set, the network LLM is replaced by a
    ``FakeBackend`` that returns a canned decompose-plan response read from that
    JSON file (see ``tests/harness/fake_backend.py``). This replaces ONLY the
    network call — the REAL decomposer / validator / skill / GoalVerifier /
    evidence-gate / verdict all still run, so a canned plan whose step verifies
    ``"True"`` STILL classifies RAN and the verdict is honest. The seam never
    bypasses any verify / permission layer.

    Absent the env var, ``create_backend`` is called unchanged.
    """
    # M1 native-loop seam: a SCRIPT of native tool_use turns (not a decompose plan).
    # Gated on its own env var so the legacy plan seam stays byte-identical.
    tools_path = os.environ.get("VECTOR_FAKE_LLM_TOOLS")
    if tools_path:
        from tests.harness.fake_backend import FakeToolScriptBackend

        return FakeToolScriptBackend.from_json_file(tools_path)
    fake_path = os.environ.get("VECTOR_FAKE_LLM")
    if fake_path:
        # TEST-ONLY import (lazy, gated on the env var) — production never reaches
        # this branch, so it never imports the test harness. The repo root is on
        # sys.path when cli is run as a module from the project dir.
        from tests.harness.fake_backend import FakeBackend

        return FakeBackend.from_json_file(fake_path)
    return create_backend(provider=provider, api_key=api_key, model=model, base_url=base_url)


@dataclass
class TurnContext:
    """Everything cli.main's REPL (and a non-interactive turn) needs to run.

    Built ONCE by ``_build_turn_context`` from the shared setup that used to live
    inline in ``main()`` (credentials -> agent -> world -> registry -> session ->
    system prompt -> engine -> init_vgg). The REPL unpacks it into the SAME local
    names it used before (byte-identical loop); ``run_one_turn`` reuses it verbatim.
    """

    args: Any
    api_key: str
    provider: str
    model: str
    base_url: str | None
    agent: Any
    world: Any
    registry: Any
    permissions: Any
    session: Any
    system_prompt: Any
    robot_ctx_provider: Any
    intent_router: Any
    hooks: Any
    engine: "VectorEngine | None"
    app_state: dict[str, Any]


def _build_turn_context(
    args: Any,
    *,
    on_vgg_step: "Callable[[Any], None] | None" = None,
    on_vgg_step_view: "Callable[[dict[str, Any]], None] | None" = None,
    tool_permission_resolver: "Callable[[str, dict[str, Any]], str] | None" = None,
) -> TurnContext:
    """Shared setup for BOTH the REPL and a non-interactive ``-p`` turn.

    Identical to the block that used to live inline in ``main()`` (cli.py setup
    1271-1486). The display callbacks are injected so the REPL can pass its live
    step renderers while a headless turn passes None (no-op). The LLM backend is
    built through ``create_backend_with_fake_seam`` so a ``VECTOR_FAKE_LLM`` test
    run drives the REAL cli.main with a deterministic plan.
    """
    # Resolve API key + provider from CLI flags > env vars > config file
    from vector_os_nano.vcli.config import resolve_credentials
    api_key, provider, model, base_url = resolve_credentials(
        cli_api_key=args.api_key,
        cli_base_url=args.base_url,
        cli_model=args.model,
    )

    no_key = not api_key
    if no_key:
        # Each console.print is parsed independently — markup tags must balance
        # within a single call (rich does not span tags across print calls).
        console.print("[yellow]No API key configured.[/]")
        console.print("[dim]  /login claude     auto-detect Claude Code subscription[/dim]")
        console.print("[dim]  /login anthropic  enter Anthropic API key[/dim]")
        console.print("[dim]  /login openrouter enter OpenRouter key[/dim]\n")

    # Agent (optional hardware) + active world. An explicit --scenario selects a
    # playground world (its verify predicates win); otherwise a connected agent
    # selects the robot world, else the default cross-platform "dev" world. The
    # resolved world flows into init_vgg(world=...) below, which sets engine._world
    # BEFORE the verifier namespace is built, so the merge picks up its predicates.
    sim_session = _init_agent(args)
    agent = sim_session.agent if sim_session is not None else None
    world = _resolve_active_world(args, agent)

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
    else:
        # Dev world: hide robot/diag/system tools. The 'sim' category (start/stop
        # _simulation) stays enabled so the user can spin up a sim conversationally
        # ("start the arm sim").
        for _c in ("robot", "diag", "system"):
            registry.disable_category(_c)

    # Permissions
    permissions = PermissionContext(no_permission=args.no_permission)

    # Session
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

    # System prompt (with live robot context)
    robot_ctx_provider = None
    try:
        from vector_os_nano.vcli.robot_context import RobotContextProvider
        base = getattr(agent, "_base", None) if agent else None
        sg = getattr(agent, "_spatial_memory", None) if agent else None
        arm = getattr(agent, "_arm", None) if agent else None
        robot_ctx_provider = RobotContextProvider(base=base, scene_graph=sg, arm=arm)
    except ImportError:
        pass
    system_prompt = build_system_prompt(
        agent=agent, cwd=Path.cwd(), robot_context=robot_ctx_provider, world=world
    )

    # Wrap in DynamicSystemPrompt so robot state refreshes each turn
    try:
        from vector_os_nano.vcli.dynamic_prompt import DynamicSystemPrompt
        system_prompt = DynamicSystemPrompt(system_prompt, robot_ctx_provider)
    except ImportError:
        pass

    # Intent router + hooks
    intent_router = None
    hooks = None
    try:
        from vector_os_nano.vcli.intent_router import IntentRouter
        intent_router = IntentRouter()
    except ImportError:
        pass
    try:
        from vector_os_nano.vcli.hooks import ToolHookRegistry
        hooks = ToolHookRegistry()
    except ImportError:
        pass

    # Explore event streaming — wired via _setup_explore_events on first explore
    _setup_explore_events(console)

    # Backend + engine (deferred if no API key — /login can set it up). The
    # backend is built through the fake-LLM seam (production-identical unless
    # VECTOR_FAKE_LLM is set) at the SINGLE create_backend site.
    engine: VectorEngine | None = None
    if api_key:
        backend = create_backend_with_fake_seam(
            provider=provider, api_key=api_key, model=model, base_url=base_url
        )
        engine = VectorEngine(
            backend=backend, registry=registry, system_prompt=system_prompt,
            permissions=permissions, intent_router=intent_router, hooks=hooks,
        )

    # Mutable app state
    _spatial_memory = getattr(agent, "_spatial_memory", None) if agent else None
    _skill_registry = getattr(agent, "_skill_registry", None) if agent else None
    # An explicit --scenario selected a playground world; its name doubles as the
    # active scenario id (banner + mid-session /scenario read/write this).
    _active_scenario = getattr(args, "scenario", None) or None
    app_state: dict[str, Any] = {
        "agent": agent,
        "registry": registry,
        "engine": engine,
        "model": model,
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
        "scene_graph": _spatial_memory,
        "skill_registry": _skill_registry,
        "robot_ctx_provider": robot_ctx_provider,
        "world": world,
        "scenario": _active_scenario,
        "sim_session": sim_session,
    }
    # Headless: an ``ask`` is denied unless the caller passes a resolver (or
    # --no-permission makes the PermissionContext allow) — never a stdin prompt.
    app_state["tool_permission_resolver"] = tool_permission_resolver

    # VGG cognitive layer (optional) — init through the SAME path as the REPL so
    # the verifier namespace / oracle set is identical whether interactive or not.
    if engine is not None:
        try:
            engine.init_vgg(
                agent=agent,
                skill_registry=_skill_registry,
                on_vgg_step=on_vgg_step,
                on_vgg_step_view=on_vgg_step_view,
                world=world,
                tool_permission_resolver=app_state["tool_permission_resolver"],
                persist_dir=(Path.home() / ".vector") if not world.is_robot() else None,
            )
        except Exception:  # noqa: BLE001
            pass

    return TurnContext(
        args=args,
        api_key=api_key,
        provider=provider,
        model=model,
        base_url=base_url,
        agent=agent,
        world=world,
        registry=registry,
        permissions=permissions,
        session=session,
        system_prompt=system_prompt,
        robot_ctx_provider=robot_ctx_provider,
        intent_router=intent_router,
        hooks=hooks,
        engine=engine,
        app_state=app_state,
    )


def run_one_turn(args: Any) -> int:
    """Run ONE turn for ``args.print_prompt`` non-interactively and return an exit code.

    This is the machine-checkable acceptance entrypoint (R2a). It reuses the SAME
    shared setup the REPL uses (``_build_turn_context``), runs the turn
    SYNCHRONOUSLY via ``engine.vgg_execute`` (never on a background thread — an
    in-flight trace would emit a verdict on incomplete evidence), then builds
    a frozen ``VerdictReport`` from the EXISTING
    ``evidence_passed(trace, verify_oracle_names(agent, engine))`` (never a second
    opinion). Under ``--json`` it emits exactly one stdout line
    ``VECTOR_VERDICT {<json>}`` (sentinel) with all Rich/banner output already
    routed to stderr by ``main``.

    Exit codes: 0 = verified, 2 = ran (not verified), 1 = error / no trace.
    """
    from vector_os_nano.vcli.cognitive.trace_store import verify_oracle_names
    from vector_os_nano.vcli.verdict import VerdictReport

    prompt = args.print_prompt
    emit_json = bool(getattr(args, "json", False))

    def _emit(report: "VerdictReport") -> int:
        if emit_json:
            # The ONE machine line on real stdout (Rich/banner are on stderr).
            print(report.to_sentinel_line(), flush=True)
        else:
            console.print(
                f"[{TEAL}]verdict[/] {report.evidence} "
                f"verified={report.verified} ({report.n_grounded}/{report.n_steps} grounded)"
            )
        return report.exit_code()

    try:
        ctx = _build_turn_context(args)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Setup error:[/] {exc}")
        return _emit(VerdictReport.no_trace(goal=prompt or "", error=str(exc)))

    engine = ctx.engine
    if engine is None:
        console.print("[yellow]No API key. Use /login to authenticate first.[/]")
        return _emit(VerdictReport.no_trace(goal=prompt or "", error="no engine (no API key)"))

    # P0 stop: the registered stop skill's words, no LLM call.
    if engine.is_stop(prompt or ""):
        _p0_stop(engine, prompt, ctx.app_state, ctx.session)
        return _emit(VerdictReport.no_trace(goal=prompt, error="stopped (P0, no LLM call)"))

    # S5b PRINT-PATH CUTOVER native-attempt-then-fallback (DEFAULT ON via
    # _print_native_enabled; also forced by --native-first / VECTOR_NATIVE_FIRST):
    # ATTEMPT the native producer first; if it DISPATCHED an action skill (routed the
    # goal) emit its verdict — otherwise it took NO action (zero skills dispatched ->
    # no half-action side-effects) and we FALL THROUGH to the EXISTING legacy
    # decompose+execute+verdict block below. Strictly ADDITIVE: a goal native cannot
    # route (e.g. the VECTOR_FAKE_LLM decompose-plan seam yields no tool_calls) falls
    # through to the byte-identical legacy verdict, so only goals native CAN route
    # change producer. VECTOR_PRINT_NATIVE in {0,false,off,no} forces the pure-legacy
    # -p path (reversible escape hatch). The fallback's vgg_decompose(prompt) uses the
    # RAW prompt, independent of any session text the native attempt may have appended.
    # The DEFAULT-ON print cutover must NOT pre-empt an EXPLICIT --native-loop (pure
    # native, no fallback) — defer to block 2 in that case; an explicit --native-first
    # still wins (preserving every prior flag behavior). So block 1 fires on: explicit
    # --native-first, OR the default cutover when --native-loop is NOT explicitly set.
    if _native_first_enabled(args) or (
        _print_native_enabled() and not _native_loop_enabled(args)
    ):
        # ask_permission=None: headless denies every 'ask' (--no-permission allows).
        try:
            trace = engine.run_turn_native(
                prompt, agent=getattr(engine, "_vgg_agent", None), session=ctx.session,
                app_state=ctx.app_state,
            )
        except Exception:  # noqa: BLE001 — raised before any action: safe to fall back
            trace = None
        if _native_owns_turn(trace):
            try:
                agent = getattr(engine, "_vgg_agent", None)
                oracle_names = verify_oracle_names(agent, engine)
                report = VerdictReport.from_trace(trace, oracle_names, gait_backend=_gait_backend(agent))
            except Exception as exc:  # noqa: BLE001
                report = VerdictReport.no_trace(goal=prompt or "", error=f"verdict failed: {exc}")
            return _emit(report)
        # Native dispatched NOTHING -> fall through to the legacy block (no return here).

    # M1 strangler-fig (flag-gated, default OFF): the frontier-model NATIVE
    # TOOL-USE producer assembles the trace instead of the legacy decompose plan.
    # The verdict block below is UNCHANGED — the native producer hands the SAME
    # ExecutionTrace shape to the SAME VerdictReport.from_trace gate.
    if _native_loop_enabled(args):
        try:
            trace = engine.run_turn_native(
                prompt, agent=getattr(engine, "_vgg_agent", None), session=ctx.session,
                app_state=ctx.app_state,
            )
        except Exception as exc:  # noqa: BLE001
            return _emit(VerdictReport.no_trace(goal=prompt or "", error=f"native loop failed: {exc}"))
        try:
            agent = getattr(engine, "_vgg_agent", None)
            oracle_names = verify_oracle_names(agent, engine)
            report = VerdictReport.from_trace(trace, oracle_names, gait_backend=_gait_backend(agent))
        except Exception as exc:  # noqa: BLE001
            report = VerdictReport.no_trace(goal=prompt or "", error=f"verdict failed: {exc}")
        return _emit(report)

    # Decompose + execute SYNCHRONOUSLY. A non-VGG (chat / tool_use) route yields
    # no GoalTree -> no deterministic trace -> NO_TRACE (fail closed, exit 1).
    try:
        goal_tree = engine.vgg_decompose(prompt)
    except Exception as exc:  # noqa: BLE001
        return _emit(VerdictReport.no_trace(goal=prompt or "", error=f"decompose failed: {exc}"))

    if goal_tree is None:
        return _emit(VerdictReport.no_trace(goal=prompt or "", error="not a VGG turn (no trace)"))

    try:
        trace = engine.vgg_execute(goal_tree)  # SYNC: the verdict reads a finished trace
    except Exception as exc:  # noqa: BLE001
        return _emit(VerdictReport.no_trace(goal=goal_tree.goal, error=f"execute failed: {exc}"))

    # Build the verdict from the EXISTING gate, single-sourced from the SAME
    # namespace GoalVerifier uses. Fail CLOSED on any error reading the gate.
    try:
        agent = getattr(engine, "_vgg_agent", None)
        oracle_names = verify_oracle_names(agent, engine)
        report = VerdictReport.from_trace(trace, oracle_names, gait_backend=_gait_backend(agent))
    except Exception as exc:  # noqa: BLE001
        report = VerdictReport.no_trace(goal=goal_tree.goal, error=f"verdict failed: {exc}")

    return _emit(report)


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
        # decided. A live sim (--sim/--sim-go2) leaves a MuJoCo physics daemon
        # thread + ROS2 stack running whose interpreter-teardown can SIGABRT/segv
        # AFTER the verdict — corrupting the process exit code so it no longer
        # matches the (already-correct) verdict. For a single-use non-interactive
        # turn there is nothing to gracefully shut down, so exit HARD with the
        # decided code (after flushing) to keep the harness invariant
        # ``verified == (exit_code == 0)`` honest. Interactive / non-sim turns are
        # unaffected (no daemon to crash). os._exit skips atexit/GC by design.
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

    # Resolve API key + provider from CLI flags > env vars > config file
    from vector_os_nano.vcli.config import resolve_credentials
    api_key, provider, model, base_url = resolve_credentials(
        cli_api_key=args.api_key,
        cli_base_url=args.base_url,
        cli_model=args.model,
    )

    no_key = not api_key
    if no_key:
        # Each console.print is parsed independently — markup tags must balance
        # within a single call (rich does not span tags across print calls).
        console.print("[yellow]No API key configured.[/]")
        console.print("[dim]  /login claude     auto-detect Claude Code subscription[/dim]")
        console.print("[dim]  /login anthropic  enter Anthropic API key[/dim]")
        console.print("[dim]  /login openrouter enter OpenRouter key[/dim]\n")

    # Agent (optional hardware) + active world. An explicit --scenario selects a
    # playground world (its verify predicates win); otherwise a connected agent
    # selects the robot world, else the default cross-platform "dev" world. The
    # resolved world flows into init_vgg(world=...) below, which sets engine._world
    # BEFORE the verifier namespace is built, so the merge picks up its predicates.
    sim_session = _init_agent(args)
    agent = sim_session.agent if sim_session is not None else None
    world = _resolve_active_world(args, agent)

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
    else:
        # Dev world: hide robot/diag/system tools. The 'sim' category (start/stop
        # _simulation) stays enabled so the user can spin up a sim conversationally
        # ("start the arm sim").
        for _c in ("robot", "diag", "system"):
            registry.disable_category(_c)

    # Permissions
    permissions = PermissionContext(no_permission=args.no_permission)

    # Session
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

    # System prompt (with live robot context)
    robot_ctx_provider = None
    try:
        from vector_os_nano.vcli.robot_context import RobotContextProvider
        base = getattr(agent, "_base", None) if agent else None
        sg = getattr(agent, "_spatial_memory", None) if agent else None
        arm = getattr(agent, "_arm", None) if agent else None
        robot_ctx_provider = RobotContextProvider(base=base, scene_graph=sg, arm=arm)
    except ImportError:
        pass
    system_prompt = build_system_prompt(
        agent=agent, cwd=Path.cwd(), robot_context=robot_ctx_provider, world=world
    )

    # Wrap in DynamicSystemPrompt so robot state refreshes each turn
    try:
        from vector_os_nano.vcli.dynamic_prompt import DynamicSystemPrompt
        system_prompt = DynamicSystemPrompt(system_prompt, robot_ctx_provider)
    except ImportError:
        pass

    # Intent router + hooks
    intent_router = None
    hooks = None
    try:
        from vector_os_nano.vcli.intent_router import IntentRouter
        intent_router = IntentRouter()
    except ImportError:
        pass
    try:
        from vector_os_nano.vcli.hooks import ToolHookRegistry
        hooks = ToolHookRegistry()
    except ImportError:
        pass

    # Explore event streaming — wired via _setup_explore_events on first explore
    _setup_explore_events(console)

    # Backend + engine (deferred if no API key — /login can set it up). Built
    # through the SINGLE fake-LLM seam (production-identical unless VECTOR_FAKE_LLM
    # is set) so the REPL and the non-interactive -p path share one create_backend
    # site (the seam is byte-identical to create_backend in production).
    engine: VectorEngine | None = None
    if api_key:
        backend = create_backend_with_fake_seam(
            provider=provider, api_key=api_key, model=model, base_url=base_url
        )
        engine = VectorEngine(
            backend=backend, registry=registry, system_prompt=system_prompt,
            permissions=permissions, intent_router=intent_router, hooks=hooks,
        )

    # Mutable app state
    _spatial_memory = getattr(agent, "_spatial_memory", None) if agent else None
    _skill_registry = getattr(agent, "_skill_registry", None) if agent else None
    # An explicit --scenario selected a playground world; its name doubles as the
    # active scenario id (banner + mid-session /scenario read/write this).
    _active_scenario = getattr(args, "scenario", None) or None
    app_state: dict[str, Any] = {
        "agent": agent,
        "registry": registry,
        "engine": engine,
        "model": model,
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
        "scene_graph": _spatial_memory,
        "skill_registry": _skill_registry,
        "robot_ctx_provider": robot_ctx_provider,
        "world": world,
        "scenario": _active_scenario,
        "sim_session": sim_session,
    }

    # VGG cognitive layer (optional)
    _vgg_step_idx = [0]
    _vgg_total = [0]

    def _vgg_step_display(step: Any) -> None:
        """Print VGG step progress to console — human-readable."""
        _vgg_step_idx[0] += 1
        idx = _vgg_step_idx[0]
        total = _vgg_total[0]
        name = getattr(step, "sub_goal_name", "?")
        strategy = getattr(step, "strategy", "?")
        success = getattr(step, "success", False)
        dur = getattr(step, "duration_sec", 0.0)
        fallback = getattr(step, "fallback_used", False)
        err = getattr(step, "error", "")

        prefix = f"[{idx}/{total}]" if total > 0 else ""

        if success:
            console.print(f"  [{TEAL}]>[/] {prefix} {name} [green]done[/] [dim]{dur:.1f}s[/]")
        elif err == "aborted":
            console.print(f"  [{TEAL}]>[/] {prefix} {name} [yellow]aborted[/]")
        else:
            # Translate common errors to friendly messages
            friendly = err
            if "No rooms learned" in err or "room_not_explored" in err:
                friendly = "no map yet — run explore first"
            elif "navigation_failed" in err or "timed out" in err:
                friendly = "navigation timed out"
            elif "Skill not found" in err:
                friendly = f"skill not available: {strategy}"
            fb_tag = " [dim](tried fallback)[/]" if fallback else ""
            console.print(f"  [{TEAL}]>[/] {prefix} {name} [red]failed[/] — {friendly}{fb_tag} [dim]{dur:.1f}s[/]")

    # Observation surface (INC8): make the verified loop VISIBLE. The per-step
    # EXPORT VIEW (JSON-safe dict) is rendered to a readable line carrying the
    # sub-goal, strategy, verify predicate and a stable RAN/FAILED marker. Runs
    # alongside _vgg_step_display (best-effort) — never affects execution. The
    # verify predicate lives on the active goal tree's sub_goals; the REPL loop
    # refreshes this map when it decomposes each task.
    _vgg_verify_by_name: dict[str, str] = {}

    def _vgg_step_view_display(view: dict[str, Any]) -> None:
        """Render one per-step EXPORT VIEW (sub-goal/strategy/verify/RAN-FAILED)."""
        try:
            from vector_os_nano.vcli.cognitive.observation import render_step_view
            verify = _vgg_verify_by_name.get(view.get("sub_goal_name"))
            line = render_step_view(view, verify)
            # The live view is ungraded (evidence is graded on the whole trace in the
            # run snapshot), so it renders [RAN]/[FAILED] and is never green.
            colour = "yellow" if view.get("success") else "red"
            # markup=False: the rendered line carries literal [RAN]/[FAILED]
            # markers that must NOT be parsed as rich style tags.
            console.print(f"    VGG {line}", style=colour, markup=False)
        except Exception:  # noqa: BLE001 — display must never break the loop
            pass

    # Stored so SimStartTool / a mid-session /scenario switch can reuse the real
    # step + view displays when they re-init VGG.
    app_state["vgg_step_callback"] = _vgg_step_display
    app_state["vgg_step_view_callback"] = _vgg_step_view_display
    # Stored so a mid-session /scenario switch (enter_scenario) re-inits VGG with the
    # SAME interactive permission prompt the launch path uses — otherwise switching
    # from a dev world would silently lose the prompt.
    app_state["tool_permission_resolver"] = lambda n, p: ask_permission(n, p)

    try:
        engine.init_vgg(
            agent=agent,
            skill_registry=_skill_registry,
            on_vgg_step=_vgg_step_display,
            on_vgg_step_view=_vgg_step_view_display,
            world=world,
            tool_permission_resolver=lambda n, p: ask_permission(n, p),
            # Learning tier (stats + templates) is dev-world only: keep the robot
            # decompose/execute path byte-identical and avoid cross-world stats
            # contamination (a dev 'tool_call' record must not promote onto a
            # robot sub-goal that has no ToolDispatcher).
            persist_dir=(Path.home() / ".vector") if not world.is_robot() else None,
        )
        if engine._vgg_enabled:
            console.print(f"[dim]  VGG cognitive layer: enabled[/dim]")
    except Exception:
        pass

    # Banner — detect auth source for display
    from vector_os_nano.vcli.config import load_claude_oauth
    _oauth = load_claude_oauth()
    if _oauth and api_key == _oauth.get("accessToken"):
        provider_display = f"Claude {_oauth.get('subscriptionType', 'auth')}"
    elif provider == "openrouter":
        provider_display = "OpenRouter"
    elif base_url and "deepseek" in base_url:
        provider_display = "DeepSeek"
    elif provider == "openai_compat":
        provider_display = "OpenAI-compatible"
    elif base_url and "localhost" in base_url:
        provider_display = f"Local ({base_url})"
    else:
        provider_display = "Anthropic"
    print_banner(model, provider_display, agent, scenario=_active_scenario)
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

    _tool_start_times: dict[str, float] = {}
    # Ctrl-C UX (R2-4): a ^C during a turn stops it and returns to the prompt (it
    # never exits); two consecutive ^C AT THE PROMPT exit cleanly.
    interrupt_pending = False
    app_state["crash_log"] = _enable_crash_log(session)

    # A line typed while a turn ran (cancel-then-run) is the next input.
    pending_input: list[str] = []

    try:
        def _engine_turn(user_input: str) -> None:
            # Every turn ATTEMPTS the native tool-use producer first (no keyword gate:
            # the model routes by the tool descriptions, sim launch included). If it
            # OWNS the turn its honest verdict is rendered; if it took NO action (chat,
            # a goal it could not route) the turn falls through to the legacy routing
            # below. VECTOR_REPL_NATIVE=0 forces the pure-legacy REPL (reversible).
            if _repl_native_enabled() and should_attempt_native(
                user_input, agent=getattr(engine, "_vgg_agent", None), engine=engine
            ):
                if _repl_attempt_native(engine, user_input, session, app_state, console):
                    return
                # native took NO action -> fall through to legacy routing (unchanged).

            try:
                # Single in-place progress region for the whole turn. A reasoning
                # model (DeepSeek) streams NO text for several seconds while it
                # thinks; the region shows one calm "thinking…" status that resolves
                # into the streamed answer — it is paused/cleared before any
                # tool-execution line is printed so box frames never stack and the
                # prompt is not duplicated.
                _turn_started = time.monotonic()

                def _status_content(text: str, elapsed: float, is_thinking: bool) -> Panel:
                    if is_thinking:
                        dots = "." * (int(elapsed) % 4)
                        label = f"thinking{dots}" + (
                            f"  [dim]{elapsed:.0f}s[/]" if elapsed >= 1 else ""
                        )
                        return Panel(
                            Text.from_markup(label, style="dim italic"),
                            title=V_LABEL,
                            title_align="left",
                            border_style=DIM_TEAL,
                            padding=(0, 1),
                            width=min(console.width, 80),
                        )
                    return Panel(
                        text,
                        title=V_LABEL,
                        title_align="left",
                        border_style=TEAL,
                        padding=(0, 1),
                        width=min(console.width, 80),
                    )

                status = TurnStatus(
                    content_factory=_status_content,
                    live_factory=lambda renderable: Live(
                        renderable,
                        console=console,
                        refresh_per_second=8,
                        transient=True,
                    ),
                )

                def on_text(chunk: str) -> None:
                    status.update_text(chunk)

                def on_reasoning(_chunk: str) -> None:
                    # Hidden reasoning trace: keep the "thinking…" status alive with
                    # an elapsed counter; never rendered as answer text.
                    status.thinking(time.monotonic() - _turn_started)

                def _format_tool_display(name: str, p: dict[str, Any]) -> str:
                    """Context-aware tool call display."""
                    if name == "file_read":
                        path = p.get("file_path", "")
                        short = path.split("/")[-1] if "/" in path else path
                        offset = p.get("offset", 0)
                        limit = p.get("limit", "")
                        loc = f":{offset}-{offset + limit}" if offset and limit else ""
                        return f"[dim]read[/] {short}{loc}"
                    if name == "file_edit":
                        path = p.get("file_path", "")
                        short = path.split("/")[-1] if "/" in path else path
                        old = str(p.get("old_string", ""))[:30]
                        new = str(p.get("new_string", ""))[:30]
                        return f"[dim]edit[/] {short}: [red]{old}[/] [dim]→[/] [green]{new}[/]"
                    if name == "file_write":
                        path = p.get("file_path", "")
                        short = path.split("/")[-1] if "/" in path else path
                        return f"[dim]write[/] {short}"
                    if name == "bash":
                        cmd = str(p.get("command", ""))[:60]
                        return f"[dim]$[/] {cmd}"
                    if name == "navigate":
                        room = p.get("room", "?")
                        return f"[dim]navigate →[/] {room}"
                    if name == "explore":
                        return "[dim]explore[/] starting background exploration"
                    if name in ("walk", "turn", "stand", "sit", "lie_down", "stop"):
                        parts = [name]
                        if p.get("direction"):
                            parts.append(str(p["direction"]))
                        if p.get("distance"):
                            parts.append(f"{p['distance']}m")
                        if p.get("angle"):
                            parts.append(f"{p['angle']}°")
                        return "[dim]" + " ".join(parts) + "[/]"
                    if name == "skill_reload":
                        return f"[dim]reload[/] {p.get('skill_name', '?')}"
                    if name == "scene_graph_query":
                        qt = p.get("query_type", "?")
                        room = p.get("room", "")
                        return f"[dim]scene_graph[/] {qt}" + (f" ({room})" if room else "")
                    if name in ("ros2_topics", "ros2_nodes"):
                        action = p.get("action", "?")
                        topic = p.get("topic", p.get("node", ""))
                        return f"[dim]{name}[/] {action}" + (f" {topic}" if topic else "")
                    if name == "ros2_log":
                        return f"[dim]log[/] {p.get('log_name', '?')}"
                    if name in ("glob", "grep"):
                        pattern = p.get("pattern", "")[:40]
                        return f"[dim]{name}[/] {pattern}"
                    if name == "nav_state":
                        return "[dim]nav_state[/]"
                    if name == "terrain_status":
                        return "[dim]terrain_status[/]"
                    if name == "start_simulation":
                        return f"[dim]sim start[/] {p.get('embodiment', '?')}"
                    # Fallback: generic display
                    if p:
                        items = [f"{k}={str(v)[:20]}" for k, v in list(p.items())[:2]]
                        return f"[dim]{name}[/]({', '.join(items)})"
                    return f"[dim]{name}[/]"

                def _format_result_summary(name: str, result: Any) -> str:
                    """Extract key info from tool result for display."""
                    if result.is_error:
                        content = result.content or ""
                        # Show first line of error + suggested action
                        lines = content.split("\n")
                        msg = lines[0][:60]
                        suggested = next((l for l in lines if l.startswith("Suggested:")), "")
                        if suggested:
                            return f"  [dim]{msg}[/]\n    [yellow]{suggested}[/]"
                        return f"  [dim]{msg}[/]"

                    if name in ("start_simulation", "stop_simulation"):
                        return f"  [dim]▸ {(result.content or '').splitlines()[0][:200]}[/]"
                    meta = result.metadata if hasattr(result, "metadata") else {}
                    if not meta:
                        return ""

                    # Navigate: show arrival info
                    if name == "navigate":
                        room = meta.get("room", "")
                        state = meta.get("robot_state_after", {})
                        pos = state.get("position", [])
                        if room and pos:
                            return f"  [dim]▸ 到达 {room} ({pos[0]}, {pos[1]})[/]"
                        if room:
                            return f"  [dim]▸ 到达 {room}[/]"
                    # Explore: show status
                    if name == "explore":
                        status = meta.get("status", "")
                        rooms = meta.get("rooms_visited", meta.get("all_rooms", []))
                        if rooms:
                            return f"  [dim]▸ {len(rooms)} rooms discovered[/]"
                    # Where am i
                    if name == "where_am_i":
                        room = meta.get("room", "")
                        pos = meta.get("position", [])
                        if room:
                            return f"  [dim]▸ {room} ({pos[0]}, {pos[1]})[/]" if pos else f"  [dim]▸ {room}[/]"
                    # Motor skills: show post-state
                    state = meta.get("robot_state_after", {})
                    if state:
                        room = state.get("room", "")
                        pos = state.get("position", [])
                        if room and pos:
                            return f"  [dim]▸ pos=({pos[0]}, {pos[1]}) room={room}[/]"

                    return ""

                _tool_displays: dict[str, str] = {}

                def on_tool_start(name: str, p: dict[str, Any]) -> None:
                    _tool_start_times[name] = time.monotonic()
                    _tool_displays[name] = _format_tool_display(name, p)

                def on_tool_end(name: str, result: Any) -> None:
                    elapsed = time.monotonic() - _tool_start_times.pop(name, time.monotonic())
                    display = _tool_displays.pop(name, name)
                    tag = "[green]ok[/]" if not result.is_error else "[red]fail[/]"
                    summary = _format_result_summary(name, result)
                    # Pause the live region BEFORE printing — printing into an active
                    # Live interleaves with its redraw and re-stacks the box frame.
                    with status.paused():
                        console.print(f"  [{TEAL}]▸[/] {display} {tag} [dim]{elapsed:.1f}s[/]")
                        if summary:
                            console.print(summary)

                    # Hook explore event callback after sim/explore tools run
                    if name in ("start_simulation", "explore"):
                        _setup_explore_events(console)

                # --- Routing (Stage 5, S5.4 cut-over) ---
                # The keyword gate is now an OPTIMIZATION HINT, not a correctness
                # fork: classify_intent (the single, observable decision the unified
                # controller uses) decides only the RENDERING shape here.
                #   * vgg route  -> the async plan renderer below (CLI stays
                #     responsive; this IS the unified plan path: vgg_decompose ->
                #     vgg_execute, deterministic verify + replan).
                #   * tool_use route -> run_turn_unified, which produces the answer
                #     via the ReAct loop AND closes the loop with a verified
                #     answer-only trace (chat is verified too; the moat is intact).
                # VECTOR_LEGACY_TURN=1 restores the exact pre-cutover fork
                # (vgg_decompose-then-run_turn) for one release as a fallback.
                _legacy_turn = os.environ.get("VECTOR_LEGACY_TURN") == "1"
                if _legacy_turn:
                    goal_tree = engine.vgg_decompose(user_input)
                else:
                    goal_tree = (
                        engine.vgg_decompose(user_input)
                        if engine.classify_intent(user_input).use_vgg
                        else None
                    )
                if _abort.is_abort_requested():
                    return  # the owner stopped this turn: nothing else runs for it
                if goal_tree is not None:
                    # Show plan BEFORE execution
                    console.print()
                    console.print(f"  [{TEAL}]>[/] [bold]VGG[/] {goal_tree.goal}")
                    for i, sg in enumerate(goal_tree.sub_goals, 1):
                        dep = f" (after {', '.join(sg.depends_on)})" if sg.depends_on else ""
                        strat = f" [dim]via {sg.strategy}[/]" if sg.strategy else ""
                        console.print(f"  [{TEAL}]>[/]   [{i}/{len(goal_tree.sub_goals)}] {sg.name} — {sg.description}{strat}{dep}")
                    console.print()

                    # Reset step counter for callback
                    _vgg_step_idx[0] = 0
                    _vgg_total[0] = len(goal_tree.sub_goals)
                    # Refresh the verify-predicate map the per-step view renderer
                    # reads (INC8) — keyed by sub_goal name for this plan.
                    _vgg_verify_by_name.clear()
                    _vgg_verify_by_name.update(
                        {sg.name: sg.verify for sg in goal_tree.sub_goals}
                    )

                    # Execute async — CLI remains responsive
                    def _on_vgg_complete(trace: Any) -> None:
                        n_steps = len(trace.steps)
                        n_ok = sum(1 for s in trace.steps if s.success)
                        dur = trace.total_duration_sec
                        # Evidence gate: a successful run only counts as *verified*
                        # when its steps actually consume a live verify-namespace
                        # oracle (world-agnostic now — the robot bypass is gone).
                        # oracle_names single-sourced from the SAME namespace
                        # GoalVerifier uses. Fail CLOSED (not verified) on any error
                        # so the moat never silently passes.
                        try:
                            from vector_os_nano.vcli.cognitive.trace_store import (
                                evidence_passed,
                                verify_oracle_names,
                            )
                            _oracle_names = verify_oracle_names(
                                getattr(engine, "_vgg_agent", None), engine
                            )
                            _evidence = evidence_passed(trace, _oracle_names)
                        except Exception:  # noqa: BLE001
                            _evidence = False
                        if trace.success and _evidence:
                            console.print(f"  [{TEAL}]>[/] [green]all {n_steps} steps done[/] [dim]{dur:.1f}s[/]")
                        elif trace.success:
                            console.print(f"  [{TEAL}]>[/] [yellow]completed without verifiable evidence[/] — {n_steps} steps ran [dim]{dur:.1f}s[/]")
                        elif n_ok > 0:
                            console.print(f"  [{TEAL}]>[/] [yellow]{n_ok}/{n_steps} steps done[/], rest failed [dim]{dur:.1f}s[/]")
                        else:
                            console.print(f"  [{TEAL}]>[/] [red]task failed[/] — 0/{n_steps} steps succeeded [dim]{dur:.1f}s[/]")

                        # Observation surface (INC8): show the full verified loop
                        # — goal tree + per-step PASS/FAIL + any replan notes +
                        # outcome — from the run snapshot (pure EXPORT VIEW; never
                        # re-derived from frozen types). Best-effort display.
                        try:
                            from vector_os_nano.vcli.cognitive.observation import (
                                render_run_snapshot,
                            )
                            snapshot = engine.vgg_run_snapshot(trace)
                            for snap_line in render_run_snapshot(snapshot).splitlines():
                                # markup=False: literal [GROUNDED]/[RAN]/[FAILED] markers.
                                console.print(f"  {snap_line}", style="dim", markup=False)
                        except Exception:  # noqa: BLE001 — display only
                            pass

                        # Record in session
                        step_summary = "\n".join(
                            f"  {s.sub_goal_name}: {'ok' if s.success else 'FAILED'}"
                            + (f" ({s.error})" if s.error else "")
                            for s in trace.steps
                        )
                        session.append_user(user_input)
                        session.append_assistant(
                            f"[VGG executed]\nGoal: {trace.goal_tree.goal}\n"
                            f"Result: {'success' if trace.success else 'partial failure'}\n"
                            f"Steps:\n{step_summary}"
                        )

                    # Synchronous on the turn worker: the REPL thread keeps reading
                    # input (stop / Ctrl-C), and only one turn drives the robot.
                    _on_vgg_complete(engine.vgg_execute(goal_tree))
                    return

                # --- Normal tool_use path ---
                # Pause the live region around any interactive permission prompt so
                # the prompt is not drawn into (and duplicated by) the live box.
                def _ask_permission_paused(n: str, p: dict[str, Any]) -> str:
                    with status.paused():
                        return ask_permission(n, p)

                # Suppress ROS2/subprocess log noise during engine turn
                _saved_stderr = sys.stderr
                try:
                    sys.stderr = open(os.devnull, "w")
                except OSError:
                    pass
                try:
                    status.start()  # one live region for the whole turn
                    if _legacy_turn:
                        # Legacy fallback (VECTOR_LEGACY_TURN=1): the open ReAct loop
                        # with no closed-loop verify — kept one release.
                        turn_result: TurnResult = engine.run_turn(
                            user_message=user_input,
                            session=session,
                            agent=app_state.get("agent"),
                            on_text=on_text,
                            on_tool_start=on_tool_start,
                            on_tool_end=on_tool_end,
                            ask_permission=_ask_permission_paused,
                            app_state=app_state,
                            on_reasoning=on_reasoning,
                        )
                    else:
                        # Unified controller: the ReAct loop produces the answer
                        # (streaming, permissions, tool hooks, P0 stop all preserved
                        # by run_turn underneath) and the harness wraps it in a
                        # verified answer-only trace — closing the loop for chat too.
                        turn_result = engine.run_turn_unified(
                            user_message=user_input,
                            session=session,
                            agent=app_state.get("agent"),
                            on_text=on_text,
                            on_tool_start=on_tool_start,
                            on_tool_end=on_tool_end,
                            ask_permission=_ask_permission_paused,
                            app_state=app_state,
                            on_reasoning=on_reasoning,
                        )
                finally:
                    # Stop/clear the single region BEFORE printing the final answer,
                    # so the box never stacks and the prompt is not duplicated.
                    status.stop()
                    sys.stderr = _saved_stderr

                # Final response: highlighted panel with braille V title
                global _last_response
                if turn_result.text:
                    _last_response = turn_result.text.strip()
                    console.print()  # spacing before response
                    console.print(render_response(
                        _last_response,
                        width=min(console.width, 80),
                    ))

                # Auto-compact (summarize old context instead of truncating)
                if len(session._entries) > 50:
                    before, after = session.compact(keep_recent=12)
                    console.print(f"[dim]  compacted {before} -> {after} entries (old context summarized)[/dim]")

                # Token usage (show in/out breakdown)
                if turn_result.usage:
                    u = turn_result.usage
                    if u.input_tokens or u.output_tokens:
                        console.print(f"[dim]  in={u.input_tokens:,} out={u.output_tokens:,}[/]")
                console.print()

            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str or "rate_limit" in err_str:
                    current_model = app_state.get("model", "?")
                    console.print(f"[yellow]  Rate limited on {current_model}.[/]")
                    console.print(f"[dim]  Try: /model claude-haiku-4-5 (lower rate limit)[/dim]")
                elif "401" in err_str or "authentication" in err_str.lower():
                    console.print(f"[yellow]  Authentication failed. Use /login to reconfigure.[/]")
                elif "404" in err_str or "not_found" in err_str:
                    console.print(f"[yellow]  Model not found: {app_state.get('model', '?')}[/]")
                    console.print(f"[dim]  Try: /model claude-haiku-4-5[/dim]")
                else:
                    console.print(f"[red]Error:[/] {exc}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()

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

            # P0 stop: the REGISTERED stop skill's own words, before any producer —
            # zero LLM calls, < 100 ms. Like any turn it runs on a worker thread while
            # this thread keeps reading input: a typed stop / Ctrl-C stops it mid-turn
            # (a Ctrl-C during the stop's speed report never exits the REPL), and
            # another command cancels it and runs next (cancel-then-run).
            if engine.is_stop(user_input):
                turn = lambda: _p0_stop(engine, user_input, app_state, session)  # noqa: E731
            else:
                turn = lambda: _engine_turn(user_input)  # noqa: E731
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
