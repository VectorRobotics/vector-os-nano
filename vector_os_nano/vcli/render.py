# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""What the CLI prints: the banner, a turn's steps and its verdict block.

One console for every face (``-p --json`` points it at stderr, so stdout carries only
the verdict line).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape

from vector_os_nano.vcli.native_loop import LIFECYCLE_TOOLS

VERSION = "0.1.0"


console = Console()


TEAL = "#00b4b4"


_LOGO_PATH = Path(__file__).resolve().parent.parent / "cli" / "logo_braille.txt"


# Braille dot-art V — embedded in panel titles
V_LABEL = f"[bold {TEAL}] ⠣⡠⠃ [/]"


def _gait_backend(agent: Any) -> str:
    """The sim body's active locomotion backend ('mpc' / 'sinusoidal'), else ''."""
    return str(getattr(getattr(agent, "_base", None), "gait_backend", "") or "")


def _trace_acted(trace: Any) -> bool:
    """True iff the turn ran a side-effecting tool (``actions_dispatched``)."""
    return int(getattr(trace, "actions_dispatched", 0) or 0) > 0


def _chat_turn(trace: Any) -> bool:
    """A turn that made no claim: no task contract, no action, no verify, no refusal,
    no stop, no LLM error.

    Its only steps (if any) are reads the model answered from, never verified. A goal
    that compiled a task contract names an end state, so its turn is always graded.
    """
    return not (
        trace.contract
        or _trace_acted(trace)
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


def render_turn(console: Any, engine: Any, trace: Any, answer: str, entries: list[dict[str, Any]]) -> None:
    """Print a REPL turn: its lifecycle lines, the model's words, then — for a turn that
    claimed something — its steps and the verdict block (``VerdictReport.from_trace``,
    the same gate ``-p`` reads)."""
    from vector_os_nano.vcli.cognitive.trace_store import goal_check, verify_oracle_names
    from vector_os_nano.vcli.verdict import VerdictReport

    # The model's own last words (the README summary, the test run's outcome): shown,
    # then the steps and the verdict, which are the evidence.
    agent = getattr(engine, "_agent", None)  # the body the turn ended on
    sub_goals = list(getattr(trace.goal_tree, "sub_goals", ()) or ())
    steps = list(getattr(trace, "steps", ()) or ())
    console.print()
    for line in _lifecycle_lines(entries):
        console.print(f"  [{TEAL}]▸[/] [dim]{escape(line)}[/]")
    if answer:
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

    # A sim start/stop alone claims nothing; a task contract always does.
    if steps or trace.contract or trace.refused or trace.cancelled or trace.llm_error:
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
