"""Vector CLI — interactive REPL entry point.

Ties together VectorEngine, Session, PermissionContext, and all tools
into an interactive agent loop. Intended to be invoked as:

    python -m vector_os_nano.vcli.cli [options]
    # or via console_scripts: vector-cli [options]

Public helpers (also used by tests):
    parse_args()         — argparse wrapper
    is_slash_command()   — detect /command input
    is_exit_command()    — detect quit/exit/q
    format_banner()      — produce startup banner text
    ask_permission()     — interactive Rich permission prompt
    main()               — entry point
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

from vector_os_nano.vcli.engine import VectorEngine, TurnResult
from vector_os_nano.vcli.session import (
    Session,
    create_session,
    get_latest_session,
    list_sessions,
    load_session,
)
from vector_os_nano.vcli.permissions import PermissionContext
from vector_os_nano.vcli.prompt import build_system_prompt
from vector_os_nano.vcli.tools import ToolRegistry, discover_all_tools

logger = logging.getLogger(__name__)
console = Console()

VERSION = "0.1.0"

EXIT_COMMANDS: frozenset[str] = frozenset({"quit", "exit", "q"})


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments and return a Namespace.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).

    Returns:
        Parsed namespace with all flags.
    """
    parser = argparse.ArgumentParser(
        prog="vector-cli",
        description="Vector CLI — Agentic CLI for Vector OS Nano",
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Start with MuJoCo arm simulation",
    )
    parser.add_argument(
        "--sim-go2",
        action="store_true",
        help="Start with Go2 quadruped simulation",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Claude model to use (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="Resume session (optionally specify session ID; omit value for latest)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--no-permission",
        action="store_true",
        help="Disable permission prompts — allow all tools",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Path to a custom system prompt file",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Input classification helpers
# ---------------------------------------------------------------------------


def is_slash_command(text: str) -> bool:
    """Return True if *text* is a /command (starts with '/' after stripping)."""
    return text.strip().startswith("/")


def is_exit_command(text: str) -> bool:
    """Return True if *text* is a recognised exit keyword (case-insensitive)."""
    return text.strip().lower() in EXIT_COMMANDS


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def format_banner(model: str, agent: Any = None) -> str:
    """Build the startup banner text for Rich Panel.

    Args:
        model: Claude model string shown in the banner.
        agent: Optional Agent instance; arm/base names are extracted if present.

    Returns:
        Rich markup string suitable for Panel content.
    """
    lines: list[str] = [
        f"[bold]Vector CLI[/bold] v{VERSION}",
        f"Model: [cyan]{model}[/cyan]",
    ]

    if agent is not None:
        arm = getattr(agent, "_arm", None)
        base = getattr(agent, "_base", None)
        if arm is not None:
            arm_name = getattr(arm, "name", type(arm).__name__)
            lines.append(f"Arm: [green]{arm_name}[/green]")
        if base is not None:
            base_name = getattr(base, "name", type(base).__name__)
            lines.append(f"Base: [green]{base_name}[/green]")

    lines.append("[dim]Type /help for commands, quit to exit[/dim]")
    return "\n".join(lines)


def ask_permission(tool_name: str, params: dict[str, Any]) -> bool:
    """Interactive Rich permission prompt.

    Args:
        tool_name: Name of the tool requesting permission.
        params:    Parameters the tool will be called with.

    Returns:
        True if the user allows execution, False to deny.
        The caller is responsible for calling permissions.add_always_allow()
        when the response is "a" (always).
    """
    console.print(f"\n[yellow bold]Permission required:[/yellow bold]")
    console.print(f"  Tool: [cyan]{tool_name}[/cyan]")
    params_str = json.dumps(params, indent=2, ensure_ascii=False)
    if len(params_str) > 200:
        params_str = params_str[:200] + "..."
    console.print(f"  Params: [dim]{params_str}[/dim]")
    response = Prompt.ask("  Allow? [y/n/a=always]", choices=["y", "n", "a"], default="y")
    return response in ("y", "a")


# ---------------------------------------------------------------------------
# Hardware initialisation (lazy — only when --sim flags passed)
# ---------------------------------------------------------------------------


def _init_agent(args: argparse.Namespace) -> Any:
    """Initialise Nano Agent from sim flags.

    Imports are deferred so the CLI starts instantly when no sim is requested.

    Returns:
        Agent instance, or None when no sim flags are set.
    """
    if not (args.sim or args.sim_go2):
        return None

    try:
        from vector_os_nano.core.agent import Agent  # type: ignore[import]

        if args.sim:
            from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm  # type: ignore[import]

            arm = MuJoCoArm()
            arm.connect()
            return Agent(arm=arm)

        # args.sim_go2
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2  # type: ignore[import]

        base = MuJoCoGo2()
        base.connect()
        return Agent(base=base)

    except Exception as exc:
        console.print(f"[yellow]Warning: Could not initialise simulation: {exc}[/yellow]")
        return None


# ---------------------------------------------------------------------------
# Slash command dispatcher
# ---------------------------------------------------------------------------


def _handle_slash_command(
    cmd: str,
    args_rest: list[str],
    registry: ToolRegistry,
) -> bool:
    """Handle a /command string. Returns True to continue REPL, False to exit."""
    if cmd == "help":
        console.print("[bold]Slash commands:[/bold]")
        console.print("  /help      — show this message")
        console.print("  /quit      — exit Vector CLI")
        console.print("  /sessions  — list saved sessions")
        console.print()
        tool_names = registry.list_tools()
        if tool_names:
            console.print(f"[bold]Registered tools ({len(tool_names)}):[/bold]")
            console.print("  " + ", ".join(tool_names))
    elif cmd in ("quit", "exit", "q"):
        return False  # signal exit
    elif cmd == "sessions":
        sessions = list_sessions()
        if not sessions:
            console.print("[dim]No saved sessions.[/dim]")
        else:
            for s in sessions:
                console.print(
                    f"  [cyan]{s.session_id}[/cyan]  {s.created_at}"
                    f"  ({s.message_count} msgs)"
                )
    else:
        console.print(f"[yellow]Unknown command: /{cmd}[/yellow]  (try /help)")
    return True  # continue


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Argument list forwarded to parse_args(). None means sys.argv[1:].
    """
    args = parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # API key — prefer explicit flag, fall back to env var
    api_key: str = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        console.print(
            "[red]Error: No API key.[/red] "
            "Set [bold]ANTHROPIC_API_KEY[/bold] or use [bold]--api-key[/bold]."
        )
        sys.exit(1)

    # Agent (optional — robot hardware / simulation)
    agent = _init_agent(args)

    # Tool registry — register all built-in tools
    registry: ToolRegistry = ToolRegistry()
    for tool in discover_all_tools():
        registry.register(tool)

    # Wrap robot skills when agent is available
    if agent is not None:
        from vector_os_nano.vcli.tools.skill_wrapper import wrap_skills

        for skill_tool in wrap_skills(agent):
            registry.register(skill_tool)

    # Permission context
    permissions = PermissionContext(no_permission=args.no_permission)

    # Session — resume or create new
    session: Session
    if args.resume is not None:
        if args.resume == "latest":
            loaded = get_latest_session()
            if loaded is None:
                console.print("[yellow]No previous session found — starting new.[/yellow]")
                session = create_session(metadata={"model": args.model})
            else:
                session = loaded
                console.print(f"[dim]Resumed session: {session.session_id}[/dim]")
        else:
            session = load_session(args.resume)
            console.print(f"[dim]Resumed session: {session.session_id}[/dim]")
    else:
        session = create_session(metadata={"model": args.model})

    # System prompt (static + dynamic hardware sections)
    system_prompt = build_system_prompt(agent=agent, cwd=Path.cwd())

    # Engine
    engine = VectorEngine(
        api_key=api_key,
        model=args.model,
        registry=registry,
        system_prompt=system_prompt,
        permissions=permissions,
    )

    # Startup banner
    console.print(
        Panel(
            format_banner(args.model, agent),
            title="Vector CLI",
            border_style="cyan",
        )
    )
    console.print(f"[dim]Session: {session.session_id}[/dim]\n")

    # Prompt-toolkit session with persistent history
    history_dir = Path.home() / ".vector"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / "history"
    pt_session: PromptSession = PromptSession(history=FileHistory(str(history_path)))

    try:
        while True:
            # ---- Read input ------------------------------------------------
            try:
                raw = pt_session.prompt("vector> ")
            except EOFError:
                break  # Ctrl-D
            except KeyboardInterrupt:
                console.print("\n[dim]Use quit or Ctrl-D to exit.[/dim]")
                continue

            user_input = raw.strip()
            if not user_input:
                continue

            # ---- Exit check ------------------------------------------------
            if is_exit_command(user_input):
                break

            # ---- Slash commands --------------------------------------------
            if is_slash_command(user_input):
                parts = user_input.split()
                cmd = parts[0][1:]  # strip leading "/"
                rest = parts[1:]
                keep_going = _handle_slash_command(cmd, rest, registry)
                if not keep_going:
                    break
                continue

            # ---- Engine turn -----------------------------------------------
            def _ask(name: str, p: dict[str, Any]) -> bool:
                allowed = ask_permission(name, p)
                # If user chose "always" (ask_permission returns True for "a")
                # we rely on the outer scope's permissions object to add to session_allow.
                # Because we can't easily detect "a" vs "y" from the return value alone,
                # we accept the slight limitation for Phase 1.
                return allowed

            try:
                text_parts: list[str] = []

                def on_text(chunk: str) -> None:
                    text_parts.append(chunk)

                def on_tool_start(name: str, p: dict[str, Any]) -> None:
                    console.print(f"  [dim]  {name}[/dim]", end="")

                def on_tool_end(name: str, result: Any) -> None:
                    status = (
                        "[green]done[/green]"
                        if not result.is_error
                        else "[red]error[/red]"
                    )
                    console.print(f" {status}")

                turn_result: TurnResult = engine.run_turn(
                    user_message=user_input,
                    session=session,
                    agent=agent,
                    on_text=on_text,
                    on_tool_start=on_tool_start,
                    on_tool_end=on_tool_end,
                    ask_permission=_ask,
                )

                # Render response as Markdown
                if turn_result.text:
                    console.print()
                    console.print(Markdown(turn_result.text))
                    console.print()

                # Token usage (verbose only)
                if args.verbose and turn_result.usage:
                    u = turn_result.usage
                    console.print(
                        f"[dim]Tokens: in={u.input_tokens} "
                        f"out={u.output_tokens} "
                        f"cache_read={u.cache_read_tokens}[/dim]"
                    )

            except KeyboardInterrupt:
                console.print("\n[yellow]Turn interrupted.[/yellow]")
            except Exception as exc:
                console.print(f"[red]Error:[/red] {exc}")
                if args.verbose:
                    import traceback

                    traceback.print_exc()

    finally:
        session.save()
        console.print(f"\n[dim]Session saved: {session.session_id}[/dim]")


if __name__ == "__main__":
    main()
