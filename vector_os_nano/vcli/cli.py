"""Vector CLI — interactive REPL entry point.

Ties together VectorEngine, Session, PermissionContext, and all tools
into an interactive agent loop with V's personality.

    python -m vector_os_nano.vcli.cli [options]
    # or via console_scripts: vector-cli [options]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import re

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
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

TEAL = "#00b4b4"
DIM_TEAL = "#006666"

EXIT_COMMANDS: frozenset[str] = frozenset({"quit", "exit", "q"})

_LOGO_PATH = Path(__file__).resolve().parent.parent / "cli" / "logo_braille.txt"

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


def _append_highlighted_text(target: Text, raw: str) -> None:
    """Append text with file paths in teal and `inline code` highlighted."""
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
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Model to use (default: claude-sonnet-4-6)")
    parser.add_argument("--resume", nargs="?", const="latest", default=None, help="Resume session")
    parser.add_argument("--api-key", default=None, help="API key (or set ANTHROPIC_API_KEY / OPENROUTER_API_KEY)")
    parser.add_argument("--base-url", default=None, help="API base URL")
    parser.add_argument("--no-permission", action="store_true", help="Allow all tools without prompts")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    parser.add_argument("--system-prompt", default=None, help="Path to custom system prompt file")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Input classification
# ---------------------------------------------------------------------------


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


def ask_permission(tool_name: str, params: dict[str, Any]) -> str:
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


def _init_agent(args: argparse.Namespace) -> Any:
    if not (args.sim or args.sim_go2):
        return None
    try:
        from vector_os_nano.core.agent import Agent  # type: ignore[import]
        if args.sim:
            from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm  # type: ignore[import]
            arm = MuJoCoArm()
            arm.connect()
            return Agent(arm=arm)
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2  # type: ignore[import]
        base = MuJoCoGo2()
        base.connect()
        return Agent(base=base)
    except Exception as exc:
        console.print(f"[yellow]Warning: Could not init simulation: {exc}[/yellow]")
        return None


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
            before = len(session._entries)
            if before > 8:
                session._entries = session._entries[-8:]
            after = len(session._entries)
            console.print(f"[dim]  Compacted: {before} -> {after} entries[/dim]")
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
                # Strip provider prefix for Anthropic direct, add for OpenRouter
                if prov == "anthropic" and "/" in new_model:
                    new_model = new_model.split("/", 1)[1]
                elif prov == "openrouter" and "/" not in new_model:
                    new_model = f"anthropic/{new_model}"
                new_backend = create_backend(
                    provider=prov,
                    api_key=app_state["api_key"],
                    model=new_model,
                    base_url=app_state.get("base_url"),
                )
                app_state["engine"]._backend = new_backend
                app_state["model"] = new_model
                console.print(f"  Switched to [{TEAL}]{new_model}[/]")

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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Resolve API key + provider from CLI flags > env vars > config file
    from vector_os_nano.vcli.config import resolve_credentials
    api_key, provider, model, base_url = resolve_credentials(
        cli_api_key=args.api_key,
        cli_base_url=args.base_url,
        cli_model=args.model if args.model != "claude-sonnet-4-6" else None,
    )

    no_key = not api_key
    if no_key:
        console.print(f"[yellow]No API key configured.[/]")
        console.print(f"[dim]  /login claude     auto-detect Claude Code subscription")
        console.print(f"  /login anthropic  enter Anthropic API key")
        console.print(f"  /login openrouter enter OpenRouter key[/dim]\n")

    # Agent (optional hardware)
    agent = _init_agent(args)

    # Tools
    registry: ToolRegistry = ToolRegistry()
    for tool in discover_all_tools():
        registry.register(tool)
    if agent is not None:
        from vector_os_nano.vcli.tools.skill_wrapper import wrap_skills
        for skill_tool in wrap_skills(agent):
            registry.register(skill_tool)

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

    # System prompt
    system_prompt = build_system_prompt(agent=agent, cwd=Path.cwd())

    # Backend + engine (deferred if no API key — /login can set it up)
    engine: VectorEngine | None = None
    if api_key:
        backend = create_backend(provider=provider, api_key=api_key, model=model, base_url=base_url)
        engine = VectorEngine(backend=backend, registry=registry, system_prompt=system_prompt, permissions=permissions)

    # Mutable app state
    app_state: dict[str, Any] = {
        "agent": agent,
        "registry": registry,
        "engine": engine,
        "model": model,
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
    }

    # Banner — detect auth source for display
    from vector_os_nano.vcli.config import load_claude_oauth
    _oauth = load_claude_oauth()
    if _oauth and api_key == _oauth.get("accessToken"):
        provider_display = f"Claude {_oauth.get('subscriptionType', 'auth')}"
    elif provider == "openrouter":
        provider_display = "OpenRouter"
    elif base_url and "localhost" in base_url:
        provider_display = f"Local ({base_url})"
    else:
        provider_display = "Anthropic"
    print_banner(model, provider_display, agent)
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

    try:
        while True:
            # ---- Read input ----
            try:
                raw = pt_session.prompt(
                    HTML(f'<style fg="{TEAL}" bold="true">vector&gt;</style> '),
                    bottom_toolbar=_get_toolbar,
                )
            except EOFError:
                break
            except KeyboardInterrupt:
                console.print("\n[dim]Use quit or Ctrl-D to exit.[/dim]")
                continue

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

            try:
                streamed_text: list[str] = []
                live_ref: Live | None = None

                def on_text(chunk: str) -> None:
                    streamed_text.append(chunk)
                    if live_ref is not None:
                        live_ref.update(
                            Panel(
                                "".join(streamed_text),
                                title=V_LABEL,
                                title_align="left",
                                border_style=TEAL,
                                padding=(0, 1),
                                width=min(console.width, 80),
                            )
                        )

                def _format_params_brief(p: dict[str, Any]) -> str:
                    if not p:
                        return ""
                    items: list[str] = []
                    for k, v in list(p.items())[:3]:
                        v_str = str(v)
                        if len(v_str) > 40:
                            v_str = v_str[:37] + "..."
                        if isinstance(v, str):
                            v_str = f'"{v_str}"'
                        items.append(f"{k}={v_str}")
                    suffix = ", ..." if len(p) > 3 else ""
                    return ", ".join(items) + suffix

                def on_tool_start(name: str, p: dict[str, Any]) -> None:
                    _tool_start_times[name] = time.monotonic()
                    ps = _format_params_brief(p)
                    if ps:
                        console.print(f"  [{TEAL}]{name}[/]([dim]{ps}[/]) ...", end="")
                    else:
                        console.print(f"  [{TEAL}]{name}[/]() ...", end="")

                def on_tool_end(name: str, result: Any) -> None:
                    elapsed = time.monotonic() - _tool_start_times.pop(name, time.monotonic())
                    tag = "[green]ok[/]" if not result.is_error else "[red]fail[/]"
                    console.print(f" {tag} [dim]{elapsed:.1f}s[/]")

                thinking_panel = Panel(
                    Text("thinking...", style="dim italic"),
                    title=V_LABEL,
                    title_align="left",
                    border_style=DIM_TEAL, padding=(0, 1),
                    width=min(console.width, 80),
                )
                # Suppress ROS2/subprocess log noise during engine turn
                _saved_stderr = sys.stderr
                try:
                    sys.stderr = open(os.devnull, "w")
                except OSError:
                    pass
                try:
                    with Live(thinking_panel, console=console, refresh_per_second=8, transient=True) as live:
                        live_ref = live
                        turn_result: TurnResult = engine.run_turn(
                            user_message=user_input,
                            session=session,
                            agent=app_state.get("agent"),
                            on_text=on_text,
                            on_tool_start=on_tool_start,
                            on_tool_end=on_tool_end,
                            ask_permission=lambda n, p: ask_permission(n, p),
                            app_state=app_state,
                        )
                finally:
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

                # Auto-compact
                if len(session._entries) > 50:
                    session._entries = session._entries[-12:]
                    console.print(f"[dim]  auto-compacted to last 12 entries[/dim]")

                # Token usage (show in/out breakdown)
                if turn_result.usage:
                    u = turn_result.usage
                    if u.input_tokens or u.output_tokens:
                        console.print(f"[dim]  in={u.input_tokens:,} out={u.output_tokens:,}[/]")
                console.print()

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted.[/yellow]")
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

    finally:
        session.save()
        console.print(f"[dim]Session saved: {session.session_id}[/dim]")


if __name__ == "__main__":
    main()
