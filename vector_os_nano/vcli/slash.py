# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The REPL's slash commands: ONE table of (name, help, handler), and its completion.

A handler takes the runtime and the command's words and returns False to leave the
REPL. ``dispatch`` runs one command; a command that fails prints its error and the
REPL lives.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Callable, NamedTuple

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from vector_os_nano.vcli.render import TEAL, V_LABEL, console
from vector_os_nano.vcli.runtime import switch_llm
from vector_os_nano.vcli.session import list_sessions

EXIT_COMMANDS: frozenset[str] = frozenset({"quit", "exit", "q"})

# Popular OpenRouter models, offered after "/model " (with the provider names).
KNOWN_MODELS: list[str] = [
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-haiku-4-5",
    "openai/gpt-4o",
    "openai/gpt-4.1",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "deepseek/deepseek-chat-v3-0324",
    "meta-llama/llama-4-maverick",
]


def _table(*rows: tuple[str, str]) -> None:
    tbl = Table(show_header=False, box=None, padding=(0, 2))
    tbl.add_column(style="dim", no_wrap=True)
    tbl.add_column()
    for row in rows:
        tbl.add_row(*row)
    console.print()
    console.print(tbl)
    console.print()


def _help(rt: Any, args: list[str]) -> None:
    _table(*((f"[{TEAL}]/{c.name}[/]", c.help) for c in COMMANDS),
           ("!<cmd>", "Run a shell command directly (e.g. !ls -la)"),
           ("quit", "Exit (also: exit, q, Ctrl-D)"))
    console.print("[dim]  /  command menu · Tab  accept · Ctrl+R  history · Ctrl+C  stop the turn · "
                  "Ctrl+D  exit[/dim]\n")


def _quit(rt: Any, args: list[str]) -> bool:
    return False


def _login(rt: Any, args: list[str]) -> None:
    """Keys live only in the environment: re-read .env and the environment for the
    current provider's variable. Nothing is typed here."""
    from vector_os_nano.vcli.config import resolve_credentials

    cur = rt.app_state["credentials"]
    creds = resolve_credentials(cur.provider, cur.model, cur.base_url)
    if switch_llm(rt.app_state, creds):
        console.print(f"  [green]{creds.key_env} found.[/] Using [{TEAL}]{creds.provider} / {creds.model}[/]")
    else:
        console.print(f"[yellow]  {escape(creds.instruction())}[/]")


def _model(rt: Any, args: list[str]) -> None:
    from vector_os_nano.vcli.config import PROVIDERS, resolve_credentials

    cur = rt.app_state["credentials"]
    if not args:
        console.print(f"  [{TEAL}]{cur.provider} / {cur.model}[/]")
        console.print(f"[dim]  /model <model>  or  /model <provider> [<model>]; providers: "
                      f"{', '.join(sorted(PROVIDERS))}[/dim]")
        return
    if args[0] in PROVIDERS:  # a provider switch: its key variable and its defaults
        creds = resolve_credentials(args[0], args[1] if len(args) > 1 else None)
    else:
        creds = resolve_credentials(cur.provider, args[0], cur.base_url)
    if switch_llm(rt.app_state, creds):
        console.print(f"  Switched to [{TEAL}]{creds.model}[/] ({creds.provider})")
    else:
        console.print(f"[yellow]  {escape(creds.instruction())}[/]")


def _config(rt: Any, args: list[str]) -> None:
    from vector_os_nano.vcli.config import _CONFIG_PATH

    c = rt.app_state["credentials"]
    _table(("Config", f"[dim]{_CONFIG_PATH}[/dim]"), ("Provider", f"[{TEAL}]{c.provider}[/]"),
           ("Model", f"[{TEAL}]{c.model}[/]"), ("Base URL", str(c.base_url or "(provider default)")),
           ("Key", f"{c.key_env} " + ("[green]set[/]" if c.api_key else "[yellow]not set[/]")))


def _tools(rt: Any, args: list[str]) -> None:
    registry = rt.registry
    _table(*((f"[{TEAL}]{name}[/]", f"[dim]{'read-only' if t.is_read_only({}) else 'write'} · "
              f"{escape(getattr(t, 'description', ''))}[/dim]")
             for name in registry.list_tools() if (t := registry.get(name)) is not None))


def _agent(rt: Any, args: list[str]) -> None:
    console.print()
    console.print(Panel(
        "V -- the AI core of Vector OS Nano.\n"
        "Built by Vector Robotics at CMU Robotics Institute.\n\n"
        "Capabilities:\n"
        "  Robot control    start sims, walk, explore, pick, place, navigate\n"
        "  Codebase work    read/write/edit files, run bash, search code\n"
        "  Perception       query world model, check hardware status\n"
        "  Web              fetch URLs for documentation and references\n\n"
        "V owns the hardware. Arms, grippers, quadrupeds, cameras -- V's body.\n"
        "V speaks your language. Safety is non-negotiable.",
        title=V_LABEL, title_align="left", border_style=TEAL, padding=(1, 2),
        width=min(console.width, 76),
    ))
    console.print()


def _named(device: Any) -> str:
    return f"[green]{getattr(device, 'name', type(device).__name__)}[/]" if device is not None else "[dim]none[/]"


def _status(rt: Any, args: list[str]) -> None:
    agent = rt.agent
    _table(("Model", f"[{TEAL}]{rt.app_state['credentials'].model}[/]"),
           ("Arm", _named(getattr(agent, "_arm", None))), ("Base", _named(getattr(agent, "_base", None))),
           ("Perception", _named(getattr(agent, "_perception", None))),
           ("Tools", str(len(rt.registry.list_tools()))), ("Messages", str(len(rt.session._entries))))


def _usage(rt: Any, args: list[str]) -> None:
    u = rt.session.token_usage
    console.print(f"  in={u.input_tokens:,}  out={u.output_tokens:,}  total={u.input_tokens + u.output_tokens:,}")


def _copy(rt: Any, args: list[str]) -> None:
    text = rt.app_state.get("last_response", "")
    if not text:
        console.print("[dim]  No response to copy.[/dim]")
        return
    for cmd in (["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"], ["pbcopy"]):
        try:
            subprocess.run(cmd, input=text.encode(), check=True, timeout=5)
        except (FileNotFoundError, subprocess.SubprocessError):
            continue
        console.print(f"[dim]  Copied to clipboard ({len(text)} chars)[/dim]")
        return
    console.print("[dim]  No clipboard tool: install xclip or xsel[/dim]")


def _export(rt: Any, args: list[str]) -> None:
    session = rt.session
    path = Path.home() / ".vector" / "exports" / f"{session.session_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Vector CLI Session\n\nSession: {session.session_id}\n"]
    for entry in session._entries:
        if entry.get("type") == "user":
            lines.append(f"\n**You:** {entry['content']}\n")
        elif entry.get("type") == "assistant":
            lines.append(f"\n**V:** {entry.get('text', '')}\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[dim]  Exported to {path}[/dim]")


def _compact(rt: Any, args: list[str]) -> None:
    before, after = rt.session.compact(keep_recent=8)
    console.print(f"[dim]  Compacted: {before} -> {after} entries (old context summarized)[/dim]")


def _clear(rt: Any, args: list[str]) -> None:
    rt.session._entries.clear()
    console.print("[dim]  Conversation cleared.[/dim]")


def _clear_memory(rt: Any, args: list[str]) -> None:
    """Forget the running body's rooms and objects: the scene graph is cleared IN PLACE
    (every holder — skills, oracles, the robot-state prompt — keeps the same object)
    and its file and the terrain map are deleted."""
    from vector_os_nano.vcli.tools.nav_tools import _TERRAIN_PATH

    sg = rt.app_state.get("scene_graph")
    if sg is None:
        console.print("[dim]  No running body has a scene graph (start a simulation first).[/dim]")
        return
    sg.clear()
    for path in (getattr(sg, "_persist_path", None), _TERRAIN_PATH):
        if path:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
    console.print("[dim]  Scene graph cleared. All rooms/objects forgotten.[/dim]")


def _reset(rt: Any, args: list[str]) -> None:
    reset_pose = getattr(getattr(rt.agent, "_base", None), "reset_pose", None)
    if not callable(reset_pose):
        console.print("[dim]  No running base to reset (a legged sim body stands up again).[/dim]")
        return
    reset_pose()
    console.print("[dim]  Robot pose reset: standing at its current position.[/dim]")


def _sessions(rt: Any, args: list[str]) -> None:
    sessions = list_sessions()
    if not sessions:
        console.print("[dim]No saved sessions.[/dim]")
    for s in sessions:
        console.print(f"  [{TEAL}]{s.session_id}[/]  {s.created_at}  ({s.message_count} msgs)")


class Command(NamedTuple):
    name: str
    help: str
    handler: Callable[[Any, list[str]], bool | None]


COMMANDS: tuple[Command, ...] = (
    Command("help", "Show all commands and shortcuts", _help),
    Command("login", "Re-read the API key from the environment / .env", _login),
    Command("model", "Show or switch the model  (/model <model> | /model <provider> [<model>])", _model),
    Command("config", "Show the LLM configuration (never the key)", _config),
    Command("tools", "List all registered tools", _tools),
    Command("agent", "Show V's identity and capabilities", _agent),
    Command("status", "Show hardware, tools, session info", _status),
    Command("usage", "Show token usage this session", _usage),
    Command("copy", "Copy last response to clipboard", _copy),
    Command("export", "Export session as markdown", _export),
    Command("compact", "Compress context window", _compact),
    Command("clear", "Reset conversation", _clear),
    Command("clear_memory", "Clear scene graph (forget all explored rooms/objects)", _clear_memory),
    Command("reset", "Reset robot pose (stand up after tip-over, sim only)", _reset),
    Command("sessions", "List saved sessions", _sessions),
    Command("quit", "Exit", _quit),
)
_BY_NAME = {c.name: c for c in COMMANDS}


def dispatch(rt: Any, line: str) -> bool:
    """Run the slash command *line*; False to leave the REPL. A failing command prints
    its error and the REPL lives."""
    name, *args = line.strip()[1:].split()
    command = _BY_NAME.get(name) or (_BY_NAME["quit"] if name in EXIT_COMMANDS else None)
    if command is None:
        console.print(f"[yellow]  Unknown: /{escape(name)}[/]  (type / + Tab)")
        return True
    try:
        return command.handler(rt, args) is not False
    except Exception as exc:  # noqa: BLE001 — a command never takes the REPL down
        console.print(f"[red]  /{name} failed:[/] {escape(f'{type(exc).__name__}: {exc}')}")
        return True


class VectorCompleter(Completer):
    """Completion for the REPL: ``/`` lists the commands, ``/model `` the providers and
    known models, and a partial exit word its completion."""

    def get_completions(self, document: Document, complete_event: Any) -> Any:
        from vector_os_nano.vcli.config import PROVIDERS

        text = document.text_before_cursor
        if text.startswith("/"):
            parts = text.split(None, 1)
            if len(parts) == 1 and not text.endswith(" "):
                for c in COMMANDS:
                    if c.name.startswith(parts[0][1:]):
                        yield Completion(f"/{c.name}", start_position=-len(parts[0]), display_meta=c.help)
            elif parts[0] == "/model":
                prefix = parts[1] if len(parts) > 1 else ""
                for m in [*sorted(PROVIDERS), *KNOWN_MODELS]:
                    if m.startswith(prefix):
                        yield Completion(m, start_position=-len(prefix))
            return
        word = text.strip().lower()
        if word and not text.startswith("!"):
            for ec in EXIT_COMMANDS:
                if ec.startswith(word) and ec != word:
                    yield Completion(ec, start_position=-len(text.lstrip()))
