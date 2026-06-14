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
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import re

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
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
from vector_os_nano.vcli.turn_status import TurnStatus
from vector_os_nano.vcli.tools import CategorizedToolRegistry, ToolRegistry, discover_categorized_tools

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
        if not text.startswith("/") and text.strip().lower() not in ("q", "qu", "qui", "qui", "ex", "exi"):
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
    # W2.1 — background-run operations (early-exit; the REPL never starts).
    parser.add_argument("--status", action="store_true",
                        help="List registered background runs and exit")
    parser.add_argument("--stop", default=None, metavar="RUN_ID",
                        help="Stop a background run (SIGTERM, then SIGKILL) and exit")
    parser.add_argument("--log", default=None, metavar="RUN_ID",
                        help="Print the tail of a background run's log and exit")
    return parser.parse_args(argv)


def _run_ops_dispatch(args: argparse.Namespace) -> bool:
    """Handle the W2.1 run-operations flags. Returns True when one ran
    (the caller exits without starting the REPL)."""
    if not (args.status or args.stop or args.log):
        return False
    from vector_os_nano.vcli.daemon import stop_run, tail_log
    from vector_os_nano.vcli.run_registry import RunRegistry

    registry = RunRegistry()
    if args.status:
        runs = registry.list_runs()
        if not runs:
            console.print("[dim]no background runs[/dim]")
        for r in runs:
            import datetime as _dt
            started = _dt.datetime.fromtimestamp(r.started_at).strftime("%H:%M:%S")
            console.print(
                f"  {r.run_id}  pid={r.pid}  world={r.world or '-'}"
                f"{('/' + r.scenario) if r.scenario else ''}  since {started}"
            )
    elif args.stop:
        ok = stop_run(registry, args.stop)
        console.print(
            f"[green]stopped {args.stop}[/green]" if ok
            else f"[red]unknown run {args.stop!r}[/red] — try --status"
        )
    elif args.log:
        entry = registry.get(args.log)
        if entry is None:
            console.print(f"[red]unknown run {args.log!r}[/red] — try --status")
        else:
            from pathlib import Path as _P
            console.print(tail_log(str(_P(entry.log_dir) / f"{entry.run_id}.log")) or "[dim](empty)[/dim]")
    return True


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
    console.print("[dim]  Type / for commands, quit to exit[/]")
    console.print()


def ask_permission(tool_name: str, params: dict[str, Any]) -> str:
    console.print("\n[yellow bold]Permission required:[/yellow bold]")
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


def _maybe_init_habitat_agent(args: argparse.Namespace, world: Any) -> Any:
    """Boot the habitat kinematic base for a habitat-backend scenario (M2).

    Dispatches on the resolved world's ``scenario.sim_backend`` — data the M1
    seam carries — so the kernel never imports a simulator from config. Lazy
    domain imports mirror the playground scenario resolution path. Returns an
    ``Agent`` with the connected base, or None (loudly) on any failure so the
    REPL still starts with fail-safe predicates.
    """
    scenario = getattr(world, "scenario", None)
    if scenario is None or getattr(scenario, "sim_backend", "mujoco") != "habitat":
        return None
    from vector_os_nano.vcli import habitat_runtime

    try:
        agent = habitat_runtime.boot_habitat_agent(
            world, on_status=lambda line: console.print(f"[dim]  {line}[/dim]")
        )

        # M4/M5 — semantic perception into the REPL agent (opt-in):
        # VECTOR_HABITAT_SYSNAV=1 starts the in-process pano/cloud/odom feed
        # (same base, thread-safe bridge) and the /object_nodes_list consumer
        # into THIS agent's world model. The SysNav nodes themselves run
        # outside the REPL (scripts/launch_sysnav_nodes.sh, or the
        # sysnav_perception tool mid-session) — heavy GPU procs.
        if os.environ.get("VECTOR_HABITAT_SYSNAV") == "1":
            try:
                habitat_runtime.wire_sysnav_feed(
                    agent,
                    on_status=lambda line: console.print(f"[dim]  {line}[/dim]"),
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]SysNav wiring skipped: {exc}[/yellow]")
        return agent
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]habitat scenario boot failed: {exc}[/red]")
        return None


def _maybe_init_g1_agent(args: argparse.Namespace, world: Any) -> Any:
    """Boot the in-process G1 policy-gait base for the g1_flat scenario.

    Campaign #5 — dispatches on the resolved world's embodiment ("g1") with
    the default MuJoCo backend; sibling of _maybe_init_habitat_agent. Lazy
    domain import; failure is loud, the REPL still starts.
    """
    scenario = getattr(world, "scenario", None)
    if scenario is None or getattr(scenario, "embodiment", "") != "g1":
        return None
    from vector_os_nano.vcli import g1_runtime

    # campaign #8 R0: open a live viewer window unless --headless. The owner
    # drives the G1 from vector-cli and must SEE the gait, so the default is a
    # window (mirrors the go2 default; --headless suppresses it, e.g. on a
    # display-less box).
    gui = not getattr(args, "headless", False)
    try:
        return g1_runtime.boot_g1_agent(
            world,
            on_status=lambda line: console.print(f"[dim]  {line}[/dim]"),
            gui=gui,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]g1 scenario boot failed: {exc}[/red]")
        return None


def _maybe_init_go2_agent(args: argparse.Namespace, world: Any) -> Any:
    """Boot the Go2 quadruped for the furnished VLM-room scenario (campaign #9
    R2, track C). Dispatches on embodiment "go2" + the furnished id, so the
    apartment go2_room (full ROS/nav stack via --sim-go2) is untouched. Lazy
    import; failure is loud, the REPL still starts."""
    scenario = getattr(world, "scenario", None)
    if scenario is None or getattr(scenario, "embodiment", "") != "go2":
        return None
    if getattr(scenario, "id", "") != "go2_room_vlm":
        return None
    from vector_os_nano.vcli import go2_runtime

    gui = not getattr(args, "headless", False)
    try:
        return go2_runtime.boot_go2_agent(
            world,
            on_status=lambda line: console.print(f"[dim]  {line}[/dim]"),
            gui=gui,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]go2 scenario boot failed: {exc}[/red]")
        return None


def _init_agent(args: argparse.Namespace) -> Any:
    if not (args.sim or args.sim_go2):
        return None
    try:
        from vector_os_nano.core.agent import Agent  # type: ignore[import]
        if args.sim:
            from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm  # type: ignore[import]
            from vector_os_nano.hardware.sim.mujoco_gripper import MuJoCoGripper  # type: ignore[import]
            from vector_os_nano.hardware.sim.mujoco_perception import MuJoCoPerception  # type: ignore[import]
            from vector_os_nano.skills.pick import SIM_PICK_CONFIG
            arm = MuJoCoArm(gui=not getattr(args, "headless", False))
            arm.connect()
            gripper = MuJoCoGripper(arm)
            perception = MuJoCoPerception(arm)
            return Agent(
                arm=arm,
                gripper=gripper,
                perception=perception,
                config={"skills": {"pick": dict(SIM_PICK_CONFIG)}},
            )

        # --- Go2 full stack: MuJoCo + ROS2 bridge + nav stack + VLM + Rerun ---
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2  # type: ignore[import]
        import os

        console.print("[dim]  Starting Go2 MuJoCo simulation...[/dim]")
        base = MuJoCoGo2(gui=True, room=True, backend="auto")
        base.connect()
        base.stand()

        # Load config for API key
        from vector_os_nano.core.config import load_config
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))), "config", "user.yaml")
        cfg = load_config(cfg_path) if os.path.exists(cfg_path) else {}
        api_key = (
            args.api_key
            or cfg.get("llm", {}).get("api_key")
            or os.environ.get("OPENROUTER_API_KEY", "")
        )

        agent = Agent(base=base, llm_api_key=api_key, config=cfg)

        # Register Go2 skills
        from vector_os_nano.skills.go2 import get_go2_skills
        for skill in get_go2_skills():
            agent._skill_registry.register(skill)

        # VLM perception (GPT-4o via OpenRouter)
        if api_key:
            try:
                from vector_os_nano.perception.vlm_go2 import Go2VLMPerception
                agent._vlm = Go2VLMPerception(config={"api_key": api_key})
                console.print("[dim]  VLM: GPT-4o via OpenRouter[/dim]")
            except Exception:
                agent._vlm = None


        # Scene graph (SysNav-style) with persistence
        import os as _os
        from vector_os_nano.core.scene_graph import SceneGraph
        _sg_path = _os.path.expanduser("~/.vector_os_nano/scene_graph.yaml")
        _os.makedirs(_os.path.dirname(_sg_path), exist_ok=True)
        _sg = SceneGraph(persist_path=_sg_path)
        _sg.load()
        _sg_stats = _sg.stats()
        if _sg_stats["rooms"] > 0:
            console.print(
                f"[dim]  Memory: scene graph restored "
                f"({_sg_stats['rooms']} rooms, {_sg_stats['objects']} objects)[/dim]"
            )
        else:
            console.print("[dim]  Memory: scene graph (rooms -> viewpoints -> objects)[/dim]")
        agent._spatial_memory = _sg

        # ROS2 bridge + nav stack (background)
        try:
            _launch_ros2_stack(base)
            console.print("[dim]  ROS2: bridge + nav stack launched[/dim]")
        except Exception as exc:
            console.print(f"[dim]  ROS2: not available ({exc})[/dim]")

        return agent

    except Exception as exc:
        console.print(f"[yellow]Warning: Could not init simulation: {exc}[/yellow]")
        import traceback
        traceback.print_exc()
        return None


def _launch_ros2_stack(go2: Any) -> None:
    """Launch ROS2 bridge + Vector Nav Stack in background.

    Starts the bridge in a daemon thread and the nav stack as a subprocess.
    Non-blocking — returns immediately after launching.
    """
    import subprocess
    import signal
    import atexit
    import os
    import threading

    # 1. Start ROS2 bridge on existing MuJoCoGo2
    try:
        import rclpy
        if not rclpy.ok():
            rclpy.init()

        import importlib.util
        import sys
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        bridge_path = os.path.join(repo, "scripts", "go2_vnav_bridge.py")
        spec = importlib.util.spec_from_file_location("_vnav_bridge", bridge_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_vnav_bridge"] = mod
        spec.loader.exec_module(mod)

        node = mod.Go2VNavBridge(go2)

        def _spin():
            try:
                rclpy.spin(node)
            except Exception:
                pass

        t = threading.Thread(target=_spin, daemon=True)
        t.start()
    except ImportError:
        raise RuntimeError("ROS2 (rclpy) not available")

    # 2. Launch nav stack nodes
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script = os.path.join(repo, "scripts", "launch_nav_only.sh")
    if os.path.isfile(script):
        log_path = "/tmp/vector_nav_only.log"
        # Truncate log if it has grown beyond 5 MB to prevent unbounded disk use
        if os.path.exists(log_path) and os.path.getsize(log_path) > 5 * 1024 * 1024:
            with open(log_path, "w") as _f:
                _f.write("")  # truncate
        log_fh = open(log_path, "a")
        proc = subprocess.Popen(
            [script], stdout=log_fh, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        # Use a list so the monitor thread can mutate the proc reference.
        proc_ref = [proc]

        def _nav_health_monitor(proc_ref: list, script: str, log_fh: Any, console: Any) -> None:
            """Daemon thread: poll nav stack every 5s, restart if crashed."""
            while True:
                time.sleep(5)
                current = proc_ref[0]
                if current.poll() is not None:
                    exit_code = current.returncode
                    console.print(f"  [red]nav stack crashed (exit {exit_code}), restarting...[/]")
                    try:
                        new_proc = subprocess.Popen(
                            [script], stdout=log_fh, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid,
                        )
                        proc_ref[0] = new_proc
                        time.sleep(3)  # let it initialize
                        console.print("  [green]nav stack restarted[/]")
                    except Exception as exc:
                        console.print(f"  [red]nav stack restart failed: {exc}[/]")

        monitor = threading.Thread(
            target=_nav_health_monitor,
            args=(proc_ref, script, log_fh, console),
            daemon=True,
        )
        monitor.start()

        def _cleanup():
            current = proc_ref[0]
            try:
                os.killpg(os.getpgid(current.pid), signal.SIGTERM)
                current.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(current.pid), signal.SIGKILL)
                except Exception:
                    pass
            log_fh.close()

        atexit.register(_cleanup)


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
            console.print("[yellow]  Usage: /login claude | /login anthropic | /login openrouter[/]")
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
                console.print("[green]  Authenticated.[/] Token saved to ~/.vector/oauth_credentials.json")
                console.print("[dim]  Restart vector-cli to use your subscription.[/dim]\n")
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
                console.print("[green]  Saved.[/] Restart vector-cli to apply.")
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
                console.print("[green]  Saved.[/] Restart vector-cli to apply.")
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
            console.print("[dim]  Conversation cleared.[/dim]")
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

        # Always delete the persist file (works even without agent)
        try:
            _os.remove(_sg_path)
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
            console.print("[dim]  Scene graph cleared. All rooms/objects forgotten.[/dim]")
        else:
            console.print("[dim]  No scene graph file found.[/dim]")

    elif cmd == "reset":
        import os as _os
        # Signal bridge to reset robot pose via file flag
        try:
            with open("/tmp/vector_reset_pose", "w") as _f:
                _f.write("1")
            console.print("[dim]  Reset signal sent. Robot will stand up at current position.[/dim]")
        except OSError as _exc:
            console.print(f"[yellow]  Failed to send reset: {_exc}[/]")

    elif cmd == "model":
        if not args_rest:
            current = app_state.get("model", "unknown") if app_state else "unknown"
            console.print(f"  [{TEAL}]{current}[/]")
            console.print("[dim]  /model <name> to switch. Tab for suggestions.[/dim]")
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
    from a background thread — Rich Console is not thread-safe.
    """
    try:
        from vector_os_nano.skills.go2.explore import set_event_callback
    except ImportError:
        return

    def _on_explore_event(event_type: str, data: dict) -> None:
        if event_type == "started":
            total = data.get("total_rooms", 8)
            print(f"  >> Exploration started ({total} rooms to discover)")

        elif event_type == "room_entered":
            room = data.get("room", "?")
            visited = data.get("visited", 0)
            total = data.get("total", 8)
            elapsed = data.get("elapsed_sec", 0)
            label = _ROOM_LABELS.get(room, room)
            bar = f"[{'#' * visited}{'.' * (total - visited)}]"
            print(f"  >> {label} {bar} {visited}/{total} ({elapsed}s)")

        elif event_type == "status":
            elapsed = data.get("elapsed_sec", 0)
            rooms = data.get("rooms_found", 0)
            total = data.get("total", 8)
            room = data.get("current_room", "?")
            pos = data.get("position", [0, 0])
            print(f"  .. {elapsed}s | {rooms}/{total} rooms | at {room} ({pos[0]}, {pos[1]})")

        elif event_type in ("completed", "complete"):
            rooms = data.get("rooms", [])
            reason = data.get("reason", "")
            elapsed = data.get("elapsed_sec", 0)
            if reason == "tare_finished":
                print(f"  >> Exploration complete ({elapsed}s, TARE: all frontiers covered). {len(rooms)} rooms.")
            else:
                print(f"  >> Exploration complete ({elapsed}s). {len(rooms)} rooms.")

        elif event_type == "stopped":
            reason = data.get("reason", "unknown")
            rooms = data.get("rooms", [])
            if reason == "cancelled":
                print(f"  >> Exploration stopped. {len(rooms)} rooms so far.")
            elif reason == "robot_fell":
                print("  >> Robot fell! Exploration aborted.")
            else:
                print(f"  >> Exploration {reason} — {len(rooms)} rooms.")

    set_event_callback(_on_explore_event)


def _wants_window(args: argparse.Namespace) -> bool:
    """Return True when the user wants a visible MuJoCo viewer window.

    A window is wanted when --sim or --sim-go2 is active (NL-triggered sims are
    handled separately in sim_tool.py) and --headless is NOT set.
    """
    return (args.sim or args.sim_go2) and not getattr(args, "headless", False)


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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # W2.1 run ops are cheap early exits — before any re-exec/credential/agent init.
    if _run_ops_dispatch(args):
        return

    # Owner finding (b): an unsourced shell must still NL-start sysnav. Re-exec
    # once with the ROS/SysNav overlays sourced when they exist on disk
    # (LD_LIBRARY_PATH can only be set at process birth). No-op everywhere else.
    from vector_os_nano.vcli.ros_bootstrap import maybe_reexec_with_ros
    maybe_reexec_with_ros()
    if os.environ.get("VECTOR_ROS_BOOTSTRAPPED"):
        console.print(
            "[dim]ROS env auto-sourced: "
            f"{os.environ['VECTOR_ROS_BOOTSTRAPPED'].replace(':', ' + ')}[/dim]"
        )

    # W2.2: tag this session so every descendant (sim subprocess, nav stack,
    # explore threads' children) inherits VECTOR_RUN_ID — the watchdog can then
    # sweep strays by tag after an abnormal exit. setdefault keeps an outer tag
    # (e.g. a daemonized parent) authoritative.
    import uuid as _uuid
    os.environ.setdefault("VECTOR_RUN_ID", f"cli-{_uuid.uuid4().hex[:8]}")

    # --- macOS mjpython re-exec guard (must be before any credential/agent init) ---
    _maybe_reexec_under_mjpython(args)

    _setup_logging(args.verbose)
    _ensure_sigint_under_mjpython()

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
    agent = _init_agent(args)
    world = _resolve_active_world(args, agent)
    # M2 (ADR-009): a habitat-backend scenario boots its own kinematic base
    # when no sim agent was requested — same lazy domain wiring as the
    # scenario resolution above; failure is loud, the REPL still starts.
    if agent is None:
        agent = _maybe_init_habitat_agent(args, world)
    if agent is None:
        agent = _maybe_init_g1_agent(args, world)   # campaign #5: g1_flat
    if agent is None:
        agent = _maybe_init_go2_agent(args, world)  # campaign #9 R2: go2_room_vlm

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
        wm = getattr(agent, "_world_model", None) if agent else None
        robot_ctx_provider = RobotContextProvider(
            base=base, scene_graph=sg, arm=arm, world=world, world_model=wm
        )
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

    # Backend + engine (deferred if no API key — /login can set it up)
    engine: VectorEngine | None = None
    if api_key:
        backend = create_backend(provider=provider, api_key=api_key, model=model, base_url=base_url)
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
    # sub-goal, strategy, verify predicate and a stable PASS/FAIL marker. Runs
    # alongside _vgg_step_display (best-effort) — never affects execution. The
    # verify predicate lives on the active goal tree's sub_goals; the REPL loop
    # refreshes this map when it decomposes each task.
    _vgg_verify_by_name: dict[str, str] = {}

    def _vgg_step_view_display(view: dict[str, Any]) -> None:
        """Render one per-step EXPORT VIEW (sub-goal/strategy/verify/PASS-FAIL)."""
        try:
            from vector_os_nano.vcli.cognitive.observation import render_step_view
            verify = _vgg_verify_by_name.get(view.get("sub_goal_name"))
            line = render_step_view(view, verify)
            ok = bool(view.get("success")) and bool(view.get("verify_result"))
            colour = "green" if ok else "red"
            # markup=False: the rendered line carries literal [PASS]/[FAIL]
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
            console.print("[dim]  VGG cognitive layer: enabled[/dim]")
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
        parts: list[str] = ["<b>V</b>"]
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
    # Ctrl-C UX (R2-4): one ^C aborts a running task and returns to the prompt;
    # a second consecutive ^C (at the prompt) exits cleanly. Reset on any input.
    interrupt_pending = False

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
                console.print("[yellow]No API key. Use /login to authenticate first.[/]")
                continue

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
                        st = p.get("sim_type", "go2")
                        return f"[dim]sim start[/] {st}"
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
                        # Evidence gate (invariant II): a successful run only
                        # counts as *verified* when its steps are backed by
                        # deterministic predicates — in EVERY world; only the
                        # world's bounded exemption set is excused.
                        try:
                            from vector_os_nano.vcli.cognitive.trace_store import evidence_passed
                            from vector_os_nano.vcli.engine import _world_exempt_strategies
                            # Read the LIVE world — the closure-captured one
                            # goes stale after an NL sim start swaps worlds.
                            _w = app_state.get("world") or world
                            _evidence = evidence_passed(
                                trace,
                                exempt_strategies=_world_exempt_strategies(_w),
                            )
                        except Exception:  # noqa: BLE001
                            _evidence = True
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
                                # markup=False: literal [PASS]/[FAIL] markers.
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

                    # Runs on a background thread (CLI stays responsive) EXCEPT
                    # when a GUI viewer is live UNDER MJPYTHON (macOS) — only
                    # there GLFW is main-thread-only, so it executes
                    # synchronously on this thread and the prompt returns once
                    # the arm finishes. Linux/Windows keep the background
                    # thread even with a viewer (REPL stays responsive).
                    engine.vgg_execute_async(goal_tree, on_complete=_on_vgg_complete)
                    continue  # next input (immediately when headless)

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

            except KeyboardInterrupt:
                # One Ctrl-C aborts the running task and returns to the prompt;
                # arm the pending-exit so an immediate second Ctrl-C quits (R2-4).
                interrupt_pending = True
                console.print("\n[yellow]Interrupted.[/yellow] [dim](Ctrl-C again or 'quit' to exit)[/dim]")
            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str or "rate_limit" in err_str:
                    current_model = app_state.get("model", "?")
                    console.print(f"[yellow]  Rate limited on {current_model}.[/]")
                    console.print("[dim]  Try: /model claude-haiku-4-5 (lower rate limit)[/dim]")
                elif "401" in err_str or "authentication" in err_str.lower():
                    console.print("[yellow]  Authentication failed. Use /login to reconfigure.[/]")
                elif "404" in err_str or "not_found" in err_str:
                    console.print(f"[yellow]  Model not found: {app_state.get('model', '?')}[/]")
                    console.print("[dim]  Try: /model claude-haiku-4-5[/dim]")
                else:
                    console.print(f"[red]Error:[/] {exc}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()

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
