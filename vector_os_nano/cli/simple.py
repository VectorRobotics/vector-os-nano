"""Vector OS Nano — Interactive CLI with AI Chat.

readline-based interactive shell with:
- Robot command execution via Agent
- AI conversation with Claude Haiku (multi-turn memory)
- Status-aware chat (AI knows robot state + objects)
- Visual distinction between commands, AI chat, and system messages

Entry point: vector-os  (configured in pyproject.toml)
"""
from __future__ import annotations

import json
import logging
import readline  # noqa: F401 — side-effect: enables line editing and history
import sys
import time
from typing import Any

logger = logging.getLogger(__name__)

from vector_os_nano.version import __version__ as _VERSION

# ---------------------------------------------------------------------------
# ANSI color codes
# ---------------------------------------------------------------------------
_TEAL = "\033[38;2;0;180;180m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


class SimpleCLI:
    """Interactive CLI with AI chat and robot control."""

    COMMANDS: dict[str, str] = {
        "pick":   "Pick an object: pick <name>",
        "place":  "Place held object",
        "home":   "Move arm to home position",
        "scan":   "Move to scan position",
        "open":   "Open gripper",
        "close":  "Close gripper",
        "detect": "Detect objects: detect [query]",
        "stop":   "Stop all tasks",
        "status": "Show system status",
        "world":  "Show world model",
        "help":   "Show commands",
        "quit":   "Exit (also: q, Ctrl-C)",
    }

    def __init__(self, agent: Any = None, verbose: bool = False) -> None:
        self._agent = agent
        self._verbose = verbose
        self._running: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main readline input loop."""
        self._running = True
        self._print_banner()

        while self._running:
            try:
                user_input = input("vector> ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                self._running = False
                break
            except Exception:
                # MuJoCo viewer thread can sometimes interrupt input()
                continue

            if not user_input:
                continue

            try:
                self._handle_input(user_input)
            except Exception as exc:
                print(f"{_RED}Error: {exc}{_RESET}")
                logger.exception("Unhandled error in _handle_input")

        print("Goodbye.")

    def _handle_input(self, text: str) -> None:
        """Route input: built-in CLI commands → Agent unified pipeline."""
        parts = text.split(None, 1)
        command = parts[0].lower()

        # Built-in CLI-only commands
        if command in ("quit", "exit", "q"):
            self._running = False
            return
        if command == "help":
            self._print_help()
            return
        if command == "status":
            self._handle_status()
            return
        if command == "skills":
            self._handle_skills()
            return
        if command == "world":
            self._handle_world()
            return

        # Everything else → unified Agent pipeline
        if self._agent is not None:
            self._execute_unified(text)
        else:
            print(f"{_DIM}No agent configured. Type 'help'.{_RESET}")

    # ------------------------------------------------------------------
    # Unified execution — all user input goes here
    # ------------------------------------------------------------------

    def _execute_unified(self, text: str) -> None:
        """Route all input through Agent's multi-stage pipeline."""
        start = time.time()
        result = self._agent.execute(text)
        elapsed = time.time() - start

        # Display based on result status
        if result.status == "chat":
            # Pure chat response from AI
            if result.message:
                print(f"\n{_TEAL}V:{_RESET} {result.message}\n")

        elif result.status == "query":
            # Query response (scan+detect happened, AI summarized)
            if result.message:
                print(f"\n{_TEAL}V:{_RESET} {result.message}\n")

        elif result.status == "clarification_needed":
            msg = result.message or result.clarification_question
            print(f"\n{_TEAL}V:{_RESET} {msg}\n")

        elif result.success:
            # Task completed
            if result.message:
                print(f"\n{_TEAL}V:{_RESET} {result.message}")
            print(f"{_GREEN}OK{_RESET} ({result.steps_completed}/{result.steps_total} steps, {elapsed:.1f}s)")
            if self._verbose and result.trace:
                for step in result.trace:
                    icon = f"{_GREEN}OK{_RESET}" if step.status == "success" else f"{_RED}{step.status}{_RESET}"
                    print(f"  [{icon}] {step.skill_name} ({step.duration_sec:.1f}s)")
            print()

        else:
            # Failed
            if result.message:
                print(f"\n{_TEAL}V:{_RESET} {result.message}")
            print(f"{_RED}FAILED: {result.failure_reason}{_RESET}")
            if self._verbose and result.trace:
                for step in result.trace:
                    icon = f"{_GREEN}OK{_RESET}" if step.status == "success" else f"{_RED}{step.status}{_RESET}"
                    print(f"  [{icon}] {step.skill_name} ({step.duration_sec:.1f}s)")
            print()

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        import pathlib as _pl
        _logo_path = _pl.Path(__file__).parent / "logo_braille.txt"
        try:
            logo_lines = _logo_path.read_text().strip().splitlines()
        except FileNotFoundError:
            logo_lines = ["VECTOR OS NANO"]

        print()
        for line in logo_lines:
            print(f"{_TEAL}{_BOLD}{line}{_RESET}")
        print(f"{_DIM}{'':>40}v{_VERSION}{_RESET}")
        print()
        print(f"  {_TEAL}Natural language robot arm control + AI chat.{_RESET}")
        print(f"  Robot commands execute directly. Other messages chat with AI.")
        print(f"  Type {_TEAL}'help'{_RESET} for commands.\n")

    def _print_help(self) -> None:
        print(f"\n{_BOLD}Robot Commands:{_RESET}")
        for cmd, desc in self.COMMANDS.items():
            print(f"  {_TEAL}{cmd:12s}{_RESET} {desc}")
        print(f"\n{_BOLD}AI Chat:{_RESET}")
        print(f"  Type anything else to chat with the AI assistant.")
        print(f"  The AI knows your robot state and can help with tasks.")
        print(f"  Use {_TEAL}'chat <msg>'{_RESET} to force chat mode.\n")

    def _handle_status(self) -> None:
        if self._agent is None:
            print("No agent configured.")
            return
        print(f"{_BOLD}Skills:{_RESET} {', '.join(self._agent.skills)}")
        if self._agent._arm is not None:
            joints = self._agent._arm.get_joint_positions()
            names = self._agent._arm.joint_names
            print(f"{_BOLD}Joints:{_RESET}")
            for n, j in zip(names, joints):
                print(f"  {n:18s} {j:+.3f} rad")
        if self._agent._gripper is not None:
            try:
                pos = self._agent._gripper.get_position()
                holding = self._agent._gripper.is_holding()
                print(f"{_BOLD}Gripper:{_RESET} {'open' if pos > 0.5 else 'closed'}{' (holding)' if holding else ''}")
            except Exception:
                pass
        if hasattr(self._agent._arm, "get_object_positions"):
            objs = self._agent._arm.get_object_positions()
            print(f"{_BOLD}Objects:{_RESET} {len(objs)}")
            for name, pos in objs.items():
                print(f"  {name:15s} ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        print(f"{_BOLD}Chat:{_RESET} {'enabled' if self._llm_client else 'disabled (no API key)'}")

    def _handle_skills(self) -> None:
        if self._agent is not None:
            print("Registered skills:", ", ".join(self._agent.skills))
        else:
            print("No agent configured.")

    def _handle_world(self) -> None:
        if self._agent is None:
            print("No agent configured.")
            return
        print(json.dumps(self._agent.world.to_dict(), indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the 'vector-os' console script."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Vector OS Nano CLI")
    parser.add_argument(
        "--port", default="/dev/ttyACM0", help="Serial port for SO-101 arm"
    )
    parser.add_argument(
        "--no-arm", action="store_true", help="Run without arm hardware"
    )
    parser.add_argument(
        "--no-perception", action="store_true", help="Run without camera"
    )
    parser.add_argument(
        "--llm-key", default=None, help="LLM API key (or set OPENROUTER_API_KEY)"
    )
    parser.add_argument(
        "--config", default=None, help="Path to config YAML"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "--sim", action="store_true", help="Use PyBullet simulation"
    )

    args = parser.parse_args()

    # Lazy import of Agent to keep startup fast
    try:
        from vector_os_nano.core.agent import Agent
    except ImportError as exc:
        print(f"Error: cannot import Agent: {exc}", file=sys.stderr)
        sys.exit(1)

    arm = None
    gripper = None
    perception = None

    if not args.no_arm and not args.sim:
        try:
            from vector_os_nano.hardware.so101 import SO101Arm
            arm = SO101Arm(port=args.port)
            arm.connect()
            print(f"Connected to SO-101 on {args.port}")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not connect to arm: {exc}")

    if args.sim:
        print("Simulation mode (PyBullet) — not yet implemented")

    api_key = args.llm_key
    if api_key is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")

    agent = Agent(
        arm=arm,
        gripper=gripper,
        perception=perception,
        llm_api_key=api_key,
        config=args.config,
    )

    cli = SimpleCLI(agent=agent, verbose=args.verbose)

    try:
        cli.run()
    finally:
        if arm is not None:
            arm.disconnect()
            print("Arm disconnected.")


if __name__ == "__main__":
    main()
