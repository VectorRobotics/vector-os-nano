# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""``vector-cli``: argument parsing and the entry point.

    python -m vector_os_nano.vcli.cli [options]
    # or via console_scripts: vector-cli [options]

Bare ``vector-cli`` runs the REPL (``vcli/repl.py``); ``-p TEXT`` runs ONE turn and
exits with its verdict. Both build their runtime with ``vcli/runtime.py``.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from dataclasses import replace
from typing import Any

from rich.markup import escape

from vector_os_nano.vcli.render import (
    NO_REPLY, TEAL, _chat_turn, _gait_backend, _setup_explore_events, _trace_acted, console,
)
from vector_os_nano.vcli.repl import _p0_stop, run_repl
from vector_os_nano.vcli.runtime import build_runtime, run_turn

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from vector_os_nano.version import __version__

    parser = argparse.ArgumentParser(
        prog="vector-cli",
        description="Vector CLI — Agentic CLI for Vector OS Nano",
    )
    parser.add_argument("--version", action="version", version=f"vector-cli {__version__}")
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
    from vector_os_nano.vcli.config import PROVIDERS
    parser.add_argument(
        "--provider", default=None, choices=sorted(PROVIDERS),
        help="LLM provider (overrides llm.provider in ~/.vector/config.yaml; default deepseek). "
        "Its key is read from the environment or .env, never from a flag",
    )
    parser.add_argument("--model", default=None, help="Model to use (overrides llm.model in ~/.vector/config.yaml)")
    parser.add_argument("--resume", nargs="?", const="latest", default=None, help="Resume session")
    parser.add_argument("--base-url", default=None, help="API base URL")
    parser.add_argument("--no-permission", action="store_true", help="Allow all tools without prompts")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
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


def _requested_embodiment(args: argparse.Namespace) -> str | None:
    """--embodiment wins; --sim-go2 / --sim are aliases for go2 / so101."""
    return (
        getattr(args, "embodiment", None)
        or ("go2" if getattr(args, "sim_go2", False) else None)
        or ("so101" if getattr(args, "sim", False) else None)
    )


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

    from vector_os_nano.hardware.sim.viewer_mode import running_under_mjpython

    if running_under_mjpython():
        return  # already under mjpython — nothing to do

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


def _build(args: Any, face: str) -> Any:
    """The face's runtime (``build_runtime``) from the parsed flags; notices on the console."""
    rt = build_runtime(
        provider=args.provider, model=args.model, base_url=args.base_url,
        embodiment=_requested_embodiment(args), gui=not args.headless,
        no_permission=args.no_permission, resume=args.resume, face=face,
        notice=lambda msg: console.print(f"[dim]{escape(msg)}[/dim]"),
    )
    # Explore event streaming (the go2 explore skill's background progress lines).
    _setup_explore_events(console)
    return rt


def run_one_turn(args: Any) -> int:
    """Run ONE turn for ``args.print_prompt`` non-interactively and return an exit code.

    The machine-checkable acceptance entrypoint. It builds the same runtime as the
    REPL (``build_runtime``), runs the turn SYNCHRONOUSLY through ``run_turn``, prints the model's own words, then builds a frozen ``VerdictReport``
    from the EXISTING gate (never a second opinion). Under ``--json`` it emits
    exactly one stdout line ``VECTOR_VERDICT {<json>}`` (all Rich output is already
    routed to stderr by ``main``): the report, whose ``acted`` says whether a
    side-effecting tool ran.

    Exit codes: 0 = verified, or a chat turn (no action, no verify: nothing claimed);
    2 = ran, not verified; 1 = error / no trace.
    """
    from vector_os_nano.vcli.cognitive.trace_store import verify_oracle_names
    from vector_os_nano.vcli.verdict import EXIT_VERIFIED, VerdictReport

    prompt = args.print_prompt or ""
    emit_json = bool(getattr(args, "json", False))

    def _emit(report: "VerdictReport", *, chat: bool = False) -> int:
        if emit_json:
            # The ONE machine line on real stdout (Rich/banner are on stderr).
            print(report.to_sentinel_line(), flush=True)
        else:
            console.print(
                f"[{TEAL}]verdict[/] {report.evidence} verified={report.verified} "
                f"acted={report.acted} ({report.n_grounded}/{report.n_steps} grounded) "
                f"goal={report.goal_check}"
            )
        return EXIT_VERIFIED if chat else report.exit_code()

    try:
        rt = _build(args, face="print")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Setup error:[/] {exc}")
        return _emit(VerdictReport.no_trace(goal=prompt, error=str(exc)))

    engine = rt.engine
    if engine._backend is None:
        key_env = rt.app_state["credentials"].key_env
        return _emit(VerdictReport.no_trace(goal=prompt, error=f"no LLM: {key_env} is not set"))

    # P0 stop: the registered stop skill's words, no LLM call.
    if engine.is_stop(prompt):
        _p0_stop(engine, prompt, rt.app_state, rt.session)
        return _emit(VerdictReport.no_trace(goal=prompt, error="stopped (P0, no LLM call)"))

    # ask_permission=None: headless denies every 'ask' (--no-permission allows).
    try:
        trace, answer, _ = run_turn(rt, prompt)
    except Exception as exc:  # noqa: BLE001
        return _emit(VerdictReport.no_trace(goal=prompt, error=f"native turn failed: {exc}"))
    if answer:
        console.print(escape(answer))
    if _chat_turn(trace):
        if not answer:
            console.print(f"[yellow]{NO_REPLY}[/]")
        return _emit(VerdictReport.no_trace(goal=prompt), chat=True)

    # Build the verdict from the EXISTING gate, single-sourced from the SAME
    # namespace GoalVerifier uses. Fail CLOSED on any error reading the gate.
    try:
        agent = getattr(engine, "_agent", None)
        oracle_names = verify_oracle_names(agent, engine)
        report = VerdictReport.from_trace(trace, oracle_names, gait_backend=_gait_backend(agent))
    except Exception as exc:  # noqa: BLE001
        report = replace(VerdictReport.no_trace(goal=prompt, error=f"verdict failed: {exc}"),
                         acted=_trace_acted(trace))
    return _emit(report)


def _exit_on_signal(signum: int, frame: Any) -> None:
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # --- macOS mjpython re-exec guard (must be before any credential/agent init) ---
    _maybe_reexec_under_mjpython(args)

    _setup_logging(args.verbose)
    _ensure_sigint_under_mjpython()
    # SIGTERM / SIGHUP take the normal exit path: the turn is stopped, the session and
    # the scene graph are saved, every sim this process started is stopped (atexit).
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _exit_on_signal)

    # Non-interactive -p/--print: run ONE turn and exit with the verdict code.
    # Under --json route ALL Rich/banner output to stderr so stdout carries ONLY
    # the single VECTOR_VERDICT sentinel line.
    if getattr(args, "print_prompt", None) is not None:
        if getattr(args, "json", False):
            console.file = sys.stderr
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
        rt = _build(args, face="repl")
    except Exception as exc:  # noqa: BLE001 — a sim that did not start, a body the engine cannot bind
        console.print(f"[red]Setup error:[/] {exc}")
        sys.exit(2)
    run_repl(rt)


if __name__ == "__main__":
    main()
