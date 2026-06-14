# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Daemon lifecycle for background runs — Phase E W2.1.

``daemonize()`` is the classic double-fork: the caller's process exits, the
detached grandchild becomes the run (new session, stdio redirected to a
per-run log), registers itself in the :class:`RunRegistry`, and installs
SIGTERM/SIGINT handlers that clean its registry entry on the way out.

``stop_run()`` is the matching operator verb: SIGTERM, wait, escalate to
SIGKILL, always clean the registry entry. ``tail_log()`` backs ``--log``.

Linux-first: the double-fork path is exercised on Linux; macOS quirks
(launchd sessions, fork-safety of some frameworks) are NOT covered here —
owner-machine verification needed before relying on --daemon on macOS.
Pure stdlib; no kernel imports.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path

from vector_os_nano.vcli.run_registry import RunEntry, RunRegistry, _pid_alive

logger = logging.getLogger(__name__)


def daemonize(
    run_id: str,
    registry_root: "str | None" = None,
    log_dir: "str | None" = None,
    world: str = "",
    scenario: str = "",
) -> RunRegistry:
    """Detach into a daemon process; returns ONLY in the daemon.

    The original process (and the intermediate child) exit with status 0 —
    the invoking shell gets its prompt back immediately. The daemon keeps the
    caller's working directory (runs depend on repo-relative paths), writes
    stdout+stderr to ``<log_dir>/<run_id>.log``, registers itself, and cleans
    its registry entry on SIGTERM/SIGINT.
    """
    if os.fork() > 0:
        os._exit(0)  # original parent: hand the shell back
    os.setsid()
    if os.fork() > 0:
        os._exit(0)  # session leader exits; grandchild can never re-acquire a tty

    # --- the daemon ---
    log_root = Path(log_dir) if log_dir else Path.home() / ".vector-os-nano" / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{run_id}.log"
    sys.stdout.flush()
    sys.stderr.flush()
    null_fd = os.open(os.devnull, os.O_RDONLY)
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(null_fd, 0)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)

    # W2.2: tag this run — every descendant inherits the env var, so the
    # watchdog can sweep strays by tag even after a SIGKILL.
    os.environ["VECTOR_RUN_ID"] = run_id

    registry = RunRegistry(root=registry_root)
    registry.register(
        RunEntry(
            run_id=run_id,
            pid=os.getpid(),
            started_at=time.time(),
            world=world,
            scenario=scenario,
            log_dir=str(log_root),
            argv=tuple(sys.argv),
        )
    )

    def _graceful_exit(signum: int, _frame: object) -> None:
        logger.info("daemon %s: signal %d — cleaning registry and exiting", run_id, signum)
        try:
            registry.remove(run_id)
        finally:
            os._exit(0)

    signal.signal(signal.SIGTERM, _graceful_exit)
    signal.signal(signal.SIGINT, _graceful_exit)
    return registry


def stop_run(registry: RunRegistry, run_id: str, term_timeout: float = 5.0) -> bool:
    """Stop a registered run: SIGTERM → wait → SIGKILL; always clean the entry.

    Returns True when the run is gone afterwards (already-dead counts as
    stopped), False when *run_id* is unknown.
    """
    entry = registry.get(run_id)
    if entry is None:
        return False
    if not _pid_alive(entry.pid):
        registry.remove(run_id)
        return True

    def _wait_dead(deadline: float) -> bool:
        while time.time() < deadline:
            if not _pid_alive(entry.pid):
                return True
            time.sleep(0.05)
        return not _pid_alive(entry.pid)

    try:
        os.kill(entry.pid, signal.SIGTERM)
    except OSError as exc:
        logger.warning("stop_run(%s): SIGTERM failed: %s", run_id, exc)
    if not _wait_dead(time.time() + term_timeout):
        logger.warning("stop_run(%s): SIGTERM ignored — escalating to SIGKILL", run_id)
        try:
            os.kill(entry.pid, signal.SIGKILL)
        except OSError as exc:
            logger.warning("stop_run(%s): SIGKILL failed: %s", run_id, exc)
        _wait_dead(time.time() + 2.0)
    # W2.2: reap any tagged descendants the main process left behind.
    try:
        from vector_os_nano.vcli.watchdog import sweep_orphans
        sweep_orphans(run_id)
    except Exception as exc:  # noqa: BLE001 — sweep is best-effort cleanup
        logger.warning("stop_run(%s): orphan sweep failed: %s", run_id, exc)
    registry.remove(run_id)
    return True


def tail_log(log_path: str, lines: int = 50) -> str:
    """Return the last *lines* lines of *log_path* ('' when absent)."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            return "".join(deque(fh, maxlen=lines))
    except OSError:
        return ""
