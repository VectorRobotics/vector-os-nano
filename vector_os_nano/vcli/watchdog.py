# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""RUN_ID watchdog orphan sweep — Phase E W2.2.

A run tags itself with the ``VECTOR_RUN_ID`` environment variable (set once,
early — every descendant inherits it for free). After an abnormal exit, the
sweep finds surviving descendants by scanning ``/proc/<pid>/environ`` for the
tag and reaps them TERM→KILL — no pidfile bookkeeping, works even after the
parent died to SIGKILL.

Stdlib-only BY DESIGN: psutil is not a project dependency and adding one is a
gated decision; ``/proc`` covers Linux (the machine that runs sim stacks).
On macOS (no /proc) the sweep is a logged no-op — owner verification needed
if --daemon ever runs there.

/tmp flag-file inventory (W2.2 survey, 2026-06-10): every ``/tmp/vector_*``
flag in the tree (9 files: go2_vnav_bridge.py, go2_ros2_proxy.py, explore.py,
navigate.py, nav_tools.py, ros2_tools.py, sim_tool.py, cli.py,
spatial_memory.py) crosses the agent↔bridge PROCESS boundary — they are the
documented cross-process channel the plan says to keep. There are currently
NO same-process-only flags, so the in-memory PubSub replacement is correctly
EMPTY until one exists.
"""
from __future__ import annotations

import logging
import os
import signal
import time

from vector_os_nano.vcli.run_registry import _pid_alive

logger = logging.getLogger(__name__)

ENV_TAG = "VECTOR_RUN_ID"


def find_tagged_pids(run_id: str) -> tuple[int, ...]:
    """Return PIDs (excluding the caller's) whose env carries this run's tag.

    Reads ``/proc/<pid>/environ`` — own-user processes only (others raise and
    are skipped). Returns () on platforms without /proc.
    """
    if not os.path.isdir("/proc"):
        logger.info("watchdog: no /proc on this platform — sweep unavailable")
        return ()
    needle = f"{ENV_TAG}={run_id}".encode()
    me = os.getpid()
    hits: list[int] = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == me:
            continue
        try:
            with open(f"/proc/{pid}/environ", "rb") as fh:
                if needle in fh.read().split(b"\0"):
                    hits.append(pid)
        except OSError:
            continue  # vanished or not ours — skip
    return tuple(hits)


def sweep_orphans(run_id: str, term_timeout: float = 3.0) -> tuple[int, ...]:
    """Reap every process still tagged with *run_id*: SIGTERM → wait → SIGKILL.

    Returns the PIDs that were swept. Safe to call repeatedly; never touches
    the calling process.
    """
    pids = find_tagged_pids(run_id)
    if not pids:
        return ()
    logger.warning("watchdog: sweeping %d orphan(s) of run %s: %s",
                   len(pids), run_id, list(pids))
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.time() + term_timeout
    while time.time() < deadline and any(_pid_alive(p) for p in pids):
        time.sleep(0.05)
    for pid in pids:
        if _pid_alive(pid):
            logger.warning("watchdog: SIGKILL escalation for orphan %d", pid)
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    return pids
