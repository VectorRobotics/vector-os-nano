# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Scoped process cleanup: kill ONLY what this test process spawned.

Replaces the old ``rosm nuke --yes`` / ``pkill -f`` teardowns, which killed every
ROS2 node and model server on the (shared) machine. Everything here is keyed on the
pytest process's own descendants, so another project's stack is never touched.
"""
from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest


def descendants(pid: int | None = None) -> set[int]:
    """All live descendants of *pid* (default: this process), via /proc children."""
    out: set[int] = set()
    stack = [os.getpid() if pid is None else pid]
    while stack:
        p = stack.pop()
        try:
            tasks = os.listdir(f"/proc/{p}/task")
        except OSError:
            continue
        for t in tasks:
            try:
                kids = Path(f"/proc/{p}/task/{t}/children").read_text().split()
            except OSError:
                continue
            for k in map(int, kids):
                if k not in out:
                    out.add(k)
                    stack.append(k)
    return {p for p in out if _alive(p)}


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")[:120].decode(errors="replace")
    except OSError:
        return str(pid)


def reap(pids: set[int]) -> None:
    """SIGKILL each pid's process group (just the pid if it shares ours), then reap."""
    own = os.getpgrp()
    for pid in pids:
        try:
            pgid = os.getpgid(pid)
            if pgid != own:
                os.killpg(pgid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    for pid in pids:
        try:
            os.waitpid(pid, os.WNOHANG)
        except OSError:
            pass


@pytest.fixture()
def sim_cleanup():
    """Kill every process the test spawned that is still running at teardown."""
    before = descendants()
    yield
    reap(descendants() - before)
