# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""W2.2 — RUN_ID watchdog orphan sweep (stdlib /proc, no psutil).

A run tags itself (and therefore every descendant, via env inheritance) with
``VECTOR_RUN_ID``. After an abnormal exit (kill -9 of the parent), the sweep
finds surviving descendants by reading ``/proc/<pid>/environ`` and reaps them
TERM→KILL. Linux-only by design (no /proc on macOS — owner-verified there).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from vector_os_nano.vcli.run_registry import _pid_alive
from vector_os_nano.vcli.watchdog import find_tagged_pids, sweep_orphans

pytestmark = pytest.mark.skipif(
    not os.path.isdir("/proc"), reason="watchdog sweep is /proc-based (Linux)"
)

_RUN_ID = "wdtest-xyz"


def _spawn_tagged_tree(run_id: str) -> tuple[subprocess.Popen, int]:
    """Spawn a TAGGED child that spawns its own grandchild, both sleeping.
    Returns (child_proc, grandchild_pid)."""
    code = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(p.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    env = {**os.environ, "VECTOR_RUN_ID": run_id}
    child = subprocess.Popen(
        [sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, text=True
    )
    grandchild_pid = int(child.stdout.readline().strip())
    return child, grandchild_pid


class TestFindTaggedPids:
    def test_finds_child_and_grandchild(self) -> None:
        child, gc_pid = _spawn_tagged_tree(_RUN_ID)
        try:
            found = find_tagged_pids(_RUN_ID)
            assert child.pid in found
            assert gc_pid in found
            assert os.getpid() not in found  # this test process is NOT tagged
        finally:
            child.kill()
            os.kill(gc_pid, 9)

    def test_unknown_run_id_finds_nothing(self) -> None:
        assert find_tagged_pids("no-such-run-id-000") == ()


class TestSweepOrphans:
    def test_sweeps_grandchild_after_parent_sigkill(self) -> None:
        child, gc_pid = _spawn_tagged_tree(_RUN_ID)
        child.kill()  # SIGKILL the parent — the grandchild is now an orphan
        child.wait(timeout=5)
        assert _pid_alive(gc_pid)  # the orphan survived the parent's death
        swept = sweep_orphans(_RUN_ID, term_timeout=1.0)
        assert gc_pid in swept
        deadline = time.time() + 5
        while time.time() < deadline and _pid_alive(gc_pid):
            time.sleep(0.05)
        assert not _pid_alive(gc_pid)

    def test_sweep_with_no_orphans_is_noop(self) -> None:
        assert sweep_orphans("no-such-run-id-000") == ()

    def test_sweep_never_kills_the_caller(self) -> None:
        """Even if the caller itself carries the tag (daemon case), the sweep
        must exclude its own process."""
        os.environ["VECTOR_RUN_ID"] = _RUN_ID
        try:
            swept = sweep_orphans(_RUN_ID)
            assert os.getpid() not in swept
            assert _pid_alive(os.getpid())
        finally:
            os.environ.pop("VECTOR_RUN_ID", None)
