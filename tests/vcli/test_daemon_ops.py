# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""W2.1 — daemon ops: stop escalation, status listing, log tail.

No double-fork in-process (pytest must survive): stop/status/log are tested
against REAL spawned subprocesses registered by hand; daemonize itself gets a
subprocess smoke test driving the real code path end-to-end.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

from vector_os_nano.vcli.daemon import stop_run, tail_log
from vector_os_nano.vcli.run_registry import RunEntry, RunRegistry


def _spawn_sleeper(ignore_sigterm: bool = False) -> subprocess.Popen:
    code = (
        "import signal, time\n"
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_sigterm else "")
        + "time.sleep(60)\n"
    )
    return subprocess.Popen([sys.executable, "-c", code])


def _register(reg: RunRegistry, proc: subprocess.Popen, run_id: str, log_dir: str) -> None:
    reg.register(
        RunEntry(
            run_id=run_id,
            pid=proc.pid,
            started_at=time.time(),
            world="dev",
            scenario="",
            log_dir=log_dir,
            argv=("test",),
        )
    )


class TestStopRun:
    def test_sigterm_stops_and_cleans_registry(self, tmp_path) -> None:
        reg = RunRegistry(root=tmp_path)
        proc = _spawn_sleeper()
        _register(reg, proc, "r1", str(tmp_path))
        assert stop_run(reg, "r1", term_timeout=5.0) is True
        proc.wait(timeout=5)
        assert reg.get("r1") is None
        assert proc.returncode != 0  # killed, not natural exit

    def test_sigkill_escalation_when_sigterm_ignored(self, tmp_path) -> None:
        reg = RunRegistry(root=tmp_path)
        proc = _spawn_sleeper(ignore_sigterm=True)
        _register(reg, proc, "r2", str(tmp_path))
        time.sleep(0.3)  # let the child install its SIG_IGN handler
        assert stop_run(reg, "r2", term_timeout=0.5) is True
        proc.wait(timeout=5)
        assert reg.get("r2") is None

    def test_stop_unknown_run_returns_false(self, tmp_path) -> None:
        assert stop_run(RunRegistry(root=tmp_path), "nope") is False

    def test_stop_dead_run_cleans_entry(self, tmp_path) -> None:
        reg = RunRegistry(root=tmp_path)
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        _register(reg, proc, "r3", str(tmp_path))
        assert stop_run(reg, "r3") is True  # already dead counts as stopped
        assert reg.get("r3") is None


class TestTailLog:
    def test_tail_returns_last_lines(self, tmp_path) -> None:
        log = tmp_path / "run.log"
        log.write_text("".join(f"line{i}\n" for i in range(50)))
        out = tail_log(str(log), lines=10)
        assert out.splitlines()[-1] == "line49"
        assert len(out.splitlines()) == 10

    def test_tail_missing_file_is_empty(self, tmp_path) -> None:
        assert tail_log(str(tmp_path / "absent.log")) == ""


class TestDaemonizeSmoke:
    def test_daemonize_detaches_registers_and_stops(self, tmp_path) -> None:
        """End-to-end on the REAL daemonize path, driven from a subprocess so
        the double-fork never touches the pytest process."""
        script = f"""
import sys, time
sys.path.insert(0, {repr(os.getcwd())})
from vector_os_nano.vcli.daemon import daemonize
from vector_os_nano.vcli.run_registry import RunRegistry
daemonize(run_id="smoke", registry_root={str(tmp_path)!r},
          log_dir={str(tmp_path)!r}, world="dev", scenario="")
time.sleep(30)
"""
        subprocess.run([sys.executable, "-c", script], timeout=15, check=True)
        reg = RunRegistry(root=tmp_path)
        deadline = time.time() + 5
        entry = None
        while time.time() < deadline:
            entry = reg.get("smoke")
            if entry is not None:
                break
            time.sleep(0.1)
        assert entry is not None, "daemonized child never registered"
        assert entry.pid != 0 and entry.pid != os.getpid()
        # The daemon must be stoppable and leave no residue.
        assert stop_run(reg, "smoke", term_timeout=5.0) is True
        assert reg.get("smoke") is None
        # Stdio was redirected to a log file inside log_dir.
        assert (tmp_path / "smoke.log").exists()
