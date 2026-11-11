# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""SIGTERM / SIGHUP to a bare ``vector-cli`` mid-turn take the normal exit path: the
turn is stopped, the session is saved, and no child the session started survives."""
from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

import pytest

from tests.harness.pty_cli import _Pty, _child_env, _kill_group, _spawn, _write_tool_script

_REPO = Path(__file__).resolve().parents[3]


def _alive(marker: str) -> list[int]:
    out = []
    for pid in os.listdir("/proc"):
        try:
            if pid.isdigit() and marker in Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode():
                out.append(int(pid))
        except OSError:
            continue
    return out


@pytest.mark.cli_main
@pytest.mark.timeout(120)
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_a_signal_mid_turn_leaves_no_child(sig: int) -> None:
    marker = f"9{sig}7.25"  # a sleep whose argv no other process on the box has
    script = _write_tool_script({"steps": [
        {"when": "run the long job", "tool_calls": [{"name": "bash", "input": {"command": f"sleep {marker}"}}]},
    ]})
    env, home = _child_env(live=False, home_prefix="vector_pty_sig_")
    env.update(VECTOR_FAKE_LLM_TOOLS=script, TERM="xterm-256color", PROMPT_TOOLKIT_NO_CPR="1")
    proc, fd = _spawn([sys.executable, "-m", "vector_os_nano.vcli.cli", "--no-permission"], env, _REPO)
    tty = _Pty(fd)
    try:
        deadline = time.monotonic() + 60
        while "vector>" not in tty.text() and time.monotonic() < deadline:
            tty.pump(0.2)
        tty.send("run the long job")
        while not _alive(marker) and time.monotonic() < deadline:
            tty.pump(0.2)
        assert _alive(marker), tty.text()[-2000:]
        os.kill(proc.pid, sig)  # the REPL alone, never its group
        proc.wait(timeout=30)
        while not tty.eof and time.monotonic() < deadline:
            tty.pump(0.2)
        assert proc.returncode == 128 + sig, tty.text()[-2000:]
        assert "Session saved" in tty.text()
        time.sleep(0.5)
        assert _alive(marker) == []
    finally:
        _kill_group(proc)
        for pid in _alive(marker):
            os.kill(pid, signal.SIGKILL)
        os.close(fd)
        os.unlink(script)
        import shutil

        shutil.rmtree(home, ignore_errors=True)
