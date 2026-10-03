# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The PTY harness itself: one deadline, process-group kill, event-driven REPL input.

Regressions: tests:pty-harness-timeouts-and-kill (the turn timeout was waited twice
and only the direct child was killed, so tests ran a machine-wide ``rosm nuke``) and
tests:repl-session-wallclock (lines were typed on fixed sleeps, and reading stopped
before the child exited).
"""
from __future__ import annotations

import os
import sys
import textwrap
import time

import pytest

from tests.harness import pty_cli


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as fh:
        return fh.read().rsplit(")", 1)[1].split()[0] != "Z"


def test_timeout_is_one_deadline_and_kills_the_whole_group(tmp_path, monkeypatch) -> None:
    # A stand-in "cli" that spawns a grandchild, never prints a verdict, never exits.
    fake = tmp_path / "fake_python"
    pidfile = tmp_path / "grandchild.pid"
    fake.write_text(f"#!/bin/sh\nsleep 300 &\necho $! > {pidfile}\nwait\n")
    fake.chmod(0o755)
    monkeypatch.setattr(pty_cli.sys, "executable", str(fake))

    t0 = time.monotonic()
    with pytest.raises(AssertionError, match="no VECTOR_VERDICT"):
        pty_cli.run_cli_turn("hi", timeout_sec=2.0, cwd=tmp_path)
    elapsed = time.monotonic() - t0

    assert elapsed < 4.0, f"timeout waited {elapsed:.1f}s for a 2s budget (double wait?)"
    grandchild = int(pidfile.read_text())
    deadline = time.monotonic() + 2.0
    while _alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(grandchild), "the child's process group survived the timeout"


_SLOW_REPL = textwrap.dedent('''
    import sys, time
    while True:
        print("vector> ", end="", flush=True)
        if not sys.stdin.readline():
            break
        time.sleep(1.5)  # every turn is silent and slow
''')


def test_repl_session_is_one_deadline_not_one_per_step(tmp_path, monkeypatch) -> None:
    # Four 1.5 s turns against a 2 s budget: a per-step deadline would let all of
    # them (and the quit) through in ~7 s; one session deadline stops at ~2 s.
    script = tmp_path / "slow_repl.py"
    script.write_text(_SLOW_REPL)
    fake = tmp_path / "fake_python"
    fake.write_text(f"#!/bin/sh\nexec {sys.executable} {script}\n")
    fake.chmod(0o755)
    monkeypatch.setattr(pty_cli.sys, "executable", str(fake))

    t0 = time.monotonic()
    r = pty_cli.run_repl_session(["a", "b", "c", "d"], timeout_sec=2.0, cwd=tmp_path)
    elapsed = time.monotonic() - t0

    assert r.timed_out, r.text
    assert elapsed < 3.5, f"session ran {elapsed:.1f}s on a 2s budget (a deadline per step?)"


_FAKE_REPL = textwrap.dedent('''
    import sys, time
    print("banner"); print("vector> ", end="", flush=True)
    line = sys.stdin.readline().strip()
    time.sleep(1.5)  # a silent, slow turn: the echo must not count as a fresh prompt
    print("Allow? [y/n/a] (y): ", end="", flush=True)
    answer = sys.stdin.readline().strip()
    print(f"ran {line} with {answer}"); print("vector> ", end="", flush=True)
    sys.stdin.readline()
    print("bye", flush=True)
''')


def test_repl_input_waits_for_prompt_and_permission_events(tmp_path) -> None:
    proc, fd = pty_cli._spawn([sys.executable, "-c", _FAKE_REPL], dict(os.environ), tmp_path)
    tty = pty_cli._Pty(fd)
    try:
        assert tty.wait_input(time.monotonic() + 10, "") == "prompt"
        tty.send("go")
        t0 = time.monotonic()
        assert tty.wait_input(time.monotonic() + 10, "go") == "permission"
        assert time.monotonic() - t0 >= 1.4, "the typed echo was mistaken for readiness"
        tty.send("y")
        assert tty.wait_input(time.monotonic() + 10, "") == "prompt"
        assert "ran go with y" in tty.text()
        tty.send("quit")
        deadline = time.monotonic() + 10
        while not tty.eof and time.monotonic() < deadline:
            tty.pump(0.2)
        assert "bye" in tty.text(), "output after the last line must be read to EOF"
    finally:
        pty_cli._kill_group(proc)
        os.close(fd)
