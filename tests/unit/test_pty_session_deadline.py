# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""S3aV-08: a REPL session's deadline ends inside its test's pytest-timeout budget,
so a stalled session returns its transcript (E5 hung past the gate's 300 s with a
420 s session deadline: pytest-timeout killed it and nothing was left to read)."""
from __future__ import annotations

import os
import sys
import time

import pytest
from pytest_timeout import get_env_settings

from tests.harness import pty_cli


@pytest.mark.timeout(60)
def test_a_stalled_session_returns_its_transcript_inside_the_pytest_budget(monkeypatch) -> None:
    spawn = pty_cli._spawn
    stall = [sys.executable, "-c", "print('booting', flush=True); import time; time.sleep(600)"]
    monkeypatch.setattr(pty_cli, "_spawn", lambda argv, env, cwd: spawn(stall, env, cwd))
    monkeypatch.setenv("PYTEST_TIMEOUT", "24")  # a 4 s session after the teardown share
    t0 = time.monotonic()
    r = pty_cli.run_repl_session(["停下"], timeout_sec=420)
    assert r.timed_out and "booting" in r.transcript, r.transcript
    assert time.monotonic() - t0 < 20


def test_the_session_budget_is_the_timeout_pytest_runs_under(request) -> None:
    """The harness reads PYTEST_TIMEOUT; the gate must run pytest under that same value
    (a bare --timeout without the export would leave every session's deadline uncut)."""
    env = os.environ.get("PYTEST_TIMEOUT")
    ran = get_env_settings(request.config).timeout
    assert (float(ran) if ran is not None else None) == (float(env) if env else None), (ran, env)
