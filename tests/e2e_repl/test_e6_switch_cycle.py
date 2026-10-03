# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E6 switch: go2_piper -> go2 -> so101 -> go2_piper by NL in ONE REPL session.

Every start must succeed (no 'already running', no error). The process-group leak
half of E6 is the suite-wide check in tests/conftest.py (pytest is a subreaper).
"""
from __future__ import annotations

import re

import pytest

from tests.e2e_repl import READY
from tests.harness.pty_cli import run_repl_session


def _switch(say: str, embodiment: str) -> list[dict]:
    # start_simulation alone switches (it stops the live session first, after validating).
    return [
        {"when": say, "tool_calls": [{"name": "start_simulation",
                                      "input": {"embodiment": embodiment, "gui": False}}]},
        {"when": "", "text": f"switched: {say}", "tool_calls": []},
    ]


_CYCLE = [
    ("切换到 go2 带机械臂 仿真 无窗口", "go2_piper"),
    ("切换到 go2 仿真 无窗口", "go2"),
    ("切换到机械臂仿真 无窗口", "so101"),
    ("再切换到 go2 带机械臂 仿真 无窗口", "go2_piper"),
]
_SCRIPT = {"steps": [s for say, e in _CYCLE for s in _switch(say, e)]}


@pytest.mark.sim
@pytest.mark.cli_main
def test_e6_switch_cycle_every_start_succeeds() -> None:
    r = run_repl_session([say for say, _ in _CYCLE], tool_script=_SCRIPT,
                         answers=["y"] * 8, timeout_sec=600)
    text = r.text
    assert not r.timed_out, text[-3000:]
    assert "already running" not in text and "Failed to start" not in text, text[-3000:]
    # Every start, in order, reports the SimSession readiness line for its embodiment.
    assert re.findall(READY, text) == [e for _, e in _CYCLE], text[-3000:]
