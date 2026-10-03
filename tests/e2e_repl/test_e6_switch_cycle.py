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
    start = {"embodiment": embodiment, "gui": False}
    return [
        {"when": say, "tool_calls": [{"name": "stop_simulation", "input": {}},
                                     {"name": "start_simulation", "input": start}]},
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
@pytest.mark.xfail(strict=True, reason="acceptance path = subprocess bridge until WP03")
def test_e6_switch_cycle_every_start_succeeds() -> None:
    r = run_repl_session([say for say, _ in _CYCLE], tool_script=_SCRIPT,
                         answers=["y"] * 8, timeout_sec=600)
    text = r.text
    assert not r.timed_out, text[-3000:]
    assert text.count("sim start") >= len(_CYCLE), text[-3000:]
    assert "already running" not in text and " fail " not in text, text[-3000:]
    # Each go2 start must report readiness (the subprocess bridge says 'ok' even
    # when the sim never came up). WP03 owns the ready-line format, as in E1.
    go2_starts = sum(1 for _, e in _CYCLE if e.startswith("go2"))
    assert len(re.findall(READY, text)) >= go2_starts, text[-3000:]
