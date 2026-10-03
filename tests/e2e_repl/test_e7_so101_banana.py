# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E7 so101: '启动机械臂仿真 无窗口' then '抓起香蕉' -> GROUNDED holding_object('banana')."""
from __future__ import annotations

import pytest

from tests.harness.pty_cli import run_repl_session

_SCRIPT = {"steps": [
    {"when": "启动机械臂仿真", "tool_calls": [
        {"name": "start_simulation", "input": {"sim_type": "arm", "gui": False}}]},
    {"when": "", "is_error": False, "text": "机械臂仿真已启动", "tool_calls": []},
    {"when": "抓起香蕉", "tool_calls": [
        {"name": "pick", "input": {"object_label": "banana", "mode": "hold"}}]},
    {"when": "", "tool_calls": [{"name": "verify", "input": {"expr": "holding_object('banana')"}}]},
    {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
]}


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_e7_nl_launch_arm_then_grasp_banana_grounded() -> None:
    r = run_repl_session(["启动机械臂仿真 无窗口", "抓起香蕉"], tool_script=_SCRIPT,
                         answers=["y"], turn_timeout_sec=180)
    text = r.text
    assert not r.timed_out, text[-3000:]
    assert "sim start arm ok" in text, f"NL launch did not start the arm sim:\n{text[-3000:]}"
    assert "pick → verify holding_object('banana') ✓ (actor=CAUSED)" in text, text[-3000:]
    assert "verdict GROUNDED verified=True (1/1 grounded)" in text, text[-3000:]
