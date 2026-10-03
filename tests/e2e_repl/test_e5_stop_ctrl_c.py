# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E5 stop: Ctrl-C mid-walk leaves the REPL alive, and the next command still grounds."""
from __future__ import annotations

import pytest

from tests.harness.pty_cli import run_repl_session

_SCRIPT = {"steps": [
    {"when": "启动 go2 仿真", "tool_calls": [
        {"name": "start_simulation", "input": {"sim_type": "go2", "with_arm": False, "gui": False}}]},
    {"when": "", "text": "go2 仿真已启动", "tool_calls": []},
    {"when": "往前走五米", "tool_calls": [
        {"name": "walk", "input": {"direction": "forward", "distance": 5.0, "speed": 0.3}}]},
    {"when": "往前走一米", "tool_calls": [
        {"name": "walk", "input": {"direction": "forward", "distance": 1.0, "speed": 0.3}}]},
    {"when": "", "is_error": False, "tool_calls": [
        {"name": "verify", "input": {"expr": "get_position()[0] > 10.0"}}]},
    {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
]}


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.xfail(strict=True, reason=(
    "WP05 (CancelToken, Ctrl-C never exits the REPL) on the WP03 in-process go2 sim"))
def test_e5_ctrl_c_mid_walk_keeps_repl_and_next_turn_works() -> None:
    r = run_repl_session(
        ["启动 go2 仿真 无窗口", "往前走五米", (3.0, "\x03"), "往前走一米"],
        tool_script=_SCRIPT, answers=["y"], timeout_sec=300,
    )
    text = r.text
    assert not r.timed_out, text[-3000:]
    assert "Interrupted" in text, f"Ctrl-C was not handled as an interrupt:\n{text[-3000:]}"
    after = text[text.rfind("往前走一米"):]
    assert "walk" in after and "verdict" in after, f"the next turn did not run:\n{after[-2000:]}"
