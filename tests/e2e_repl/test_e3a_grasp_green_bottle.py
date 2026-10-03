# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E3a grasp in view: NL launch go2_piper, then '拿起绿色的瓶子' -> GROUNDED holding."""
from __future__ import annotations

import pytest

from tests.e2e_repl import LAUNCH, LAUNCH_STEPS
from tests.harness.pty_cli import run_repl_session

_TARGET = "pickable_bottle_green"
_SCRIPT = {"steps": LAUNCH_STEPS + [
    {"when": "拿起绿色的瓶子", "tool_calls": [
        {"name": "perception_grasp", "input": {"query": "绿色的瓶子"}}]},
    {"when": "", "tool_calls": [{"name": "verify", "input": {"expr": f"holding_object('{_TARGET}')"}}]},
    {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
]}


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.xfail(strict=True, reason="acceptance path = subprocess bridge until WP03")
def test_e3a_grasp_green_bottle_grounded() -> None:
    r = run_repl_session([LAUNCH, "拿起绿色的瓶子"], tool_script=_SCRIPT, answers=["y"], turn_timeout_sec=240)
    text = r.text
    assert not r.timed_out, text[-3000:]
    assert f"verify holding_object('{_TARGET}') ✓ (actor=CAUSED)" in text, text[-3000:]
    assert "verdict GROUNDED verified=True" in text, text[-3000:]
