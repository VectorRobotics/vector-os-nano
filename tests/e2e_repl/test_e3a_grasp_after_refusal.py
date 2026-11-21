# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E3a after a refusal (SP1, charter 3.5): the live model's '把绿色的瓶子拿起来' turn.

Its first call, pick_top_down by object id, is refused (object_not_found) before the
robot moves; it then detects, grasps with perception_grasp and verifies. A refused call
is not an action: the verdict is GROUNDED, the goal proven, and the refusal is shown.
"""
from __future__ import annotations

import pytest

from tests.e2e_repl import LAUNCH, LAUNCH_STEPS
from tests.harness.pty_cli import run_repl_session

_TARGET = "pickable_bottle_green"
_SCRIPT = {"steps": LAUNCH_STEPS + [
    {"when": "把绿色的瓶子拿起来", "tool_calls": [
        {"name": "pick_top_down", "input": {"object_id": _TARGET}}]},
    {"when": "object_not_found", "tool_calls": [{"name": "detect", "input": {"query": "green bottle"}}]},
    {"when": "", "tool_calls": [{"name": "perception_grasp", "input": {"query": "绿色的瓶子"}}]},
    {"when": "", "tool_calls": [{"name": "verify", "input": {
        "expr": f"holding_object('{_TARGET}')", "goal_id": "grasp_green_bottle"}}]},
    {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
]}


@pytest.mark.sim
@pytest.mark.cli_main
def test_e3a_a_refused_pick_then_a_grasp_is_grounded() -> None:
    r = run_repl_session([LAUNCH, "把绿色的瓶子拿起来"], tool_script=_SCRIPT, answers=["y"], timeout_sec=420)
    text = " ".join(r.text.split())
    assert not r.timed_out, text[-3000:]
    assert "refused pick_top_down: Cannot locate target object" in text, text[-3000:]
    assert f"verify holding_object('{_TARGET}') ✓ (actor=CAUSED)" in text, text[-3000:]
    assert "verdict GROUNDED verified=True" in text, text[-3000:]
    assert "goal=proven" in text and f"goal proven: holding_object('{_TARGET}')" in text, text[-3000:]
