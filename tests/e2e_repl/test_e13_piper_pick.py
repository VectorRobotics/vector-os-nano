# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E13 piper: 'start the piper simulation' then 'pick up the red box' -> goal proven.

The 6-DoF arm alone on the workbench, every device a generic driver (sim/drivers), the
scene composed from the world manifest (sim/scene/compose.py). The grasp is perceived
(colour blob + depth through the body camera, no ground truth) and graded on the world
side: held by every finger, attached, lifted."""
from __future__ import annotations

import re
import copy

import pytest

from tests.harness.pty_cli import run_repl_session

_TARGET = "red_box"
_SCRIPT = {"steps": [
    {"when": "start the piper simulation", "tool_calls": [
        {"name": "start_simulation", "input": {"embodiment": "piper", "gui": False}}]},
    {"when": "", "is_error": False, "text": "the piper simulation is up", "tool_calls": []},
    {"when": "pick up the red box", "tool_calls": [
        {"name": "grasp", "input": {"query": "red box"}}]},
    {"when": "", "tool_calls": [{"name": "verify", "input": {"expr": f"holding_object('{_TARGET}')"}}]},
    {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
]}


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.parametrize("backend", ["mujoco", "pybullet"])
def test_e13_nl_launch_piper_then_pick_the_red_box_goal_proven(backend) -> None:
    script = copy.deepcopy(_SCRIPT)
    script["steps"][0]["tool_calls"][0]["input"]["backend"] = backend
    r = run_repl_session(["start the piper simulation", "pick up the red box"], tool_script=script,
                         answers=["y"], timeout_sec=240)
    text = r.text
    assert not r.timed_out, text[-3000:]
    assert re.search(r"Started piper simulation, ready in [0-9.]+s: .*; Arm", text), text[-3000:]
    assert f"backend={backend}" in text
    assert f"verify holding_object('{_TARGET}') ✓ (actor=CAUSED)" in text, text[-3000:]
    assert "verdict GROUNDED verified=True" in text, text[-3000:]
    assert f"goal proven: holding_object('{_TARGET}')" in text, text[-3000:]  # a closed grasp form
