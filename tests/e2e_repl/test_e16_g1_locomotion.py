# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""G1 through the real CLI, scripted producer; physical room/point audit is independent."""
import json
import re

import pytest

from tests.harness.pty_cli import run_repl_session

LAUNCH = "启动 g1 仿真 无窗口"


def action(when, name, args, proof):
    return [
        {"when": when, "tool_calls": [{"name": name, "input": args}]},
        {"when": "", "tool_calls": [{"name": "verify", "input": {"expr": proof}}]},
        {"when": "", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]


SCRIPT = {"steps": [
    {"when": LAUNCH, "tool_calls": [{"name": "start_simulation", "input": {"embodiment": "g1", "gui": False}}]},
    {"when": "", "text": "g1 已启动", "tool_calls": []},
    *action("站好", "stand", {}, "base_standing()"),
    *action("往前走一米", "walk", {"direction": "forward", "distance": 1.0, "speed": .4},
            "at_position(11.0, 3.0, 0.5)"),
    *action("向左转90度", "turn", {"direction": "left", "angle": 90}, "facing(1.5707963267948966)"),
    *action("去厨房", "navigate", {"x": 17.0, "y": 1.7}, "in_room('kitchen')"),
]}


@pytest.mark.sim
@pytest.mark.cli_main
def test_stand_walk_turn_and_room_are_honest(g1_assets):
    r = run_repl_session([LAUNCH, "站好", "往前走一米", "向左转90度", "去厨房"],
                         tool_script=SCRIPT, timeout_sec=240,
                         extra_env={"VECTOR_SIM_LOCKSTEP": "1"})
    text = " ".join(r.text.split())
    assert not r.timed_out and r.exit_code == 0, text[-4000:]
    assert "gait=torchscript" in text
    # Already standing at launch: a state pass must not claim a new physical effect.
    assert "stand → verify base_standing() ✓ (actor=UNCAUSED)" in text
    for proof in ("at_position(11.0, 3.0, 0.5)", "facing(1.5707963267948966)", "in_room('kitchen')"):
        assert f"verify {proof} ✓ (actor=CAUSED)" in text, text[-4000:]
    assert "goal proven: in_room('kitchen')" in text


@pytest.mark.sim
@pytest.mark.cli_main
def test_typed_stop_and_ctrl_c_interrupt_walking_then_navigation_recovers(g1_assets, tmp_path):
    log = tmp_path / "calls.jsonl"
    script = {"steps": SCRIPT["steps"][:2] + [
        {"when": "后退五米", "tool_calls": [
            {"name": "walk", "input": {"direction": "backward", "distance": 5, "speed": .3}}]},
        *action("检查站立", "verify", {"expr": "base_standing()"}, "base_standing()"),
        {"when": "往前走五米", "tool_calls": [
            {"name": "walk", "input": {"direction": "forward", "distance": 5, "speed": .3}}]},
        *action("检查站立", "verify", {"expr": "base_standing()"}, "base_standing()"),
        *action("走到坐标 (9.0, 3.0)", "navigate", {"x": 9., "y": 3.}, "at_position(9.0, 3.0)"),
    ]}
    r = run_repl_session(
        [LAUNCH, "后退五米", (3., "停"), "检查站立", "往前走五米", (3., "\x03"),
         "检查站立", "走到坐标 (9.0, 3.0)"], tool_script=script, timeout_sec=260,
        extra_env={"VECTOR_FAKE_LLM_LOG": str(log)})
    text = " ".join(r.text.split())
    assert not r.timed_out and r.exit_code == 0, text[-5000:]
    stops = re.findall(r"■ stopped \((\d+) ms, no LLM call\)", text)
    assert len(stops) == 2 and max(map(int, stops)) < 100, text
    speeds = re.findall(
        r"base ([\d.]+) m/s at 0\.2 sim-s, peak ([\d.]+) m/s to 1\.0 sim-s, ([\d.]+) m/s at 1\.0 sim-s", text)
    assert len(speeds) == 2, text
    # The learned biped balances by stepping in place; full-cycle drift is checked
    # separately by test_learned_base. Here reject a runaway or a stop lurch.
    assert all(float(a) < .35 and float(peak) < .4 and float(b) < .2 for a, peak, b in speeds), speeds
    assert text.count("walk → aborted") == 2 and "Interrupted" in text, text
    assert text.count("verify base_standing() ✓") >= 2, text
    assert "verify at_position(9.0, 3.0) ✓ (actor=CAUSED)" in text, text
    assert "verdict GROUNDED verified=True" in text, text
    calls = [json.loads(line)["obs"].strip() for line in log.read_text().splitlines()]
    assert "停" not in calls
