# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""SO-101 through the bare REPL: home, cancelled grasp, home, camera-driven grasp.

The independent hold oracle requires both fingers, attachment and lift. The stopped
attempt has no success claim; the next command must still run and be caused.
"""
from __future__ import annotations

import re
import copy

import pytest

from tests.harness.pty_cli import run_repl_session

_SCRIPT = {"steps": [
    {"when": "启动机械臂仿真", "tool_calls": [
        {"name": "start_simulation", "input": {"embodiment": "so101", "gui": False}}]},
    {"when": "", "text": "机械臂仿真已启动", "tool_calls": []},
    {"when": "回到初始位置", "tool_calls": [{"name": "home", "input": {}}]},
    {"when": "", "tool_calls": [{"name": "verify", "input": {"expr": "arm_at_home()"}}]},
    {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    {"when": "尝试抓起香蕉", "tool_calls": [{"name": "grasp", "input": {"query": "banana"}}]},
    {"when": "回到初始位置", "tool_calls": [{"name": "home", "input": {}}]},
    {"when": "", "tool_calls": [{"name": "verify", "input": {"expr": "arm_at_home()"}}]},
    {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    {"when": "抓起香蕉", "tool_calls": [{"name": "grasp", "input": {"query": "banana"}}]},
    {"when": "", "tool_calls": [{"name": "verify", "input": {"expr": "holding_object('banana')"}}]},
    {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
]}
_STOPPED = re.compile(r"■ stopped \((\d+) ms, no LLM call\)")
_HOME_CAUSED = "home → verify arm_at_home() ✓ (actor=CAUSED)"


@pytest.fixture(scope="module", params=["mujoco", "pybullet"])
def e7(request):
    script = copy.deepcopy(_SCRIPT)
    script["steps"][0]["tool_calls"][0]["input"]["backend"] = request.param
    # The initial view equals home; let the observed grasp progress leave that preparation phase.
    r = run_repl_session(["启动机械臂仿真 无窗口", "回到初始位置", "尝试抓起香蕉", ("grasp 4s", "停"), "回到初始位置", "抓起香蕉"],
                         tool_script=script, answers=["y"], timeout_sec=240)
    assert f"backend={request.param}" in r.text
    return r, r.text


def _section(text: str, start: str, end: str | None = None, nth: int = 0) -> str:
    i = -1
    for _ in range(nth + 1):
        i = text.find(f"vector> {start}", i + 1)
        assert i != -1, f"{start!r} (#{nth}) was never typed at a prompt:\n{text[-3000:]}"
    j = text.find(f"vector> {end}", i + 1) if end else -1
    return text[i: j if j != -1 else None]


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_e7_nl_launch_arm_then_home_is_grounded_and_caused(e7) -> None:
    r, text = e7
    assert not r.timed_out, text[-3000:]
    assert re.search(r"Started so101 simulation, ready in", text), f"NL launch did not start the arm sim:\n{text[-3000:]}"
    home = _section(text, "回到初始位置", "尝试抓起香蕉")
    assert _HOME_CAUSED in home and "verdict GROUNDED verified=True" in home, home


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_e7_grasp_banana_is_grounded_by_contact(e7) -> None:
    _r, text = e7
    pick = _section(text, "抓起香蕉")
    assert "grasp → verify holding_object('banana') ✓ (actor=CAUSED)" in pick, pick
    assert "verdict GROUNDED verified=True (1/1 grounded)" in pick, pick
    assert "goal proven: holding_object('banana')" in pick, pick   # 抓起香蕉: a closed grasp form


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_e7_a_typed_stop_preempts_the_arm_mid_move(e7) -> None:
    r, text = e7
    attempt = _section(text, "尝试抓起香蕉", "回到初始位置", nth=0)
    m = _STOPPED.search(attempt)
    assert m and int(m.group(1)) < 100 and ("cancelled" in attempt or "aborted" in attempt), attempt
    # The grasp was interrupted after leaving the view pose.
    again = _section(text, "回到初始位置", "抓起香蕉", nth=1)
    assert _HOME_CAUSED in again and "verdict GROUNDED verified=True" in again, again
    assert r.exit_code == 0
