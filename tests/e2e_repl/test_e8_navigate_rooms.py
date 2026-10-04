# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E8 room navigation, bare REPL go2: '去厨房' and '去餐厅' GROUNDED, N=3 each.

Both rooms were unreachable before WP08: their layout points sat inside the kitchen
island and the dining table, and the under-inflating planner refused the goal.
Each trial is one session: launch, 去厨房, 回到起点, 去餐厅 (every leg starts where
its verify is false). The room points are read from the layout the product uses.
A Ctrl-C mid-navigate (third session) leaves the REPL alive with the base stopped:
the next navigate starts from where the dog halted and still grounds.
"""
from __future__ import annotations

import pytest

from tests.harness.pty_cli import run_repl_session
from vector_os_nano.embodiments.config import load_embodiment_config
from vector_os_nano.sim.nav.coverage import room_layout

_ROOMS = room_layout()["rooms"]
_K, _D = _ROOMS["kitchen"], _ROOMS["dining_room"]
_S = tuple(load_embodiment_config("go2").spawn.xy)
_LAUNCH = "启动 go2 仿真 无窗口"


def _leg(when: str, xy: tuple[float, float]) -> list[dict]:
    x, y = xy
    return [
        {"when": when, "tool_calls": [{"name": "navigate", "input": {"x": x, "y": y}}]},
        {"when": f"(target ({x}, {y}))", "tool_calls": [
            {"name": "verify", "input": {"expr": f"at_position({x}, {y})"}}]},
        {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]


_SCRIPT = {"steps": [
    {"when": "启动 go2 仿真", "tool_calls": [
        {"name": "start_simulation", "input": {"embodiment": "go2", "gui": False}}]},
    {"when": "", "text": "go2 仿真已启动", "tool_calls": []},
    *_leg("去厨房", _K), *_leg("回到起点", _S), *_leg("去餐厅", _D),
]}


def _grounded(text: str, xy: tuple[float, float]) -> bool:
    return f"verify at_position({xy[0]}, {xy[1]}) ✓ (actor=CAUSED)" in text


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.parametrize("trial", range(3))
def test_e8_kitchen_and_dining_room_ground(trial: int) -> None:
    r = run_repl_session([_LAUNCH, "去厨房", "回到起点", "去餐厅"], tool_script=_SCRIPT,
                         answers=["y"], timeout_sec=420)
    text = r.text
    assert not r.timed_out, text[-4000:]
    kitchen = text[text.rfind("vector> 去厨房"):text.rfind("vector> 回到起点")]
    dining = text[text.rfind("vector> 去餐厅"):]
    assert _grounded(kitchen, _K) and "verdict GROUNDED verified=True" in kitchen, kitchen[-3000:]
    assert _grounded(dining, _D) and "verdict GROUNDED verified=True" in dining, dining[-3000:]


_CTRL_C_SCRIPT = {"steps": [
    {"when": "启动 go2 仿真", "tool_calls": [
        {"name": "start_simulation", "input": {"embodiment": "go2", "gui": False}}]},
    {"when": "", "text": "go2 仿真已启动", "tool_calls": []},
    # the study is a ~9 m leg: Ctrl-C lands mid-way
    {"when": "去书房", "tool_calls": [
        {"name": "navigate", "input": {"x": _ROOMS["study"][0], "y": _ROOMS["study"][1]}}]},
    *_leg("回到起点", _S),
]}


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.xfail(strict=True, reason=(
    "WP05: the REPL's native attempt runs outside its KeyboardInterrupt handler, so a "
    "Ctrl-C mid-turn exits the REPL (the follower does stop the dog first: "
    "tests/hardware/sim/test_go2_nav_stop.py)"))
def test_e8_ctrl_c_mid_navigate_stops_and_the_repl_goes_on() -> None:
    r = run_repl_session([_LAUNCH, "去书房", (6.0, "\x03"), "回到起点"], tool_script=_CTRL_C_SCRIPT,
                         answers=["y"], timeout_sec=300)
    text = r.text
    assert not r.timed_out, text[-4000:]
    study = text[text.rfind("vector> 去书房"):text.rfind("vector> 回到起点")]
    assert "Interrupted" in study and "Traceback" not in study, study[-3000:]
    back = text[text.rfind("vector> 回到起点"):]
    assert _grounded(back, _S) and "verdict GROUNDED verified=True" in back, back[-3000:]
