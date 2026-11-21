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
from vector_os_nano.schemas.world import WORLDS_DIR
_LAYOUT_YAML = WORLDS_DIR / "apartment" / "world.yaml"

import copy
import pytest

from tests.harness.pty_cli import run_repl_session
from vector_os_nano.schemas.world import load_world
from vector_os_nano.sim.nav.coverage import room_layout

_ROOMS = room_layout(str(_LAYOUT_YAML))["rooms"]
_K, _D = _ROOMS["kitchen"], _ROOMS["dining_room"]
_S = tuple(load_world("apartment").spawn[:2])
_LAUNCH = "启动 go2 仿真 无窗口"


def _leg(when: str, xy: tuple[float, float], expr: str = "", room: str = "") -> list[dict]:
    """Navigate to *xy*, then verify *expr* (default: arriving at *xy*). A room goal's
    contract is in_room(<room>) (K1), so its leg verifies the room."""
    x, y = xy
    return [
        {"when": when, "tool_calls": [{"name": "go_to_room" if room else "navigate",
                                      "input": {"room": room} if room else {"x": x, "y": y}}]},
        {"when": f"go_to_room({room}) ->" if room else f"(target ({x}, {y}))", "tool_calls": [
            {"name": "verify", "input": {"expr": expr or f"at_position({x}, {y})"}}]},
        {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]


_SCRIPT = {"steps": [
    {"when": "启动 go2 仿真", "tool_calls": [
        {"name": "start_simulation", "input": {"embodiment": "go2", "gui": False}}]},
    {"when": "", "text": "go2 仿真已启动", "tool_calls": []},
    *_leg("去厨房", _K, "in_room('kitchen')", "kitchen"), *_leg("回到起点", _S),
    *_leg("去餐厅", _D, "in_room('dining_room')", "dining_room"),
]}


def _grounded(text: str, xy: tuple[float, float]) -> bool:
    return f"verify at_position({xy[0]}, {xy[1]}) ✓ (actor=CAUSED)" in text


def _in_room(text: str, room: str) -> bool:
    return f"verify in_room('{room}') ✓ (actor=CAUSED)" in text and f"goal proven: in_room('{room}')" in text


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.parametrize("trial,backend", [(0, "mujoco"), (1, "mujoco"), (2, "mujoco"), (0, "pybullet")])
def test_e8_kitchen_and_dining_room_ground(trial: int, backend: str) -> None:
    # Lockstep (WP06): physics advances only inside the follower's sim-time waits,
    # so every trial replays the same ground truth, as fast as the box allows.
    script = copy.deepcopy(_SCRIPT)
    script["steps"][0]["tool_calls"][0]["input"]["backend"] = backend
    r = run_repl_session([_LAUNCH, "去厨房", "回到起点", "去餐厅"], tool_script=script,
                         answers=["y"], timeout_sec=420, extra_env={"VECTOR_SIM_LOCKSTEP": "1"})
    text = r.text
    assert not r.timed_out, text[-4000:]
    assert f"backend={backend}" in text
    kitchen = text[text.rfind("vector> 去厨房"):text.rfind("vector> 回到起点")]
    dining = text[text.rfind("vector> 去餐厅"):]
    assert _in_room(kitchen, "kitchen") and "verdict GROUNDED verified=True" in kitchen, kitchen[-3000:]
    assert _in_room(dining, "dining_room") and "verdict GROUNDED verified=True" in dining, dining[-3000:]


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
def test_e8_ctrl_c_mid_navigate_stops_and_the_repl_goes_on() -> None:
    r = run_repl_session([_LAUNCH, "去书房", (6.0, "\x03"), "回到起点"], tool_script=_CTRL_C_SCRIPT,
                         answers=["y"], timeout_sec=300)
    text = r.text
    assert not r.timed_out, text[-4000:]
    study = text[text.rfind("vector> 去书房"):text.rfind("vector> 回到起点")]
    assert "Interrupted" in study and "Traceback" not in study, study[-3000:]
    back = text[text.rfind("vector> 回到起点"):]
    assert _grounded(back, _S) and "verdict GROUNDED verified=True" in back, back[-3000:]
