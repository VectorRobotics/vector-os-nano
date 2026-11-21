# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E12 recover by retry, bare REPL go2 (WP07 attempt semantics + ActorEpoch).

Turn 1 '走到坐标 (<kitchen>)': the scripted model first navigates to the WRONG room
(dining), so its at_position(kitchen) verify FAILs; it navigates again to the kitchen
and re-verifies the SAME predicate, which PASSes. The FAIL row stays on screen as a
superseded attempt and the turn is GROUNDED — before WP07 a recorded FAIL made every
recovered turn RAN. Turn 2 '回到起点': one navigate and NO model verify — the
navigate's own verify_hint auto-verifies the target and the in-process navigate step
is actor-CAUSED (D14's "navigate is RAN-only" no longer holds in-process).
"""
from __future__ import annotations
from vector_os_nano.schemas.world import WORLDS_DIR
_LAYOUT_YAML = WORLDS_DIR / "apartment" / "world.yaml"

import pytest

from tests.harness.pty_cli import run_repl_session
from vector_os_nano.schemas.world import load_world
from vector_os_nano.sim.nav.coverage import room_layout

_ROOMS = room_layout(str(_LAYOUT_YAML))["rooms"]
_K, _D = _ROOMS["kitchen"], _ROOMS["dining_room"]
_S = tuple(load_world("apartment").spawn[:2])
_GOAL = f"走到坐标 ({_K[0]},{_K[1]})"


def _nav(when: str, xy) -> dict:
    return {"when": when, "tool_calls": [{"name": "navigate", "input": {"x": xy[0], "y": xy[1]}}]}


def _verify(when: str, xy) -> dict:
    return {"when": when, "tool_calls": [{"name": "verify", "input": {"expr": f"at_position({xy[0]}, {xy[1]})"}}]}


_SCRIPT = {"steps": [
    {"when": "启动 go2 仿真", "tool_calls": [
        {"name": "start_simulation", "input": {"embodiment": "go2", "gui": False}}]},
    {"when": "", "text": "go2 仿真已启动", "tool_calls": []},
    _nav("走到坐标", _D),                                   # the wrong room: lands short of the goal
    _verify(f"(target ({_D[0]}, {_D[1]}))", _K),            # -> FAIL
    _nav("-> FAIL", _K),                                    # recover: navigate again
    _verify(f"(target ({_K[0]}, {_K[1]}))", _K),            # the SAME predicate -> PASS
    {"when": "-> PASS (result=True", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    _nav("回到起点", _S),                                   # no model verify: the hint verifies
    {"when": f"(target ({_S[0]}, {_S[1]}))", "tool_calls": [{"name": "finish", "input": {}}],
     "stop_reason": "end_turn"},
]}


@pytest.mark.sim
@pytest.mark.cli_main
def test_e12_fail_retry_pass_is_grounded_and_navigate_is_caused() -> None:
    r = run_repl_session(["启动 go2 仿真 无窗口", _GOAL, "回到起点"], tool_script=_SCRIPT,
                         answers=["y"], timeout_sec=420, extra_env={"VECTOR_SIM_LOCKSTEP": "1"})
    text = r.text
    assert not r.timed_out, text[-4000:]
    retry = text[text.rfind(f"vector> {_GOAL}"):text.rfind("vector> 回到起点")]
    kx, ky = _K
    assert f"Task contract: at_position({kx}, {ky})" in retry, retry[-3000:]
    assert f"verify at_position({kx}, {ky}) · (actor=CAUSED, superseded)" in retry, retry[-3000:]
    assert f"verify at_position({kx}, {ky}) ✓ (actor=CAUSED)" in retry, retry[-3000:]
    assert "verdict GROUNDED verified=True (1/1 grounded)" in retry, retry[-3000:]
    back = text[text.rfind("vector> 回到起点"):]
    assert f"navigate → verify at_position({_S[0]}, {_S[1]}) ✓ (actor=CAUSED)" in back, back[-3000:]
    assert "verdict GROUNDED verified=True" in back, back[-3000:]
