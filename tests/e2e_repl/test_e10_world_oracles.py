# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E10 world GT oracles (WP10) through bare vector-cli on go2_piper.

1. '找到绿色瓶子' -> look (the detector over the world vocabulary, no VLM) ->
   verify(localized('green bottle')) GROUNDED: the scene-graph instance the look
   wrote this turn sits where ground truth says the bottle is.
2. '把绿色瓶子放到桌子上' with no placing action: the bottle IS on the table, but the
   actor caused nothing -> UNCAUSED -> RAN (a pre-placed object never grounds a place).
   The re-verify needs an action first (WP07 probe refusal): a short walk, no release.
3. '把绿色瓶子拿过来' in ONE turn: grasp -> navigate to the user (the world persona's
   delivery point) -> kept_holding + near_object('user') GROUNDED against the task
   contract kept_holding('pickable_bottle_green') and near_object('user') (I4). Then
   '再拿过来一点' with the bottle already held at turn start -> RAN (kept_holding is
   caused only by a grasp in the turn), and a scripted drop during a carry -> RAN.
"""
from __future__ import annotations

import pytest

from tests.e2e_repl import LAUNCH, LAUNCH_STEPS
from tests.harness.pty_cli import run_repl_session

_LOOK = "找到绿色瓶子"
_PLACE = "把绿色瓶子放到桌子上"
_FIND_SCRIPT = {"steps": LAUNCH_STEPS + [
    {"when": _LOOK, "tool_calls": [{"name": "look", "input": {}}]},
    {"when": "", "tool_calls": [{"name": "verify", "input": {"expr": "localized('green bottle')"}}]},
    {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    # the place turn: verify straight away, then after an action that is no release
    {"when": _PLACE, "tool_calls": [
        {"name": "verify", "input": {"expr": "object_on('green bottle', 'pick_table')"}}]},
    {"when": "actor=UNCAUSED", "tool_calls": [
        {"name": "walk", "input": {"direction": "backward", "distance": 0.2, "speed": 0.3}}]},
    {"when": "", "tool_calls": [
        {"name": "verify", "input": {"expr": "object_on('green bottle', 'pick_table')"}}]},
    {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
]}

_FETCH, _AGAIN, _DROP = "把绿色瓶子拿过来", "再拿过来一点", "继续拿过来"
_CARRY_VERIFY = "kept_holding('green bottle') and near_object('user')"
_CONTRACT = "Task contract: kept_holding('pickable_bottle_green') and near_object('user')"
_CARRY_SCRIPT = {"steps": LAUNCH_STEPS + [
    # one fetch turn: grasp, carry to the user's spot (config/worlds/apartment.yaml user.xy)
    {"when": _FETCH, "tool_calls": [{"name": "perception_grasp", "input": {"query": "绿色的瓶子"}}]},
    {"when": "", "tool_calls": [{"name": "verify", "input": {"expr": "holding_object('green bottle')"}}]},
    {"when": "PASS", "tool_calls": [{"name": "navigate", "input": {"x": 8.0, "y": 3.0}}]},
    {"when": "", "tool_calls": [{"name": "verify", "input": {"expr": _CARRY_VERIFY}}]},
    {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    # held since the last turn: a short carry proves no fetch of this turn
    {"when": _AGAIN, "tool_calls": [
        {"name": "walk", "input": {"direction": "backward", "distance": 0.3, "speed": 0.3}}]},
    {"when": "", "tool_calls": [{"name": "verify", "input": {"expr": _CARRY_VERIFY}}]},
    {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    # the scripted drop mid-carry
    {"when": _DROP, "tool_calls": [
        {"name": "walk", "input": {"direction": "backward", "distance": 0.3, "speed": 0.3}}]},
    {"when": "", "tool_calls": [{"name": "gripper_open", "input": {}}]},
    {"when": "", "tool_calls": [{"name": "verify", "input": {"expr": _CARRY_VERIFY}}]},
    {"when": "FAIL", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
]}


def _turn(text: str, cmd: str) -> str:
    """The transcript from *cmd* being typed up to the next prompt's command."""
    i = text.find(cmd + "\r")
    if i == -1:
        i = text.find(cmd)
    assert i != -1, f"{cmd!r} never typed:\n{text[-3000:]}"
    return text[i:]


@pytest.mark.sim
@pytest.mark.cli_main
def test_e10_find_localizes_and_a_preplaced_object_never_grounds_a_place() -> None:
    r = run_repl_session([LAUNCH, _LOOK, _PLACE], tool_script=_FIND_SCRIPT, answers=["y"], timeout_sec=420)
    text = r.text
    assert not r.timed_out, text[-3000:]
    find = _turn(text, _LOOK)
    find = find[: find.find(_PLACE)] if _PLACE in find else find
    assert "look → verify localized('green bottle') ✓" in find, find[-3000:]
    assert "verdict GROUNDED verified=True" in find, find[-3000:]
    place = _turn(text, _PLACE)
    assert "verify object_on('green bottle', 'pick_table') ✓ (actor=UNCAUSED)" in place, place[-3000:]
    assert "walk → verify object_on('green bottle', 'pick_table') ✓ (actor=UNCAUSED)" in place, place[-3000:]
    assert "verdict RAN verified=False" in place, place[-3000:]


@pytest.mark.sim
@pytest.mark.cli_main
def test_e10_carry_grounds_and_a_drop_during_carry_is_never_green() -> None:
    r = run_repl_session([LAUNCH, _FETCH, _AGAIN, _DROP], tool_script=_CARRY_SCRIPT,
                         answers=["y", "y", "y", "y"], timeout_sec=600)
    text = r.text
    assert not r.timed_out, text[-3000:]
    fetch = _turn(text, _FETCH)
    fetch = fetch[: fetch.find(_AGAIN)] if _AGAIN in fetch else fetch
    assert _CONTRACT in fetch, fetch[-3000:]
    assert "verify holding_object('green bottle') ✓ (actor=CAUSED)" in fetch, fetch[-3000:]
    assert f"navigate → verify {_CARRY_VERIFY} ✓ (actor=CAUSED)" in fetch, fetch[-3000:]
    assert "verdict GROUNDED verified=True" in fetch, fetch[-3000:]
    again = _turn(text, _AGAIN)
    again = again[: again.find(_DROP)] if _DROP in again else again
    assert f"walk → verify {_CARRY_VERIFY} ✓ (actor=UNCAUSED)" in again, again[-3000:]
    assert "already satisfied at turn start: kept_holding('pickable_bottle_green')" in again, again[-3000:]
    assert "verdict RAN verified=False" in again, again[-3000:]
    drop = _turn(text, _DROP)
    assert f"gripper_open → verify {_CARRY_VERIFY} ·" in drop, drop[-3000:]
    assert "verdict RAN verified=False" in drop, drop[-3000:]
