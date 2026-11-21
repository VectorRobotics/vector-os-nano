# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E15 the e2e audit catches what it is for (A3), with the scripted model.

Each session asks for a task the scripted model then does NOT do. The product rejects
every one of these turns today, and the independent grade (tests/harness/gt_audit, run
by the harness after the session against tests/e2e_repl/gt_expect.yaml) agrees that
nothing happened. A product whose verify or task contract were broken would claim the
turn, and the audit, never an assertion of this file, turns the session red:

  * '抓起马克杯' (the mug): the model picks the banana and verifies the mug. A holding
    oracle that ignores which object is held would verify it.
  * '抓起马克杯': the model picks the banana and verifies the banana. A task contract
    bound to another object than the sentence names would prove it.
  * '去厨房': the model stops on the hallway side of the kitchen door and verifies a
    point inside the kitchen 0.78 m from it. A doubled arrival tolerance would verify it.

This file asserts only that the audit graded the turn and found nothing happened.
"""
from __future__ import annotations

import pytest

from tests.harness.pty_cli import run_repl_session

_ARM, _MUG = "启动 so101 机械臂仿真 无窗口", "抓起马克杯"
_GO2, _KITCHEN = "启动 go2 仿真 无窗口", "去厨房"


def _launch(say: str, body: str) -> list[dict]:
    return [{"when": say, "tool_calls": [{"name": "start_simulation", "input": {"embodiment": body, "gui": False}}]},
            {"when": "", "text": "ok", "tool_calls": []}]


def _then(task: str, act: dict, expr: str) -> list[dict]:
    return [{"when": task, "tool_calls": [act]},
            {"when": "", "tool_calls": [{"name": "verify", "input": {"expr": expr}}]},
            {"when": "", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"}]


def _graded_nothing_happened(r, task: str) -> None:
    assert not r.timed_out, r.text[-3000:]
    (rec,) = [a for a in r.audit if a["sentence"] == task]
    assert rec["grade"] is False and rec["outcome"] == "agree", rec


_PICK_BANANA = {"name": "pick", "input": {"object_label": "banana", "mode": "hold"}}


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.parametrize("verify", ["mug", "banana"])
def test_e15_a_pick_of_the_wrong_object_is_never_a_pick_of_the_mug(verify: str) -> None:
    script = {"steps": _launch(_ARM, "so101") + _then(_MUG, _PICK_BANANA, f"holding_object('{verify}')")}
    r = run_repl_session([_ARM, _MUG], tool_script=script, answers=["y"], timeout_sec=240)
    _graded_nothing_happened(r, _MUG)


# The kitchen-hallway door is (13.5, 3.0) (config/worlds/apartment.yaml); the kitchen floor
# starts at x = 14.0. Lockstep, the base stops at (13.27, 2.94), outside the kitchen and
# 0.78 m from the point it verifies: beyond the 0.5 m arrival tolerance, within twice it.
_DOOR, _INSIDE = (13.5, 3.0), (14.05, 3.0)


@pytest.mark.sim
@pytest.mark.cli_main
def test_e15_a_base_at_the_door_is_never_in_the_kitchen() -> None:
    go = {"name": "navigate", "input": {"x": _DOOR[0], "y": _DOOR[1]}}
    script = {"steps": _launch(_GO2, "go2") + _then(_KITCHEN, go, f"at_position({_INSIDE[0]}, {_INSIDE[1]})")}
    r = run_repl_session([_GO2, _KITCHEN], tool_script=script, answers=["y"], timeout_sec=300,
                         extra_env={"VECTOR_SIM_LOCKSTEP": "1"})
    _graded_nothing_happened(r, _KITCHEN)
