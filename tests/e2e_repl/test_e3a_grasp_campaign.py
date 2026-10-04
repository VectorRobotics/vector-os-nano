# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E3a as a paired-seed campaign: each seed nudges the dog (walk back, sidestep, turn)
by NL before '拿起绿色的瓶子'. The grasp is physics, so its rate is the measure:

    VECTOR_GRASP_CAMPAIGN=12 with-sim-slot run_tests.sh <out> tests/e2e_repl/test_e3a_grasp_campaign.py

passes / N is the scripted grasp success rate (same seeds = paired before/after).
Skipped unless VECTOR_GRASP_CAMPAIGN is set.
"""
from __future__ import annotations

import os
import random

import pytest

from tests.e2e_repl import LAUNCH, LAUNCH_STEPS
from tests.harness.pty_cli import run_repl_session

_N = int(os.environ.get("VECTOR_GRASP_CAMPAIGN", "0") or 0)
_SEED0 = int(os.environ.get("VECTOR_GRASP_CAMPAIGN_SEED0", "0") or 0)
_TARGET = "pickable_bottle_green"


def _script(seed: int) -> dict:
    r = random.Random(seed)
    back, side_dir, side = r.uniform(0.0, 0.25), r.choice(["left", "right"]), r.uniform(0.0, 0.10)
    turn_dir, angle = r.choice(["left", "right"]), r.uniform(0.0, 12.0)
    return {"steps": LAUNCH_STEPS + [
        {"when": "挪一下", "tool_calls": [{"name": "walk", "input": {"direction": "backward", "distance": round(back, 3)}}]},
        {"when": "", "tool_calls": [{"name": "walk", "input": {"direction": side_dir, "distance": round(side, 3)}}]},
        {"when": "", "tool_calls": [{"name": "turn", "input": {"direction": turn_dir, "angle": round(angle, 1)}}]},
        {"when": "", "text": "ok", "tool_calls": []},
        {"when": "拿起绿色的瓶子", "tool_calls": [{"name": "perception_grasp", "input": {"query": "绿色的瓶子"}}]},
        {"when": "", "tool_calls": [{"name": "verify", "input": {"expr": f"holding_object('{_TARGET}')"}}]},
        {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]}


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.skipif(not _N, reason="set VECTOR_GRASP_CAMPAIGN=N to run the N-seed grasp campaign")
@pytest.mark.parametrize("seed", range(_SEED0, _SEED0 + max(_N, 1)))
def test_grasp_campaign_seed(seed: int) -> None:
    r = run_repl_session([LAUNCH, "挪一下", "拿起绿色的瓶子"], tool_script=_script(seed),
                         answers=["y"] * 6, timeout_sec=420)
    assert not r.timed_out, r.text[-3000:]
    assert f"verify holding_object('{_TARGET}') ✓" in r.text, r.text[-3000:]
    assert "verdict GROUNDED verified=True" in r.text, r.text[-3000:]
