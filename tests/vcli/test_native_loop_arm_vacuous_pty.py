# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""WP02 e2e acceptance on the arm world, through the REAL ``cli.main`` (``--sim``).

(1) Five real ``home`` steps, each "verified" by a VACUOUS predicate that evaluates
    True whatever the world state -> every step RAN, the turn RAN, exit 2. Before
    WP02, 4 of the 5 graded GROUNDED.
(2) A real ``pick`` + ``holding_object('banana')`` -> GROUNDED, exit 0.

Scripted native tool-use (no live LLM); serialized, headless. The arm sim lives in the
vector-cli process, so nothing outlives the turn and no machine-wide ``rosm nuke`` runs.
"""
from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from tests.harness.pty_cli import run_cli_turn  # noqa: E402

_SIM_TIMEOUT_SEC = 360.0

# Each is TRUE whatever the world state, over the live GT oracles (WP10 removed the
# detect_objects / describe_scene / placed_count listings that used to make the set).
_VACUOUS = [
    "released() >= 0",
    "object_on('banana', 'table') or True",
    "not holding_object('banana')",
    "arm_at_home() == arm_at_home()",
    "holding_object('banana') is not None",
]


def _turns(*pairs):  # noqa: ANN002, ANN202
    turns = []
    for tool, expr in pairs:
        turns.append({"tool_calls": [tool]})
        turns.append({"tool_calls": [{"name": "verify", "input": {"expr": expr}}]})
    turns.append({"tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"})
    return {"turns": turns}


@pytest.mark.sim
@pytest.mark.cli_main
def test_arm_vacuous_verifies_grade_ran() -> None:
    r = run_cli_turn(
        "回到初始位置",
        sim=True,
        timeout_sec=_SIM_TIMEOUT_SEC,
        extra_args=["--headless"],
        tool_script=_turns(*[({"name": "home", "input": {}}, e) for e in _VACUOUS]),
    )
    assert r.verified is False and r.exit_code == 2, r.verdict
    assert r.evidence == "RAN", r.verdict
    steps = r.verdict["per_step"]
    assert [s["verify"] for s in steps] == _VACUOUS, steps
    # Each predicate is TRUE — only the classifier's structure check rejects it.
    assert all(s["verify_result"] is True for s in steps), steps
    assert all(s["evidence"] == "RAN" for s in steps), steps


@pytest.mark.sim
@pytest.mark.cli_main
def test_arm_honest_grasp_grades_grounded() -> None:
    r = run_cli_turn(
        "把香蕉抓起来拿在手里",
        sim=True,
        timeout_sec=_SIM_TIMEOUT_SEC,
        extra_args=["--headless"],
        tool_script=_turns(
            ({"name": "pick", "input": {"object_label": "banana", "mode": "hold"}},
             "holding_object('banana')"),
        ),
    )
    assert r.verified is True and r.exit_code == 0, r.verdict
    assert r.evidence == "GROUNDED", r.verdict
    (step,) = r.verdict["per_step"]
    assert step["strategy"] == "pick" and step["evidence"] == "GROUNDED", step
