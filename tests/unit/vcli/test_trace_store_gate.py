# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The trace store's evidence gate (moved from tests/vcli/test_level64_verify_as_eval.py
when vector-eval was deleted).

- the evidence gate grades pass/fail.
- a verify="True"-only (sentinel) trace is NOT counted as verified evidence; robot
  traces do not bypass the gate.
- per step: a sentinel verify is never evidence.

Pure kernel logic — no robot, no network, no mujoco fixtures.
"""
from __future__ import annotations

from vector_os_nano.vcli.cognitive.trace_store import (
    classify_step_evidence,
    evidence_passed,
)
from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)


# The live verify-namespace callable names a dev+robot world exposes, passed to
# the R1 evidence gate (replaces the old ``is_robot`` flag). file_exists /
# path_contains anchor the dev predicates; the robot oracles (at_position,
# arm_at_home, ...) anchor the sim path. Mirrors verify_oracle_names(agent, engine).
ORACLES = frozenset({
    "at_position", "facing", "visited", "holding_object", "arm_at_home",
    "file_exists", "path_contains", "get_position", "get_heading",
    "describe_scene", "detect_objects", "placed_count", "nearest_room",
    "objects_in_room", "find_object", "room_coverage",
})


# ---------------------------------------------------------------------------
# Trace builders
# ---------------------------------------------------------------------------


def _trace(
    *,
    verify: str = "file_exists('a.txt')",
    success: bool = True,
    verify_result: bool = True,
    goal: str = "make a.txt",
) -> ExecutionTrace:
    sg = SubGoal(
        name="step1",
        description="create a.txt",
        verify=verify,
        strategy="tool_call",
    )
    step = StepRecord(
        sub_goal_name="step1",
        strategy="tool_call",
        success=success,
        verify_result=verify_result,
        duration_sec=0.05,
    )
    return ExecutionTrace(
        goal_tree=GoalTree(goal=goal, sub_goals=(sg,)),
        steps=(step,),
        success=success,
        total_duration_sec=0.05,
    )


# ---------------------------------------------------------------------------
# Evidence gate: pass/fail over deterministic predicates
# ---------------------------------------------------------------------------


def test_evidence_passed_real_predicate() -> None:
    assert evidence_passed(_trace(verify="file_exists('a.txt')"), ORACLES) is True


def test_evidence_rejects_true_sentinel() -> None:
    assert evidence_passed(_trace(verify="True"), ORACLES) is False


def test_evidence_rejects_empty_verify() -> None:
    assert evidence_passed(_trace(verify=""), ORACLES) is False


def test_evidence_robot_world_now_requires_evidence() -> None:
    # R1 FLIP: the old ``if is_robot: return True`` bypass is GONE. A robot motor
    # step with the sentinel verify="True" now classifies RAN, not GROUNDED, so it
    # no longer auto-passes the done-gate — every world must back success with
    # deterministic evidence.
    assert evidence_passed(_trace(verify="True"), ORACLES) is False


def test_evidence_requires_verify_result_true() -> None:
    tr = _trace(verify="file_exists('a.txt')", success=True, verify_result=False)
    assert evidence_passed(tr, ORACLES) is False


def _step_and_goal(
    *,
    verify: str = "file_exists('a.txt')",
    success: bool = True,
    verify_result: bool = True,
    strategy: str = "tool_call",
) -> tuple[StepRecord, SubGoal]:
    sg = SubGoal(
        name="step1",
        description="do a thing",
        verify=verify,
        strategy=strategy,
    )
    step = StepRecord(
        sub_goal_name="step1",
        strategy=strategy,
        success=success,
        verify_result=verify_result,
        duration_sec=0.05,
    )
    return step, sg


def test_w1_1_sentinel_verify_blocks_evidence_dev() -> None:
    # T1: sub_goal.verify is the sentinel 'True' -> no deterministic evidence.
    step, sg = _step_and_goal(verify="True", success=True, verify_result=True)
    assert (classify_step_evidence(step, sg, ORACLES) == "GROUNDED") is False


def test_w1_1_real_predicate_has_evidence_dev() -> None:
    # T3: a real predicate + verify_result True -> evidence.
    step, sg = _step_and_goal(verify="file_exists('a.txt')", success=True, verify_result=True)
    assert (classify_step_evidence(step, sg, ORACLES) == "GROUNDED") is True


def test_w1_1_robot_world_now_requires_evidence_gate() -> None:
    # R1 FLIP: the robot bypass is GONE. A robot motor step with verify="True" now
    # classifies RAN, not GROUNDED — it is not GROUNDED for BOTH the passing
    # sentinel step and the failed one (FAILED is also not GROUNDED).
    passing, sg = _step_and_goal(verify="True", success=True, verify_result=True)
    assert (classify_step_evidence(passing, sg, ORACLES) == "GROUNDED") is False
    failed, sg_f = _step_and_goal(verify="True", success=False, verify_result=False)
    assert (classify_step_evidence(failed, sg_f, ORACLES) == "GROUNDED") is False
