# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""R2a PART A — VerdictReport is built ONLY from the existing evidence gate.

The honest verdict the engine computes — ``evidence_passed(trace,
verify_oracle_names(agent, engine))`` — must escape cli.main as a machine signal
that can ONLY AGREE with that gate, never a second opinion. These tests pin:

(1) CONTRACT: ``VerdictReport.from_trace(trace, oracle).verified`` equals
    ``evidence_passed(trace, oracle)`` for grounded, ran, failed, and empty-oracle
    traces — the machine signal IS the moat, not a re-derivation.
(2) per-step evidence equals ``classify_step_evidence`` for every step.
(3) the exit-code contract: ``verified == (exit_code() == 0)``; RAN -> 2; NO_TRACE -> 1.
(4) the sentinel line round-trips to a dict carrying the contract fields.

Pure kernel logic on deterministic StepRecord/SubGoal — no cli.main, no LLM.
"""
from __future__ import annotations

import json
import pytest

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
from vector_os_nano.vcli.verdict import (
    EVIDENCE_FAILED,
    EVIDENCE_GROUNDED,
    EVIDENCE_NO_TRACE,
    EVIDENCE_RAN,
    VERDICT_SENTINEL,
    VerdictReport,
)

# A robot oracle set including at_position (mirrors verify_oracle_names live keys).
ORACLES = frozenset({"at_position", "facing", "visited", "arm_at_home"})


def _trace(verify: str, success: bool, verify_result: bool, goal: str = "g") -> ExecutionTrace:
    sg = SubGoal(name="s1", description="d", verify=verify, strategy="walk_forward")
    step = StepRecord(
        sub_goal_name="s1",
        strategy="walk_forward",
        success=success,
        verify_result=verify_result,
        duration_sec=0.1,
    )
    tree = GoalTree(goal=goal, sub_goals=(sg,))
    return ExecutionTrace(goal_tree=tree, steps=(step,), success=success, total_duration_sec=0.1)


# ---------------------------------------------------------------------------
# (1) CONTRACT — verified IS evidence_passed, never re-derived
# ---------------------------------------------------------------------------


def test_verdict_matches_evidence_passed_grounded() -> None:
    # A real oracle predicate, true post-state -> GROUNDED -> verified.
    trace = _trace("at_position(2.0, 0.0)", success=True, verify_result=True)
    report = VerdictReport.from_trace(trace, ORACLES)
    assert report.verified == evidence_passed(trace, ORACLES)
    assert report.verified is True
    assert report.evidence == EVIDENCE_GROUNDED
    assert report.n_grounded == 1


def test_verdict_matches_evidence_passed_ran_sentinel() -> None:
    # The RED case the spec demands: a "succeeded" step whose verify is the
    # sentinel "True" carries NO grounded evidence -> RAN -> verified False, even
    # though the trace "succeeded".
    trace = _trace("True", success=True, verify_result=True)
    report = VerdictReport.from_trace(trace, ORACLES)
    assert report.verified == evidence_passed(trace, ORACLES)
    assert report.verified is False
    assert report.evidence == EVIDENCE_RAN
    assert report.n_grounded == 0


def test_verdict_matches_evidence_passed_failed() -> None:
    trace = _trace("at_position(2.0, 0.0)", success=False, verify_result=False)
    report = VerdictReport.from_trace(trace, ORACLES)
    assert report.verified == evidence_passed(trace, ORACLES)
    assert report.verified is False
    assert report.evidence == EVIDENCE_FAILED


def test_verdict_matches_evidence_passed_empty_oracle_fails_closed() -> None:
    # Empty oracle set -> every predicate classifies RAN -> verified False
    # (fail closed, moat only stricter).
    trace = _trace("at_position(2.0, 0.0)", success=True, verify_result=True)
    report = VerdictReport.from_trace(trace, frozenset())
    assert report.verified == evidence_passed(trace, frozenset())
    assert report.verified is False


# ---------------------------------------------------------------------------
# (2) per-step evidence equals classify_step_evidence
# ---------------------------------------------------------------------------


def test_per_step_evidence_equals_classifier() -> None:
    trace = _trace("at_position(2.0, 0.0)", success=True, verify_result=True)
    report = VerdictReport.from_trace(trace, ORACLES)
    assert len(report.per_step) == 1
    sg = trace.goal_tree.sub_goals[0]
    step = trace.steps[0]
    assert report.per_step[0].evidence == classify_step_evidence(step, sg, ORACLES)
    assert report.per_step[0].name == "s1"
    assert report.per_step[0].verify == "at_position(2.0, 0.0)"


# ---------------------------------------------------------------------------
# (3) exit-code contract: verified == (exit == 0)
# ---------------------------------------------------------------------------


def test_exit_code_verified_is_zero() -> None:
    trace = _trace("at_position(2.0, 0.0)", success=True, verify_result=True)
    report = VerdictReport.from_trace(trace, ORACLES)
    assert report.exit_code() == 0
    assert report.verified == (report.exit_code() == 0)


def test_exit_code_ran_not_verified_is_two() -> None:
    trace = _trace("True", success=True, verify_result=True)
    report = VerdictReport.from_trace(trace, ORACLES)
    assert report.exit_code() == 2
    assert report.verified == (report.exit_code() == 0)


def test_exit_code_no_trace_is_one() -> None:
    report = VerdictReport.no_trace(goal="chat only", error="no trace")
    assert report.evidence == EVIDENCE_NO_TRACE
    assert report.verified is False
    assert report.exit_code() == 1
    assert report.verified == (report.exit_code() == 0)


# ---------------------------------------------------------------------------
# (4) sentinel line round-trips to the contract dict
# ---------------------------------------------------------------------------


def test_sentinel_line_roundtrips() -> None:
    trace = _trace("at_position(2.0, 0.0)", success=True, verify_result=True)
    report = VerdictReport.from_trace(trace, ORACLES)
    line = report.to_sentinel_line()
    assert line.startswith(VERDICT_SENTINEL + " ")
    payload = json.loads(line[len(VERDICT_SENTINEL) + 1 :])
    # Contract fields all present.
    for key in (
        "verified", "success", "evidence", "goal",
        "n_steps", "n_grounded", "oracle_names", "per_step",
    ):
        assert key in payload
    assert payload["verified"] is True
    assert payload["goal"] == "g"
    assert isinstance(payload["oracle_names"], list)
    assert isinstance(payload["per_step"], list)
    assert payload["per_step"][0]["evidence"] == "GROUNDED"


def test_output_schema_and_step_provenance_are_pinned():
    from dataclasses import replace
    trace = _trace('at_position(2.0, 0.0)', success=True, verify_result=True)
    one = {'embodiment': 'go2', 'backend': 'pybullet', 'session': 7}
    step = replace(trace.steps[0], result_data={'output': {'session_info': one}})
    report = VerdictReport.from_trace(replace(trace, steps=(step,)), ORACLES,
                                      session_info={'world': 'room_b', 'backend_version': '1.2', 'perception_kind': 'rgbd'})
    data = report.to_dict()
    assert set(data) == {'verified', 'success', 'evidence', 'goal', 'n_steps', 'n_grounded', 'oracle_names',
                         'per_step', 'error', 'gait_backend', 'llm_unavailable', 'model', 'reject_reason',
                         'acted', 'scene', 'contract', 'goal_check', 'schema', 'world', 'backend_version', 'perception_kind'}
    assert set(data['per_step'][0]) == {'name', 'strategy', 'success', 'verify', 'verify_result', 'evidence',
                                      'actor_caused', 'reject_reason', 'superseded', 'failed_attempt', 'refused',
                                      'embodiment', 'backend', 'session'}
    assert all(data['per_step'][0][k] == v for k, v in one.items())
    assert data['schema'] == 1 and data['backend_version'] == '1.2' and data['perception_kind'] == 'rgbd'


def test_harness_refuses_unknown_verdict_schema():
    from tests.harness.pty_cli import _find_verdict_line
    from tests.harness.gt_audit import json_verdict
    for schema in (None, True, '1', 2):
        with pytest.raises(ValueError, match='unsupported verdict schema'):
            _find_verdict_line('VECTOR_VERDICT ' + json.dumps({'schema': schema}))
        with pytest.raises(ValueError, match='unsupported verdict schema'):
            json_verdict({'schema': schema})
