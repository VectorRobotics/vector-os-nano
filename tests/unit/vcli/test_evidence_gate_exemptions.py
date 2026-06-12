# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Invariant II — per-step evidence-gate exemptions (design review #4).

The old gate had a WORLD-level bypass (`if is_robot: return True`) and every
playground world hard-codes is_robot()=True — so 'verified done' collapsed to
trace.success exactly where the owner live-tests. The bypass is gone: worlds
now declare a bounded set of strategies that legitimately carry no symbolic
post-condition (`evidence_exempt_strategies`, mechanism modeled on the
answer_only exemption); PlaygroundWorld declares the EMPTY set (it has a full
sim oracle). Exempt steps still require verify_result=True and no visual
override — the exemption never launders a failed or VLM-graded step.
"""
from __future__ import annotations

from vector_os_nano.vcli.cognitive.trace_store import (
    evidence_passed,
    step_evidence_ok,
)
from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)


def _trace(verify: str, strategy: str = "walk_skill", verify_result: bool = True,
           visual_override: bool = False) -> ExecutionTrace:
    sg = SubGoal(name="s1", description="d", verify=verify, strategy=strategy)
    step = StepRecord(
        sub_goal_name="s1", strategy=strategy.removesuffix("_skill"),
        success=True, verify_result=verify_result, duration_sec=0.1,
        visual_override=visual_override,
    )
    tree = GoalTree(goal="g", sub_goals=(sg,))
    return ExecutionTrace(goal_tree=tree, steps=(step,), success=True,
                          total_duration_sec=0.1)


class TestWorldBypassRemoved:
    def test_is_robot_no_longer_bypasses(self):
        # The single most important flip: a sentinel-verify step in a robot
        # world is NOT evidence any more.
        assert evidence_passed(_trace(verify="True"), is_robot=True) is False

    def test_real_predicate_still_passes(self):
        assert evidence_passed(_trace(verify="at_position(1, 2)"),
                               is_robot=True) is True


class TestPerStepExemption:
    def test_exempt_strategy_with_sentinel_verify_passes(self):
        t = _trace(verify="True", strategy="explore_skill")
        assert evidence_passed(t, exempt_strategies=frozenset({"explore"})) is True

    def test_exemption_never_launders_failed_verify(self):
        t = _trace(verify="True", strategy="explore_skill", verify_result=False)
        assert evidence_passed(t, exempt_strategies=frozenset({"explore"})) is False

    def test_exemption_never_launders_visual_override(self):
        t = _trace(verify="True", strategy="explore_skill", visual_override=True)
        assert evidence_passed(t, exempt_strategies=frozenset({"explore"})) is False

    def test_non_exempt_sentinel_still_fails(self):
        t = _trace(verify="True", strategy="walk_skill")
        assert evidence_passed(t, exempt_strategies=frozenset({"explore"})) is False


class TestStepEvidenceOk:
    def _pair(self, verify: str, strategy="explore"):
        sg = SubGoal(name="s1", description="d", verify=verify,
                     strategy=f"{strategy}_skill")
        step = StepRecord(sub_goal_name="s1", strategy=strategy, success=True,
                          verify_result=True, duration_sec=0.1)
        return step, sg

    def test_is_robot_no_longer_bypasses(self):
        step, sg = self._pair("True")
        assert step_evidence_ok(step, sg, is_robot=True) is False

    def test_exempt_strategy_ok(self):
        step, sg = self._pair("True")
        assert step_evidence_ok(step, sg,
                                exempt_strategies=frozenset({"explore"})) is True


class TestWorldDeclarations:
    def test_playground_world_declares_empty_set(self):
        from vector_os_nano.playground.world import PlaygroundWorld

        assert PlaygroundWorld().evidence_exempt_strategies() == frozenset()

    def test_robot_world_declares_transitional_motor_set(self):
        # The REAL-hardware world keeps its async motor skills exempt until
        # invariant III gives them real verify hints (transitional, bounded).
        from vector_os_nano.vcli.worlds.robot import RobotWorld

        exempt = RobotWorld().evidence_exempt_strategies()
        assert "explore" in exempt and "patrol" in exempt
        assert "pick" not in exempt  # manipulation must carry real predicates
