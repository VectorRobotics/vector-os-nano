# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Batch 1 (campaign #4) — dependency-failure skip, unified semantics (#11).

The harness comment promised "steps with depends_on referencing the failed
step will skip" but no such mechanism existed: the harness blindly executed
every remaining step (on hardware: pick from the wrong place after a failed
navigate), while the executor aborted the WHOLE tree on first failure — two
opposite semantics for the same GoalTree. Both paths now share one contract:
a failed step poisons its dependents (transitively) with a skipped
StepRecord (failure_class="dep_skipped", never executed); independent steps
still run. max_redecompose was a dead config knob promising a Layer-2 that
was never implemented — deleted per the design review (§5: do NOT build
mid-tree re-decompose).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.blackboard import Blackboard
from vector_os_nano.vcli.cognitive.types import (
    FAILURE_CLASSES,
    GoalTree,
    StepRecord,
    SubGoal,
)
from vector_os_nano.vcli.cognitive.vgg_harness import HarnessConfig, VGGHarness


class _Selector:
    def select(self, sub_goal):
        from vector_os_nano.vcli.cognitive.strategy_selector import StrategyResult

        return StrategyResult("primitive", "noop", {})


def _executor(calls: list) -> GoalExecutor:
    ex = GoalExecutor(
        strategy_selector=_Selector(),
        verifier=GoalVerifier({}),
        primitives={"noop": lambda: calls.append("noop") or True},
    )
    ex.blackboard = Blackboard()
    return ex


def _sg(name: str, verify: str = "True", depends: tuple = ()) -> SubGoal:
    return SubGoal(name=name, description=name, verify=verify,
                   timeout_sec=10, depends_on=depends)


class TestFailureClassEnum:
    def test_dep_skipped_is_a_legal_class(self):
        assert "dep_skipped" in FAILURE_CLASSES


class TestExecutorDepSkip:
    def test_dependent_skipped_independent_runs(self):
        calls: list = []
        tree = GoalTree(goal="t", sub_goals=(
            _sg("a", verify="False"),            # fails at verify
            _sg("b", depends=("a",)),            # must be skipped, not run
            _sg("d"),                            # independent — must run
        ))
        trace = _executor(calls).execute(tree)

        assert trace.success is False
        by_name = {s.sub_goal_name: s for s in trace.steps}
        assert set(by_name) == {"a", "b", "d"}
        assert by_name["b"].failure_class == "dep_skipped"
        assert by_name["b"].success is False
        assert "a" in by_name["b"].error              # names the blocker
        assert by_name["d"].success is True
        # a ran, d ran, b never executed its primitive
        assert len(calls) == 2

    def test_skip_is_transitive(self):
        calls: list = []
        tree = GoalTree(goal="t", sub_goals=(
            _sg("a", verify="False"),
            _sg("b", depends=("a",)),
            _sg("c", depends=("b",)),             # blocked through b
        ))
        trace = _executor(calls).execute(tree)

        by_name = {s.sub_goal_name: s for s in trace.steps}
        assert by_name["b"].failure_class == "dep_skipped"
        assert by_name["c"].failure_class == "dep_skipped"
        assert len(calls) == 1                    # only a executed

    def test_all_success_unchanged(self):
        calls: list = []
        tree = GoalTree(goal="t", sub_goals=(
            _sg("a"), _sg("b", depends=("a",)),
        ))
        trace = _executor(calls).execute(tree)
        assert trace.success is True
        assert len(calls) == 2


def _step(name: str, success: bool) -> StepRecord:
    return StepRecord(
        sub_goal_name=name, strategy="s", success=success,
        verify_result=success, duration_sec=0.0,
        error="" if success else "boom",
    )


def _harness(tree: GoalTree, results: dict) -> tuple[VGGHarness, MagicMock]:
    executor = MagicMock()
    executor._stats = None
    executor._topological_sort.return_value = list(tree.sub_goals)
    executor._execute_sub_goal.side_effect = (
        lambda sg: results[sg.name]
    )
    decomposer = MagicMock()
    decomposer.decompose.return_value = tree
    h = VGGHarness(
        decomposer=decomposer,
        executor=executor,
        config=HarnessConfig(max_step_retries=0, max_pipeline_retries=0),
    )
    return h, executor


class TestHarnessDepSkip:
    def test_dependent_skipped_independent_runs(self):
        tree = GoalTree(goal="t", sub_goals=(
            _sg("a"), _sg("b", depends=("a",)), _sg("d"),
        ))
        h, executor = _harness(
            tree, {"a": _step("a", False), "d": _step("d", True)})
        trace = h.run("task", "ctx")

        by_name = {s.sub_goal_name: s for s in trace.steps}
        assert by_name["b"].failure_class == "dep_skipped"
        executed = [c.args[0].name for c in
                    executor._execute_sub_goal.call_args_list]
        assert executed == ["a", "d"]             # b never reached the executor

    def test_skipped_step_feeds_replan_context(self):
        tree = GoalTree(goal="t", sub_goals=(
            _sg("a"), _sg("b", depends=("a",)),
        ))
        h, _ = _harness(tree, {"a": _step("a", False)})
        trace = h.run("task", "ctx")
        assert trace.success is False
        skipped = [s for s in trace.steps
                   if s.failure_class == "dep_skipped"]
        assert len(skipped) == 1


class TestDeadConfigDeleted:
    def test_max_redecompose_is_gone(self):
        assert not hasattr(HarnessConfig(), "max_redecompose")
