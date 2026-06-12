# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Batch 1 (campaign #4) — #17 post-hoc timeout honesty + #16 async navigate.

#17: a slow-but-COMPLETED physical action used to be recorded as a bare
timeout failure — verify never ran, output was never captured, and the
replan context carried only the failure, so the whole tree re-played
motions that had actually succeeded (double motion/time/LLM cost). Now a
timed-out step that executed still runs its verify: verified → an honest
PASS carrying a timing warning; unverified → still a "timeout" failure.
The re-decompose context names the already-completed steps so the replan
plans only the remainder.

#16: a paced navigate held the server op thread (and the repo-side bridge
lock) for its whole wall-time — renders/panos starved for 30 s. The server
now exposes navigate_start/navigate_status (motion worker + ticket): the
op thread is free between polls, so renders interleave.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

from vector_os_nano.vcli.cognitive.blackboard import Blackboard
from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.types import GoalTree, StepRecord, SubGoal
from vector_os_nano.vcli.cognitive.vgg_harness import HarnessConfig, VGGHarness


class _Selector:
    def __init__(self, primitive: str = "slow"):
        self._p = primitive

    def select(self, sub_goal):
        from vector_os_nano.vcli.cognitive.strategy_selector import StrategyResult

        return StrategyResult("primitive", self._p, {})


def _executor(primitives: dict) -> GoalExecutor:
    name = next(iter(primitives))
    ex = GoalExecutor(
        strategy_selector=_Selector(name),
        verifier=GoalVerifier({}),
        primitives=primitives,
    )
    ex.blackboard = Blackboard()
    return ex


def _slow(out=None):
    def _p():
        time.sleep(0.05)
        return out if out is not None else True
    return _p


class TestPostHocTimeoutVerify:
    def test_timed_out_but_verified_step_passes_with_warning(self):
        ex = _executor({"slow": _slow({"x": 1})})
        sg = SubGoal(name="s", description="s",
                     verify="step_output('x') == 1", timeout_sec=0.01)
        step = ex._execute_sub_goal(sg)
        assert step.success is True
        assert step.verify_result is True
        assert "timeout" in step.result_data.get("timing_warning", "")
        assert step.failure_class == ""

    def test_timed_out_verified_output_is_captured(self):
        ex = _executor({"slow": _slow({"x": 1})})
        sg = SubGoal(name="s", description="s",
                     verify="step_output('x') == 1", timeout_sec=0.01)
        ex._execute_sub_goal(sg)
        # blackboard capture must have happened (the replan/binding seed)
        assert ex.blackboard.resolve("${s.output.x}") == 1

    def test_timed_out_and_unverified_is_still_timeout_failure(self):
        ex = _executor({"slow": _slow({"x": 2})})
        sg = SubGoal(name="s", description="s",
                     verify="step_output('x') == 1", timeout_sec=0.01)
        step = ex._execute_sub_goal(sg)
        assert step.success is False
        assert step.failure_class == "timeout"
        assert "timeout" in step.error

    def test_fast_step_has_no_warning(self):
        ex = _executor({"fast": lambda: {"x": 1}})
        sg = SubGoal(name="s", description="s",
                     verify="step_output('x') == 1", timeout_sec=10)
        step = ex._execute_sub_goal(sg)
        assert step.success is True
        assert "timing_warning" not in step.result_data


def _step(name: str, success: bool) -> StepRecord:
    return StepRecord(
        sub_goal_name=name, strategy="strat_" + name, success=success,
        verify_result=success, duration_sec=0.0,
        error="" if success else "boom",
    )


class TestReplanCompletedInjection:
    def test_replan_context_names_completed_steps(self):
        tree = GoalTree(goal="t", sub_goals=(
            SubGoal(name="a", description="a", verify="True", timeout_sec=10),
            SubGoal(name="b", description="b", verify="True", timeout_sec=10),
        ))
        executor = MagicMock()
        executor._stats = None
        executor._topological_sort.return_value = list(tree.sub_goals)
        executor._execute_sub_goal.side_effect = lambda sg: _step(
            sg.name, sg.name == "a")

        decomposer = MagicMock()
        decomposer.decompose.return_value = tree

        h = VGGHarness(
            decomposer=decomposer, executor=executor,
            config=HarnessConfig(max_step_retries=0, max_pipeline_retries=1),
        )
        h.run("task", "ctx")

        # the retry decompose (2nd call) must see the completed-step block
        assert decomposer.decompose.call_count == 2
        retry_ctx = decomposer.decompose.call_args_list[1].args[1]
        assert "COMPLETED" in retry_ctx
        assert "a" in retry_ctx
        assert "plan only" in retry_ctx.lower()

    def test_no_completed_block_when_nothing_succeeded(self):
        tree = GoalTree(goal="t", sub_goals=(
            SubGoal(name="a", description="a", verify="True", timeout_sec=10),
        ))
        executor = MagicMock()
        executor._stats = None
        executor._topological_sort.return_value = list(tree.sub_goals)
        executor._execute_sub_goal.side_effect = lambda sg: _step(sg.name, False)
        decomposer = MagicMock()
        decomposer.decompose.return_value = tree
        h = VGGHarness(
            decomposer=decomposer, executor=executor,
            config=HarnessConfig(max_step_retries=0, max_pipeline_retries=1),
        )
        h.run("task", "ctx")
        retry_ctx = decomposer.decompose.call_args_list[1].args[1]
        assert "COMPLETED" not in retry_ctx


class TestTimeoutCarriesExecError:
    """R11 (smoke walk_20m finding): a timed-out FAILED execution returned a
    bare 'timeout after Xs' — the underlying exec error (e.g. moved_short
    after hitting a wall) was masked, misleading the diagnosis again."""

    def test_exec_error_surfaces_in_timeout_message(self):
        def _slow_fail():
            time.sleep(0.05)
            raise RuntimeError("walk moved 4.8m of the requested 20.0m")

        ex = _executor({"slowfail": _slow_fail})
        sg = SubGoal(name="s", description="s", verify="True",
                     timeout_sec=0.01)
        step = ex._execute_sub_goal(sg)
        assert step.success is False
        assert step.failure_class == "timeout"
        assert "timeout" in step.error
        # the underlying execution error must ride along, not be replaced
        assert "exec:" in step.error
