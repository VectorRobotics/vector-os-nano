# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Batch 2 R10 — Layer-1 retry must not degrade the step (live root cause).

GUI smoke finding: attempt 1 failed with the REAL diagnosable error ("world
model is EMPTY — start sysnav"), then the retry cleared the strategy, the
selector's registry-alias match re-routed to the same skill WITH EMPTY
PARAMS, and attempts 2-3 failed as 'requires a label' — the surfaced error
was the DEGRADED one, two layers from the truth. Contract now:
(1) the alias-match route carries the sub_goal's explicit strategy_params
    (re-routing never strips bindings);
(2) a failed step's record surfaces the FIRST attempt's error alongside the
    last when they differ (result_data["attempt_errors"]).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from vector_os_nano.vcli.cognitive.strategy_selector import StrategySelector
from vector_os_nano.vcli.cognitive.types import GoalTree, StepRecord, SubGoal
from vector_os_nano.vcli.cognitive.vgg_harness import HarnessConfig, VGGHarness


class TestAliasMatchKeepsParams:
    def test_registry_alias_route_carries_explicit_params(self):
        match = MagicMock()
        match.skill_name = "navigate_to"
        registry = MagicMock()
        registry.match.return_value = match
        sel = StrategySelector(skill_registry=registry, has_base=False)

        sg = SubGoal(name="nav", description="approach the target zone",
                     verify="True", timeout_sec=10,
                     strategy="", strategy_params={"label": "kitchen"})
        result = sel.select(sg)
        assert result.executor_type == "skill"
        assert result.params == {"label": "kitchen"}

    def test_no_params_stays_empty(self):
        match = MagicMock()
        match.skill_name = "navigate_to"
        registry = MagicMock()
        registry.match.return_value = match
        sel = StrategySelector(skill_registry=registry, has_base=False)
        sg = SubGoal(name="nav", description="approach the target zone",
                     verify="True", timeout_sec=10)
        assert sel.select(sg).params == {}


def _failing_step(name, error):
    return StepRecord(sub_goal_name=name, strategy="navigate_to",
                      success=False, verify_result=False, duration_sec=0.0,
                      error=error)


class TestFirstErrorSurfaces:
    def _run(self, errors):
        tree = GoalTree(goal="t", sub_goals=(
            SubGoal(name="nav", description="nav", verify="True",
                    timeout_sec=10, strategy="navigate_to_skill",
                    strategy_params={"label": "kitchen"}),
        ))
        executor = MagicMock()
        executor._stats = None
        executor._topological_sort.return_value = list(tree.sub_goals)
        seq = [_failing_step("nav", e) for e in errors]
        executor._execute_sub_goal.side_effect = seq
        decomposer = MagicMock()
        decomposer.decompose.return_value = tree
        h = VGGHarness(
            decomposer=decomposer, executor=executor,
            config=HarnessConfig(max_step_retries=len(errors) - 1,
                                 max_pipeline_retries=0),
        )
        trace = h.run("task", "ctx")
        return trace.steps[0]

    def test_differing_first_error_kept(self):
        step = self._run([
            "world model is EMPTY: start sysnav first",
            "navigate_to requires a label OR numeric x and y",
        ])
        assert "requires a label" in step.error          # last attempt
        assert "world model is EMPTY" in step.error       # FIRST attempt kept
        assert step.result_data["attempt_errors"][0].startswith(
            "world model is EMPTY")

    def test_identical_errors_unchanged(self):
        step = self._run(["same error", "same error"])
        assert step.error == "same error"
        assert "attempt_errors" not in step.result_data
