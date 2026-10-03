# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""H-1a — a foreach run through the VGGHarness records each child strategy ONCE.

REAL BUG: GoalExecutor._execute_foreach already records per-child StrategyStats
internally. The harness's foreach branch USED TO re-record the same children,
double-counting every foreach child whenever a live StrategyStats object was
attached to the executor. This test runs a real foreach plan through the harness
with a live StrategyStats and asserts each child's (strategy, pattern) attempt
count is EXACTLY the number of items — not double.

Hermetic: no LLM, no MuJoCo. A pre-built GoalTree is passed to the harness so no
decompose backend is needed; the executor expands the foreach and the harness
drives it.
"""
from __future__ import annotations

from typing import Any

from vector_os_nano.vcli.cognitive.blackboard import Blackboard
from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.oracle_registry import oracle
from vector_os_nano.vcli.cognitive.strategy_selector import StrategyResult
from vector_os_nano.vcli.cognitive.strategy_stats import StrategyStats
from vector_os_nano.vcli.cognitive.types import ForEachSpec, GoalTree, SubGoal
from vector_os_nano.vcli.cognitive.vgg_harness import HarnessConfig, VGGHarness


class _DirectSelector:
    def select(self, sub_goal: SubGoal) -> StrategyResult:
        return StrategyResult(
            "primitive", sub_goal.strategy, dict(sub_goal.strategy_params)
        )


def _build(objects: list[dict[str, Any]]) -> tuple[GoalExecutor, StrategyStats]:
    state: dict[str, Any] = {"picked": []}

    def detect(**_: Any) -> dict[str, Any]:
        return {"objects": list(objects), "count": len(objects)}

    def pick(object_label: str | None = None, **_: Any) -> dict[str, Any]:
        state["picked"].append(object_label)
        return {"picked": object_label}

    # R1: the per-step reward gate now requires deterministic evidence — a bare
    # call to a recognised PREDICATE oracle. ``holding_object`` is such an oracle
    # (goal-conditioned bool), so the foreach body's verify grounds honestly and
    # the gate rewards each verified child. (A bespoke ``picked()`` would classify
    # RAN under the honest gate and never count as a success.)
    namespace = {
        # WP02: only @oracle-tagged callables count as evidence. holding_object is a
        # robot GT oracle, so tag it; the reward gate then grounds each verified child.
        "holding_object": oracle(scope="robot", channel="gripper")(
            lambda label=None: label in state["picked"]
        ),
        "detected": lambda: len(objects) > 0,
    }
    # In-memory only StrategyStats (persist_path=None -> no file I/O).
    stats = StrategyStats()
    executor = GoalExecutor(
        strategy_selector=_DirectSelector(),
        verifier=GoalVerifier(namespace),
        primitives={"detect": detect, "pick": pick},
        stats=stats,
    )
    return executor, stats


def _foreach_tree() -> GoalTree:
    return GoalTree(
        goal="pick up every object",
        sub_goals=(
            SubGoal(
                name="detect",
                description="detect every object",
                verify="detected()",
                strategy="detect",
                depends_on=(),
            ),
            SubGoal(
                name="pick_each",
                description="pick up each detected object",
                verify="True",
                strategy="",
                depends_on=("detect",),
                foreach=ForEachSpec(
                    source_step="detect",
                    source_path="objects",
                    var="obj",
                    body=(
                        SubGoal(
                            name="grasp_one",
                            description="pick up the current object",
                            verify="holding_object('${obj.name}')",
                            strategy="pick",
                            strategy_params={"object_label": "${obj.name}"},
                        ),
                    ),
                ),
            ),
        ),
    )


def test_harness_foreach_records_each_child_strategy_exactly_once() -> None:
    objects = [{"name": "mug"}, {"name": "banana"}, {"name": "block"}]
    executor, stats = _build(objects)

    harness = VGGHarness(
        decomposer=None,  # tree supplied directly — no decompose needed
        executor=executor,
        config=HarnessConfig(max_pipeline_retries=0),
    )
    trace = harness.run(
        "pick up every object", "scene", goal_tree=_foreach_tree()
    )

    assert trace.success is True

    # The expanded children are named "pick_each[i].grasp_one", so they bucket
    # under the "pick_*" pattern (extract_pattern uses the first underscore stem).
    child_pattern = StrategyStats.extract_pattern("pick_each[0].grasp_one")

    # The foreach body strategy "pick" must be recorded once PER ITEM
    # (== len(objects)), not double (== 2 * len(objects)) as the old harness did.
    pick_rec = stats.get_stats("pick", child_pattern)
    assert pick_rec is not None
    assert pick_rec.total_attempts == len(objects)
    assert pick_rec.successes == len(objects)

    # The producing step ran once and was recorded once.
    detect_rec = stats.get_stats("detect", StrategyStats.extract_pattern("detect"))
    assert detect_rec is not None
    assert detect_rec.total_attempts == 1
