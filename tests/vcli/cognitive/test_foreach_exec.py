# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""S4-2 — FOREACH execution-time expansion.

S4-1 modelled + parsed + validated the FOREACH control-flow IR. S4-2 makes the
executor EXPAND a foreach node at runtime: it reads the producing step's list off
the run Blackboard, instantiates the body templates once per item (binding the
iteration var via the Blackboard's PURE dict/list path traversal — never eval),
and executes each concrete child in order with its own per-step verify.

These tests pin (deterministic, no LLM, no MuJoCo):

  - N produced items -> exactly N expanded children execute, each verified.
  - Per-item binding resolves correctly: child i sees item i's fields
    (``${obj.name}`` -> item i's name), proven by the side effect each child has.
  - An empty producing list yields zero children (not an error); the run still
    succeeds.
  - depends_on: the foreach node runs AFTER its producing step (it consumes that
    step's captured output).
"""
from __future__ import annotations

from typing import Any

from vector_os_nano.vcli.cognitive.blackboard import Blackboard
from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.strategy_selector import StrategyResult
from vector_os_nano.vcli.cognitive.types import ForEachSpec, GoalTree, SubGoal


# ---------------------------------------------------------------------------
# A selector that routes each sub_goal's explicit strategy to a primitive.
# ---------------------------------------------------------------------------


class _DirectSelector:
    def select(self, sub_goal: SubGoal) -> StrategyResult:
        return StrategyResult(
            "primitive", sub_goal.strategy, dict(sub_goal.strategy_params)
        )


# ---------------------------------------------------------------------------
# A verifier that evaluates a tiny expression against a live namespace. We reuse
# the real GoalVerifier so the per-step predicate is the real deterministic gate.
# ---------------------------------------------------------------------------


def _build(objects: list[dict[str, Any]]) -> tuple[GoalExecutor, dict[str, Any]]:
    """Wire an executor whose ``detect`` step yields *objects*, plus a foreach
    body that 'picks' each item. Returns (executor, state) where ``state`` records
    the order of picked names so per-item binding can be asserted.
    """
    from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier

    state: dict[str, Any] = {"picked": []}

    def detect(**_: Any) -> dict[str, Any]:
        # Producing step output. result_data is captured wrapped under "output",
        # so the foreach resolves ${detect.output.objects}.
        return {"objects": list(objects), "count": len(objects)}

    def pick(object_label: str | None = None, **_: Any) -> dict[str, Any]:
        state["picked"].append(object_label)
        return {"picked": object_label}

    primitives = {"detect": detect, "pick": pick}

    # verify namespace: picked(label) -> bool checks the label was just picked.
    namespace = {
        "picked": lambda label: label in state["picked"],
        "detected": lambda: len(objects) > 0,
    }
    verifier = GoalVerifier(namespace)

    executor = GoalExecutor(
        strategy_selector=_DirectSelector(),
        verifier=verifier,
        primitives=primitives,
    )
    # A fresh run-scoped blackboard (the harness normally attaches this).
    executor.blackboard = Blackboard()
    return executor, state


def _foreach_tree() -> GoalTree:
    return GoalTree(
        goal="pick up every object on the table",
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
                            # Per-item verify: the just-picked label must match
                            # this iteration's object name (bound via the bb).
                            verify="picked('${obj.name}')",
                            strategy="pick",
                            strategy_params={"object_label": "${obj.name}"},
                        ),
                    ),
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# N items -> N expanded children, each verified.
# ---------------------------------------------------------------------------


def test_foreach_expands_one_child_per_item() -> None:
    objects = [{"name": "mug"}, {"name": "banana"}, {"name": "block"}]
    executor, state = _build(objects)

    trace = executor.execute(_foreach_tree())

    assert trace.success is True
    # detect + one grasp per item.
    names = [s.sub_goal_name for s in trace.steps]
    assert names == [
        "detect",
        "pick_each[0].grasp_one",
        "pick_each[1].grasp_one",
        "pick_each[2].grasp_one",
    ]
    # Every step both executed and verified deterministically.
    for s in trace.steps:
        assert s.success is True
        assert s.verify_result is True
        assert s.visual_override is False


def test_foreach_binds_each_item_in_order() -> None:
    objects = [{"name": "mug"}, {"name": "banana"}, {"name": "block"}]
    executor, state = _build(objects)

    executor.execute(_foreach_tree())

    # The per-item binding resolved ${obj.name} to item i's name, in order — the
    # pick primitive saw each label. (Proven via the recorded side effect, not the
    # step name.)
    assert state["picked"] == ["mug", "banana", "block"]


def test_foreach_child_sees_its_own_item_not_a_shared_ref() -> None:
    # Two items with distinct fields: each child must bind ITS row.
    objects = [{"name": "cup", "slot": 1}, {"name": "plate", "slot": 2}]
    executor, state = _build(objects)

    trace = executor.execute(_foreach_tree())

    assert trace.success is True
    assert state["picked"] == ["cup", "plate"]
    # The grasp child's captured output carries the resolved per-item label.
    grasp_steps = [s for s in trace.steps if "grasp_one" in s.sub_goal_name]
    labels = [s.result_data.get("output", {}).get("picked") for s in grasp_steps]
    assert labels == ["cup", "plate"]


# ---------------------------------------------------------------------------
# Empty list -> zero children, run still succeeds.
# ---------------------------------------------------------------------------


def test_foreach_empty_list_yields_zero_children() -> None:
    executor, state = _build([])  # detect produces an empty objects list

    # detect's verify (detected()) is False on an empty list — to isolate the
    # empty-foreach behaviour, make detect always verify True here.
    tree = _foreach_tree()
    detect = tree.sub_goals[0]
    detect_ok = SubGoal(
        name=detect.name,
        description=detect.description,
        verify="True",
        strategy=detect.strategy,
        depends_on=detect.depends_on,
    )
    tree = GoalTree(goal=tree.goal, sub_goals=(detect_ok, tree.sub_goals[1]))

    trace = executor.execute(tree)

    assert trace.success is True
    # Only the producing step ran — the foreach expanded to nothing.
    assert [s.sub_goal_name for s in trace.steps] == ["detect"]
    assert state["picked"] == []


def test_foreach_missing_producer_yields_zero_children() -> None:
    # Batch 2 #10 (design review): an UNRESOLVED producer is now a LOUD,
    # replannable failure — the old silent zero-iteration PASS structurally
    # broke the closed loop (no FailureRecord, replan could never correct).
    executor, _ = _build([{"name": "mug"}])
    loop = SubGoal(
        name="pick_each",
        description="pick each",
        verify="True",
        strategy="",
        foreach=ForEachSpec(
            source_step="never_ran",
            source_path="objects",
            var="obj",
            body=(
                SubGoal(
                    name="grasp_one",
                    description="pick",
                    verify="True",
                    strategy="pick",
                    strategy_params={"object_label": "${obj.name}"},
                ),
            ),
        ),
    )
    trace = executor.execute(GoalTree(goal="g", sub_goals=(loop,)))

    assert trace.success is False
    assert len(trace.steps) == 1
    assert trace.steps[0].failure_class == "exec_error"
    assert "did not resolve" in trace.steps[0].error


# ---------------------------------------------------------------------------
# depends_on: the foreach runs AFTER its producer (it consumes its output).
# ---------------------------------------------------------------------------


def test_foreach_runs_after_producing_step() -> None:
    objects = [{"name": "mug"}, {"name": "banana"}]
    executor, state = _build(objects)

    trace = executor.execute(_foreach_tree())

    names = [s.sub_goal_name for s in trace.steps]
    # detect first; every grasp child after it.
    assert names[0] == "detect"
    assert all("grasp_one" in n for n in names[1:])
    assert state["picked"] == ["mug", "banana"]


# ---------------------------------------------------------------------------
# Regression (CR): intra-BODY depends_on orders the body even when the body
# templates are listed out of dependency order. Previously the body ran in raw
# list order and silently ignored template-to-template depends_on, so a consumer
# could run before its producer.
# ---------------------------------------------------------------------------


def test_foreach_body_honors_intra_body_depends_on() -> None:
    from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier

    order: list[str] = []

    def detect(**_: Any) -> dict[str, Any]:
        return {"objects": [{"name": "mug"}], "count": 1}

    def producer(**_: Any) -> dict[str, Any]:
        order.append("producer")
        return {"ran": "producer"}

    def consumer(**_: Any) -> dict[str, Any]:
        order.append("consumer")
        return {"ran": "consumer"}

    primitives = {"detect": detect, "producer": producer, "consumer": consumer}
    namespace = {
        "detected": lambda: True,
        # consumer's verify only holds once the producer has already run, so an
        # out-of-order execution would make this verify fail.
        "producer_ran": lambda: "producer" in order,
        "ok": lambda: True,
    }
    executor = GoalExecutor(
        strategy_selector=_DirectSelector(),
        verifier=GoalVerifier(namespace),
        primitives=primitives,
    )
    executor.blackboard = Blackboard()

    tree = GoalTree(
        goal="ordered body",
        sub_goals=(
            SubGoal(
                name="detect",
                description="d",
                verify="detected()",
                strategy="detect",
            ),
            SubGoal(
                name="loop",
                description="l",
                verify="True",
                strategy="",
                depends_on=("detect",),
                foreach=ForEachSpec(
                    source_step="detect",
                    source_path="objects",
                    var="obj",
                    body=(
                        # consumer is listed FIRST but depends on producer.
                        SubGoal(
                            name="consumer",
                            description="consume",
                            verify="producer_ran()",
                            strategy="consumer",
                            depends_on=("producer",),
                        ),
                        SubGoal(
                            name="producer",
                            description="produce",
                            verify="ok()",
                            strategy="producer",
                        ),
                    ),
                ),
            ),
        ),
    )

    trace = executor.execute(tree)

    assert trace.success is True
    # producer ran before consumer despite the body listing consumer first.
    assert order == ["producer", "consumer"]
    names = [s.sub_goal_name for s in trace.steps]
    assert names == ["detect", "loop[0].producer", "loop[0].consumer"]
