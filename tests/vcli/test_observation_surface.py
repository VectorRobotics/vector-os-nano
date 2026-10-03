# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Observation surface (INC5) — the verified-loop's serializable view.

This covers the SHARED-PRELUDE item (3): the structured, JSON-serializable
snapshot a front-end (the playground view) renders. The tests exercise:

- ``step_view`` carries exactly the per-step fields the UI needs and is JSON-safe
  even when a strategy emits an exotic ``result_data`` value.
- ``run_snapshot`` over a REAL decompose+execute (tiny hand-built GoalTree driven
  through a real ``GoalExecutor`` with fake selector/verifier) carries the goal
  tree, every step's success+verify_result+result_data, and json.dumps succeeds.
- ``validation_notes`` (replan feedback) survive into the snapshot.
- The engine fans the per-step view out to the ``on_vgg_step_view`` sink and the
  run-complete snapshot is reachable via ``vgg_run_snapshot``.

No MuJoCo / network — deterministic fakes only.
"""

from __future__ import annotations

import json
from typing import Any

from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.observation import (
    SNAPSHOT_VERSION,
    goal_tree_view,
    run_snapshot,
    step_view,
)
from vector_os_nano.vcli.cognitive.strategy_selector import StrategyResult
from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)


# ---------------------------------------------------------------------------
# Deterministic fakes — drive a REAL GoalExecutor without sim/network.
# ---------------------------------------------------------------------------


class _FakeSelector:
    """Routes every sub_goal to a named primitive we inject below."""

    def __init__(self, primitive_name: str) -> None:
        self._name = primitive_name

    def select(self, sub_goal: SubGoal) -> StrategyResult:
        return StrategyResult("primitive", self._name, dict(sub_goal.strategy_params))


class _FakeVerifier:
    """Deterministic verifier that returns a fixed (bool, value)."""

    def __init__(self, ok: bool, value: Any) -> None:
        self._ok = ok
        self._value = value

    def evaluate(self, expression: str) -> tuple[bool, Any]:
        return self._ok, self._value

    def verify(self, expression: str) -> bool:
        return self._ok


def _build_executor(primitive_name: str, primitive_fn: Any, ok: bool, value: Any) -> GoalExecutor:
    return GoalExecutor(
        strategy_selector=_FakeSelector(primitive_name),
        verifier=_FakeVerifier(ok, value),
        primitives={primitive_name: primitive_fn},
    )


# ---------------------------------------------------------------------------
# step_view
# ---------------------------------------------------------------------------


def test_step_view_carries_ui_fields_and_is_json_safe() -> None:
    step = StepRecord(
        sub_goal_name="pick_banana",
        strategy="grasp",
        success=True,
        verify_result=True,
        duration_sec=0.5,
        error="",
        result_data={"output": {"value": object()}, "verify_value": True},
    )
    view = step_view(step)

    assert view["sub_goal_name"] == "pick_banana"
    assert view["strategy"] == "grasp"
    assert view["success"] is True
    assert view["verify_result"] is True
    assert view["error"] == ""
    # result_data is preserved but coerced to JSON-safe primitives (the
    # non-serializable object() became a string, never an exception).
    assert isinstance(view["result_data"]["output"]["value"], str)
    # Round-trips cleanly.
    json.dumps(view)


def test_step_view_handles_empty_result_data() -> None:
    step = StepRecord(
        sub_goal_name="home",
        strategy="home",
        success=False,
        verify_result=False,
        duration_sec=0.1,
        error="boom",
    )
    view = step_view(step)
    assert view["success"] is False
    assert view["error"] == "boom"
    assert view["result_data"] == {}
    json.dumps(view)


# ---------------------------------------------------------------------------
# goal_tree_view
# ---------------------------------------------------------------------------


def test_goal_tree_view_shape() -> None:
    tree = GoalTree(
        goal="tidy the table",
        sub_goals=(
            SubGoal(name="a", description="step a", verify="True", strategy="s1"),
            SubGoal(
                name="b",
                description="step b",
                verify="placed_count() == 1",
                strategy="s2",
                depends_on=("a",),
            ),
        ),
        validation_notes=("strategy 'look_skill' is not valid; cleared",),
    )
    view = goal_tree_view(tree)
    assert view["goal"] == "tidy the table"
    assert [sg["name"] for sg in view["sub_goals"]] == ["a", "b"]
    assert view["sub_goals"][1]["depends_on"] == ["a"]
    assert view["sub_goals"][1]["verify"] == "placed_count() == 1"
    assert view["validation_notes"] == ["strategy 'look_skill' is not valid; cleared"]
    json.dumps(view)


# ---------------------------------------------------------------------------
# run_snapshot over a REAL execute
# ---------------------------------------------------------------------------


def test_run_snapshot_over_real_execute() -> None:
    captured_steps: list[dict[str, Any]] = []

    def move(**_: Any) -> dict[str, Any]:
        return {"moved_to": [0.1, 0.2, 0.3]}

    executor = _build_executor("move", move, ok=True, value=True)

    tree = GoalTree(
        goal="move the arm twice",
        sub_goals=(
            SubGoal(name="first", description="move once", verify="arm_at_home()", strategy="move"),
            SubGoal(
                name="second",
                description="move again",
                verify="arm_at_home()",
                strategy="move",
                depends_on=("first",),
            ),
        ),
    )

    trace = executor.execute(tree, on_step=lambda s: captured_steps.append(step_view(s)))

    snapshot = run_snapshot(trace)

    # Goal tree present.
    assert snapshot["snapshot_version"] == SNAPSHOT_VERSION
    assert snapshot["goal"] == "move the arm twice"
    assert [sg["name"] for sg in snapshot["goal_tree"]["sub_goals"]] == ["first", "second"]

    # Each step carries success + verify_result + result_data.
    assert len(snapshot["steps"]) == 2
    for sv in snapshot["steps"]:
        assert sv["success"] is True
        assert sv["verify_result"] is True
        assert "result_data" in sv
        assert sv["result_data"]["output"]["moved_to"] == [0.1, 0.2, 0.3]

    assert snapshot["success"] is True
    # WP02: the run-complete snapshot grades each step (evidence key); the live
    # per-step view is pre-grading and carries no evidence. They match otherwise.
    for sv in snapshot["steps"]:
        assert sv["evidence"] in ("GROUNDED", "RAN", "FAILED")
    stripped = [{k: v for k, v in sv.items() if k != "evidence"} for sv in snapshot["steps"]]
    assert captured_steps == stripped

    # Full snapshot round-trips through json.dumps.
    json.dumps(snapshot)


def test_run_snapshot_includes_validation_notes_on_replan() -> None:
    # A replan/validator-dropped strategy records human-readable notes on the
    # GoalTree; the snapshot must surface them for the UI.
    tree = GoalTree(
        goal="grab the mug",
        sub_goals=(
            SubGoal(name="grab", description="grab mug", verify="holding_object('mug')", strategy=""),
        ),
        validation_notes=(
            "strategy 'teleport_skill' is not valid; cleared to ''",
        ),
    )
    trace = ExecutionTrace(
        goal_tree=tree,
        steps=(
            StepRecord(
                sub_goal_name="grab",
                strategy="grasp",
                success=False,
                verify_result=False,
                duration_sec=0.2,
                error="failed after fallback",
                result_data={"output": {}, "verify_value": False},
            ),
        ),
        success=False,
        total_duration_sec=0.2,
    )

    snapshot = run_snapshot(trace)

    assert snapshot["validation_notes"] == [
        "strategy 'teleport_skill' is not valid; cleared to ''"
    ]
    # Mirrored inside the tree view too.
    assert snapshot["goal_tree"]["validation_notes"] == snapshot["validation_notes"]
    assert snapshot["success"] is False
    json.dumps(snapshot)


# ---------------------------------------------------------------------------
# Engine wiring — observation surface fans out without changing behaviour.
# ---------------------------------------------------------------------------


def test_engine_emits_step_view_and_run_snapshot() -> None:
    from vector_os_nano.vcli.engine import VectorEngine

    eng = VectorEngine.__new__(VectorEngine)  # skip heavy __init__

    raw_steps: list[Any] = []
    view_steps: list[dict[str, Any]] = []

    eng._vgg_step_callback = raw_steps.append
    eng._vgg_step_view_callback = view_steps.append

    step = StepRecord(
        sub_goal_name="pick",
        strategy="grasp",
        success=True,
        verify_result=True,
        duration_sec=0.3,
        result_data={"output": {"grabbed": "mug"}, "verify_value": True},
    )

    eng._on_vgg_step(step)

    # Raw callback unchanged; view callback got the JSON-safe export view.
    assert raw_steps == [step]
    assert len(view_steps) == 1
    assert view_steps[0]["sub_goal_name"] == "pick"
    assert view_steps[0]["success"] is True
    json.dumps(view_steps[0])

    # Run-complete accessor yields the full snapshot.
    trace = ExecutionTrace(
        goal_tree=GoalTree(goal="g", sub_goals=()),
        steps=(step,),
        success=True,
        total_duration_sec=0.3,
    )
    snapshot = eng.vgg_run_snapshot(trace)
    assert snapshot["goal"] == "g"
    assert snapshot["steps"][0]["sub_goal_name"] == "pick"
    json.dumps(snapshot)


def test_engine_run_snapshot_grades_against_the_world_scoped_tagged_oracles() -> None:
    """vgg_run_snapshot grades with verify_oracle_names: the engine's namespace, its
    @oracle tags, filtered by the active world's scope (a dev predicate in a robot
    world, or an untagged callable, never grounds)."""
    from types import SimpleNamespace

    from vector_os_nano.vcli.cognitive.oracle_registry import oracle
    from vector_os_nano.vcli.engine import VectorEngine

    eng = VectorEngine.__new__(VectorEngine)  # skip heavy __init__
    eng._build_verifier_namespace = lambda agent: {
        "holding_object": oracle(scope="robot")(lambda *a: True),
        "file_exists": oracle(scope="dev")(lambda *a: True),
        "nearest_room": lambda: "kitchen",
    }
    eng._world = SimpleNamespace(is_robot=lambda: True)

    def _snap(verify: str) -> dict[str, Any]:
        sg = SubGoal(name="s", description="d", verify=verify)
        step = StepRecord(
            sub_goal_name="s", strategy="x", success=True, verify_result=True, duration_sec=0.1
        )
        trace = ExecutionTrace(
            goal_tree=GoalTree(goal="g", sub_goals=(sg,)), steps=(step,), success=True,
            total_duration_sec=0.1,
        )
        return eng.vgg_run_snapshot(trace)

    snap = _snap("holding_object('mug')")
    assert snap["steps"][0]["evidence"] == "GROUNDED" and snap["verified"] is True
    for verify in ("file_exists('a.txt')", "nearest_room() == 'kitchen'"):
        snap = _snap(verify)
        assert snap["steps"][0]["evidence"] == "RAN" and snap["verified"] is False, verify
