# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Playground (INC6, PART A) — the verified loop on a real hand-authored chain.

Proves plan -> execute -> verify END-TO-END on a fixed tabletop chain BEFORE NL
drives it. A static GoalTree (a tuple of SubGoals — no foreach/loop) carries a
playground verify predicate per step:

    home   -> arm_at_home()
    detect -> len(detect_objects()) > 0
    grasp  -> holding_object()
    place  -> placed_count(region) >= 1

The chain runs through the REAL GoalExecutor with the REAL GoalVerifier off the
engine's MERGED playground verify namespace (engine._world = PlaygroundWorld). A
deterministic stub arm/gripper advances the sim-oracle state as each primitive
executes, so verify legitimately flips False -> True step by step. The run must
reach verified-done and emit a sane INC5 observation snapshot.

No MuJoCo / network — the arm + gripper are deterministic stubs sharing the same
oracle surface the predicates read. The real MuJoCoArm shape is guarded
separately by ``test_playground_real_arm.py`` (Part B).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from vector_os_nano.playground import PlaygroundWorld
from vector_os_nano.playground.verify.arm_predicates import _HOME_JOINTS
from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.observation import run_snapshot, step_view
from vector_os_nano.vcli.cognitive.strategy_selector import StrategyResult
from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal


# ---------------------------------------------------------------------------
# Deterministic stub arm + gripper — a mutable sim oracle the chain advances.
# ---------------------------------------------------------------------------
#
# These share the exact oracle surface the playground predicates read
# (get_object_positions / get_joint_positions / fk on the arm; is_holding on the
# gripper). Each primitive in the chain mutates this state, so the very same
# verify predicate reads False before its step and True after it.

_NOT_HOME_JOINTS: tuple[float, ...] = tuple(j + 0.5 for j in _HOME_JOINTS)
_EE = [0.20, 0.10, 0.25]  # end-effector xyz used by holding_object()
_REGION = (0.0, 0.0, 0.5, 0.5)  # place target region (x_min, y_min, x_max, y_max)


class StubArm:
    """Mutable stand-in for MuJoCoArm's oracle surface (no MuJoCo)."""

    def __init__(self) -> None:
        # Start away from home, with the mug resting on the table (z below the
        # lift height). banana is a distractor that stays put.
        self._joints: list[float] = list(_NOT_HOME_JOINTS)
        self._objects: dict[str, list[float]] = {
            "mug": [0.21, 0.10, 0.06],
            "banana": [0.40, 0.40, 0.06],
        }
        self._connected = True

    def get_joint_positions(self) -> list[float]:
        return list(self._joints)

    def get_object_positions(self) -> dict[str, list[float]]:
        return {k: list(v) for k, v in self._objects.items()}

    def fk(self, joint_positions: list[float]):
        return list(_EE), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    # --- mutations driven by the chain's primitives ---

    def go_home(self) -> None:
        self._joints = list(_HOME_JOINTS)

    def lift(self, name: str) -> None:
        # Lift the named object up next to the EE (above the lift height).
        self._objects[name] = [_EE[0] + 0.01, _EE[1], _EE[2]]

    def place(self, name: str, xy: tuple[float, float]) -> None:
        # Lower the object back onto the table inside the target region.
        self._objects[name] = [xy[0], xy[1], 0.06]


class StubGripper:
    def __init__(self) -> None:
        self._holding = False

    def is_holding(self) -> bool:
        return self._holding

    def close(self) -> None:
        self._holding = True

    def open(self) -> None:
        self._holding = False


# ---------------------------------------------------------------------------
# A selector that routes each SubGoal to its named primitive (no rules needed).
# ---------------------------------------------------------------------------


class _DirectSelector:
    """Routes a sub_goal's explicit strategy straight to a primitive."""

    def select(self, sub_goal: SubGoal) -> StrategyResult:
        return StrategyResult(
            "primitive", sub_goal.strategy, dict(sub_goal.strategy_params)
        )


# ---------------------------------------------------------------------------
# The hand-authored tabletop chain (static — NO foreach/loop construct).
# ---------------------------------------------------------------------------


def _tabletop_chain() -> GoalTree:
    return GoalTree(
        goal="home, detect the table, grasp the mug, place it in the bin",
        sub_goals=(
            SubGoal(
                name="home_arm",
                description="move the arm to its home pose",
                verify="arm_at_home()",
                strategy="go_home",
            ),
            SubGoal(
                name="detect_scene",
                description="detect objects on the table",
                verify="len(detect_objects()) > 0",
                strategy="observe",
                depends_on=("home_arm",),
            ),
            SubGoal(
                name="grasp_mug",
                description="grasp the mug and lift it",
                verify="holding_object()",
                strategy="grasp_lift",
                depends_on=("detect_scene",),
            ),
            SubGoal(
                name="place_mug",
                description="place the mug in the target region",
                verify="placed_count((0.0, 0.0, 0.5, 0.5)) >= 1",
                strategy="place_in_region",
                depends_on=("grasp_mug",),
            ),
        ),
    )


def _build(arm: StubArm, gripper: StubGripper) -> GoalExecutor:
    """Wire the REAL executor + REAL verifier over the merged playground ns."""
    agent: Any = SimpleNamespace(_arm=arm, _gripper=gripper)

    # The verify namespace is the playground world's predicates, exactly as the
    # engine merges them (PlaygroundWorld.build_verify_namespace bound to agent).
    namespace = PlaygroundWorld().build_verify_namespace(agent)
    verifier = GoalVerifier(namespace)

    # Primitives are the chain's side effects: each advances the shared oracle so
    # the next verify reads the new ground truth. They return None (the executor
    # treats a non-bool, non-exception return as success).
    def go_home(**_: Any) -> dict[str, Any]:
        arm.go_home()
        return {"action": "home"}

    def observe(**_: Any) -> dict[str, Any]:
        # A no-op perception step: the objects are already on the table; detect
        # reads them through the oracle. Returns the count for the trace.
        return {"detected": len(arm.get_object_positions())}

    def grasp_lift(**_: Any) -> dict[str, Any]:
        gripper.close()
        arm.lift("mug")
        return {"grabbed": "mug"}

    def place_in_region(**_: Any) -> dict[str, Any]:
        arm.place("mug", (_REGION[0] + 0.1, _REGION[1] + 0.1))
        gripper.open()
        return {"placed": "mug"}

    primitives = {
        "go_home": go_home,
        "observe": observe,
        "grasp_lift": grasp_lift,
        "place_in_region": place_in_region,
    }
    return GoalExecutor(
        strategy_selector=_DirectSelector(),
        verifier=verifier,
        primitives=primitives,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_chain_reaches_verified_done() -> None:
    """The full home->detect->grasp->place chain reaches verified-done, with each
    step's deterministic predicate passing through the real GoalVerifier."""
    arm = StubArm()
    gripper = StubGripper()
    executor = _build(arm, gripper)

    trace = executor.execute(_tabletop_chain())

    assert trace.success is True
    assert [s.sub_goal_name for s in trace.steps] == [
        "home_arm",
        "detect_scene",
        "grasp_mug",
        "place_mug",
    ]
    # Every step both EXECUTED and VERIFIED via the deterministic predicate (not
    # a visual override — this is real evidence).
    for s in trace.steps:
        assert s.success is True
        assert s.verify_result is True
        assert s.visual_override is False


def test_verify_flips_false_to_true_as_chain_progresses() -> None:
    """Before its step runs, each step's predicate is False against the oracle;
    after the chain advances the oracle, it is True. Proves verify is grounded in
    ground truth, not a fixed stub."""
    arm = StubArm()
    gripper = StubGripper()
    agent: Any = SimpleNamespace(_arm=arm, _gripper=gripper)
    gv = GoalVerifier(PlaygroundWorld().build_verify_namespace(agent))

    # Initial oracle: away from home, nothing held, mug resting (already placed).
    assert gv.verify("arm_at_home()") is False
    assert gv.verify("holding_object()") is False

    # detect is True from the start (objects are on the table).
    assert gv.verify("len(detect_objects()) > 0") is True

    # Drive the oracle the way the chain's primitives do and re-check.
    arm.go_home()
    assert gv.verify("arm_at_home()") is True

    gripper.close()
    arm.lift("mug")
    assert gv.verify("holding_object()") is True

    arm.place("mug", (0.1, 0.1))
    gripper.open()
    assert gv.verify("holding_object()") is False
    assert gv.verify("placed_count((0.0, 0.0, 0.5, 0.5)) >= 1") is True


def test_chain_emits_sane_observation_snapshot() -> None:
    """The run produces a JSON-safe INC5 snapshot carrying the goal tree, every
    step's success + verify_result, and round-trips through json.dumps."""
    arm = StubArm()
    gripper = StubGripper()
    executor = _build(arm, gripper)

    captured: list[dict[str, Any]] = []
    trace = executor.execute(
        _tabletop_chain(), on_step=lambda s: captured.append(step_view(s))
    )

    snapshot = run_snapshot(trace)

    assert snapshot["goal"].startswith("home, detect")
    assert [sg["name"] for sg in snapshot["goal_tree"]["sub_goals"]] == [
        "home_arm",
        "detect_scene",
        "grasp_mug",
        "place_mug",
    ]
    # Each sub-goal carries its playground verify predicate in the snapshot.
    verifies = [sg["verify"] for sg in snapshot["goal_tree"]["sub_goals"]]
    assert verifies == [
        "arm_at_home()",
        "len(detect_objects()) > 0",
        "holding_object()",
        "placed_count((0.0, 0.0, 0.5, 0.5)) >= 1",
    ]

    assert snapshot["success"] is True
    assert len(snapshot["steps"]) == 4
    for sv in snapshot["steps"]:
        assert sv["success"] is True
        assert sv["verify_result"] is True

    # WP02: snapshot steps carry a graded evidence key the live view lacks.
    stripped = [{k: v for k, v in sv.items() if k != "evidence"} for sv in snapshot["steps"]]
    assert captured == stripped
    # Full snapshot round-trips cleanly.
    json.dumps(snapshot)
