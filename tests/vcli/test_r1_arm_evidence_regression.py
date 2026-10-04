# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""R1 regression — the honest evidence gate, fail-closed, no hardware.

Campaign #13 R1 deleted the ``if is_robot: return True`` bypass: a robot step is
no longer auto-passed. These tests pin the NEW honest behaviour against a
deterministic FAKE sim arm/base (no MuJoCo, no /dev/ttyACM*), proving the gate
fails CLOSED in exactly the cases the bypass used to wave through:

(a) KNOWN-BAD: a connected sim arm OFF home -> ``arm_at_home()`` is False AND the
    step classifies non-GROUNDED (even though it "succeeded").
(b) ABSENT ORACLE: drop ``arm_at_home`` from oracle_names -> the SAME passing
    arm-at-home step classifies RAN, never GROUNDED (fail closed -> stricter).
(c) LIVE NAMESPACE: ``arm_at_home`` IS in ``verify_oracle_names(agent, engine)``
    for an arm-connected agent (single-sourced from the verifier namespace).
(d) ROBOT REACHES GROUNDED: the re-keyed robot examples (goal_decomposer /
    vocab_from_registry now emit ``at_position(...)`` not ``"True"``) let a robot
    step reach GROUNDED — a base AT the target with verify_result True classifies
    GROUNDED, so honest robot verification is achievable, not merely stricter.

Pure kernel logic on deterministic fakes — no robot, no network, no mujoco.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from vector_os_nano.vcli.cognitive.evidence_classifier import classify_verify_expr
from vector_os_nano.vcli.cognitive.trace_store import (
    classify_step_evidence,
    step_evidence_ok,
    verify_oracle_names,
)
from vector_os_nano.vcli.cognitive.types import StepRecord, SubGoal
from vector_os_nano.vcli.engine import VectorEngine
from vector_os_nano.vcli.permissions import PermissionContext
from vector_os_nano.vcli.tools.base import CategorizedToolRegistry
from tests.harness.fake_gt import bind_fake_sim, fake_namespace
from vector_os_nano.embodiments import registry
from vector_os_nano.vcli.worlds.robot import RobotWorld

# The SO-101 home pose (its manifest stance; the arm_at_home oracle reads it there).
_HOME_JOINTS = tuple(registry.load("so101").stance.values())

# Robot verify-namespace oracle names (the sim arm/base subset). Mirrors what
# verify_oracle_names(agent, engine) single-sources from the live namespace.
ROBOT_ORACLES = frozenset({
    "at_position", "facing", "visited", "holding_object", "arm_at_home",
    "describe_scene", "detect_objects", "placed_count",
})


# ---------------------------------------------------------------------------
# Deterministic fakes (no MuJoCo, no hardware) — duck-typed to the oracle surface
# ---------------------------------------------------------------------------


class _FakeArm:
    """Connected sim-arm stand-in exposing the oracle surface the predicates read.

    Mirrors tests/vcli/test_playground_predicates.py::FakeArm — no MuJoCo.
    """

    def __init__(self, joints: list[float], objects: dict[str, list[float]] | None = None) -> None:
        self._joints = list(joints)
        self._objects = objects or {}
        self._connected = True

    def get_object_positions(self) -> dict[str, list[float]]:
        return {k: list(v) for k, v in self._objects.items()}

    def get_joint_positions(self) -> list[float]:
        return list(self._joints)

    def fk(self, joint_positions: list[float]):
        return [0.2, 0.0, 0.2], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


class _FakeBase:
    """Connected sim-base stand-in: deterministic ground-truth pose."""

    def __init__(self, x: float, y: float, yaw: float = 0.0) -> None:
        self._pos = (float(x), float(y), 0.0)
        self._yaw = float(yaw)
        self._connected = True

    def get_position(self):
        return self._pos

    def get_heading(self) -> float:
        return self._yaw


def _arm_agent(arm: Any) -> SimpleNamespace:
    return SimpleNamespace(_arm=arm, _gripper=None, _base=None)


def _motor_step(strategy: str = "home_skill") -> StepRecord:
    """A step that SUCCEEDED at the motor level (the bypass used to wave this)."""
    return StepRecord(
        sub_goal_name="go_home",
        strategy=strategy,
        success=True,
        verify_result=True,
        duration_sec=0.1,
    )


def _sub_goal(verify: str, strategy: str = "home_skill") -> SubGoal:
    return SubGoal(name="go_home", description="return the arm home", verify=verify, strategy=strategy)


# ---------------------------------------------------------------------------
# (a) KNOWN-BAD: arm OFF home -> predicate False AND step non-GROUNDED
# ---------------------------------------------------------------------------


def test_arm_off_home_predicate_is_false() -> None:
    off_home = [j + 0.5 for j in _HOME_JOINTS]  # well beyond _HOME_TOL_RAD (0.10)
    arm = _FakeArm(joints=off_home)
    arm_at_home = fake_namespace(_arm_agent(arm), "so101")["arm_at_home"]
    assert arm_at_home() is False


def test_arm_off_home_step_is_not_grounded() -> None:
    # The motor step "succeeded" and the sub-goal's verify reads a real oracle
    # (arm_at_home()), but the arm is OFF home so verify_result is the truth: with
    # an honest verify_result=False, the step is NOT GROUNDED. This is the case the
    # old ``if is_robot: return True`` bypass used to pass — it now fails closed.
    off_home = [j + 0.5 for j in _HOME_JOINTS]
    arm = _FakeArm(joints=off_home)
    arm_at_home = fake_namespace(_arm_agent(arm), "so101")["arm_at_home"]
    # Predicate is the ground truth the verifier would compute.
    real_verify_result = arm_at_home()
    assert real_verify_result is False
    step = StepRecord(
        sub_goal_name="go_home",
        strategy="home_skill",
        success=True,
        verify_result=real_verify_result,  # honest: the arm is not home
        duration_sec=0.1,
    )
    sg = _sub_goal("arm_at_home()")
    assert classify_step_evidence(step, sg, ROBOT_ORACLES) != "GROUNDED"
    assert step_evidence_ok(step, sg, ROBOT_ORACLES) is False
    # And via the structural classifier directly: arm_at_home() over the oracle set
    # IS a GROUNDED-shaped predicate (a bare predicate-oracle call), so the ONLY
    # thing keeping this step out of GROUNDED is the honest verify_result=False.
    assert classify_verify_expr("arm_at_home()", ROBOT_ORACLES) == "GROUNDED"


# ---------------------------------------------------------------------------
# (b) ABSENT ORACLE: drop arm_at_home -> RAN, never GROUNDED (fail closed)
# ---------------------------------------------------------------------------


def test_absent_oracle_collapses_to_ran() -> None:
    # Even a step whose verify_result is True classifies RAN when arm_at_home is
    # NOT in the oracle set: an absent namespace can only make the gate stricter.
    oracles_without_arm = ROBOT_ORACLES - {"arm_at_home"}
    step = _motor_step()  # verify_result True
    sg = _sub_goal("arm_at_home()")
    assert classify_verify_expr("arm_at_home()", oracles_without_arm) == "RAN"
    assert classify_step_evidence(step, sg, oracles_without_arm) == "RAN"
    assert step_evidence_ok(step, sg, oracles_without_arm) is False


# ---------------------------------------------------------------------------
# (c) LIVE NAMESPACE: arm_at_home IS in verify_oracle_names(agent, engine)
# ---------------------------------------------------------------------------


def test_arm_at_home_in_live_verify_oracle_names() -> None:
    engine = VectorEngine(
        backend=None,
        registry=CategorizedToolRegistry(),
        permissions=PermissionContext(),
    )
    engine._world = RobotWorld()  # robot world merges the sim-arm oracle namespace
    # A connected sim arm exposing get_object_positions -> RobotWorld adds the arm
    # predicates (arm_at_home / holding_object / ...) on top of the engine bindings.
    agent = _arm_agent(_FakeArm(joints=list(_HOME_JOINTS), objects={"mug": [0.2, 0.1, 0.06]}))
    bind_fake_sim(agent, "so101")  # the live sim session's GT oracles
    names = verify_oracle_names(agent, engine)
    assert "arm_at_home" in names
    # Single-sourcing sanity: it is the SAME namespace the verifier uses.
    assert names == frozenset(engine._build_verifier_namespace(agent).keys())


def test_verify_oracle_names_fails_closed_without_engine() -> None:
    # No engine -> empty set (fail closed). The moat only ever gets stricter.
    assert verify_oracle_names(_arm_agent(_FakeArm(joints=list(_HOME_JOINTS))), None) == frozenset()


# ---------------------------------------------------------------------------
# (d) ROBOT REACHES GROUNDED: the re-keyed examples let a robot step verify
# ---------------------------------------------------------------------------


def test_robot_base_step_reaches_grounded() -> None:
    # The decomposer's robot example now emits verify="at_position(2.0, 0.0)" (not
    # the old "True" sentinel). A base AT the target, verify_result True, over a
    # robot oracle set including at_position -> GROUNDED. >=1 robot task can now
    # honestly verify (the gate is not merely stricter — it is reachable).
    sg = SubGoal(
        name="walk_forward_2m",
        description="向前走2米",
        verify="at_position(2.0, 0.0)",
        strategy="walk_forward",
    )
    step = StepRecord(
        sub_goal_name="walk_forward_2m",
        strategy="walk_forward",
        success=True,
        verify_result=True,  # the base actually reached (2.0, 0.0)
        duration_sec=0.1,
    )
    assert classify_verify_expr("at_position(2.0, 0.0)", ROBOT_ORACLES) == "GROUNDED"
    assert classify_step_evidence(step, sg, ROBOT_ORACLES) == "GROUNDED"
    assert step_evidence_ok(step, sg, ROBOT_ORACLES) is True


def test_robot_base_at_position_predicate_is_grounded_against_live_oracle() -> None:
    # End-to-end on the live base oracle: a base AT (2.0, 0.0) makes at_position
    # True, OFF it makes it False — the predicate is real, not a tautology.
    def make_at_position(agent):
        return fake_namespace(agent, "go2")["at_position"]

    at_target = SimpleNamespace(_base=_FakeBase(2.0, 0.0), _arm=None, _gripper=None)
    off_target = SimpleNamespace(_base=_FakeBase(0.0, 0.0), _arm=None, _gripper=None)
    assert make_at_position(at_target)(2.0, 0.0) is True
    assert make_at_position(off_target)(2.0, 0.0) is False
