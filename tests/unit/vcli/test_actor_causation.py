# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""R2b unit tests — actor-causation grading (deterministic fakes, no MuJoCo).

Pin the actor-causation mechanism in isolation, on duck-typed fakes (no sim):

(snapshot)  capture(agent, session) reads ONLY the drivers' commanded-motion counters
            and the session's GROUND-TRUTH pose; a (0,0,0) "write" (zero magnitude)
            does NOT satisfy causation; a baseline from another session never causes.
(grade)     honest / no-op / teleport / missing-baseline(fail-closed) -> the right
            CAUSED/UNCAUSED grade.
(gate)      a robot at_position step is GROUNDED only when CAUSED (single-source
            through classify_step_evidence).
(holding)   weld eq_active 0->1 grades CAUSED; no transition grades UNCAUSED.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vector_os_nano.vcli.cognitive.actor_causation import (
    ActorCaused,
    capture,
    grade,
    is_robot_predicate,
)
from vector_os_nano.vcli.cognitive.oracle_registry import OracleTag
from vector_os_nano.vcli.cognitive.trace_store import classify_step_evidence
from vector_os_nano.vcli.cognitive.types import StepRecord, SubGoal

# The live evidence oracles with their @oracle tags (channel / transition), as
# trace_store.verify_oracles returns them.
_BASE = OracleTag(scope="robot", channel="base", transition="rise")
ROBOT_ORACLES = {
    "at_position": _BASE, "facing": _BASE, "visited": _BASE,
    "holding_object": OracleTag(scope="robot", channel="gripper", transition="rise"),
    "arm_at_home": OracleTag(scope="robot", channel="arm", transition="rise"),
}


# ---------------------------------------------------------------------------
# Fakes — duck-typed to the actor-causation capture surface
# ---------------------------------------------------------------------------


class _FakeBase:
    """A go2-base stand-in exposing the instrumented surface + pose."""

    def __init__(self, x: float, y: float, yaw: float = 0.0, cmd_motion: float = 0.0) -> None:
        self._x, self._y, self._yaw = float(x), float(y), float(yaw)
        self._cmd_motion = float(cmd_motion)
        self._connected = True

    def get_position(self) -> list[float]:
        return [self._x, self._y, 0.0]

    def get_heading(self) -> float:
        return self._yaw

    def cmd_motion(self) -> float:
        return self._cmd_motion

    # --- test helpers to simulate commands / motion ---
    def command(self, mag: float) -> None:
        self._cmd_motion += float(mag)

    def teleport(self, x: float, y: float) -> None:
        self._x, self._y = float(x), float(y)


def _agent(base=None, arm=None, gripper=None) -> SimpleNamespace:
    return SimpleNamespace(_base=base, _arm=arm, _gripper=gripper)


def _cap(agent, uid=1):
    """capture() in sim session *uid*, whose ground truth is the fakes' own state."""
    from tests.harness.fake_gt import FakeGT

    return capture(agent, SimpleNamespace(gt=FakeGT(agent), uid=uid))


# ---------------------------------------------------------------------------
# (snapshot) capture reads ungated counters; a (0,0,0) write doesn't count
# ---------------------------------------------------------------------------


def test_capture_snapshots_counter_and_pose() -> None:
    base = _FakeBase(10.0, 3.0, cmd_motion=1.5)
    b = _cap(_agent(base=base))
    assert b.base_cmd_motion == 1.5
    assert b.base_pos == (10.0, 3.0, 0.0)


def test_capture_none_agent_is_all_none() -> None:
    b = capture(None)
    assert b.base_cmd_motion is None and b.base_pos is None and b.arm_joints is None


def test_zero_command_does_not_advance_motion() -> None:
    # The go2 contract: set_velocity(0,0,0) adds a write but ZERO magnitude. Model
    # that here — a base whose cmd_motion is unchanged after a stop -> UNCAUSED.
    base = _FakeBase(10.0, 3.0, cmd_motion=2.0)
    before = _cap(_agent(base=base))
    base.command(0.0)  # a (0,0,0) stop: magnitude 0
    after = _cap(_agent(base=base))
    assert after.base_cmd_motion == before.base_cmd_motion  # no advance
    g = grade(before, after, "at_position(10.0, 3.0)", ROBOT_ORACLES)
    assert g == ActorCaused.UNCAUSED


# ---------------------------------------------------------------------------
# (grade) honest / no-op / teleport / missing-baseline
# ---------------------------------------------------------------------------


def test_grade_honest_walk_is_caused() -> None:
    base = _FakeBase(10.0, 3.0, cmd_motion=0.0)
    before = _cap(_agent(base=base))
    # The skill commands motion AND the base displaces ~1m.
    base.command(0.3)
    base.teleport(11.0, 3.0)
    after = _cap(_agent(base=base))
    assert grade(before, after, "at_position(11.0, 3.0)", ROBOT_ORACLES) == ActorCaused.CAUSED


def test_grade_noop_is_uncaused() -> None:
    # The target is already satisfied at baseline; no command, no displacement.
    base = _FakeBase(10.0, 3.0, cmd_motion=0.0)
    before = _cap(_agent(base=base))
    after = _cap(_agent(base=base))  # nothing happened
    assert grade(before, after, "at_position(10.0, 3.0)", ROBOT_ORACLES) == ActorCaused.UNCAUSED


def test_grade_teleport_is_uncaused() -> None:
    # The pose JUMPS far, but NO command was issued (qpos poked) -> UNCAUSED.
    base = _FakeBase(10.0, 3.0, cmd_motion=0.0)
    before = _cap(_agent(base=base))
    base.teleport(11.0, 3.0)  # large displacement, zero commanded motion
    after = _cap(_agent(base=base))
    assert grade(before, after, "at_position(11.0, 3.0)", ROBOT_ORACLES) == ActorCaused.UNCAUSED


def test_grade_missing_baseline_fails_closed() -> None:
    base = _FakeBase(11.0, 3.0, cmd_motion=5.0)
    post = _cap(_agent(base=base))
    assert grade(None, post, "at_position(11.0, 3.0)", ROBOT_ORACLES) == ActorCaused.UNCAUSED


def test_grade_command_without_displacement_is_uncaused() -> None:
    # Commanded motion but the robot did NOT move (e.g. wall, fell) -> UNCAUSED:
    # both conjuncts (command AND pose change) are required.
    base = _FakeBase(10.0, 3.0, cmd_motion=0.0)
    before = _cap(_agent(base=base))
    base.command(0.5)  # commanded, but no teleport => no displacement
    after = _cap(_agent(base=base))
    assert grade(before, after, "at_position(11.0, 3.0)", ROBOT_ORACLES) == ActorCaused.UNCAUSED


# ---------------------------------------------------------------------------
# (gate) robot at_position step GROUNDED only when CAUSED (single-source)
# ---------------------------------------------------------------------------


def _grounded_step(actor_caused: ActorCaused) -> tuple[StepRecord, SubGoal]:
    sg = SubGoal(name="walk", description="d", verify="at_position(11.0, 3.0)", strategy="walk_forward")
    step = StepRecord(
        sub_goal_name="walk",
        strategy="walk_forward",
        success=True,
        verify_result=True,
        duration_sec=0.1,
        actor_caused=actor_caused,
    )
    return step, sg


def test_gate_caused_is_grounded() -> None:
    step, sg = _grounded_step(ActorCaused.CAUSED)
    assert classify_step_evidence(step, sg, ROBOT_ORACLES) == "GROUNDED"


def test_gate_uncaused_downgrades_to_ran() -> None:
    step, sg = _grounded_step(ActorCaused.UNCAUSED)
    assert classify_step_evidence(step, sg, ROBOT_ORACLES) == "RAN"


def test_gate_not_graded_stays_grounded_legacy() -> None:
    # The legacy/default path: a step the executor never graded must classify
    # EXACTLY as before R2b (GROUNDED) — zero regression.
    step, sg = _grounded_step(ActorCaused.NOT_GRADED)
    assert classify_step_evidence(step, sg, ROBOT_ORACLES) == "GROUNDED"


def test_gate_failed_step_unaffected_by_grade() -> None:
    sg = SubGoal(name="walk", description="d", verify="at_position(11.0, 3.0)", strategy="walk_forward")
    step = StepRecord(
        sub_goal_name="walk", strategy="walk_forward", success=False,
        verify_result=False, duration_sec=0.1, actor_caused=ActorCaused.CAUSED,
    )
    assert classify_step_evidence(step, sg, ROBOT_ORACLES) == "FAILED"


# ---------------------------------------------------------------------------
# (holding) weld eq_active 0->1 -> CAUSED; no transition -> UNCAUSED
# ---------------------------------------------------------------------------


class _FakeGripper:
    def __init__(self, welds: dict[str, bool]) -> None:
        self._welds = dict(welds)

    def weld_is_active(self) -> dict[str, bool]:
        return dict(self._welds)


def test_holding_fresh_grasp_is_caused() -> None:
    grip = _FakeGripper({"mug": False})
    before = _cap(_agent(gripper=grip))
    grip._welds["mug"] = True  # weld goes 0 -> 1 this step
    after = _cap(_agent(gripper=grip))
    assert grade(before, after, "holding_object('mug')", ROBOT_ORACLES) == ActorCaused.CAUSED


def test_holding_no_transition_is_uncaused() -> None:
    # Already welded at baseline (no fresh grasp) -> UNCAUSED.
    grip = _FakeGripper({"mug": True})
    before = _cap(_agent(gripper=grip))
    after = _cap(_agent(gripper=grip))
    assert grade(before, after, "holding_object('mug')", ROBOT_ORACLES) == ActorCaused.UNCAUSED


# ---------------------------------------------------------------------------
# misc — predicate detection
# ---------------------------------------------------------------------------


def test_is_robot_predicate_detection() -> None:
    assert is_robot_predicate("at_position(11.0, 3.0)", ROBOT_ORACLES) is True
    assert is_robot_predicate("arm_at_home()", ROBOT_ORACLES) is True
    # A dev predicate (not a robot oracle) is not graded.
    assert is_robot_predicate("file_exists('/tmp/x')", ROBOT_ORACLES) is False
    # A robot predicate ABSENT from the live oracle set is not graded.
    assert is_robot_predicate("at_position(1, 2)", frozenset()) is False



# ---------------------------------------------------------------------------
# (cross-channel, STEP-12 fix) a base predicate grades on the dimension it
# MEASURES: a forward walk must NOT launder a satisfied-at-baseline facing(h)
# no-op to CAUSED, and a pure turn must NOT launder at_position. The old
# max(planar, yaw) metric accepted the irrelevant channel.
# ---------------------------------------------------------------------------


def test_facing_uncaused_after_forward_walk_no_turn() -> None:
    # Forward walk: position moves ~1.5m, heading UNCHANGED. facing(h0) was already
    # true at baseline; the planar move must NOT grade the heading no-op CAUSED.
    base = _FakeBase(10.0, 3.0, yaw=0.5, cmd_motion=0.0)
    before = _cap(_agent(base=base))
    base.command(0.3)
    base.teleport(11.5, 3.0)  # dxy ~1.5, dyaw 0
    after = _cap(_agent(base=base))
    assert grade(before, after, "facing(0.5)", ROBOT_ORACLES) == ActorCaused.UNCAUSED


def test_facing_caused_after_real_turn() -> None:
    # A real turn: heading changes ~0.5 rad (>_YAW_CAUSE_EPS), position unchanged.
    base = _FakeBase(10.0, 3.0, yaw=0.0, cmd_motion=0.0)
    before = _cap(_agent(base=base))
    base.command(0.3)
    base._yaw = 0.5  # ~29deg intentional turn
    after = _cap(_agent(base=base))
    assert grade(before, after, "facing(0.5)", ROBOT_ORACLES) == ActorCaused.CAUSED


def test_facing_uncaused_on_drift_below_yaw_threshold() -> None:
    # Gait yaw-drift below the ~10deg threshold must NOT count as an intentional turn
    # for facing — even alongside a planar move (a forward walk that wobbles).
    base = _FakeBase(10.0, 3.0, yaw=0.0, cmd_motion=0.0)
    before = _cap(_agent(base=base))
    base.command(0.3)
    base.teleport(11.5, 3.0)
    base._yaw = 0.05  # ~2.9deg drift < _YAW_CAUSE_EPS (~10deg)
    after = _cap(_agent(base=base))
    assert grade(before, after, "facing(0.0)", ROBOT_ORACLES) == ActorCaused.UNCAUSED


def test_at_position_still_caused_after_forward_walk() -> None:
    # The planar channel is unaffected by the fix: a forward walk grades at_position
    # CAUSED exactly as before.
    base = _FakeBase(10.0, 3.0, yaw=0.0, cmd_motion=0.0)
    before = _cap(_agent(base=base))
    base.command(0.3)
    base.teleport(11.0, 3.0)
    after = _cap(_agent(base=base))
    assert grade(before, after, "at_position(11.0, 3.0)", ROBOT_ORACLES) == ActorCaused.CAUSED


def test_at_position_uncaused_after_pure_turn() -> None:
    # STEP-12 bonus tightening: a pure turn (no planar move) must NOT grade an
    # at_position predicate CAUSED — the position channel didn't move.
    base = _FakeBase(10.0, 3.0, yaw=0.0, cmd_motion=0.0)
    before = _cap(_agent(base=base))
    base.command(0.3)
    base._yaw = 1.0  # turned in place, position unchanged
    after = _cap(_agent(base=base))
    assert grade(before, after, "at_position(10.0, 3.0)", ROBOT_ORACLES) == ActorCaused.UNCAUSED


# ---------------------------------------------------------------------------
# WP10: the GT oracle set is causation-graded on its channel (a new predicate is
# never NOT_GRADED, which would let a pre-existing state ground for free).
# ---------------------------------------------------------------------------

def _sim_oracle_tags() -> dict:
    """The WP10 GT oracles with their REAL @oracle tags (vcli/worlds/sim_oracles)."""
    from tests.harness.fake_gt import FakeGT, fake_namespace
    from vector_os_nano.vcli.cognitive.oracle_registry import oracle_tags

    agent = SimpleNamespace(_base=_FakeBase(0.0, 0.0))
    ns = fake_namespace(agent, "go2_piper", gt=FakeGT(agent, objects={"pickable_bottle_green": [10.9, 3.0, 0.32]}))
    return oracle_tags(ns, "robot")


_WP10 = {**ROBOT_ORACLES, **_sim_oracle_tags()}


@pytest.mark.parametrize("expr", [
    "in_room('kitchen')", "near_object('user')", "in_view('red can')", "lifted('mug')",
    "kept_holding('mug')", "released('mug')", "object_at('mug', 1, 2)", "object_on('mug', 'table')",
])
def test_wp10_predicates_are_graded(expr) -> None:
    assert is_robot_predicate(expr, _WP10)


def test_object_on_already_true_with_no_release_is_uncaused() -> None:
    grip = _FakeGripper({"mug": False})  # the mug was resting on the table all along
    before = _cap(_agent(gripper=grip))
    after = _cap(_agent(gripper=grip))
    assert grade(before, after, "object_on('mug', 'table')", _WP10) == ActorCaused.UNCAUSED
    assert grade(before, after, "released('mug')", _WP10) == ActorCaused.UNCAUSED


def test_object_on_after_a_release_is_caused() -> None:
    grip = _FakeGripper({"mug": True})
    before = _cap(_agent(gripper=grip))
    grip._welds["mug"] = False  # placed: weld 1 -> 0 this step
    after = _cap(_agent(gripper=grip))
    for expr in ("object_on('mug', 'table')", "object_at('mug', 1, 2)", "released('mug')"):
        assert grade(before, after, expr, _WP10) == ActorCaused.CAUSED, expr
    # a release never proves a grasp
    assert grade(before, after, "lifted('mug')", _WP10) == ActorCaused.UNCAUSED


def test_carrying_a_pre_held_object_never_causes_kept_holding() -> None:
    # WP07 grades EVERY channel (spine:multi-channel-first-match): the carry's base
    # motion no longer stands in for the gripper. The mug was held before the step.
    base, grip = _FakeBase(10.0, 3.0), _FakeGripper({"mug": True})
    before = _cap(_agent(base=base, gripper=grip))
    after = _cap(_agent(base=base, gripper=grip))
    assert grade(before, after, "kept_holding('mug')", _WP10) == ActorCaused.UNCAUSED
    base.command(0.3)
    base.teleport(8.0, 3.0)
    after = _cap(_agent(base=base, gripper=grip))
    expr = "kept_holding('mug') and near_object('user', 0.8)"
    assert grade(before, after, expr, _WP10) == ActorCaused.UNCAUSED
    assert grade(before, after, "near_object('user', 0.8)", _WP10) == ActorCaused.CAUSED


def test_in_view_is_caused_by_a_turn_or_a_walk_not_by_looking() -> None:
    base = _FakeBase(10.0, 3.0)
    before = _cap(_agent(base=base))
    assert grade(before, _cap(_agent(base=base)), "in_view('red can')", _WP10) == ActorCaused.UNCAUSED
    base.command(0.3)
    base._yaw = 1.0  # turned in place
    assert grade(before, _cap(_agent(base=base)), "in_view('red can')", _WP10) == ActorCaused.CAUSED


# INT2: localized is graded on the perception channel. A look is read-only (it never
# actuates), so the no-action rule graded every look -> localized UNCAUSED (E10 RAN),
# while an unrelated walk let a map entry from an earlier step ground (NOT_GRADED).
# I23: the channel is WP11's (SceneGraph.observation_count); INT2's cases write the
# belief the way it counts it (a localized merge_object), not by add_object.

def _sg_agent(*seed: tuple[float, float]):
    from vector_os_nano.core.scene_graph import SceneGraph

    sg = SceneGraph()
    for x, y in seed:
        sg.merge_object("green bottle", "", "", x=x, y=y)
    return SimpleNamespace(_base=None, _arm=None, _gripper=None, _spatial_memory=sg), sg


def test_localized_is_graded_on_the_perception_channel() -> None:
    assert is_robot_predicate("localized('green bottle')", _WP10)
    assert not is_robot_predicate("localized('green bottle')", ROBOT_ORACLES)  # not bound


def test_localized_caused_only_by_a_belief_write_this_step() -> None:
    agent, sg = _sg_agent()
    before = _cap(agent)
    sg.merge_object("green bottle", "", "", x=10.9, y=3.0)  # the look observed it this step
    assert grade(before, _cap(agent), "localized('green bottle')", _WP10) == ActorCaused.CAUSED
    agent, sg = _sg_agent((10.9, 3.0))
    before = _cap(agent)
    sg.merge_object("green bottle", "", "", x=10.95, y=3.0)  # re-observed: the instance moved
    assert grade(before, _cap(agent), "localized('green bottle')", _WP10) == ActorCaused.CAUSED


def test_localized_from_memory_is_uncaused() -> None:
    agent, sg = _sg_agent((10.9, 3.0))  # on the map before the step began
    before = _cap(agent)
    assert grade(before, _cap(agent), "localized('green bottle')", _WP10) == ActorCaused.UNCAUSED
    assert grade(None, _cap(agent), "localized('green bottle')", _WP10) == ActorCaused.UNCAUSED
    no_map = SimpleNamespace(_base=None, _arm=None, _gripper=None, _spatial_memory=None)
    assert grade(_cap(no_map), _cap(agent), "localized('green bottle')", _WP10) == ActorCaused.UNCAUSED


def test_localized_add_object_is_not_a_perceive() -> None:
    # I23 (stricter than INT2's row diff): inserting a node by hand (a restore, a tool
    # write) is not a localized observation, so it never causes localized.
    from vector_os_nano.core.scene_graph import ObjectNode

    agent, sg = _sg_agent()
    before = _cap(agent)
    sg.add_object(ObjectNode(object_id="obj_g", category="green bottle", x=10.9, y=3.0))
    assert grade(before, _cap(agent), "localized('green bottle')", _WP10) == ActorCaused.UNCAUSED


# ---------------------------------------------------------------------------
# WP11: localized is graded on the PERCEPTION channel — caused only by a localized
# observation written this step, never by memory (adversarial cases first).
# ---------------------------------------------------------------------------


def _perceiver():
    from vector_os_nano.core.scene_graph import SceneGraph

    agent = _agent()
    agent._spatial_memory = SceneGraph()
    return agent


def test_localized_from_memory_alone_is_uncaused() -> None:
    agent = _perceiver()
    agent._spatial_memory.merge_object("green bottle", "hallway", "vp", x=10.88, y=3.0)
    before = _cap(agent)  # the step starts with the object already on the map
    assert is_robot_predicate("localized('green bottle')", _WP10)
    assert grade(before, _cap(agent), "localized('green bottle')", _WP10) == ActorCaused.UNCAUSED
    agent._spatial_memory.merge_object("sofa", "hallway", "vp")  # a name with no position
    assert grade(before, _cap(agent), "localized('green bottle')", _WP10) == ActorCaused.UNCAUSED
    agent._spatial_memory.clear()
    agent._spatial_memory.load()  # nothing persisted: still nothing perceived this step
    assert grade(before, _cap(agent), "localized('green bottle')", _WP10) == ActorCaused.UNCAUSED


def test_localized_after_a_perceive_wrote_the_belief_is_caused() -> None:
    agent = _perceiver()
    before = _cap(agent)
    agent._spatial_memory.merge_object("green bottle", "hallway", "vp", x=10.88, y=3.0)
    assert grade(before, _cap(agent), "localized('green bottle')", _WP10) == ActorCaused.CAUSED


def test_localized_without_a_belief_or_a_baseline_fails_closed() -> None:
    assert grade(None, _cap(_perceiver()), "localized('x')", _WP10) == ActorCaused.UNCAUSED
    assert grade(_cap(_agent()), _cap(_agent()), "localized('x')", _WP10) == ActorCaused.UNCAUSED


# ---------------------------------------------------------------------------
# S1: the pose half of a baseline is ground truth, stamped with its session
# ---------------------------------------------------------------------------


class _Truth:
    """A session's GroundTruth stand-in that the devices cannot write."""

    has_base = True

    def __init__(self, x: float, y: float, yaw: float = 0.0) -> None:
        self.pose = (x, y, 0.3, yaw)
        self.joints = [0.0] * 6
        self.held: dict[str, str] = {}

    def base_pose(self):
        return self.pose

    def joint_positions(self, joints):
        return list(self.joints[:len(joints)])

    def attachments(self):
        return dict(self.held)


class _LyingArm:
    joint_names = [f"j{i}" for i in range(6)]

    def __init__(self) -> None:
        self.motion = 0.0

    def ctrl_motion(self) -> float:
        return self.motion

    def get_joint_positions(self) -> list[float]:
        return [1.0] * 6  # whatever the device claims


class _LyingGripper:
    def weld_is_active(self) -> dict[str, bool]:
        return {"mug": True}  # claims a grasp the world never made


def test_capture_reads_the_pose_from_ground_truth_never_from_the_devices() -> None:
    base, arm, grip = _FakeBase(10.0, 3.0), _LyingArm(), _LyingGripper()
    truth = _Truth(10.0, 3.0)
    session = SimpleNamespace(gt=truth, uid=7)
    agent = _agent(base=base, arm=arm, gripper=grip)
    before = capture(agent, session)
    assert (before.base_pos, before.arm_joints, before.gripper_welds) == ((10.0, 3.0, 0.3), (0.0,) * 6, {})
    # The devices report a walk, an arm move and a grasp; the world saw none of them.
    base.command(0.5)
    base.teleport(12.0, 3.0)
    arm.motion = 1.0
    after = capture(agent, session)
    for expr in ("at_position(12.0, 3.0)", "arm_at_home()", "holding_object('mug')"):
        assert grade(before, after, expr, ROBOT_ORACLES) == ActorCaused.UNCAUSED, expr
    # The world moved: now the same commands are the cause.
    truth.pose, truth.joints, truth.held = (12.0, 3.0, 0.3, 0.0), [0.5] * 6, {"mug": "piper_link6"}
    after = capture(agent, session)
    for expr in ("at_position(12.0, 3.0)", "holding_object('mug')"):
        assert grade(before, after, expr, ROBOT_ORACLES) == ActorCaused.CAUSED, expr
    assert grade(before, after, "not arm_at_home()", ROBOT_ORACLES) == ActorCaused.CAUSED


def test_without_a_session_no_robot_channel_can_cause() -> None:
    base = _FakeBase(10.0, 3.0)
    before = capture(_agent(base=base))
    base.command(0.5)
    base.teleport(11.0, 3.0)
    after = capture(_agent(base=base))
    assert before.base_pos is None and after.base_cmd_motion == 0.5
    assert grade(before, after, "at_position(11.0, 3.0)", ROBOT_ORACLES) == ActorCaused.UNCAUSED


def test_a_baseline_from_another_session_never_causes() -> None:
    base = _FakeBase(10.0, 3.0)
    before = _cap(_agent(base=base), uid=1)
    base.command(0.5)
    base.teleport(11.0, 3.0)  # an honest walk ...
    after = _cap(_agent(base=base), uid=2)  # ... graded in the next session (the body was switched)
    assert grade(before, after, "at_position(11.0, 3.0)", ROBOT_ORACLES) == ActorCaused.UNCAUSED
    from vector_os_nano.vcli.cognitive.actor_causation import grade_channels

    assert grade_channels(before, after, "at_position(11.0, 3.0)", ROBOT_ORACLES)[1] == (
        "baseline from another session",)
    same = _cap(_agent(base=base), uid=1)
    assert grade(before, same, "at_position(11.0, 3.0)", ROBOT_ORACLES) == ActorCaused.CAUSED
    # a turn-window oracle: the turn baseline of the old session never causes it either
    grip = _FakeGripper({"mug": False})
    turn = _cap(_agent(gripper=grip), uid=1)
    grip._welds["mug"] = True
    step, post = _cap(_agent(gripper=grip), uid=2), _cap(_agent(gripper=grip), uid=2)
    assert grade(step, post, "kept_holding('mug')", _WP10, turn_baseline=turn) == ActorCaused.UNCAUSED
    assert grade(step, post, "kept_holding('mug')", _WP10,
                 turn_baseline=_cap(_agent(gripper=_FakeGripper({"mug": False})), uid=2)) == ActorCaused.CAUSED
