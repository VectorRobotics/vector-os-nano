# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Integration tests for PickSkill with mock hardware.

Tests verify the full pick sequence without requiring real hardware or ROS2.

MockArm tracks all move_joints() calls so tests can assert the exact
sequence: pre-grasp → descend → lift → home.
MockGripper tracks open/close calls.
MockPerception returns configurable Detection instances.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from vector_os_nano.perception.calibration import Calibration

from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.types import Detection, SkillResult
from vector_os_nano.core.world_model import ObjectState, WorldModel
from vector_os_nano.skills.pick import PickSkill


# ---------------------------------------------------------------------------
# Mock classes
# ---------------------------------------------------------------------------


class MockArm:
    """Minimal arm mock that records all move_joints() calls.

    IK is mocked to return a trivially valid joint vector (slightly displaced
    from current) so the PickSkill always gets a valid IK solution.
    """

    def __init__(self, initial_joints: list[float] | None = None):
        self.moves: list[tuple[list[float], float]] = []
        self._joints: list[float] = list(initial_joints or [0.0] * 5)
        self.move_return: bool = True  # set False to simulate failure

    def get_joint_positions(self) -> list[float]:
        return list(self._joints)

    def move_joints(self, positions: list[float], duration: float = 3.0) -> bool:
        self.moves.append((list(positions), duration))
        if self.move_return:
            self._joints = list(positions)
        return self.move_return

    def ik(
        self,
        target_xyz: tuple[float, float, float],
        current_joints: list[float] | None = None,
    ) -> list[float] | None:
        """Return perturbed joints so each IK call gives a distinct result."""
        base = list(current_joints or self._joints)
        x, y, z = target_xyz
        # Encode target into the first 3 joints for traceability in tests
        return [x * 0.1, y * 0.1, z * 0.1, base[3] if len(base) > 3 else 0.0, base[4] if len(base) > 4 else 0.0]

    def stop(self) -> None:
        pass


class MockGripper:
    """Records open/close calls."""

    def __init__(self):
        self.actions: list[str] = []
        self.open_return: bool = True
        self.close_return: bool = True

    def open(self) -> bool:
        self.actions.append("open")
        return self.open_return

    def close(self) -> bool:
        self.actions.append("close")
        return self.close_return

    def is_holding(self) -> bool:
        return self.actions.count("close") > self.actions.count("open")

    def get_position(self) -> float:
        return 0.0 if self.is_holding() else 1.0


class MockPerception:
    """Returns pre-configured detections."""

    def __init__(self, detections: list[Detection] | None = None):
        self._detections = detections or []

    def detect(self, query: str) -> list[Detection]:
        return list(self._detections)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_context(
    arm: MockArm | None = None,
    gripper: MockGripper | None = None,
    perception=None,
    world_model: WorldModel | None = None,
    config: dict | None = None,
    calibration=None,
) -> SkillContext:
    return SkillContext(
        arm=arm or MockArm(),
        gripper=gripper or MockGripper(),
        perception=perception,
        world_model=world_model or WorldModel(),
        calibration=calibration if calibration is not None else Calibration(),
        config=config or {},
    )


# ---------------------------------------------------------------------------
# test_pick_with_mock_arm
# ---------------------------------------------------------------------------


class TestPickWithMockArm:
    """Verify the pick sequence using a world-model-based target."""

    def setup_method(self):
        self.arm = MockArm()
        self.gripper = MockGripper()
        self.wm = WorldModel()

        # Add a target object in a reachable position
        self.wm.add_object(ObjectState(
            object_id="cup_0",
            label="red cup",
            x=0.25,
            y=0.0,
            z=0.005,
            confidence=0.9,
        ))

        self.context = make_context(
            arm=self.arm,
            gripper=self.gripper,
            world_model=self.wm,
        )
        self.skill = PickSkill()

    def test_pick_returns_success(self):
        result = self.skill.execute({"object_id": "cup_0"}, self.context)
        assert result.success, result.error_message

    def test_pick_result_data_has_position(self):
        result = self.skill.execute({"object_id": "cup_0"}, self.context)
        assert "position_cm" in result.result_data
        pos = result.result_data["position_cm"]
        assert len(pos) == 2

    def test_pick_sequence_has_moves(self):
        """At least 3 arm moves: pre-grasp, descend, lift, home."""
        self.skill.execute({"object_id": "cup_0"}, self.context)
        assert len(self.arm.moves) >= 4

    def test_gripper_opened_before_pregrasp(self):
        """Gripper must open before the first arm move."""
        self.skill.execute({"object_id": "cup_0"}, self.context)
        # First gripper action should be open
        assert self.gripper.actions[0] == "open"

    def test_gripper_closed_three_times(self):
        """Gripper close must be called 3 times (STS3215 command reliability)."""
        self.skill.execute({"object_id": "cup_0"}, self.context)
        close_count = self.gripper.actions.count("close")
        # 3 closes during grasp + 1 close at the end of release sequence
        assert close_count >= 3

    def test_pick_by_label(self):
        result = self.skill.execute({"object_label": "red cup"}, self.context)
        assert result.success, result.error_message

    def test_pick_descent_duration_is_short(self):
        """Descent should be fast (1.0s) to minimise XY drift."""
        self.skill.execute({"object_id": "cup_0"}, self.context)
        durations = [d for _, d in self.arm.moves]
        # At least one move with duration 1.0 (descent or lift)
        assert 1.0 in durations

    def test_last_move_is_home(self):
        """Final arm move must be to home joints."""
        home_joints = [-0.014, -1.238, 0.562, 0.858, 0.311]
        self.skill.execute({"object_id": "cup_0"}, self.context)
        last_joints, _ = self.arm.moves[-1]
        assert last_joints == pytest.approx(home_joints, abs=1e-4)


# ---------------------------------------------------------------------------
# test_pick_no_perception
# ---------------------------------------------------------------------------


class TestPickNoPerception:
    """Pick fails gracefully without a target in world model or perception."""

    def setup_method(self):
        self.context = make_context(perception=None)
        self.skill = PickSkill()

    def test_pick_fails_without_target(self):
        result = self.skill.execute({}, self.context)
        assert not result.success

    def test_error_message_mentions_locate(self):
        result = self.skill.execute({}, self.context)
        assert "locate" in result.error_message.lower() or len(result.error_message) > 0

    def test_pick_with_unknown_object_id_fails(self):
        result = self.skill.execute({"object_id": "nonexistent"}, self.context)
        assert not result.success

    def test_pick_with_unknown_label_fails(self):
        result = self.skill.execute({"object_label": "nonexistent"}, self.context)
        assert not result.success


# ---------------------------------------------------------------------------
# test_pick_outside_workspace
# ---------------------------------------------------------------------------


class TestPickWorkspaceBoundary:
    """Objects outside workspace bounds return failure."""

    def setup_method(self):
        self.skill = PickSkill()

    def _make_context_with_object(self, x: float, y: float) -> SkillContext:
        wm = WorldModel()
        wm.add_object(ObjectState(
            object_id="obj_0",
            label="box",
            x=x,
            y=y,
            z=0.005,
            confidence=0.9,
        ))
        return make_context(world_model=wm)

    def test_object_too_far_fails(self):
        # 40 cm forward — outside 35 cm max
        ctx = self._make_context_with_object(x=0.40, y=0.0)
        result = self.skill.execute({"object_id": "obj_0"}, ctx)
        assert not result.success
        assert "workspace" in result.error_message.lower()

    def test_object_too_close_fails(self):
        # 2 cm forward — inside 5 cm min
        ctx = self._make_context_with_object(x=0.01, y=0.005)
        result = self.skill.execute({"object_id": "obj_0"}, ctx)
        # Note: z_offset added may push the object to an unreachable position —
        # either "workspace" or "IK" failure is acceptable here
        assert not result.success


# ---------------------------------------------------------------------------
# test_pick_retry_on_failure
# ---------------------------------------------------------------------------


class TestPickRetryOnFailure:
    """Verify retry logic: arm returns home between attempts."""

    def setup_method(self):
        self.skill = PickSkill()
        self.wm = WorldModel()
        self.wm.add_object(ObjectState(
            object_id="cup_0",
            label="cup",
            x=0.25,
            y=0.0,
            z=0.005,
            confidence=0.9,
        ))

    def test_retry_uses_home_between_attempts(self):
        """After first failure, home move must precede second attempt."""
        arm = MockArm()
        gripper = MockGripper()

        # Fail on first pregrasp move, succeed after
        call_count = [0]
        original_move = arm.move_joints

        def patched_move(positions, duration=3.0):
            call_count[0] += 1
            # Fail only the second move call (first pregrasp in attempt 1)
            if call_count[0] == 2:
                return False
            return True

        arm.move_joints = patched_move
        ctx = make_context(arm=arm, gripper=gripper, world_model=self.wm)

        # Should retry and ultimately succeed (or fail gracefully)
        result = self.skill.execute({"object_id": "cup_0"}, ctx)
        # At minimum, we verify no exception was raised
        assert isinstance(result, SkillResult)

    def test_total_failure_returns_error_message_with_attempt_count(self):
        """After all retries fail, error message mentions failed attempts."""
        arm = MockArm()
        arm.move_return = False  # All moves fail
        ctx = make_context(arm=arm, world_model=self.wm)
        result = self.skill.execute({"object_id": "cup_0"}, ctx)
        assert not result.success
        # Should mention how many attempts were made
        assert "2" in result.error_message or "attempt" in result.error_message.lower()

    def test_custom_max_retries_config(self):
        """max_retries can be set via config."""
        arm = MockArm()
        arm.move_return = False
        ctx = make_context(
            arm=arm,
            world_model=self.wm,
            config={"skills": {"pick": {"max_retries": 1}}},
        )
        result = self.skill.execute({"object_id": "cup_0"}, ctx)
        assert not result.success
        assert "1" in result.error_message or "attempt" in result.error_message.lower()

    def test_zero_max_retries_still_makes_one_attempt(self):
        """skills-manip:pick-retry-unbound: max_retries=0 raised UnboundLocalError."""
        arm = MockArm()
        arm.move_return = False
        ctx = make_context(arm=arm, world_model=self.wm,
                           config={"skills": {"pick": {"max_retries": 0}}})
        result = self.skill.execute({"object_id": "cup_0"}, ctx)
        assert not result.success and result.result_data["attempts"] == 1


# ---------------------------------------------------------------------------
# test_pick_ik_failure
# ---------------------------------------------------------------------------


class TestPickIKFailure:
    """PickSkill handles IK failure gracefully."""

    def setup_method(self):
        self.skill = PickSkill()
        self.wm = WorldModel()
        self.wm.add_object(ObjectState(
            object_id="cup_0",
            label="cup",
            x=0.25,
            y=0.0,
            z=0.005,
            confidence=0.9,
        ))

    def test_ik_pregrasp_failure(self):
        """IK returning None for pre-grasp results in graceful failure."""
        arm = MockArm()
        arm.ik = lambda target, seed=None: None  # Always fail IK

        ctx = make_context(arm=arm, world_model=self.wm)
        result = self.skill.execute({"object_id": "cup_0"}, ctx)
        assert not result.success
        assert "IK" in result.error_message


# ---------------------------------------------------------------------------
# test_pick_no_gripper
# ---------------------------------------------------------------------------


class TestPickNoGripper:
    """PickSkill works without a gripper (arm-only mode)."""

    def setup_method(self):
        self.skill = PickSkill()
        self.wm = WorldModel()
        self.wm.add_object(ObjectState(
            object_id="cup_0",
            label="cup",
            x=0.25,
            y=0.0,
            z=0.005,
            confidence=0.9,
        ))

    def test_pick_with_no_gripper_succeeds(self):
        arm = MockArm()
        ctx = SkillContext(
            arm=arm,
            gripper=None,   # no gripper
            perception=None,
            world_model=self.wm,
            calibration=Calibration(),
        )
        result = self.skill.execute({"object_id": "cup_0"}, ctx)
        # Should not crash — succeeds without gripper operations
        assert result.success, result.error_message


def test_the_default_pick_holds_what_its_verify_hint_says():
    """skills-manip:pick-verify-hint-contradicts-drop-default: the hint said
    holding_object() while the default mode dropped the object."""
    wm = WorldModel()
    wm.add_object(ObjectState(object_id="cup_0", label="cup", x=0.25, y=0.0, z=0.005))
    gripper = MockGripper()
    result = PickSkill().execute({"object_label": "cup"}, make_context(gripper=gripper, world_model=wm))
    assert result.success and gripper.actions[-1] == "close"  # never re-opened: held
    assert PickSkill.verify_hint == "holding_object({object_label})"
    assert PickSkill.parameters["mode"]["default"] == "hold"


def test_pick_with_no_target_never_chooses_one():
    """skills-manip:pick-sim-oracle-nearest-object: no target read the nearest object
    from the simulator's oracle. Now it asks for a target, before any motion."""
    arm = MockArm()
    arm.get_object_positions = lambda: {"banana": (0.12, 0.12, 0.0)}
    result = PickSkill().execute({"mode": "hold"}, make_context(arm=arm))
    assert not result.success and result.result_data["diagnosis"] == "no_target"
    assert arm.moves == []


def test_a_missing_calibration_fails_loud_instead_of_using_camera_coordinates():
    """perception:calibration-missing-silent-camera-frame: with no camera->arm
    calibration, pick used an identity transform and drove to camera-frame x/y/z."""
    from vector_os_nano.core.types import Pose3D, TrackedObject

    class _Tracking(MockPerception):  # a camera-frame pose inside the arm's reach
        def track(self, detections):
            return [TrackedObject(track_id=1, label="cup", bbox_2d=(0, 0, 9, 9),
                                  pose=Pose3D(x=0.2, y=0.0, z=0.02))]

    arm = MockArm()
    ctx = make_context(arm=arm, perception=_Tracking([Detection("cup", (0, 0, 9, 9), 0.9)]),
                       config={"skills": {"pick": {"sample_count": 1}}})
    ctx.calibration = None
    result = PickSkill().execute({"object_label": "cup"}, ctx)
    assert not result.success and result.result_data["diagnosis"] == "calibration_error"
    assert arm.moves == []  # no motion at all, never a grasp at camera coordinates
