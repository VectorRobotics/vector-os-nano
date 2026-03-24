"""Unit tests for PlaceSkill diagnostic result_data — TDD RED phase.

Covers every return path in PlaceSkill.execute():
- no_arm       : context.arm is None
- ik_unreachable (above-place): arm.ik returns None on first call
- move_failed (approach)      : arm.move_joints returns False on first call
- ik_unreachable (place)      : arm.ik returns None on second call
- move_failed (descend)       : arm.move_joints returns False on second call
- move_failed (lift)          : arm.move_joints returns False on third call
- ok                          : full success path
"""
from __future__ import annotations

from unittest.mock import Mock

import pytest

from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.world_model import WorldModel
from vector_os_nano.skills.place import PlaceSkill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(arm=None, gripper=None) -> SkillContext:
    """Build a minimal SkillContext for unit tests."""
    return SkillContext(
        arm=arm,
        gripper=gripper,
        perception=None,
        world_model=WorldModel(),
        calibration=None,
        config={},
    )


def _default_arm() -> Mock:
    """Arm mock that succeeds for all calls by default."""
    arm = Mock()
    arm.get_joint_positions = Mock(return_value=[0.0] * 5)
    arm.ik = Mock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
    arm.move_joints = Mock(return_value=True)
    return arm


def _default_gripper() -> Mock:
    gripper = Mock()
    gripper.open = Mock()
    gripper.close = Mock()
    return gripper


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPlaceSkillFailureModes:
    """PlaceSkill must expose failure_modes as a class attribute."""

    def test_failure_modes_attribute_exists(self):
        skill = PlaceSkill()
        assert hasattr(skill, "failure_modes")

    def test_failure_modes_contains_required_codes(self):
        skill = PlaceSkill()
        assert "no_arm" in skill.failure_modes
        assert "ik_unreachable" in skill.failure_modes
        assert "move_failed" in skill.failure_modes


class TestPlaceSkillDiagnosticsNoArm:
    def test_place_failure_no_arm_returns_diagnosis(self):
        """Return path 1: context.arm is None → diagnosis == 'no_arm'."""
        skill = PlaceSkill()
        context = _make_context(arm=None)
        result = skill.execute({"location": "front"}, context)

        assert result.success is False
        assert result.result_data.get("diagnosis") == "no_arm"


class TestPlaceSkillDiagnosticsIKAbove:
    def test_place_failure_ik_above_returns_diagnosis(self):
        """Return path 2: first arm.ik call returns None → ik_unreachable."""
        arm = _default_arm()
        arm.ik = Mock(return_value=None)  # always fails

        skill = PlaceSkill()
        context = _make_context(arm=arm)
        result = skill.execute({"location": "front"}, context)

        assert result.success is False
        assert result.result_data.get("diagnosis") == "ik_unreachable"
        assert "target_cm" in result.result_data
        assert "above_target_cm" in result.result_data
        assert result.result_data.get("hint") is not None

    def test_place_failure_ik_above_target_cm_values(self):
        """target_cm should round coords to 1 decimal in cm."""
        arm = _default_arm()
        arm.ik = Mock(return_value=None)

        skill = PlaceSkill()
        # Use explicit x, y, z so we can predict the values
        context = _make_context(arm=arm)
        result = skill.execute({"x": 0.3, "y": 0.0, "z": 0.04}, context)

        assert result.result_data["target_cm"] == [30.0, 0.0, 4.0]
        # above_target_cm z should be tz + pre_grasp_height in cm
        assert result.result_data["above_target_cm"][0] == 30.0
        assert result.result_data["above_target_cm"][1] == 0.0
        # default pre_grasp_height is 0.06, so 4.0 + 6.0 = 10.0
        assert result.result_data["above_target_cm"][2] == 10.0


class TestPlaceSkillDiagnosticsMoveApproach:
    def test_place_failure_approach_returns_diagnosis(self):
        """Return path 3: first move_joints returns False → move_failed / approach."""
        arm = _default_arm()
        arm.move_joints = Mock(return_value=False)  # always fails

        skill = PlaceSkill()
        context = _make_context(arm=arm)
        result = skill.execute({"location": "front"}, context)

        assert result.success is False
        assert result.result_data.get("diagnosis") == "move_failed"
        assert result.result_data.get("phase") == "approach"


class TestPlaceSkillDiagnosticsIKPlace:
    def test_place_failure_ik_place_returns_diagnosis(self):
        """Return path 4: second arm.ik call returns None → ik_unreachable for place."""
        arm = _default_arm()
        # First IK call (above) succeeds, second (place) fails
        arm.ik = Mock(side_effect=[[0.1, 0.2, 0.3, 0.4, 0.5], None])

        skill = PlaceSkill()
        context = _make_context(arm=arm)
        result = skill.execute({"location": "front"}, context)

        assert result.success is False
        assert result.result_data.get("diagnosis") == "ik_unreachable"
        assert "target_cm" in result.result_data
        # place IK failure has no above_target_cm key
        assert "above_target_cm" not in result.result_data
        assert result.result_data.get("hint") is not None


class TestPlaceSkillDiagnosticsDescend:
    def test_place_failure_descend_returns_diagnosis(self):
        """Return path 5: second move_joints returns False → move_failed / descend."""
        arm = _default_arm()
        # First move (approach) succeeds, second (descend) fails
        arm.move_joints = Mock(side_effect=[True, False])

        skill = PlaceSkill()
        context = _make_context(arm=arm)
        result = skill.execute({"location": "front"}, context)

        assert result.success is False
        assert result.result_data.get("diagnosis") == "move_failed"
        assert result.result_data.get("phase") == "descend"


class TestPlaceSkillDiagnosticsLift:
    def test_place_failure_lift_returns_diagnosis(self):
        """Return path 6: third move_joints returns False → move_failed / lift."""
        arm = _default_arm()
        # approach=True, descend=True, lift=False
        arm.move_joints = Mock(side_effect=[True, True, False])

        skill = PlaceSkill()
        gripper = _default_gripper()
        context = _make_context(arm=arm, gripper=gripper)
        result = skill.execute({"location": "front"}, context)

        assert result.success is False
        assert result.result_data.get("diagnosis") == "move_failed"
        assert result.result_data.get("phase") == "lift"


class TestPlaceSkillDiagnosticsSuccess:
    def test_place_success_returns_diagnosis_ok(self):
        """Return path 7: full success → diagnosis == 'ok' + placed_at present."""
        arm = _default_arm()
        gripper = _default_gripper()

        skill = PlaceSkill()
        context = _make_context(arm=arm, gripper=gripper)
        result = skill.execute({"location": "front"}, context)

        assert result.success is True
        assert result.result_data.get("diagnosis") == "ok"
        assert "placed_at" in result.result_data

    def test_place_success_placed_at_values(self):
        """placed_at should reflect the resolved coordinates."""
        arm = _default_arm()
        gripper = _default_gripper()
        # move_joints called 3 times (approach, descend, lift) + 1 home (no check)
        arm.move_joints = Mock(return_value=True)

        skill = PlaceSkill()
        context = _make_context(arm=arm, gripper=gripper)
        result = skill.execute({"x": 0.25, "y": 0.05, "z": 0.04}, context)

        assert result.success is True
        assert result.result_data["placed_at"] == [0.25, 0.05, 0.04]
        assert result.result_data["diagnosis"] == "ok"


# ---------------------------------------------------------------------------
# ScanSkill diagnostics
# ---------------------------------------------------------------------------

from vector_os_nano.skills.scan import ScanSkill


class TestScanSkillFailureModes:
    def test_failure_modes_attribute_exists(self):
        skill = ScanSkill()
        assert hasattr(skill, "failure_modes")

    def test_failure_modes_contains_required_codes(self):
        skill = ScanSkill()
        assert "no_arm" in skill.failure_modes
        assert "move_failed" in skill.failure_modes


class TestScanSkillDiagnosticsNoArm:
    def test_scan_no_arm_returns_diagnosis(self):
        """context.arm is None -> diagnosis == 'no_arm'."""
        skill = ScanSkill()
        context = _make_context(arm=None)
        result = skill.execute({}, context)

        assert result.success is False
        assert result.result_data.get("diagnosis") == "no_arm"


class TestScanSkillDiagnosticsMoveFailed:
    def test_scan_move_failed_returns_diagnosis(self):
        """arm.move_joints returns False -> diagnosis == 'move_failed'."""
        arm = Mock()
        arm.move_joints = Mock(return_value=False)
        skill = ScanSkill()
        context = _make_context(arm=arm)
        result = skill.execute({}, context)

        assert result.success is False
        assert result.result_data.get("diagnosis") == "move_failed"


class TestScanSkillDiagnosticsSuccess:
    def test_scan_success_returns_diagnosis_ok(self):
        """Full success path -> diagnosis == 'ok' and joint_values present."""
        arm = Mock()
        arm.move_joints = Mock(return_value=True)
        skill = ScanSkill()
        context = _make_context(arm=arm)
        result = skill.execute({}, context)

        assert result.success is True
        assert result.result_data.get("diagnosis") == "ok"
        assert "joint_values" in result.result_data


# ---------------------------------------------------------------------------
# HomeSkill diagnostics
# ---------------------------------------------------------------------------

from vector_os_nano.skills.home import HomeSkill


class TestHomeSkillFailureModes:
    def test_failure_modes_attribute_exists(self):
        skill = HomeSkill()
        assert hasattr(skill, "failure_modes")

    def test_failure_modes_contains_required_codes(self):
        skill = HomeSkill()
        assert "no_arm" in skill.failure_modes
        assert "move_failed" in skill.failure_modes


class TestHomeSkillDiagnosticsNoArm:
    def test_home_no_arm_returns_diagnosis(self):
        """context.arm is None -> diagnosis == 'no_arm'."""
        skill = HomeSkill()
        context = _make_context(arm=None)
        result = skill.execute({}, context)

        assert result.success is False
        assert result.result_data.get("diagnosis") == "no_arm"


class TestHomeSkillDiagnosticsMoveFailed:
    def test_home_move_failed_returns_diagnosis(self):
        """arm.move_joints returns False -> diagnosis == 'move_failed'."""
        arm = Mock()
        arm.move_joints = Mock(return_value=False)
        skill = HomeSkill()
        context = _make_context(arm=arm)
        result = skill.execute({}, context)

        assert result.success is False
        assert result.result_data.get("diagnosis") == "move_failed"


class TestHomeSkillDiagnosticsSuccess:
    def test_home_success_returns_diagnosis_ok(self):
        """Full success path -> diagnosis == 'ok' and joint_values present."""
        arm = Mock()
        arm.move_joints = Mock(return_value=True)
        skill = HomeSkill()
        context = _make_context(arm=arm)
        result = skill.execute({}, context)

        assert result.success is True
        assert result.result_data.get("diagnosis") == "ok"
        assert "joint_values" in result.result_data


# ---------------------------------------------------------------------------
# GripperOpenSkill / GripperCloseSkill diagnostics
# ---------------------------------------------------------------------------

from vector_os_nano.skills.gripper import GripperCloseSkill, GripperOpenSkill


class TestGripperOpenSkillFailureModes:
    def test_failure_modes_attribute_exists(self):
        skill = GripperOpenSkill()
        assert hasattr(skill, "failure_modes")

    def test_failure_modes_contains_no_arm(self):
        skill = GripperOpenSkill()
        assert "no_arm" in skill.failure_modes


class TestGripperCloseSkillFailureModes:
    def test_failure_modes_attribute_exists(self):
        skill = GripperCloseSkill()
        assert hasattr(skill, "failure_modes")

    def test_failure_modes_contains_no_arm(self):
        skill = GripperCloseSkill()
        assert "no_arm" in skill.failure_modes


class TestGripperOpenSkillDiagnosticsNoGripper:
    def test_gripper_open_no_gripper_returns_diagnosis(self):
        """context.gripper is None -> diagnosis == 'no_arm'."""
        skill = GripperOpenSkill()
        context = _make_context(arm=None, gripper=None)
        result = skill.execute({}, context)

        assert result.success is False
        assert result.result_data.get("diagnosis") == "no_arm"


class TestGripperOpenSkillDiagnosticsSuccess:
    def test_gripper_open_success_returns_diagnosis_ok(self):
        """Gripper present -> diagnosis == 'ok'."""
        gripper = _default_gripper()
        world_model = Mock()
        world_model.update_robot_state = Mock()
        context = SkillContext(
            arm=None,
            gripper=gripper,
            perception=None,
            world_model=world_model,
            calibration=None,
            config={},
        )
        skill = GripperOpenSkill()
        result = skill.execute({}, context)

        assert result.success is True
        assert result.result_data.get("diagnosis") == "ok"


class TestGripperCloseSkillDiagnosticsSuccess:
    def test_gripper_close_success_returns_diagnosis_ok(self):
        """Gripper present -> diagnosis == 'ok'."""
        gripper = _default_gripper()
        world_model = Mock()
        world_model.update_robot_state = Mock()
        context = SkillContext(
            arm=None,
            gripper=gripper,
            perception=None,
            world_model=world_model,
            calibration=None,
            config={},
        )
        skill = GripperCloseSkill()
        result = skill.execute({}, context)

        assert result.success is True
        assert result.result_data.get("diagnosis") == "ok"


# ---------------------------------------------------------------------------
# WaveSkill diagnostics
# ---------------------------------------------------------------------------

from vector_os_nano.skills.wave import WaveSkill


class TestWaveSkillFailureModes:
    def test_failure_modes_attribute_exists(self):
        skill = WaveSkill()
        assert hasattr(skill, "failure_modes")

    def test_failure_modes_contains_required_codes(self):
        skill = WaveSkill()
        assert "no_arm" in skill.failure_modes
        assert "move_failed" in skill.failure_modes


class TestWaveSkillDiagnosticsNoArm:
    def test_wave_no_arm_returns_diagnosis(self):
        """context.arm is None -> diagnosis == 'no_arm'."""
        skill = WaveSkill()
        context = _make_context(arm=None)
        result = skill.execute({}, context)

        assert result.success is False
        assert result.result_data.get("diagnosis") == "no_arm"


class TestWaveSkillDiagnosticsMoveFailed:
    def test_wave_raise_failed_returns_diagnosis(self):
        """First arm.move_joints (raise) returns False -> diagnosis == 'move_failed'."""
        arm = Mock()
        arm.move_joints = Mock(return_value=False)
        skill = WaveSkill()
        context = _make_context(arm=arm)
        result = skill.execute({}, context)

        assert result.success is False
        assert result.result_data.get("diagnosis") == "move_failed"


class TestWaveSkillDiagnosticsSuccess:
    def test_wave_success_returns_diagnosis_ok(self):
        """Full success path -> diagnosis == 'ok'."""
        arm = Mock()
        arm.move_joints = Mock(return_value=True)
        skill = WaveSkill()
        context = _make_context(arm=arm)
        result = skill.execute({}, context)

        assert result.success is True
        assert result.result_data.get("diagnosis") == "ok"


# ---------------------------------------------------------------------------
# PickSkill diagnostics
# ---------------------------------------------------------------------------

import numpy as np
from vector_os_nano.skills.pick import PickSkill
from vector_os_nano.core.world_model import WorldModel, ObjectState


def _make_pick_context(
    arm=None,
    gripper=None,
    perception=None,
    world: WorldModel | None = None,
) -> SkillContext:
    """Build a SkillContext suitable for PickSkill unit tests.

    hardware_offsets=False so workspace check runs on raw coordinates.
    max_retries=1 so tests don't loop.
    """
    if world is None:
        world = WorldModel()
    return SkillContext(
        arm=arm,
        gripper=gripper,
        perception=perception,
        world_model=world,
        calibration=np.eye(4),
        config={
            "skills": {
                "pick": {
                    "hardware_offsets": False,
                    "z_offset": 0.0,
                    "pre_grasp_height": 0.04,
                    "max_retries": 1,
                },
                "home": {
                    "joint_values": [0.0, 0.0, 0.0, 0.0, 0.0],
                },
            },
        },
    )


def _default_pick_arm() -> Mock:
    arm = Mock()
    arm.get_joint_positions = Mock(return_value=[0.0, 0.0, 0.0, 0.0, 0.0])
    arm.ik = Mock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
    arm.move_joints = Mock(return_value=True)
    return arm


def _default_pick_gripper() -> Mock:
    gripper = Mock()
    gripper.open = Mock()
    gripper.close = Mock()
    return gripper


class TestPickSkillFailureModes:
    """PickSkill must expose failure_modes as a class attribute."""

    def test_failure_modes_attribute_exists(self):
        skill = PickSkill()
        assert hasattr(skill, "failure_modes")

    def test_failure_modes_contains_required_codes(self):
        skill = PickSkill()
        assert "no_arm" in skill.failure_modes
        assert "object_not_found" in skill.failure_modes
        assert "out_of_workspace" in skill.failure_modes
        assert "ik_unreachable" in skill.failure_modes
        assert "move_failed" in skill.failure_modes


class TestPickSkillDiagnosticsNoArm:
    def test_pick_no_arm_returns_diagnosis(self):
        """context.arm is None -> diagnosis == 'no_arm'."""
        skill = PickSkill()
        ctx = _make_pick_context(arm=None)
        result = skill.execute({"object_label": "mug"}, ctx)

        assert result.success is False
        assert result.result_data.get("diagnosis") == "no_arm"


class TestPickSkillDiagnosticsObjectNotFound:
    def test_pick_object_not_found_returns_diagnosis(self):
        """No perception, empty world model -> diagnosis == 'object_not_found'."""
        skill = PickSkill()
        arm = _default_pick_arm()
        ctx = _make_pick_context(arm=arm, perception=None)
        # world model is empty — no objects
        result = skill.execute({"object_label": "mug"}, ctx)

        assert result.success is False
        assert result.result_data.get("diagnosis") == "object_not_found"
        assert "query" in result.result_data
        assert "world_model_objects" in result.result_data


class TestPickSkillDiagnosticsOutOfWorkspace:
    def test_pick_workspace_returns_diagnosis(self):
        """Object at x=0.5 (50cm) is outside 35cm workspace limit."""
        skill = PickSkill()
        arm = _default_pick_arm()
        world = WorldModel()
        world.add_object(ObjectState(
            object_id="obj_0",
            label="mug",
            x=0.5,   # 50cm — outside _WORKSPACE_MAX_DIST=0.35
            y=0.0,
            z=0.0,
            confidence=0.9,
            state="on_table",
        ))
        ctx = _make_pick_context(arm=arm, world=world)
        result = skill.execute({"object_label": "mug"}, ctx)

        assert result.success is False
        assert result.result_data.get("diagnosis") == "out_of_workspace"
        assert "target_base_cm" in result.result_data
        assert "distance_cm" in result.result_data
        assert "workspace_bounds_cm" in result.result_data


class TestPickSkillDiagnosticsIKFailure:
    def test_pick_ik_failure_returns_diagnosis(self):
        """Object in workspace but arm.ik returns None -> diagnosis == 'ik_unreachable'."""
        skill = PickSkill()
        arm = _default_pick_arm()
        arm.ik = Mock(return_value=None)  # IK always fails
        world = WorldModel()
        world.add_object(ObjectState(
            object_id="obj_0",
            label="mug",
            x=0.20,   # 20cm — inside workspace
            y=0.0,
            z=0.0,
            confidence=0.9,
            state="on_table",
        ))
        ctx = _make_pick_context(arm=arm, world=world)
        result = skill.execute({"object_label": "mug"}, ctx)

        assert result.success is False
        assert result.result_data.get("diagnosis") == "ik_unreachable"


class TestPickSkillDiagnosticsSuccess:
    def test_pick_success_returns_diagnosis_ok(self):
        """Full mock success path -> diagnosis == 'ok' and position_cm present."""
        skill = PickSkill()
        arm = _default_pick_arm()
        gripper = _default_pick_gripper()
        world = WorldModel()
        world.add_object(ObjectState(
            object_id="obj_0",
            label="mug",
            x=0.20,
            y=0.0,
            z=0.0,
            confidence=0.9,
            state="on_table",
        ))
        ctx = _make_pick_context(arm=arm, gripper=gripper, world=world)
        result = skill.execute({"object_label": "mug"}, ctx)

        assert result.success is True
        assert result.result_data.get("diagnosis") == "ok"
        assert "position_cm" in result.result_data


class TestPickSkillWorldModelFix:
    def test_pick_removes_only_picked_object(self):
        """After successful pick, only the picked object is removed from world model."""
        skill = PickSkill()
        arm = _default_pick_arm()
        gripper = _default_pick_gripper()
        world = WorldModel()
        world.add_object(ObjectState(
            object_id="mug_0",
            label="mug",
            x=0.20,
            y=0.0,
            z=0.0,
            confidence=0.9,
            state="on_table",
        ))
        world.add_object(ObjectState(
            object_id="cup_0",
            label="cup",
            x=0.25,
            y=0.05,
            z=0.0,
            confidence=0.9,
            state="on_table",
        ))
        ctx = _make_pick_context(arm=arm, gripper=gripper, world=world)
        result = skill.execute({"object_label": "mug"}, ctx)

        assert result.success is True
        # mug_0 should be gone, cup_0 should still be there
        assert world.get_object("mug_0") is None
        assert world.get_object("cup_0") is not None


# ---------------------------------------------------------------------------
# DetectSkill diagnostics (T6)
# ---------------------------------------------------------------------------

from vector_os_nano.skills.detect import DetectSkill
from vector_os_nano.core.types import Detection, TrackedObject, Pose3D


def _make_detect_ctx(perception=None, world=None) -> SkillContext:
    """Minimal SkillContext for DetectSkill diagnostics tests."""
    return SkillContext(
        arm=None,
        gripper=None,
        perception=perception,
        world_model=world if world is not None else WorldModel(),
        calibration=None,
        config={},
    )


class TestDetectSkillFailureModes:
    """DetectSkill must expose failure_modes class attribute."""

    def test_failure_modes_attribute_exists(self):
        skill = DetectSkill()
        assert hasattr(skill, "failure_modes")

    def test_failure_modes_contains_required_codes(self):
        skill = DetectSkill()
        assert "no_perception" in skill.failure_modes
        assert "no_detections" in skill.failure_modes


class TestDetectSkillDiagnosticsNoPerception:
    def test_detect_no_perception_returns_diagnosis(self):
        """context.perception is None -> diagnosis == 'no_perception'."""
        skill = DetectSkill()
        context = _make_detect_ctx(perception=None)
        result = skill.execute({"query": "banana"}, context)

        assert result.success is False
        assert result.result_data.get("diagnosis") == "no_perception"


class TestDetectSkillDiagnosticsSuccess:
    def test_detect_success_returns_diagnosis_ok(self):
        """Successful detection -> diagnosis == 'ok' and merged_count in result_data."""
        mock_perception = Mock()
        mock_perception.detect = Mock(return_value=[
            Detection(label="banana", bbox=(100, 100, 200, 200), confidence=0.9),
        ])
        mock_perception.track = Mock(return_value=[])

        skill = DetectSkill()
        context = _make_detect_ctx(perception=mock_perception)
        result = skill.execute({"query": "banana"}, context)

        assert result.success is True
        assert result.result_data.get("diagnosis") == "ok"
        assert "merged_count" in result.result_data

    def test_detect_no_detections_returns_diagnosis(self):
        """perception.detect returns [] -> diagnosis == 'no_detections' with query."""
        mock_perception = Mock()
        mock_perception.detect = Mock(return_value=[])

        skill = DetectSkill()
        context = _make_detect_ctx(perception=mock_perception)
        result = skill.execute({"query": "banana"}, context)

        assert result.success is True
        assert result.result_data.get("diagnosis") == "no_detections"
        assert result.result_data.get("query") == "banana"
