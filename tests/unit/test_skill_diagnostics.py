# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

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
        arm = Mock(dof=5)
        arm.home = Mock(return_value=False)
        skill = HomeSkill()
        context = _make_context(arm=arm)
        result = skill.execute({}, context)

        assert result.success is False
        assert result.result_data.get("diagnosis") == "move_failed"


class TestHomeSkillDiagnosticsSuccess:
    def test_home_success_returns_diagnosis_ok(self):
        """Full success path -> diagnosis == 'ok' and joint_values present."""
        arm = Mock(dof=5)
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
        assert "no_gripper" in skill.failure_modes


class TestGripperCloseSkillFailureModes:
    def test_failure_modes_attribute_exists(self):
        skill = GripperCloseSkill()
        assert hasattr(skill, "failure_modes")

    def test_failure_modes_contains_no_arm(self):
        skill = GripperCloseSkill()
        assert "no_gripper" in skill.failure_modes


class TestGripperOpenSkillDiagnosticsNoGripper:
    def test_gripper_open_no_gripper_returns_diagnosis(self):
        """context.gripper is None -> diagnosis == 'no_arm'."""
        skill = GripperOpenSkill()
        context = _make_context(arm=None, gripper=None)
        result = skill.execute({}, context)

        assert result.success is False
        assert result.result_data.get("diagnosis") == "no_gripper"


class TestGripperOpenSkillDiagnosticsSuccess:
    def test_gripper_open_success_returns_diagnosis_ok(self):
        """Gripper present -> diagnosis == 'ok'."""
        gripper = _default_gripper()
        context = SkillContext(
            arm=None,
            gripper=gripper,
            perception=None,
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
        context = SkillContext(
            arm=None,
            gripper=gripper,
            perception=None,
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











# ---------------------------------------------------------------------------
# PickSkill diagnostics
# ---------------------------------------------------------------------------



def test_place_uses_configured_home_and_reports_home_failure():
    arm = _default_arm()
    arm.home.return_value = False
    result = PlaceSkill().execute({'x': .3, 'y': 0., 'z': .04}, _make_context(arm, _default_gripper()))
    assert not result.success and result.result_data == {'diagnosis': 'move_failed', 'phase': 'home'}
    arm.home.assert_called_once_with()
    assert arm.move_joints.call_count == 3
