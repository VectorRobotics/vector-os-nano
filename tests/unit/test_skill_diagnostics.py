# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Home and gripper diagnostic regressions; named placement lives in skills/test_place.py."""
from __future__ import annotations

from unittest.mock import Mock

import pytest

from vector_os_nano.core.skill import SkillContext

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

def _default_gripper() -> Mock:
    gripper = Mock()
    gripper.open = Mock()
    gripper.close = Mock()
    return gripper

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

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
