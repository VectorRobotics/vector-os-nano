# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tests for pick skill workspace bounds and target resolution.

Covers:
- Objects inside / outside the 5–35cm XY workspace radius
- Objects at (0,0,0) filtered by the >0.01 guard
- World model fallback when perception is None
- Error messages contain expected keywords

All tests use mocked hardware — no real arm or camera required.

Note on hardware_offsets:
  PickSkill applies +2cm X and +2cm Y when hardware_offsets=True (the default).
  The workspace check runs on the OFFSET position. Test coordinates are chosen
  to remain unambiguous after the offset is applied.
"""
from __future__ import annotations

import pytest

from vector_os_nano.perception.calibration import Calibration

from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.types import SkillResult
from vector_os_nano.core.world_model import ObjectState, WorldModel
from vector_os_nano.skills.pick import PickSkill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_arm():
    """Minimal arm mock with working IK (returns non-None joint vector)."""
    from unittest.mock import MagicMock

    arm = MagicMock()
    arm.get_joint_positions.return_value = [0.0] * 5
    arm.move_joints.return_value = True
    # IK returns a valid 5-DOF joint vector
    arm.ik.return_value = [0.1, -1.2, 0.5, 0.8, 0.3]
    return arm


def _make_gripper():
    from unittest.mock import MagicMock

    gripper = MagicMock()
    gripper.open.return_value = True
    gripper.close.return_value = True
    return gripper


def _make_context(
    base_pos: list[float] | None = None,
    perception=None,
    config: dict | None = None,
) -> SkillContext:
    """Build a SkillContext with optional object in world model.

    Args:
        base_pos: [x, y, z] in metres to place a test object, or None for empty model.
        perception: perception mock, or None.
        config: optional config override.
    """
    arm = _make_arm()
    gripper = _make_gripper()
    world = WorldModel()

    if base_pos is not None:
        world.add_object(ObjectState(
            object_id="test_0",
            label="test",
            x=float(base_pos[0]),
            y=float(base_pos[1]),
            z=float(base_pos[2]),
            confidence=0.9,
            state="on_table",
        ))

    default_config: dict = {
        "skills": {
            "pick": {
                "z_offset": 0.10,
                "max_retries": 1,
                # Disable hardware offsets so workspace check uses raw base_pos
                "hardware_offsets": False,
            },
            "home": {
                "joint_values": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
        },
    }
    if config is not None:
        # Shallow merge at skills.pick level
        default_config["skills"]["pick"].update(
            config.get("skills", {}).get("pick", {})
        )

    return SkillContext(
        arm=arm,
        gripper=gripper,
        perception=perception,
        world_model=world,
        calibration=Calibration(),
        config=default_config,
    )


# ---------------------------------------------------------------------------
# TestPickWorkspace
# ---------------------------------------------------------------------------


class TestPickWorkspace:
    """Workspace boundary enforcement in PickSkill._single_pick_attempt()."""

    def test_object_in_workspace_proceeds(self) -> None:
        """Object at (0.20, 0.05) is inside 5–35cm — pick should not reject it.

        With hardware_offsets=False the workspace check uses the raw XY position.
        dist = sqrt(0.2^2 + 0.05^2) ≈ 20.6cm — well inside the 5–35cm range.
        """
        skill = PickSkill()
        ctx = _make_context(base_pos=[0.20, 0.05, 0.005])
        result = skill.execute({"object_label": "test"}, ctx)
        assert "outside workspace" not in (result.error_message or ""), (
            f"Expected workspace check to pass, got: {result.error_message!r}"
        )

    def test_object_outside_workspace_radially_rejected(self) -> None:
        """Object at (0.289, -0.780) has dist ≈ 83cm — outside 35cm max."""
        skill = PickSkill()
        ctx = _make_context(base_pos=[0.289, -0.780, 0.005])
        result = skill.execute({"object_label": "test"}, ctx)
        assert not result.success
        assert "outside workspace" in (result.error_message or ""), (
            f"Expected 'outside workspace' in error, got: {result.error_message!r}"
        )

    def test_object_too_far_along_x_rejected(self) -> None:
        """Object at (0.40, 0.0) has dist = 40cm — outside 35cm max."""
        skill = PickSkill()
        ctx = _make_context(base_pos=[0.40, 0.0, 0.005])
        result = skill.execute({"object_label": "test"}, ctx)
        assert not result.success
        assert "outside workspace" in (result.error_message or "")

    def test_object_too_close_rejected(self) -> None:
        """Object at (0.02, 0.02) has dist ≈ 2.8cm — below 5cm minimum."""
        skill = PickSkill()
        ctx = _make_context(base_pos=[0.02, 0.02, 0.005])
        result = skill.execute({"object_label": "test"}, ctx)
        assert not result.success
        assert "outside workspace" in (result.error_message or ""), (
            f"Expected 'outside workspace' in error, got: {result.error_message!r}"
        )

    def test_object_at_boundary_25cm_accepted(self) -> None:
        """Object at (0.25, 0.0) = exactly 25cm — firmly inside workspace."""
        skill = PickSkill()
        ctx = _make_context(base_pos=[0.25, 0.0, 0.005])
        result = skill.execute({"object_label": "test"}, ctx)
        assert "outside workspace" not in (result.error_message or ""), (
            f"25cm object should pass workspace check, got: {result.error_message!r}"
        )

    def test_object_at_max_boundary_35cm_accepted(self) -> None:
        """Object at exactly 35cm from origin — right at the max boundary.

        dist = 0.35 so the check `dist_xy > 0.35` is False — accepted.
        """
        skill = PickSkill()
        ctx = _make_context(base_pos=[0.35, 0.0, 0.005])
        result = skill.execute({"object_label": "test"}, ctx)
        assert "outside workspace" not in (result.error_message or ""), (
            f"35cm boundary object should pass workspace check: {result.error_message!r}"
        )

    def test_object_just_beyond_max_rejected(self) -> None:
        """Object at (0.351, 0.0) — 35.1cm from origin, just over the 35cm max."""
        skill = PickSkill()
        ctx = _make_context(base_pos=[0.351, 0.0, 0.005])
        result = skill.execute({"object_label": "test"}, ctx)
        assert not result.success
        assert "outside workspace" in (result.error_message or "")


# ---------------------------------------------------------------------------
# TestPickTargetResolution
# ---------------------------------------------------------------------------


class TestPickTargetResolution:
    """World model fallback and target resolution in PickSkill._get_target_base_pos()."""

    def test_no_object_found_returns_cannot_locate(self) -> None:
        """Empty world model and no perception → 'Cannot locate' failure."""
        skill = PickSkill()
        ctx = _make_context(perception=None)  # no object, no perception
        result = skill.execute({"object_label": "nonexistent"}, ctx)
        assert not result.success
        assert "Cannot locate" in (result.error_message or ""), (
            f"Expected 'Cannot locate' error, got: {result.error_message!r}"
        )

    def test_world_model_fallback_used_when_no_perception(self) -> None:
        """With no perception, pick uses world model to find the object."""
        skill = PickSkill()
        ctx = _make_context(base_pos=[0.25, 0.0, 0.005], perception=None)
        result = skill.execute({"object_label": "test"}, ctx)
        # Workspace check must pass (25cm inside bounds)
        assert "outside workspace" not in (result.error_message or ""), (
            f"World model fallback should resolve to 25cm: {result.error_message!r}"
        )

    def test_world_model_zero_coords_filtered_out(self) -> None:
        """Objects at exact (0, 0, z) are filtered by the abs(x)>0.01 or abs(y)>0.01 guard.

        The pick skill ignores objects where both |x| <= 0.01 and |y| <= 0.01 because
        they're likely stale/zeroed placeholders, not real detections.
        """
        skill = PickSkill()
        ctx = _make_context(base_pos=[0.0, 0.0, 0.0], perception=None)
        result = skill.execute({"object_label": "test"}, ctx)
        assert not result.success
        assert "Cannot locate" in (result.error_message or ""), (
            f"Zero-coord object should be filtered, got: {result.error_message!r}"
        )

    def test_world_model_small_x_zero_y_filtered(self) -> None:
        """Object at (0.005, 0.0, 0.0) — |x|=0.005 ≤ 0.01 and |y|=0 ≤ 0.01 → filtered."""
        skill = PickSkill()
        ctx = _make_context(base_pos=[0.005, 0.0, 0.0], perception=None)
        result = skill.execute({"object_label": "test"}, ctx)
        assert not result.success
        assert "Cannot locate" in (result.error_message or "")

    def test_world_model_valid_y_not_filtered(self) -> None:
        """Object at (0.005, 0.20, 0.0): |y|=0.20 > 0.01 → NOT filtered by zero guard."""
        skill = PickSkill()
        ctx = _make_context(base_pos=[0.005, 0.20, 0.0], perception=None)
        result = skill.execute({"object_label": "test"}, ctx)
        # Object passes the zero-coord filter; dist ≈ 20cm which is in workspace
        assert "Cannot locate" not in (result.error_message or ""), (
            f"Object with valid Y should not be filtered: {result.error_message!r}"
        )

    def test_lookup_by_object_id(self) -> None:
        """Object can be located using object_id instead of object_label."""
        skill = PickSkill()
        ctx = _make_context(base_pos=[0.25, 0.0, 0.005], perception=None)
        result = skill.execute({"object_id": "test_0"}, ctx)
        assert "Cannot locate" not in (result.error_message or ""), (
            f"Lookup by object_id should find test_0: {result.error_message!r}"
        )

    def test_missing_arm_returns_no_arm_error(self) -> None:
        """PickSkill with arm=None returns 'No arm connected'."""
        skill = PickSkill()
        world = WorldModel()
        world.add_object(ObjectState(
            object_id="test_0", label="test",
            x=0.25, y=0.0, z=0.005,
            confidence=0.9, state="on_table",
        ))
        ctx = SkillContext(
            arm=None,
            gripper=None,
            perception=None,
            world_model=world,
            calibration=Calibration(),
            config={},
        )
        result = skill.execute({"object_label": "test"}, ctx)
        assert not result.success
        assert "No arm" in (result.error_message or ""), (
            f"Expected 'No arm' error, got: {result.error_message!r}"
        )


# ---------------------------------------------------------------------------
# TestPickWorkspaceWithHardwareOffsets
# ---------------------------------------------------------------------------


class TestPickWorkspaceWithHardwareOffsets:
    """Verify workspace check applies correctly when hardware_offsets=True.

    With hardware_offsets=True, PickSkill adds +0.02m to X and +0.02m to Y
    BEFORE the workspace distance check. Tests verify the offset shifts the
    effective position so boundary cases still behave correctly.
    """

    def _make_hw_context(self, base_pos: list[float]) -> SkillContext:
        """Context with hardware_offsets=True (the real-hardware default)."""
        arm = _make_arm()
        gripper = _make_gripper()
        world = WorldModel()
        world.add_object(ObjectState(
            object_id="test_0", label="test",
            x=float(base_pos[0]), y=float(base_pos[1]), z=float(base_pos[2]),
            confidence=0.9, state="on_table",
        ))
        return SkillContext(
            arm=arm, gripper=gripper,
            perception=None,
            world_model=world,
            calibration=Calibration(),
            config={
                "skills": {
                    "pick": {
                        "z_offset": 0.10,
                        "max_retries": 1,
                        "hardware_offsets": True,
                    },
                    "home": {"joint_values": [0.0] * 5},
                },
            },
        )

    def test_hardware_offset_shifts_effective_position(self) -> None:
        """With hardware_offsets=True, raw (0.30, 0.02) + offset = (0.32, 0.04).

        dist = sqrt(0.32^2 + 0.04^2) ≈ 32.3cm — still inside workspace.
        """
        skill = PickSkill()
        ctx = self._make_hw_context(base_pos=[0.30, 0.02, 0.005])
        result = skill.execute({"object_label": "test"}, ctx)
        assert "outside workspace" not in (result.error_message or ""), (
            f"Expected to pass workspace check: {result.error_message!r}"
        )

    def test_hardware_offset_can_push_object_outside_workspace(self) -> None:
        """Raw position (0.33, 0.0) + offset (+0.02 X) = (0.35, 0.02).

        dist = sqrt(0.35^2 + 0.02^2) ≈ 35.1cm — just outside 35cm max.
        """
        skill = PickSkill()
        ctx = self._make_hw_context(base_pos=[0.33, 0.0, 0.005])
        result = skill.execute({"object_label": "test"}, ctx)
        # After +0.02 offset: dist ≈ 35.1cm > 35cm → rejected
        assert not result.success
        assert "outside workspace" in (result.error_message or ""), (
            f"Expected workspace rejection after offset: {result.error_message!r}"
        )
