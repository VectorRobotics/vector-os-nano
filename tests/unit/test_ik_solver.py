# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Unit tests for vector_os_nano.hardware.so101.ik_solver — TDD RED phase.

Tests require pinocchio. If not installed, entire module is skipped.
All tests use the bundled SO-101 URDF (hardware/urdf/so101.urdf).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pinocchio = pytest.importorskip("pinocchio", reason="pinocchio not installed")


# Home pose from skill_node_v2 — known-good configuration
HOME_JOINTS = [-0.014, -1.238, 0.562, 0.858, 0.311]


@pytest.fixture(scope="module")
def ik():
    from vector_os_nano.hardware.so101.ik_solver import IKSolver
    return IKSolver()  # the bundled URDF: no external ROS workspace needed


class TestFKHomePose:
    def test_fk_returns_position_and_rotation(self, ik):
        pos, rot = ik.fk(HOME_JOINTS)
        assert isinstance(pos, np.ndarray)
        assert pos.shape == (3,)
        assert isinstance(rot, np.ndarray)
        assert rot.shape == (3, 3)

    def test_fk_home_position_reasonable(self, ik):
        """Home position should be in front of the robot, elevated."""
        pos, _ = ik.fk(HOME_JOINTS)
        # x should be positive (forward) and within 0–40 cm
        assert 0.0 < pos[0] < 0.4
        # z should be positive (above table)
        assert pos[2] > 0.0

    def test_fk_zero_joints_returns_array(self, ik):
        pos, rot = ik.fk([0.0, 0.0, 0.0, 0.0, 0.0])
        assert pos.shape == (3,)
        assert rot.shape == (3, 3)

    def test_fk_gripper_tip_differs_from_fk(self, ik):
        """Gripper tip frame is 9.8 cm from gripper_link — must differ."""
        pos_ee, _ = ik.fk(HOME_JOINTS)
        pos_tip, _ = ik.fk_gripper_tip(HOME_JOINTS)
        dist = float(np.linalg.norm(pos_tip - pos_ee))
        # Should be close to 9.8 cm (within a few cm given frame offsets)
        assert dist > 0.01

    def test_fk_gripper_tip_returns_correct_shapes(self, ik):
        pos, rot = ik.fk_gripper_tip(HOME_JOINTS)
        assert pos.shape == (3,)
        assert rot.shape == (3, 3)


class TestIKRoundtrip:
    def test_fk_ik_roundtrip_within_tolerance(self, ik):
        """IK(FK(joints)) should recover joints within 2mm positional error."""
        target_pos, _ = ik.fk(HOME_JOINTS)
        solution, residual = ik.ik_position(
            tuple(target_pos.tolist()), HOME_JOINTS
        )
        assert solution is not None, "IK should find solution at reachable target"
        # Verify FK of solution is within 2mm of target
        pos_check, _ = ik.fk(solution)
        pos_err = float(np.linalg.norm(pos_check - target_pos))
        assert pos_err < 0.002, f"FK(IK(target)) error {pos_err*1000:.2f}mm > 2mm"

    def test_ik_returns_residual_tuple(self, ik):
        """ik_position must return (solution, residual_error) tuple."""
        target_pos, _ = ik.fk(HOME_JOINTS)
        result = ik.ik_position(tuple(target_pos.tolist()), HOME_JOINTS)
        assert isinstance(result, tuple), "ik_position must return a tuple"
        assert len(result) == 2, "Tuple must have exactly 2 elements"
        solution, residual = result
        assert isinstance(residual, float), "Residual must be a float"
        assert residual >= 0.0, "Residual must be non-negative"

    def test_ik_solution_has_5_joints(self, ik):
        target_pos, _ = ik.fk(HOME_JOINTS)
        solution, _ = ik.ik_position(tuple(target_pos.tolist()), HOME_JOINTS)
        assert solution is not None
        assert len(solution) == 5


class TestIKUnreachable:
    def test_ik_far_away_returns_none(self, ik):
        """Target far outside workspace should return (None, error)."""
        far_target = (5.0, 5.0, 5.0)  # 5 meters away — impossible
        solution, residual = ik.ik_position(far_target, HOME_JOINTS)
        assert solution is None, "IK should fail for unreachable target"
        assert residual > 0.01, "Residual should be large for unreachable target"

    def test_ik_below_table_returns_none(self, ik):
        """Target directly below base is unreachable."""
        below_target = (0.0, 0.0, -1.0)
        solution, residual = ik.ik_position(below_target, HOME_JOINTS)
        assert solution is None


class TestInterpolateTrajectory:
    def test_trajectory_step_count(self, ik):
        """num_steps=10 should produce 11 waypoints (including start)."""
        q_start = [0.0] * 5
        q_end = [0.5] * 5
        traj = ik.interpolate_trajectory(q_start, q_end, num_steps=10)
        assert len(traj) == 11

    def test_trajectory_default_steps(self, ik):
        """Default num_steps=50 produces 51 waypoints."""
        q_start = HOME_JOINTS
        q_end = [0.0] * 5
        traj = ik.interpolate_trajectory(q_start, q_end)
        assert len(traj) == 51

    def test_trajectory_timing(self, ik):
        """First waypoint at t=0, last at duration_sec."""
        duration = 3.0
        traj = ik.interpolate_trajectory([0.0]*5, [1.0]*5, num_steps=10, duration_sec=duration)
        first = traj[0]
        last = traj[-1]
        assert first["time_from_start"] == pytest.approx(0.0)
        assert last["time_from_start"] == pytest.approx(duration)

    def test_trajectory_waypoint_format(self, ik):
        """Each waypoint must be a dict with 'positions' and 'time_from_start'."""
        traj = ik.interpolate_trajectory([0.0]*5, [1.0]*5, num_steps=5)
        for wp in traj:
            assert isinstance(wp, dict)
            assert "positions" in wp
            assert "time_from_start" in wp
            assert len(wp["positions"]) == 5

    def test_trajectory_start_end_values(self, ik):
        """First waypoint matches q_start, last matches q_end."""
        q_start = [0.1, 0.2, 0.3, 0.4, 0.5]
        q_end = [0.5, 0.4, 0.3, 0.2, 0.1]
        traj = ik.interpolate_trajectory(q_start, q_end, num_steps=20)
        assert traj[0]["positions"] == pytest.approx(q_start, abs=1e-9)
        assert traj[-1]["positions"] == pytest.approx(q_end, abs=1e-9)

    def test_trajectory_interpolation_midpoint(self, ik):
        """At t=0.5, positions should be midpoint of q_start and q_end."""
        q_start = [0.0] * 5
        q_end = [1.0] * 5
        traj = ik.interpolate_trajectory(q_start, q_end, num_steps=10)
        mid = traj[5]  # index 5 = t=0.5
        assert mid["positions"] == pytest.approx([0.5] * 5, abs=1e-9)


class TestIKWithNoneStart:
    def test_ik_uses_zero_when_current_is_none(self, ik):
        """ik_position should accept None current_radians and use zeros."""
        target_pos, _ = ik.fk([0.0, -1.0, 0.5, 0.8, 0.3])
        solution, residual = ik.ik_position(tuple(target_pos.tolist()), None)
        # May or may not converge from zeros, but must not raise
        assert isinstance(residual, float)
