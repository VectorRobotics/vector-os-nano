# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Unit tests for SO101Arm protocol compliance and basic properties.

Tests that SO101Arm satisfies the ArmProtocol interface contract:
- Properties (name, joint_names, dof) return correct values
- Operations before connect() raise RuntimeError or ConnectionError
- disconnect() is idempotent (no error if never connected)

No serial hardware required — all tests run against mocked serial bus.

Run with: pytest tests/unit/test_arm_protocol.py -v
"""

import pytest
from unittest.mock import MagicMock, patch

from vector_os_nano.hardware.so101.arm import SO101Arm
from vector_os_nano.hardware.so101.joint_config import ARM_JOINT_NAMES


# ---------------------------------------------------------------------------
# 1. Property tests
# ---------------------------------------------------------------------------

class TestSO101ArmProperties:
    """Static properties must return correct values regardless of connection."""

    def test_name_is_so101(self):
        arm = SO101Arm()
        assert arm.name == "so101"

    def test_joint_names_matches_arm_joint_names(self):
        arm = SO101Arm()
        assert arm.joint_names == ARM_JOINT_NAMES

    def test_dof_is_5(self):
        arm = SO101Arm()
        assert arm.dof == 5

    def test_joint_names_length(self):
        arm = SO101Arm()
        assert len(arm.joint_names) == 5

    def test_joint_names_order(self):
        arm = SO101Arm()
        expected = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        assert arm.joint_names == expected

    def test_default_port(self):
        arm = SO101Arm()
        assert arm.port == "/dev/ttyACM0"

    def test_custom_port(self):
        arm = SO101Arm(port="/dev/ttyUSB0")
        assert arm.port == "/dev/ttyUSB0"

    def test_default_baudrate(self):
        arm = SO101Arm()
        assert arm.baudrate == 1000000

    def test_custom_baudrate(self):
        arm = SO101Arm(baudrate=500000)
        assert arm.baudrate == 500000


# ---------------------------------------------------------------------------
# 2. Requires connection tests
# ---------------------------------------------------------------------------

class TestSO101ArmRequiresConnection:
    """Operations that require an open serial connection must raise before connect()."""

    def test_get_joint_positions_before_connect_raises(self):
        arm = SO101Arm()
        with pytest.raises((RuntimeError, ConnectionError, OSError)):
            arm.get_joint_positions()

    def test_move_joints_before_connect_raises(self):
        arm = SO101Arm()
        with pytest.raises((RuntimeError, ConnectionError, OSError)):
            arm.move_joints([0.0] * 5, duration=1.0)

    def test_stop_before_connect_raises(self):
        arm = SO101Arm()
        with pytest.raises((RuntimeError, ConnectionError, OSError)):
            arm.stop()

    def test_move_cartesian_before_connect_raises(self):
        arm = SO101Arm()
        with pytest.raises((RuntimeError, ConnectionError, OSError)):
            arm.move_cartesian((0.1, 0.0, 0.15))

    def test_disconnect_before_connect_is_safe(self):
        """disconnect() before connect() must not raise."""
        arm = SO101Arm()
        arm.disconnect()  # should be a no-op, not an exception


# ---------------------------------------------------------------------------
# 3. Connected state tests (mocked serial bus)
# ---------------------------------------------------------------------------

class TestSO101ArmConnected:
    """Tests with a mocked SerialBus to verify connected-state behavior."""

    def _make_mock_bus(self, read_positions=None):
        """Create a SerialBus mock that returns given encoder positions."""
        mock_bus = MagicMock()
        mock_bus.connect.return_value = True
        mock_bus._connected = True

        if read_positions is None:
            # Default: midpoint for each joint
            from vector_os_nano.hardware.so101.joint_config import JOINT_CONFIG, ARM_JOINT_NAMES
            read_positions = [
                (JOINT_CONFIG[n]["enc_min"] + JOINT_CONFIG[n]["enc_max"]) // 2
                for n in ARM_JOINT_NAMES
            ]
        mock_bus.read_position.side_effect = iter(read_positions * 100)  # enough for repeated calls
        mock_bus.write_position.return_value = True
        mock_bus.set_torque.return_value = True
        return mock_bus

    def test_get_joint_positions_returns_list_of_5(self):
        arm = SO101Arm()
        mock_bus = self._make_mock_bus()
        arm._bus = mock_bus
        arm._connected = True
        positions = arm.get_joint_positions()
        assert isinstance(positions, list)
        assert len(positions) == 5

    def test_get_joint_positions_returns_floats(self):
        arm = SO101Arm()
        mock_bus = self._make_mock_bus()
        arm._bus = mock_bus
        arm._connected = True
        positions = arm.get_joint_positions()
        for p in positions:
            assert isinstance(p, float)

    def test_a_failed_servo_read_is_retried_never_faked(self):
        from vector_os_nano.hardware.so101.joint_config import ARM_JOINT_NAMES, JOINT_CONFIG, enc_to_rad

        mids = [(JOINT_CONFIG[n]["enc_min"] + JOINT_CONFIG[n]["enc_max"]) // 2 for n in ARM_JOINT_NAMES]
        arm = SO101Arm()
        arm._bus = self._make_mock_bus([-1] + mids)  # the first read of joint 0 fails once
        arm._connected = True
        assert arm.get_joint_positions()[0] == pytest.approx(enc_to_rad(ARM_JOINT_NAMES[0], mids[0]))

    def test_a_persistent_read_failure_never_moves(self):
        arm = SO101Arm()
        mock_bus = self._make_mock_bus([-1])
        arm._bus = mock_bus
        arm._connected = True
        with pytest.raises(RuntimeError, match="read_position failed"):
            arm.get_joint_positions()
        assert arm.move_joints([0.0] * 5, duration=0.01) is False
        assert not mock_bus.write_position.called

    def test_move_joints_returns_bool(self):
        arm = SO101Arm()
        mock_bus = self._make_mock_bus()
        arm._bus = mock_bus
        arm._connected = True
        result = arm.move_joints([0.0] * 5, duration=0.01)
        assert isinstance(result, bool)

    def test_move_joints_calls_write_position(self):
        arm = SO101Arm()
        mock_bus = self._make_mock_bus()
        arm._bus = mock_bus
        arm._connected = True
        arm.move_joints([0.0] * 5, duration=0.01)
        assert mock_bus.write_position.called

    def test_stop_writes_current_positions(self):
        """stop() should write current positions as goals (freeze in place)."""
        arm = SO101Arm()
        mock_bus = self._make_mock_bus()
        arm._bus = mock_bus
        arm._connected = True
        arm.stop()
        assert mock_bus.write_position.called

    def test_set_ik_solver(self):
        arm = SO101Arm()
        mock_solver = MagicMock()
        arm.set_ik_solver(mock_solver)
        assert arm._ik_solver is mock_solver


# ---------------------------------------------------------------------------
# 4. move_joints input validation
# ---------------------------------------------------------------------------

class TestSO101ArmMoveJointsValidation:
    """move_joints must validate input dimensions."""

    def _connected_arm(self):
        from vector_os_nano.hardware.so101.joint_config import JOINT_CONFIG, ARM_JOINT_NAMES
        arm = SO101Arm()
        mock_bus = MagicMock()
        mock_bus.connect.return_value = True
        mock_bus.read_position.return_value = (
            JOINT_CONFIG[ARM_JOINT_NAMES[0]]["enc_min"] +
            JOINT_CONFIG[ARM_JOINT_NAMES[0]]["enc_max"]
        ) // 2
        mock_bus.write_position.return_value = True
        mock_bus.set_torque.return_value = True
        arm._bus = mock_bus
        arm._connected = True
        return arm

    def test_move_joints_wrong_count_raises(self):
        arm = self._connected_arm()
        with pytest.raises((ValueError, AssertionError)):
            arm.move_joints([0.0] * 3, duration=1.0)  # wrong count

    def test_move_joints_correct_count_ok(self):
        arm = self._connected_arm()
        # Should not raise
        arm.move_joints([0.0] * 5, duration=0.01)
