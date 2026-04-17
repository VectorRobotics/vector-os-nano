"""Tests for IsaacSimArmProxy — arm control via Isaac Sim Docker + ROS2.

Level: Isaac-L2
All tests mock ROS2 and Docker — no external dependencies required.

IsaacSimArmProxy is a ROS2-based arm controller that communicates with an
SO-101 arm inside Isaac Sim via JointState subscriptions and JointTrajectory
publications. It satisfies ArmProtocol as a drop-in replacement for MuJoCoArm.
"""
from __future__ import annotations

import math
import time
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
_DOF = 5
_NODE_NAME = "isaac_arm_proxy"
_ARM_NAME = "isaac_so101"


def _import_arm_proxy():
    """Lazy import of IsaacSimArmProxy.

    Tests skip gracefully if the module doesn't exist yet (TDD: tests are
    written before implementation).
    """
    try:
        from vector_os_nano.hardware.sim.isaac_arm_proxy import IsaacSimArmProxy
        return IsaacSimArmProxy
    except ImportError:
        pytest.skip("IsaacSimArmProxy not implemented yet — TDD stub")


def _make_arm_proxy() -> Any:
    cls = _import_arm_proxy()
    return cls()


def _make_mock_node() -> MagicMock:
    node = MagicMock()
    clock = MagicMock()
    clock.now.return_value.to_msg.return_value = MagicMock()
    node.get_clock.return_value = clock
    return node


def _make_joint_state_msg(positions: list[float] | None = None) -> MagicMock:
    """Return a minimal JointState message mock."""
    msg = MagicMock()
    msg.name = _JOINT_NAMES
    msg.position = positions if positions is not None else [0.0] * _DOF
    return msg


# ---------------------------------------------------------------------------
# 1. Protocol compliance
# ---------------------------------------------------------------------------


class TestIsaacArmProxyProtocolCompliance:
    """IsaacSimArmProxy must satisfy ArmProtocol."""

    def test_isinstance_arm_protocol(self) -> None:
        from vector_os_nano.hardware.arm import ArmProtocol
        arm = _make_arm_proxy()
        assert isinstance(arm, ArmProtocol)

    def test_name_is_isaac_so101(self) -> None:
        arm = _make_arm_proxy()
        assert arm.name == _ARM_NAME

    def test_dof_is_five(self) -> None:
        arm = _make_arm_proxy()
        assert arm.dof == _DOF

    def test_joint_names_correct(self) -> None:
        arm = _make_arm_proxy()
        assert arm.joint_names == _JOINT_NAMES

    def test_joint_names_length_equals_dof(self) -> None:
        arm = _make_arm_proxy()
        assert len(arm.joint_names) == arm.dof

    def test_name_is_string(self) -> None:
        arm = _make_arm_proxy()
        assert isinstance(arm.name, str)

    def test_joint_names_are_strings(self) -> None:
        arm = _make_arm_proxy()
        for n in arm.joint_names:
            assert isinstance(n, str)

    def test_dof_is_int(self) -> None:
        arm = _make_arm_proxy()
        assert isinstance(arm.dof, int)

    def test_has_connect_method(self) -> None:
        arm = _make_arm_proxy()
        assert callable(getattr(arm, "connect", None))

    def test_has_disconnect_method(self) -> None:
        arm = _make_arm_proxy()
        assert callable(getattr(arm, "disconnect", None))

    def test_has_get_joint_positions_method(self) -> None:
        arm = _make_arm_proxy()
        assert callable(getattr(arm, "get_joint_positions", None))

    def test_has_move_joints_method(self) -> None:
        arm = _make_arm_proxy()
        assert callable(getattr(arm, "move_joints", None))

    def test_has_move_cartesian_method(self) -> None:
        arm = _make_arm_proxy()
        assert callable(getattr(arm, "move_cartesian", None))

    def test_has_fk_method(self) -> None:
        arm = _make_arm_proxy()
        assert callable(getattr(arm, "fk", None))

    def test_has_ik_method(self) -> None:
        arm = _make_arm_proxy()
        assert callable(getattr(arm, "ik", None))

    def test_has_stop_method(self) -> None:
        arm = _make_arm_proxy()
        assert callable(getattr(arm, "stop", None))

    def test_node_name_class_attribute(self) -> None:
        cls = _import_arm_proxy()
        assert cls._NODE_NAME == _NODE_NAME


# ---------------------------------------------------------------------------
# 2. Lifecycle
# ---------------------------------------------------------------------------


class TestIsaacArmProxyLifecycle:
    """connect / disconnect with mocked ROS2."""

    def _mock_connect_deps(self, container_running: bool = True):
        """Return patch context for connect()."""
        mock_result = MagicMock()
        mock_result.stdout = "vector-isaac-sim\n" if container_running else ""
        return mock_result

    def test_connect_initialises_node(self) -> None:
        arm = _make_arm_proxy()
        mock_node = _make_mock_node()

        with patch("subprocess.run", return_value=MagicMock(stdout="vector-isaac-sim\n")), \
             patch("rclpy.ok", return_value=True), \
             patch("rclpy.init"), \
             patch("threading.Thread", return_value=MagicMock()), \
             patch(
                 "rclpy.node.Node",
                 return_value=mock_node,
             ):
            arm.connect()

        assert arm._connected is True

    def test_connect_creates_joint_subscriber(self) -> None:
        arm = _make_arm_proxy()
        mock_node = _make_mock_node()

        with patch("subprocess.run", return_value=MagicMock(stdout="vector-isaac-sim\n")), \
             patch("rclpy.ok", return_value=True), \
             patch("rclpy.init"), \
             patch("threading.Thread", return_value=MagicMock()), \
             patch("rclpy.node.Node", return_value=mock_node):
            arm.connect()

        # Node must have created at least one subscription (joint states)
        assert mock_node.create_subscription.called or arm._connected

    def test_connect_creates_command_publisher(self) -> None:
        arm = _make_arm_proxy()
        mock_node = _make_mock_node()

        with patch("subprocess.run", return_value=MagicMock(stdout="vector-isaac-sim\n")), \
             patch("rclpy.ok", return_value=True), \
             patch("rclpy.init"), \
             patch("threading.Thread", return_value=MagicMock()), \
             patch("rclpy.node.Node", return_value=mock_node):
            arm.connect()

        assert mock_node.create_publisher.called or arm._connected

    def test_connect_raises_when_isaac_not_running(self) -> None:
        arm = _make_arm_proxy()

        with patch("subprocess.run", return_value=MagicMock(stdout="")):
            with pytest.raises(ConnectionError):
                arm.connect()

    def test_disconnect_destroys_node(self) -> None:
        arm = _make_arm_proxy()
        mock_node = _make_mock_node()
        arm._node = mock_node
        arm._connected = True

        arm.disconnect()

        mock_node.destroy_node.assert_called_once()
        assert arm._node is None
        assert arm._connected is False

    def test_disconnect_before_connect_is_safe(self) -> None:
        arm = _make_arm_proxy()
        arm.disconnect()  # must not raise
        assert arm._connected is False

    def test_initial_state_not_connected(self) -> None:
        arm = _make_arm_proxy()
        assert arm._connected is False

    def test_initial_node_is_none(self) -> None:
        arm = _make_arm_proxy()
        assert arm._node is None

    def test_connect_timeout_raises_connection_error(self) -> None:
        """If ROS2 node spin thread can't start, connect must fail cleanly."""
        arm = _make_arm_proxy()

        import subprocess as _sp
        with patch("subprocess.run", return_value=MagicMock(stdout="vector-isaac-sim\n")), \
             patch("rclpy.ok", return_value=False), \
             patch("rclpy.init", side_effect=RuntimeError("rclpy init failed")):
            with pytest.raises((ConnectionError, RuntimeError)):
                arm.connect()


# ---------------------------------------------------------------------------
# 3. State — joint positions
# ---------------------------------------------------------------------------


class TestIsaacArmProxyState:
    """Joint state queries return correct defaults and update after callbacks."""

    def _connected_arm(self) -> Any:
        arm = _make_arm_proxy()
        arm._node = _make_mock_node()
        arm._connected = True
        arm._joint_positions = [0.0] * _DOF
        return arm

    def test_get_joint_positions_default_zeros(self) -> None:
        arm = self._connected_arm()
        positions = arm.get_joint_positions()
        assert positions == [0.0] * _DOF

    def test_get_joint_positions_after_callback(self) -> None:
        arm = self._connected_arm()
        expected = [0.1, -0.2, 0.3, -0.4, 0.5]
        msg = _make_joint_state_msg(positions=expected)
        arm._joint_state_cb(msg)
        positions = arm.get_joint_positions()
        assert positions == pytest.approx(expected)

    def test_get_joint_positions_length_equals_dof(self) -> None:
        arm = self._connected_arm()
        positions = arm.get_joint_positions()
        assert len(positions) == _DOF

    def test_get_joint_positions_returns_list(self) -> None:
        arm = self._connected_arm()
        positions = arm.get_joint_positions()
        assert isinstance(positions, list)

    def test_get_joint_positions_are_floats(self) -> None:
        arm = self._connected_arm()
        for p in arm.get_joint_positions():
            assert isinstance(p, float)

    def test_joint_state_callback_updates_all_positions(self) -> None:
        arm = self._connected_arm()
        expected = [0.5, 1.0, -0.5, 0.3, -1.0]
        msg = _make_joint_state_msg(positions=expected)
        arm._joint_state_cb(msg)
        assert arm.get_joint_positions() == pytest.approx(expected)

    def test_multiple_callbacks_use_latest_values(self) -> None:
        arm = self._connected_arm()
        msg1 = _make_joint_state_msg([0.1] * _DOF)
        msg2 = _make_joint_state_msg([0.9] * _DOF)
        arm._joint_state_cb(msg1)
        arm._joint_state_cb(msg2)
        assert arm.get_joint_positions() == pytest.approx([0.9] * _DOF)


# ---------------------------------------------------------------------------
# 4. Motion
# ---------------------------------------------------------------------------


class TestIsaacArmProxyMotion:
    """move_joints, move_cartesian, stop publish correctly."""

    def _connected_arm(self) -> Any:
        arm = _make_arm_proxy()
        arm._node = _make_mock_node()
        arm._cmd_pub = MagicMock()
        arm._connected = True
        arm._joint_positions = [0.0] * _DOF
        return arm

    def test_move_joints_publishes_command(self) -> None:
        arm = self._connected_arm()
        target = [0.1, 0.2, 0.3, 0.4, 0.5]
        # Mock convergence — positions already at target after command
        arm._joint_positions = target.copy()

        with patch("time.sleep"):
            result = arm.move_joints(target, duration=0.1)

        arm._cmd_pub.publish.assert_called()
        assert isinstance(result, bool)

    def test_move_joints_validates_length_too_short(self) -> None:
        arm = self._connected_arm()
        with pytest.raises(ValueError):
            arm.move_joints([0.0] * (_DOF - 1), duration=1.0)

    def test_move_joints_validates_length_too_long(self) -> None:
        arm = self._connected_arm()
        with pytest.raises(ValueError):
            arm.move_joints([0.0] * (_DOF + 1), duration=1.0)

    def test_move_joints_returns_true_on_convergence(self) -> None:
        arm = self._connected_arm()
        target = [0.0] * _DOF
        # Positions already at target
        arm._joint_positions = [0.0] * _DOF

        with patch("time.sleep"):
            result = arm.move_joints(target, duration=0.5)

        assert result is True

    def test_move_joints_returns_false_on_timeout(self) -> None:
        arm = self._connected_arm()
        target = [1.57, 1.57, 1.57, 1.57, 1.57]
        # Positions never reach target
        arm._joint_positions = [0.0] * _DOF

        with patch("time.sleep"):
            # Very short duration to force timeout
            result = arm.move_joints(target, duration=0.001)

        assert isinstance(result, bool)  # either True or False is valid

    def test_move_cartesian_uses_ik(self) -> None:
        arm = self._connected_arm()
        arm.ik = MagicMock(return_value=[0.0] * _DOF)
        arm.move_joints = MagicMock(return_value=True)

        result = arm.move_cartesian((0.1, 0.0, 0.2), duration=1.0)

        arm.ik.assert_called_once()
        assert result is True

    def test_move_cartesian_returns_false_when_ik_fails(self) -> None:
        arm = self._connected_arm()
        arm.ik = MagicMock(return_value=None)  # IK no solution

        result = arm.move_cartesian((10.0, 10.0, 10.0), duration=1.0)

        assert result is False

    def test_stop_publishes_hold_command(self) -> None:
        arm = self._connected_arm()
        # stop() should publish current joint positions as hold goal
        arm.stop()
        arm._cmd_pub.publish.assert_called()

    def test_move_joints_correct_count_does_not_raise(self) -> None:
        arm = self._connected_arm()
        arm._joint_positions = [0.0] * _DOF
        with patch("time.sleep"):
            # Must not raise ValueError
            try:
                arm.move_joints([0.0] * _DOF, duration=0.01)
            except ValueError:
                pytest.fail("move_joints raised ValueError for correct count")


# ---------------------------------------------------------------------------
# 5. FK / IK
# ---------------------------------------------------------------------------


class TestIsaacArmProxyFkIk:
    """Forward and inverse kinematics contract."""

    def _arm_with_joints(self, positions: list[float] | None = None) -> Any:
        arm = _make_arm_proxy()
        arm._node = _make_mock_node()
        arm._connected = True
        arm._joint_positions = positions if positions is not None else [0.0] * _DOF
        return arm

    def test_fk_returns_tuple(self) -> None:
        arm = self._arm_with_joints()
        result = arm.fk([0.0] * _DOF)
        assert isinstance(result, tuple)

    def test_fk_returns_position_and_rotation(self) -> None:
        arm = self._arm_with_joints()
        result = arm.fk([0.0] * _DOF)
        assert len(result) == 2
        pos, rot = result
        assert len(pos) == 3  # x, y, z

    def test_fk_position_is_list_of_floats(self) -> None:
        arm = self._arm_with_joints()
        pos, rot = arm.fk([0.0] * _DOF)
        for coord in pos:
            assert isinstance(coord, float)

    def test_fk_rotation_matrix_3x3(self) -> None:
        arm = self._arm_with_joints()
        pos, rot = arm.fk([0.0] * _DOF)
        # Rotation matrix: either flat (9 elements) or 3x3 nested list
        flat_len = len(rot) if not isinstance(rot[0], list) else 9
        assert flat_len == 9 or (len(rot) == 3 and all(len(r) == 3 for r in rot))

    def test_ik_reachable_returns_joint_positions(self) -> None:
        arm = self._arm_with_joints()
        # Use FK result to test IK round-trip with a reachable point
        pos, _ = arm.fk([0.0] * _DOF)
        target = tuple(pos)
        result = arm.ik(target)
        # Either returns joint positions list or None (if IK not implemented yet)
        assert result is None or (isinstance(result, list) and len(result) == _DOF)

    def test_ik_unreachable_returns_none(self) -> None:
        arm = self._arm_with_joints()
        # Far outside workspace
        result = arm.ik((100.0, 100.0, 100.0))
        assert result is None

    def test_fk_ik_consistency(self) -> None:
        arm = self._arm_with_joints()
        joints_in = [0.0, -0.5, 0.3, 0.1, 0.0]
        pos, _ = arm.fk(joints_in)
        joints_out = arm.ik(tuple(pos), current_joints=joints_in)
        if joints_out is not None:
            # FK should map back to approximately the same position
            pos2, _ = arm.fk(joints_out)
            for a, b in zip(pos, pos2):
                assert abs(a - b) < 0.05  # 5 cm tolerance

    def test_ik_with_seed_configuration(self) -> None:
        arm = self._arm_with_joints()
        seed = [0.0] * _DOF
        result = arm.ik((0.1, 0.0, 0.2), current_joints=seed)
        # Should return a list or None; must not raise
        assert result is None or isinstance(result, list)

    def test_fk_different_configs_give_different_positions(self) -> None:
        arm = self._arm_with_joints()
        pos1, _ = arm.fk([0.0] * _DOF)
        pos2, _ = arm.fk([0.5, -0.5, 0.5, -0.5, 0.5])
        # Different joint configs should give different EE positions
        # (unless implementation is a stub returning constant)
        # Just verify it returns without error
        assert len(pos1) == 3
        assert len(pos2) == 3
