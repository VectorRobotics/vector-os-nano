# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""SO-101 robot arm driver — pure Python, no ROS2.

Implements the ArmProtocol interface using SerialBus for low-level
SCS communication. Trajectory execution uses linear interpolation in
joint space, ported from vector_ws hardware_bridge.py execute_trajectory().

Connect/disconnect lifecycle:
  1. connect()  — open serial, read-before-write, enable torque
  2. use arm     — get_joint_positions(), move_joints(), stop(), ...
  3. disconnect() — disable torque, close serial
"""

import logging
import time
from typing import Any, Optional

from vector_os_nano.hardware.so101.joint_config import (
    ALL_JOINT_NAMES,
    ARM_JOINT_NAMES,
    JOINT_CONFIG,
    enc_to_rad,
    rad_to_enc,
)
from vector_os_nano.hardware.so101.serial_bus import SerialBus

logger = logging.getLogger(__name__)

# Default trajectory interpolation parameters
_DEFAULT_WAYPOINTS = 50   # number of linear interpolation steps
_SETTLE_DELAY = 0.5       # seconds to wait after final waypoint
_READ_ATTEMPTS = 3        # servo position reads per joint before giving up


class SO101Arm:
    """SO-101 robot arm driver.

    Pure Python implementation of the ArmProtocol interface.
    Controls the 5 arm joints (motors 1-5) via Feetech STS3215 servos.
    The gripper (motor 6) is controlled separately by SO101Gripper.

    Usage:
        arm = SO101Arm(port="/dev/ttyACM0")
        arm.connect()
        arm.move_joints([0.0, -1.0, 0.5, 0.8, 0.3], duration=3.0)
        arm.disconnect()
    """

    def __init__(self, port: str = "/dev/ttyACM0", baudrate: int = 1000000) -> None:
        self.port = port
        self.baudrate = baudrate
        self._bus = SerialBus(port=port, baudrate=baudrate)
        self._connected: bool = False
        self._ik_solver: Optional[Any] = None

    # ------------------------------------------------------------------
    # ArmProtocol properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "so101"

    @property
    def joint_names(self) -> list[str]:
        return ARM_JOINT_NAMES

    @property
    def dof(self) -> int:
        return 5

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to servos.

        Steps (ported from hardware_bridge._connect()):
        1. Open serial port via SerialBus.
        2. For each motor: read current position while torque is off.
        3. Write current position as the goal BEFORE enabling torque
           (prevents startup jump to old goal position).
        4. Enable torque — motor holds current position.

        Raises:
            RuntimeError: if port cannot be opened or baudrate fails.
        """
        if not self._bus.connect():
            raise RuntimeError(
                f"SO101Arm: cannot open serial port {self.port}. "
                "Check cable, permissions (adduser $USER dialout), and port path."
            )

        logger.info("SO101Arm: connected to %s, enabling motors", self.port)

        # Read-before-write for ALL joints (including gripper) to prevent jump
        for name, cfg in JOINT_CONFIG.items():
            motor_id = cfg["id"]
            pos = self._bus.read_position(motor_id)
            if pos >= 0:
                # Write current position as goal before enabling torque
                self._bus.write_position(motor_id, pos)
            ok = self._bus.set_torque(motor_id, enable=True)
            if not ok:
                logger.warning("SO101Arm: torque enable failed for %s (id=%d)", name, motor_id)

        self._connected = True
        logger.info("SO101Arm: all motors torque-enabled")

    def disconnect(self) -> None:
        """Disable torque on all motors and close serial port.

        Safe to call even if connect() was never called or already disconnected.
        """
        if self._connected:
            logger.info("SO101Arm: disabling torque on all motors")
            for name, cfg in JOINT_CONFIG.items():
                ok = self._bus.set_torque(cfg["id"], enable=False)
                if not ok:
                    logger.warning(
                        "SO101Arm: torque disable failed for %s (id=%d)", name, cfg["id"]
                    )
        self._bus.disconnect()
        self._connected = False

    def _require_connection(self) -> None:
        """Raise RuntimeError if not connected."""
        if not self._connected:
            raise RuntimeError(
                "SO101Arm: not connected. Call connect() first."
            )

    # ------------------------------------------------------------------
    # Joint state
    # ------------------------------------------------------------------

    def get_joint_positions(self) -> list[float]:
        """Read all arm joint positions from servos.

        Returns a list of 5 floats in radians, one per arm joint in
        ARM_JOINT_NAMES order. A failed servo read is retried; it is never
        replaced by a made-up value (a fake 0.0 start makes move_joints jump).

        Raises:
            RuntimeError: if not connected, or a servo read keeps failing.
        """
        self._require_connection()
        positions: list[float] = []
        for name in ARM_JOINT_NAMES:
            motor_id = JOINT_CONFIG[name]["id"]
            for _ in range(_READ_ATTEMPTS):
                enc = self._bus.read_position(motor_id)
                if enc >= 0:
                    break
            else:
                raise RuntimeError(
                    f"SO101Arm: read_position failed {_READ_ATTEMPTS}x for {name} (id={motor_id})"
                )
            positions.append(enc_to_rad(name, enc))
        return positions

    # ------------------------------------------------------------------
    # Motion commands
    # ------------------------------------------------------------------

    def move_joints(self, positions: list[float], duration: float = 3.0) -> bool:
        """Move arm to target joint positions via linear interpolation.

        Ported from hardware_bridge.execute_trajectory(): generates
        _DEFAULT_WAYPOINTS linearly-spaced waypoints between the current
        position and the target, then writes each waypoint with an
        inter-waypoint delay to achieve the requested duration.

        Args:
            positions: Target joint positions in radians, length must equal dof (5).
            duration: Total motion time in seconds (default 3.0).

        Returns:
            True if all waypoints were written successfully.

        Raises:
            RuntimeError: if not connected.
            ValueError: if len(positions) != dof.
        """
        self._require_connection()
        if len(positions) != self.dof:
            raise ValueError(
                f"SO101Arm.move_joints: expected {self.dof} positions, "
                f"got {len(positions)}"
            )

        try:
            current = self.get_joint_positions()
        except RuntimeError as exc:
            logger.error("%s — not moving", exc)
            return False
        n_waypoints = _DEFAULT_WAYPOINTS
        delay = duration / n_waypoints if n_waypoints > 0 else 0.0

        for step in range(1, n_waypoints + 1):
            alpha = step / n_waypoints
            waypoint = [
                current[i] + alpha * (positions[i] - current[i])
                for i in range(self.dof)
            ]
            success = self._write_arm_positions(waypoint)
            if not success:
                logger.error("SO101Arm: write failed at waypoint %d/%d", step, n_waypoints)
                return False
            if delay > 0.0:
                time.sleep(delay)

        # Allow servos to settle at final position
        time.sleep(_SETTLE_DELAY)
        return True

    def move_cartesian(
        self,
        target_xyz: tuple[float, float, float],
        duration: float = 3.0,
    ) -> bool:
        """Move end-effector to target Cartesian position.

        Requires an IK solver to be set via set_ik_solver(). The solver is
        called with target_xyz and the current joint positions as seed.

        Args:
            target_xyz: Target (x, y, z) in metres in the base frame.
            duration: Motion duration in seconds.

        Returns:
            True on success, False if IK has no solution or solver not set.

        Raises:
            RuntimeError: if not connected.
        """
        self._require_connection()
        if self._ik_solver is None:
            logger.error(
                "SO101Arm: move_cartesian called without an IK solver. "
                "Call set_ik_solver() first."
            )
            raise RuntimeError("SO101Arm: IK solver not set. Call set_ik_solver() first.")

        current = self.get_joint_positions()
        joint_targets = self._ik_solver.ik(target_xyz, current)
        if joint_targets is None:
            logger.warning("SO101Arm: IK returned no solution for target %s", target_xyz)
            return False

        return self.move_joints(list(joint_targets), duration=duration)

    def stop(self) -> None:
        """Emergency stop — write current encoder positions as goals.

        Freezes the arm in its current position by setting each motor's
        goal to its present position, halting any ongoing motion.

        Raises:
            RuntimeError: if not connected.
        """
        self._require_connection()
        logger.warning("SO101Arm: STOP called — freezing all joints")
        for name in ARM_JOINT_NAMES:
            motor_id = JOINT_CONFIG[name]["id"]
            enc = self._bus.read_position(motor_id)
            if enc >= 0:
                self._bus.write_position(motor_id, enc)

    # ------------------------------------------------------------------
    # IK solver injection
    # ------------------------------------------------------------------

    def fk(
        self,
        joint_positions: list[float],
    ) -> tuple[list[float], list[list[float]]]:
        """Forward kinematics: joint positions -> end-effector pose.

        Delegates to the IK solver if set. Returns a zero pose otherwise.

        Args:
            joint_positions: Joint positions in radians, length == dof.

        Returns:
            (position_xyz, rotation_3x3)
        """
        if self._ik_solver is not None:
            return self._ik_solver.fk(joint_positions)
        logger.warning("SO101Arm.fk: IK solver not set, returning zero pose")
        return ([0.0, 0.0, 0.0], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def ik(
        self,
        target_xyz: tuple[float, float, float],
        current_joints: list[float] | None = None,
        *, target_rotation: list[list[float]] | None = None,
    ) -> list[float] | None:
        """Inverse kinematics: Cartesian target -> joint positions.

        Delegates to the IK solver if set. Returns None otherwise.

        Args:
            target_xyz: Target (x, y, z) in metres, robot base frame.
            current_joints: Seed configuration (uses current if None).

        Returns:
            Joint positions in radians if reachable, None if no solution.
        """
        if target_rotation is not None:
            raise NotImplementedError("SO101 hardware driver does not support oriented IK")
        if self._ik_solver is None:
            logger.warning("SO101Arm.ik: IK solver not set")
            return None
        seed = current_joints
        if seed is None and self._connected:
            seed = self.get_joint_positions()
        # IKSolver exposes ik_position() which returns (solution, residual)
        if hasattr(self._ik_solver, 'ik_position'):
            result, error = self._ik_solver.ik_position(target_xyz, seed)
            return result
        # Fallback for solvers with plain .ik() interface
        return self._ik_solver.ik(target_xyz, seed)

    def set_ik_solver(self, solver: Any) -> None:
        """Set the IK solver (injected to avoid circular dependency).

        The solver must implement:
            solver.ik(target_xyz, current_joints) -> list[float] | None
            solver.fk(joint_positions) -> tuple[list[float], list[list[float]]]
        """
        self._ik_solver = solver

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_arm_positions(self, positions: list[float]) -> bool:
        """Write radian positions for all arm joints to their servos.

        Args:
            positions: 5-element list of joint positions in radians.

        Returns:
            True if all writes succeeded.
        """
        all_ok = True
        for i, name in enumerate(ARM_JOINT_NAMES):
            enc = rad_to_enc(name, positions[i])
            ok = self._bus.write_position(JOINT_CONFIG[name]["id"], enc)
            if not ok:
                logger.warning("SO101Arm: write failed for joint %s", name)
                all_ok = False
        return all_ok
