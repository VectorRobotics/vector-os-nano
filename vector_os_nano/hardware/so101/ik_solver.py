# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Pinocchio-based FK/IK solver for the SO-101 arm.

Ported from vector_ws/src/so101_skills/so101_skills/pinocchio_ik.py.
All ROS2 imports removed — uses Python logging. FK/IK logic is identical
to the original; ik_position() now returns (solution, residual_error) tuple.

Usage:
    solver = IKSolver()
    pos, rot = solver.fk([0.0, -1.2, 0.5, 0.8, 0.3])
    solution, error = solver.ik_position((0.2, 0.0, 0.05), current_radians)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from vector_os_nano.hardware.so101.joint_config import ARM_JOINT_NAMES

logger = logging.getLogger(__name__)


# The bundled SO-101 URDF (onshape-to-robot export, the calibrated so101_new_calib).
_URDF = Path(__file__).with_name("so101.urdf")


class IKSolver:
    """Pinocchio-based FK/IK solver for SO-101.

    Constants match the original pinocchio_ik.py exactly so that joint
    trajectories produced here are consistent with hardware_bridge.
    """

    IK_MAX_ITERS: int = 300
    IK_STEP_SIZE: float = 0.3
    IK_DAMPING: float = 1e-4
    IK_POS_TOL: float = 0.002  # 2 mm position tolerance
    IK_STAGNATION_ITERS: int = 30
    IK_STAGNATION_TOL: float = 0.001

    def __init__(self, urdf_path: str | None = None) -> None:
        """Load URDF and initialise Pinocchio model.

        Args:
            urdf_path: explicit path to the SO-101 URDF; None loads the bundled one.

        Raises:
            ImportError: if pinocchio is not installed.
            FileNotFoundError: if the URDF cannot be located.
        """
        try:
            import pinocchio as pin  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pinocchio is required for IKSolver. "
                "Install it: pip install pin"
            ) from exc

        self._pin = pin

        urdf_path = str(urdf_path or _URDF)
        # Kinematics only: the URDF's package:// meshes are not needed for FK/IK.
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # Ordered list of joint names as Pinocchio sees them (skips universe)
        self._joint_order: list[str] = [
            self.model.names[i] for i in range(1, self.model.njoints)
        ]
        self._ee_frame_id: int = self.model.getFrameId("gripper_link")
        self._gripper_frame_id: int = self.model.getFrameId("gripper_frame_link")

        logger.info("IKSolver loaded URDF: %s", urdf_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _radians_to_q(self, joint_radians: list[float]) -> np.ndarray:
        """Map 5 arm-joint radians to a full Pinocchio configuration vector."""
        q = np.zeros(self.model.nq)
        for i, jname in enumerate(self._joint_order):
            if i < len(joint_radians) and jname in ARM_JOINT_NAMES:
                idx = ARM_JOINT_NAMES.index(jname)
                q[i] = joint_radians[idx]
        return q

    def _q_to_radians(self, q: np.ndarray) -> list[float]:
        """Extract 5 arm-joint radians from a full Pinocchio configuration vector."""
        result: list[float] = []
        for jname in ARM_JOINT_NAMES:
            if jname in self._joint_order:
                idx = self._joint_order.index(jname)
                result.append(float(q[idx]))
            else:
                result.append(0.0)
        return result

    def _clamp_to_limits(self, q: np.ndarray) -> np.ndarray:
        """Clamp joint angles to URDF position limits."""
        return np.clip(
            q,
            self.model.lowerPositionLimit,
            self.model.upperPositionLimit,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fk(self, joint_radians: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics to the gripper_link frame.

        Args:
            joint_radians: 5 arm joint values in radians.

        Returns:
            Tuple of (position_xyz, rotation_3x3) both as numpy arrays.
            position_xyz is shape (3,) in meters relative to base_link.
            rotation_3x3 is shape (3, 3).
        """
        pin = self._pin
        q = self._radians_to_q(joint_radians)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        T = self.data.oMf[self._ee_frame_id]
        return np.array(T.translation), np.array(T.rotation)

    def fk_gripper_tip(self, joint_radians: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics to gripper jaw tips (9.8 cm from gripper_link).

        Args:
            joint_radians: 5 arm joint values in radians.

        Returns:
            Tuple of (position_xyz, rotation_3x3) at the gripper_frame_link.
        """
        pin = self._pin
        q = self._radians_to_q(joint_radians)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        T = self.data.oMf[self._gripper_frame_id]
        return np.array(T.translation), np.array(T.rotation)

    def ik_position(
        self,
        target_xyz: tuple[float, float, float],
        current_radians: list[float] | None = None,
        max_iters: int | None = None,
        tol: float | None = None,
    ) -> tuple[list[float] | None, float]:
        """Position-only IK using damped least-squares Jacobian method.

        Warm-started from current_radians (or zeros if None).

        Args:
            target_xyz: desired (x, y, z) in base_link frame, meters.
            current_radians: current 5-joint configuration for warm-start.
                If None, starts from zero configuration.
            max_iters: override IK_MAX_ITERS.
            tol: override IK_POS_TOL (meters).

        Returns:
            (solution, residual_error_meters) tuple.
            solution is a list of 5 joint radians, or None if IK failed.
            residual_error is always returned so the caller knows quality.
        """
        pin = self._pin
        _max_iters = max_iters if max_iters is not None else self.IK_MAX_ITERS
        _tol = tol if tol is not None else self.IK_POS_TOL

        target = np.array(target_xyz, dtype=float)
        q_init = current_radians if current_radians is not None else [0.0] * 5
        q = self._radians_to_q(q_init)

        err_norm: float = float("inf")
        prev_err: float = float("inf")

        for iteration in range(_max_iters):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            pin.computeJointJacobians(self.model, self.data, q)

            T_current = self.data.oMf[self._ee_frame_id]
            pos_err = target - T_current.translation
            err_norm = float(np.linalg.norm(pos_err))

            if err_norm < _tol:
                q = self._clamp_to_limits(q)
                logger.debug(
                    "IK converged in %d iters, residual=%.4fm", iteration, err_norm
                )
                return self._q_to_radians(q), err_norm

            # Position-only: top 3 rows of the frame Jacobian
            J_full = pin.getFrameJacobian(
                self.model,
                self.data,
                self._ee_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            J = J_full[:3, :]  # (3, nv)

            # Damped least-squares update
            dq = J.T @ np.linalg.solve(
                J @ J.T + self.IK_DAMPING * np.eye(3), pos_err
            )
            q = q + self.IK_STEP_SIZE * dq
            q = self._clamp_to_limits(q)

            # Stagnation detection: abort early if progress stalls
            if iteration > self.IK_STAGNATION_ITERS and prev_err > 0:
                if (prev_err - err_norm) / prev_err < self.IK_STAGNATION_TOL:
                    logger.debug(
                        "IK stagnated at iter %d, residual=%.4fm", iteration, err_norm
                    )
                    break
            prev_err = err_norm

        # Best-effort: accept if within 1 cm
        if err_norm < 0.01:
            logger.debug("IK best-effort (%.1fmm)", err_norm * 1000)
            return self._q_to_radians(q), err_norm

        logger.warning("IK failed, residual=%.4fm", err_norm)
        return None, err_norm

    @staticmethod
    def interpolate_trajectory(
        q_start: list[float],
        q_end: list[float],
        num_steps: int = 50,
        duration_sec: float = 3.0,
    ) -> list[dict]:
        """Linear joint-space interpolation between two configurations.

        Args:
            q_start: starting 5-joint configuration.
            q_end: ending 5-joint configuration.
            num_steps: number of interpolation intervals (produces num_steps+1 waypoints).
            duration_sec: total motion duration in seconds.

        Returns:
            List of dicts with keys:
                - "positions": list of 5 joint values (floats)
                - "time_from_start": time in seconds (float)
        """
        q_s = np.array(q_start)
        q_e = np.array(q_end)
        trajectory: list[dict] = []
        for i in range(num_steps + 1):
            t = i / num_steps
            q = q_s + t * (q_e - q_s)
            trajectory.append({
                "positions": q.tolist(),
                "time_from_start": t * duration_sec,
            })
        return trajectory
