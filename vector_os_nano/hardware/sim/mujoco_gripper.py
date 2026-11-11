# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""MuJoCo-backed simulated gripper.

Implements GripperProtocol. Uses weld constraints for reliable grasping —
when close() is called and the EE is near an object, a weld constraint
attaches the object to the gripper body. On open(), the constraint is
released. The visual jaw mesh rotates via a separate revolute actuator.

This approach is standard in robotics simulation (avoids contact/friction
alignment issues with 5-DOF arms). It is the documented exception to the
world-side grasp decision (sim/grasp.py): the SO-101 jaws are visual only (no
collision), so this driver keeps welding by proximity and writes the weld itself,
inside a runtime job; the runtime logs the change from ``world.attached()`` (derived
from ``eq_active``) when that job ends. Every step goes through the arm's runtime.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm

logger = logging.getLogger(__name__)

# Visual jaw revolute joint angles (for claw animation)
_JAW_VISUAL_OPEN: float = 0.8
_JAW_VISUAL_CLOSED: float = -0.17
_JAW_VISUAL_ACTUATOR: str = "act_jaw_visual"

# Max distance (meters) from EE to object center for grasp to succeed
_GRASP_RADIUS: float = 0.05

# Steps to animate gripper open/close
_GRIPPER_SETTLE_STEPS: int = 400


class MuJoCoGripper:
    """Simulated gripper using weld constraints for grasping.

    On close(): finds the nearest object within _GRASP_RADIUS of the EE site,
    enables a pre-defined weld constraint to attach it to the gripper body.
    On open(): disables all weld constraints, releasing any held object.

    The visual jaw mesh animates open/close via a revolute actuator.

    Args:
        mujoco_arm: A connected MuJoCoArm instance.
    """

    def __init__(self, mujoco_arm: "MuJoCoArm") -> None:
        self._arm = mujoco_arm
        self._is_open: bool = True
        self._held_object: str | None = None

    # ------------------------------------------------------------------
    # GripperProtocol implementation
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Open the gripper and release any held object."""
        self._is_open = True
        self._release_all()
        self._animate_jaw(_JAW_VISUAL_OPEN)
        logger.debug("MuJoCoGripper: open (released %s)", self._held_object)
        self._held_object = None
        return True

    def close(self) -> bool:
        """Close the gripper. If an object is within grasp range, attach it."""
        self._is_open = False
        self._animate_jaw(_JAW_VISUAL_CLOSED)
        self._try_grasp()
        logger.debug("MuJoCoGripper: close (holding=%s)", self._held_object)
        return True

    def is_holding(self) -> bool:
        """Return True if an object is currently welded to the gripper."""
        return self._held_object is not None

    def get_position(self) -> float:
        """Return normalised gripper position (1.0=open, 0.0=closed)."""
        return 1.0 if self._is_open else 0.0

    def get_force(self) -> float | None:
        """Grip force — returns 5.0N when holding, 0 otherwise."""
        return 5.0 if self._held_object else 0.0

    # ------------------------------------------------------------------
    # Internal: weld constraint grasping
    # ------------------------------------------------------------------

    def _try_grasp(self) -> None:
        """Find nearest object within grasp radius and enable its weld constraint."""
        if not self._arm._connected:
            return

        try:
            best_name = self._arm.runtime.call(self._weld_nearest)
            if best_name is None:
                return
            self._held_object = best_name
            # Step to let weld settle
            self._arm.runtime.step(50)
            if self._arm._viewer is not None:
                self._arm._viewer.sync()
        except Exception as exc:  # noqa: BLE001
            logger.warning("MuJoCoGripper: grasp failed: %s", exc)

    def _weld_nearest(self, w: Any) -> str | None:
        """At a step boundary: weld the free body nearest the EE (within the grasp radius)."""
        import mujoco  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        from vector_os_nano.hardware.sim.mujoco_arm import object_positions

        model, data = w.model, w.data
        # Get EE position
        ee_pos = data.site_xpos[self._arm._ee_site_id].copy()

        # Find nearest free-body object
        best_name: str | None = None
        best_dist = _GRASP_RADIUS

        for name, pos in object_positions(model, data).items():
            dist = float(np.linalg.norm(np.array(pos) - ee_pos))
            if dist < best_dist:
                best_dist = dist
                best_name = name

        if best_name is None:
            logger.debug("MuJoCoGripper: no object within %.0fmm of EE", _GRASP_RADIUS * 1000)
            return None

        # Find and enable the weld constraint for this object
        for i in range(model.neq):
            eq_type = model.eq_type[i]
            if eq_type != mujoco.mjtEq.mjEQ_WELD:
                continue
            body2_id = model.eq_obj2id[i]
            body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
            if body2_name == best_name:
                # Set weld anchor to CURRENT relative pose so object
                # stays in place (no snap/fly-away on activation).
                body1_id = model.eq_obj1id[i]
                p1 = data.xpos[body1_id]
                R1 = data.xmat[body1_id].reshape(3, 3)
                p2 = data.xpos[body2_id]
                R2 = data.xmat[body2_id].reshape(3, 3)

                # Relative position of body2 in body1 frame
                rel_pos = R1.T @ (p2 - p1)

                # Relative rotation as quaternion (body2 in body1 frame)
                rel_R = R1.T @ R2
                # Rotation matrix to MuJoCo quaternion [w, x, y, z]
                rel_quat = np.zeros(4)
                mujoco.mju_mat2Quat(rel_quat, rel_R.flatten())

                # eq_data layout for weld: anchor(3) + pos(3) + quat(4) + torquescale(1) = 11
                model.eq_data[i, :3] = 0.0          # anchor at body1 origin
                model.eq_data[i, 3:6] = rel_pos     # relative position
                model.eq_data[i, 6:10] = rel_quat   # relative quaternion
                # data[10] = torquescale (keep default 1.0)

                data.eq_active[i] = 1
                logger.info(
                    "MuJoCoGripper: grasped '%s' (%.0fmm from EE)",
                    best_name, best_dist * 1000,
                )
                return best_name

        logger.warning("MuJoCoGripper: no weld constraint for '%s'", best_name)
        return None

    def _release_all(self) -> None:
        """Disable all weld constraints, releasing any held object."""
        if not self._arm._connected:
            return

        def release(w: Any) -> None:
            import mujoco  # noqa: PLC0415

            for i in range(w.model.neq):
                if w.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                    w.data.eq_active[i] = 0

        try:
            self._arm.runtime.call(release)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Internal: jaw animation
    # ------------------------------------------------------------------

    def _animate_jaw(self, target_angle: float) -> None:
        """Smoothly animate the visual jaw to target angle."""
        if not self._arm._connected:
            return

        try:
            import mujoco  # noqa: PLC0415

            act_id = mujoco.mj_name2id(
                self._arm._model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                _JAW_VISUAL_ACTUATOR,
            )
            if act_id < 0:
                return

            rt = self._arm.runtime
            start_val = rt.call(lambda w: float(w.data.ctrl[act_id]))
            tick = [0]

            def ramp(w: Any) -> None:
                t = (tick[0] + 1) / _GRIPPER_SETTLE_STEPS
                tick[0] += 1
                w.data.ctrl[act_id] = start_val + t * (target_angle - start_val)

            rt.step(_GRIPPER_SETTLE_STEPS, controller=self._arm._paced(ramp))

        except Exception as exc:  # noqa: BLE001
            logger.warning("MuJoCoGripper: jaw animation failed: %s", exc)
