# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""MuJoCo Piper parallel-jaw gripper.

Implements GripperProtocol. Piper's gripper is a single-actuator parallel
jaw: one position actuator on joint7 (piper_gripper), and joint8 follows via
an equality constraint defined in the Menagerie MJCF. Range is 0 (fully
closed) to 0.035m (fully open, ~35 mm jaw separation).

Shares MjModel/MjData with the parent MuJoCoGo2; every write (jaw ctrl, weld
on/off) goes through the parent's SimRuntime at a step boundary.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ACTUATOR_NAME: str = "piper_gripper"
_JOINT_NAME: str = "piper_joint7"

# joint7 range per piper.xml is [0, 0.035]; ctrl uses the same units (meters
# of jaw half-opening). Fully closed commands exactly zero; fully open is
# the upper joint limit.
_CLOSED_POS: float = 0.0
_OPEN_POS: float = 0.035

# Heuristic: jaws considered "holding something" when commanded closed but
# the physics-driven joint position is kept open by the gripped object.
_HOLDING_THRESHOLD: float = 0.005  # 5 mm jaw separation while cmd=closed

# Max distance (m) from the EE site to an object centre for the weld grasp to
# fire (mirrors MuJoCoGripper._GRASP_RADIUS). The perceived grasp point is ~2cm
# accurate and the IK drives the EE site to it, so 6cm comfortably covers IK
# residual without grabbing a neighbour (cylinders are 15cm apart).
_GRASP_RADIUS: float = 0.06

# Lazy MuJoCo import (same pattern as the other sim modules)
_mujoco: Any = None


def _get_mujoco() -> Any:
    global _mujoco
    if _mujoco is None:
        import mujoco  # noqa: PLC0415

        _mujoco = mujoco
    return _mujoco


# ---------------------------------------------------------------------------
# MuJoCoPiperGripper
# ---------------------------------------------------------------------------


class MuJoCoPiperGripper:
    """Piper parallel-jaw gripper in MuJoCo.

    Args:
        go2: A connected MuJoCoGo2 instance. Must have been launched with
            with_arm=True so the piper_gripper actuator exists.
    """

    def __init__(self, go2: "MuJoCoGo2") -> None:
        self._go2 = go2
        self._connected: bool = False
        self._actuator_id: int = -1
        self._joint_qpos_adr: int = -1
        self._ee_site_id: int = -1
        self._held_object: str | None = None  # weld-backed grasp state (the honest signal)
        self._free_bodies: dict[str, int] = {}
        self._gt: Any = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if not getattr(self._go2, "_connected", False):
            raise RuntimeError(
                "MuJoCoPiperGripper: parent MuJoCoGo2 must be connected first"
            )
        mj = _get_mujoco()
        model = self._go2._mj.model

        aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, _ACTUATOR_NAME)
        if aid < 0:
            raise RuntimeError(
                f"MuJoCoPiperGripper: actuator {_ACTUATOR_NAME!r} not in loaded "
                "MJCF. Ensure sim was launched with with_arm=True."
            )
        self._actuator_id = aid

        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, _JOINT_NAME)
        if jid < 0:
            raise RuntimeError(
                f"MuJoCoPiperGripper: joint {_JOINT_NAME!r} not in loaded MJCF."
            )
        self._joint_qpos_adr = int(model.jnt_qposadr[jid])

        # EE site for the weld grasp (nearest-object pick). Mirrors MuJoCoGripper's
        # use of the arm EE site; absent -> grasp degrades to a no-op (logged).
        self._ee_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "piper_ee_site")
        # Free bodies (grasp candidates) by id, resolved once.
        self._free_bodies = {}
        for b in range(model.nbody):
            j = int(model.body_jntadr[b])
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b)
            if j >= 0 and name and model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
                self._free_bodies[name] = b
        from vector_os_nano.sim.ground_truth import GroundTruth

        self._gt = GroundTruth(self._go2.runtime)

        self._connected = True
        logger.info(
            "MuJoCoPiperGripper connected (actuator=%s, closed=%.3f, open=%.3f)",
            _ACTUATOR_NAME, _CLOSED_POS, _OPEN_POS,
        )

    def disconnect(self) -> None:
        self._connected = False

    def _require_connection(self) -> None:
        if not self._connected:
            raise RuntimeError(
                "MuJoCoPiperGripper: not connected. Call connect() first."
            )

    # ------------------------------------------------------------------
    # GripperProtocol
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Open the jaws fully (35 mm) and RELEASE any welded object (one step boundary)."""
        self._require_connection()
        aid = self._actuator_id

        def _open(model: Any, data: Any) -> None:
            data.ctrl[aid] = _OPEN_POS
            self._release_all(model, data)

        self._go2.runtime.call(_open)
        self._held_object = None
        return True

    def close(self) -> bool:
        """Command jaws closed AND, if an object is within grasp range of the EE,
        activate its weld so it physically attaches + lifts with the arm (mirrors
        the proven SO-101 MuJoCoGripper). The weld is what makes holding_object
        grade GROUNDED — the object actually moves with the gripper. Both writes
        happen at one step boundary.
        """
        self._require_connection()
        aid = self._actuator_id

        def _close(model: Any, data: Any) -> None:
            data.ctrl[aid] = _CLOSED_POS
            self._try_grasp(model, data)

        try:
            self._go2.runtime.call(_close)
        except Exception as exc:  # noqa: BLE001 — a failed grasp is a no-hold, never a crash
            logger.warning("MuJoCoPiperGripper: grasp failed: %s", exc)
        return True

    def is_holding(self) -> bool:
        """True iff an object is WELDED to the gripper (the honest, weld-backed
        signal). No more position heuristic — a weld is the real physical grasp the
        oracle's lift+near-EE test corroborates and actor-causation reads as a 0->1.
        """
        return self._held_object is not None

    def weld_is_active(self) -> dict[str, bool]:
        """``{welded-body-name -> data.eq_active[i] != 0}`` over the live weld eqs —
        the SAME body2-keyed map MuJoCoGripper exposes, read by actor_causation
        (_capture_gripper) for the 0->1 fresh-grasp transition. Fail-safe ``{}``.
        """
        if not self._connected:
            return {}
        try:
            return self._gt.weld_active()
        except Exception as exc:  # noqa: BLE001
            logger.debug("MuJoCoPiperGripper: weld_is_active read failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Internal: weld-constraint grasping (runs at a step boundary)
    # ------------------------------------------------------------------

    def _try_grasp(self, model: Any, data: Any) -> None:
        """Find the nearest free object within _GRASP_RADIUS of the EE and activate
        its weld (anchor pinned to the live relative pose so it does not snap).
        Runs inside runtime.call, i.e. between two physics steps.
        """
        if self._ee_site_id < 0:
            logger.warning("MuJoCoPiperGripper: no piper_ee_site — cannot grasp")
            return
        mj = _get_mujoco()
        import numpy as np  # noqa: PLC0415

        ee_pos = np.array(data.site_xpos[self._ee_site_id], dtype=float).copy()
        best_name, best_dist = None, _GRASP_RADIUS
        # Track the TRUE nearest free body (regardless of radius) for the
        # miss diagnostic — a miss otherwise only logs the radius, hiding the
        # actual EE-to-object gap a grasp-reliability hypothesis loop needs.
        near_name, near_dist = None, float("inf")
        for name, bid in self._free_bodies.items():
            d = float(np.linalg.norm(np.array(data.xpos[bid]) - ee_pos))
            if d < near_dist:
                near_dist, near_name = d, name
            if d < best_dist:
                best_dist, best_name = d, name
        if best_name is None:
            logger.info(
                "MuJoCoPiperGripper: no object within %.0fmm of EE "
                "(nearest '%s' at %.0fmm)",
                _GRASP_RADIUS * 1000, near_name, near_dist * 1000)
            return
        for i in range(model.neq):
            if model.eq_type[i] != mj.mjtEq.mjEQ_WELD:
                continue
            b2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.eq_obj2id[i]))
            if b2 != best_name:
                continue
            b1 = int(model.eq_obj1id[i])
            b2id = int(model.eq_obj2id[i])
            p1 = data.xpos[b1]; R1 = data.xmat[b1].reshape(3, 3)
            p2 = data.xpos[b2id]; R2 = data.xmat[b2id].reshape(3, 3)
            rel_pos = R1.T @ (p2 - p1)
            rel_quat = np.zeros(4)
            mj.mju_mat2Quat(rel_quat, (R1.T @ R2).flatten())
            model.eq_data[i, :3] = 0.0
            model.eq_data[i, 3:6] = rel_pos
            model.eq_data[i, 6:10] = rel_quat
            data.eq_active[i] = 1
            self._held_object = best_name
            logger.info("MuJoCoPiperGripper: grasped '%s' (%.0fmm)", best_name, best_dist * 1000)
            return
        logger.warning("MuJoCoPiperGripper: no weld constraint for '%s'", best_name)

    @staticmethod
    def _release_all(model: Any, data: Any) -> None:
        """Disable all weld constraints (release any held object)."""
        mj = _get_mujoco()
        for i in range(model.neq):
            if model.eq_type[i] == mj.mjtEq.mjEQ_WELD:
                data.eq_active[i] = 0

    def get_position(self) -> float:
        """Normalized jaw position: 0.0 closed, 1.0 fully open."""
        self._require_connection()
        pos = float(self._go2.runtime.snapshot().qpos[self._joint_qpos_adr])
        return max(0.0, min(1.0, pos / _OPEN_POS))

    def get_force(self) -> float | None:
        """Piper has no force sensor in the Menagerie MJCF — return None."""
        return None
