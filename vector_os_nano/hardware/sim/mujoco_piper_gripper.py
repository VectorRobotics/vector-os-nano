# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""MuJoCo Piper parallel-jaw gripper.

Implements GripperProtocol. Piper's gripper is a single-actuator parallel
jaw: one position actuator on joint7 (piper_gripper), and joint8 follows via
an equality constraint defined in the Menagerie MJCF. Names and range come
from the manifest arm block's ``gripper`` (``embodiments/go2_piper/robot.yaml``).

A grasp is physical first: ``close()`` drives the jaws shut, waits (sim time)
for them to settle, and welds an object ONLY if it is in contact with EVERY
finger body while the jaws are held open by it. The weld then keeps the soft
position-controlled grip from slipping; it never attracts an object that is
merely near the gripper.

Shares MjModel/MjData with the parent MuJoCoGo2; every write (jaw ctrl, weld
on/off) goes through the parent's SimRuntime at a step boundary.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from vector_os_nano.embodiments.config import GripperSpec
    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2

logger = logging.getLogger(__name__)

# The jaws get up to _SETTLE_S sim seconds to close; they are settled once they
# move less than _SETTLE_EPS (m) over one _SETTLE_POLL_S poll.
_SETTLE_S: float = 1.0
_SETTLE_POLL_S: float = 0.02
_SETTLE_EPS: float = 2e-4

# Lazy MuJoCo import (same pattern as the other sim modules)
_mujoco: Any = None


def _get_mujoco() -> Any:
    global _mujoco
    if _mujoco is None:
        import mujoco  # noqa: PLC0415

        _mujoco = mujoco
    return _mujoco


class MuJoCoPiperGripper:
    """Piper parallel-jaw gripper in MuJoCo.

    Args:
        go2: A connected MuJoCoGo2 instance launched with the arm.
        spec: The manifest gripper block; default: the go2_piper manifest's.
    """

    def __init__(self, go2: "MuJoCoGo2", spec: "GripperSpec | None" = None) -> None:
        if spec is None:
            from vector_os_nano.embodiments import registry

            spec = registry.load("go2_piper").arm.gripper
        self._go2 = go2
        self._spec = spec
        self._connected: bool = False
        self._actuator_id: int = -1
        self._joint_qpos_adr: int = -1
        self._finger_geoms: list[frozenset[int]] = []  # collision geoms, one set per finger
        self._welds: dict[int, int] = {}  # object body id -> its weld equality index
        self._held_object: str | None = None  # body welded by the last close()
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
        spec = self._spec

        aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, spec.actuator)
        if aid < 0:
            raise RuntimeError(
                f"MuJoCoPiperGripper: actuator {spec.actuator!r} not in loaded "
                "MJCF. Ensure sim was launched with with_arm=True."
            )
        self._actuator_id = aid
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, spec.joint)
        if jid < 0:
            raise RuntimeError(f"MuJoCoPiperGripper: joint {spec.joint!r} not in loaded MJCF.")
        self._joint_qpos_adr = int(model.jnt_qposadr[jid])

        self._finger_geoms = []
        for body in spec.finger_bodies:
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body)
            geoms = frozenset(
                g for g in range(model.ngeom)
                if model.geom_bodyid[g] == bid and (model.geom_contype[g] or model.geom_conaffinity[g])
            )
            if bid < 0 or not geoms:
                raise RuntimeError(f"MuJoCoPiperGripper: finger body {body!r} has no collision geom")
            self._finger_geoms.append(geoms)
        if not self._finger_geoms:
            raise RuntimeError("MuJoCoPiperGripper: the manifest names no finger_bodies")

        weld_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, spec.weld_body)
        self._welds = {
            int(model.eq_obj2id[i]): i for i in range(model.neq)
            if model.eq_type[i] == mj.mjtEq.mjEQ_WELD and int(model.eq_obj1id[i]) == weld_bid
        }
        from vector_os_nano.sim.ground_truth import GroundTruth

        self._gt = GroundTruth(self._go2.runtime, robot_root_bid=self._go2._mj.layout.root_bid)
        self._connected = True
        logger.info(
            "MuJoCoPiperGripper connected (actuator=%s, closed=%.3f, open=%.3f, %d weldable objects)",
            spec.actuator, spec.closed, spec.open, len(self._welds),
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
        """Open the jaws and release every grasp weld (one step boundary), verified.

        Returns False (and keeps the held state) if a weld is still active after
        the release or the write failed.
        """
        self._require_connection()
        aid, eqs = self._actuator_id, list(self._welds.values())

        def _open(_model: Any, data: Any) -> list[int]:
            data.ctrl[aid] = self._spec.open
            data.eq_active[eqs] = 0
            return [i for i in eqs if int(data.eq_active[i]) != 0]

        try:
            still = self._go2.runtime.call(_open)
        except Exception as exc:  # noqa: BLE001 — reported, never a silent success
            logger.error("MuJoCoPiperGripper: open/release failed: %s", exc)
            return False
        if still:
            logger.error("MuJoCoPiperGripper: welds %s still active after release", still)
            return False
        self._held_object = None
        return True

    def close(self) -> bool:
        """Close the jaws; weld the object they actually grip (see module docstring).

        Returns True once the jaws were commanded and settled (whether or not they
        gripped anything — is_holding() says that); False if the sim refused.
        """
        self._require_connection()
        aid = self._actuator_id
        rt = self._go2.runtime
        try:
            rt.call(lambda _m, d: d.ctrl.__setitem__(aid, self._spec.closed))
            self._wait_jaws_settled()
            held = rt.call(self._weld_if_gripped)
        except Exception as exc:  # noqa: BLE001 — a failed grasp is a no-hold, never a crash
            logger.warning("MuJoCoPiperGripper: close failed: %s", exc)
            return False
        if held is not None:
            self._held_object = held
        return True

    def is_holding(self, name: str | None = None) -> bool:
        """True iff a grasp weld is ACTIVE in the sim for the gripped object.

        ``name`` makes it target-aware: only that object (case-insensitive scene
        body name) counts.
        """
        held = self._held_object
        if held is None or (name is not None and str(name).strip().lower() != held.lower()):
            return False
        return bool(self.weld_is_active().get(held, False))

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

    def get_position(self) -> float:
        """Normalized jaw position: 0.0 closed, 1.0 fully open."""
        self._require_connection()
        pos = float(self._go2.runtime.snapshot().qpos[self._joint_qpos_adr])
        return max(0.0, min(1.0, pos / self._spec.open))

    def get_force(self) -> float | None:
        """Piper has no force sensor in the Menagerie MJCF — return None."""
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _wait_jaws_settled(self) -> None:
        rt = self._go2.runtime
        deadline = rt.sim_time + _SETTLE_S
        last = float(rt.snapshot().qpos[self._joint_qpos_adr])
        rt.wait_sim(5 * _SETTLE_POLL_S)  # let the servo start moving
        while rt.sim_time < deadline:
            now = float(rt.snapshot().qpos[self._joint_qpos_adr])
            if abs(now - last) < _SETTLE_EPS:
                return
            last = now
            if not rt.wait_sim(_SETTLE_POLL_S):
                return

    def _weld_if_gripped(self, model: Any, data: Any) -> str | None:
        """At a step boundary: weld the object every finger touches while the jaws
        are held open by it (anchor = the live relative pose, so nothing snaps)."""
        mj = _get_mujoco()
        opening = float(data.qpos[self._joint_qpos_adr]) - self._spec.closed
        n = int(data.ncon)
        g1 = np.asarray(data.contact.geom1[:n])
        g2 = np.asarray(data.contact.geom2[:n])
        touching: list[set[int]] = []
        for geoms in self._finger_geoms:
            bodies: set[int] = set()
            for a, b in zip(g1, g2):
                if a in geoms:
                    bodies.add(int(model.geom_bodyid[b]))
                elif b in geoms:
                    bodies.add(int(model.geom_bodyid[a]))
            touching.append(bodies)
        gripped = sorted(set.intersection(*touching) & set(self._welds))
        names = {
            b: mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b)
            for b in set().union(*touching)
        }
        if not gripped or opening < self._spec.min_opening:
            logger.info(
                "MuJoCoPiperGripper: no grasp — jaw opening %.1f mm, finger contacts %s",
                opening * 1000, [sorted(names[b] or "?" for b in t) for t in touching])
            return None
        bid = gripped[0]
        i = self._welds[bid]
        b1 = int(model.eq_obj1id[i])
        p1, R1 = data.xpos[b1], data.xmat[b1].reshape(3, 3)
        p2, R2 = data.xpos[bid], data.xmat[bid].reshape(3, 3)
        rel_quat = np.zeros(4)
        mj.mju_mat2Quat(rel_quat, (R1.T @ R2).flatten())
        model.eq_data[i, :3] = 0.0
        model.eq_data[i, 3:6] = R1.T @ (p2 - p1)
        model.eq_data[i, 6:10] = rel_quat
        data.eq_active[i] = 1
        logger.info("MuJoCoPiperGripper: grasped '%s' (jaw opening %.1f mm)", names[bid], opening * 1000)
        return names[bid]
