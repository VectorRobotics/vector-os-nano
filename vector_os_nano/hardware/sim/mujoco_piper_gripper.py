# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""MuJoCo Piper parallel-jaw gripper.

Implements GripperProtocol. Piper's gripper is a single-actuator parallel
jaw: one position actuator on joint7 (piper_gripper), and joint8 follows via
an equality constraint defined in the Menagerie MJCF. Names and range come
from the manifest arm block's ``gripper`` (``embodiments/go2_piper/robot.yaml``).

A grasp is physical first: ``close()`` drives the jaws shut, waits (sim time)
for them to settle, and asks the runtime for the world-side grasp decision
(``sim/grasp.py``): an object is attached ONLY if it is in contact with EVERY
finger body while the jaws are held open by it. The attachment then keeps the soft
position-controlled grip from slipping; it never attracts an object that is
merely near the gripper. This driver never attaches or detaches anything itself.

Shares the parent MuJoCoGo2's world; every write (jaw ctrl, grasp, release) goes
through its SimRuntime at a step boundary.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from vector_os_nano.sim.grasp import GraspSpec

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
        self._grasp: GraspSpec | None = None  # the manifest gripper in canonical ids
        self._held_object: str | None = None  # body attached by the last close()

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

        for body in spec.finger_bodies:
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body)
            if bid < 0 or not any(
                model.geom_bodyid[g] == bid and (model.geom_contype[g] or model.geom_conaffinity[g])
                for g in range(model.ngeom)
            ):
                raise RuntimeError(f"MuJoCoPiperGripper: finger body {body!r} has no collision geom")
        if not spec.finger_bodies:
            raise RuntimeError("MuJoCoPiperGripper: the manifest names no finger_bodies")
        scene = self._go2.runtime.world.scene
        self._grasp = GraspSpec(
            weld_body=scene.body(spec.weld_body),
            finger_bodies=tuple(scene.body(b) for b in spec.finger_bodies),
            jaw_joint=scene.joint(spec.joint),
            closed=float(spec.closed),
            min_opening=float(spec.min_opening),
        )
        self._connected = True
        logger.info(
            "MuJoCoPiperGripper connected (actuator=%s, closed=%.3f, open=%.3f, %d graspable objects)",
            spec.actuator, spec.closed, spec.open,
            sum(p == self._grasp.weld_body for p, _c in scene.grasp_pairs),
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
        """Open the jaws and release whatever the gripper holds, verified.

        Returns False (and keeps the held state) if something is still attached
        after the release or the write failed.
        """
        self._require_connection()
        aid, rt = self._actuator_id, self._go2.runtime
        try:
            rt.call(lambda w: w.data.ctrl.__setitem__(aid, self._spec.open))
            still = rt.release(self._grasp.weld_body)
        except Exception as exc:  # noqa: BLE001 — reported, never a silent success
            logger.error("MuJoCoPiperGripper: open/release failed: %s", exc)
            return False
        if still:
            logger.error("MuJoCoPiperGripper: bodies %s still attached after release", still)
            return False
        self._held_object = None
        return True

    def close(self) -> bool:
        """Close the jaws; the runtime attaches the object they actually grip (module docstring).

        Returns True once the jaws were commanded and settled (whether or not they
        gripped anything — is_holding() says that); False if the sim refused.
        """
        self._require_connection()
        aid = self._actuator_id
        rt = self._go2.runtime
        try:
            rt.call(lambda w: w.data.ctrl.__setitem__(aid, self._spec.closed))
            self._wait_jaws_settled()
            held = rt.grasp(self._grasp)
        except Exception as exc:  # noqa: BLE001 — a failed grasp is a no-hold, never a crash
            logger.warning("MuJoCoPiperGripper: close failed: %s", exc)
            return False
        if held is not None:
            self._held_object = rt.world.scene.body_name(held)
        return True

    def is_holding(self, name: str | None = None) -> bool:
        """True iff the gripped object is still attached to the gripper in the sim.

        ``name`` makes it target-aware: only that object (case-insensitive scene
        body name) counts.
        """
        held = self._held_object
        if held is None or (name is not None and str(name).strip().lower() != held.lower()):
            return False
        return bool(self.weld_is_active().get(held, False))

    def weld_is_active(self) -> dict[str, bool]:
        """``{graspable body name -> attached to this gripper}`` as the runtime last
        published it — the SAME body-keyed map MuJoCoGripper exposes, read by
        actor_causation (_capture_gripper) for the 0->1 fresh-grasp transition.
        Fail-safe ``{}``.
        """
        if not self._connected:
            return {}
        rt, parent = self._go2.runtime, self._grasp.weld_body
        scene, held = rt.world.scene, rt.attachments()
        return {scene.body_name(c): held.get(c) == parent for p, c in scene.grasp_pairs if p == parent}

    def get_position(self) -> float:
        """Normalized jaw position: 0.0 closed, 1.0 fully open."""
        self._require_connection()
        pos = float(self._go2.runtime.snapshot().q[self._joint_qpos_adr])
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
        last = float(rt.snapshot().q[self._joint_qpos_adr])
        rt.wait_sim(5 * _SETTLE_POLL_S)  # let the servo start moving
        while rt.sim_time < deadline:
            now = float(rt.snapshot().q[self._joint_qpos_adr])
            if abs(now - last) < _SETTLE_EPS:
                return
            last = now
            if not rt.wait_sim(_SETTLE_POLL_S):
                return
