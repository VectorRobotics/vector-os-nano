# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""MuJoCo-based AgileX Piper 6-DoF arm.

Implements ArmProtocol. The Piper is a nested body tree inside the combined
Go2+Piper MJCF, so we share MjModel/MjData with the parent MuJoCoGo2 rather
than owning our own. Every name, the home pose and the IK seeds come from the
manifest arm block (``embodiments/go2_piper/robot.yaml``).

Thread model
------------
The parent MuJoCoGo2's SimRuntime owns MjData and stepping. This class:
- reads joint state from runtime.snapshot() (one instant, a copy)
- writes ctrl through runtime.submit (applied at a step boundary)
- performs fk/ik on an isolated MjModel + MjData, behind a lock, with the base
  pose copied from the live snapshot on every call (the arm rides the dog)
- times move_joints in SIM seconds (runtime.wait_sim)

We never step physics from here.

Usage:
    go2 = MuJoCoGo2(..., room=True, with_arm=True)
    go2.connect()
    piper = MuJoCoPiper(go2)
    piper.connect()
    q = piper.ik_top_down((10.4, 3.0, 0.30))
    if q is not None and piper.move_joints(q, duration=3.0):
        ...  # the arm settled at q
"""
from __future__ import annotations

import logging
import math
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from vector_os_nano.embodiments.config import ArmSpec
    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy MuJoCo import (matches mujoco_go2.py / mujoco_arm.py pattern)
# ---------------------------------------------------------------------------

_mujoco: Any = None


def _get_mujoco() -> Any:
    global _mujoco
    if _mujoco is None:
        import mujoco  # noqa: PLC0415

        _mujoco = mujoco
    return _mujoco


# ---------------------------------------------------------------------------
# IK tuning
# ---------------------------------------------------------------------------

_IK_MAX_ITER: int = 200
_IK_POS_TOL: float = 2e-3          # 2 mm
_IK_ROT_TOL: float = 2e-2          # rad (~1.1 deg) of the TRUE rotation angle
_IK_STEP_SIZE: float = 0.4
_IK_DAMPING: float = 5e-3

# Top-down grasp frame for jaw yaw 0: tool z = world -Z (fingers down), tool x =
# world +X. ik_top_down rotates it about world Z by each jaw-yaw candidate.
_R_TOP_DOWN: np.ndarray = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)

# move_joints writes the servo set-points at this SIM rate; after the
# interpolation the joints get up to _SETTLE_S sim seconds to come within the
# manifest's track_tol of the target.
_MOVE_UPDATE_HZ: float = 50.0
_SETTLE_S: float = 1.0
_SETTLE_POLL_S: float = 0.02


def _rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _jaw_yaws(heading: float) -> list[float]:
    """The body heading, +pi (the parallel jaw is symmetric), then world X and +pi.

    Heading first: the jaws then open across the dog's forward direction, the way
    it approaches an object it faces.
    """
    out: list[float] = []
    for y in (heading, heading + math.pi, 0.0, math.pi):
        if all(abs(math.remainder(y - o, 2 * math.pi)) > 1e-3 for o in out):
            out.append(y)
    return out


def _yaw_of(quat_wxyz: Any) -> float:
    w, x, y, z = (float(v) for v in quat_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rot_angle(dR: np.ndarray) -> float:
    """Rotation angle of *dR* (log map) — a 180 deg flip reads pi, not ~0."""
    return math.acos(max(-1.0, min(1.0, (float(np.trace(dR)) - 1.0) / 2.0)))


# ---------------------------------------------------------------------------
# MuJoCoPiper
# ---------------------------------------------------------------------------


class MuJoCoPiper:
    """AgileX Piper arm in MuJoCo, sharing state with the parent Go2.

    Args:
        go2: A connected MuJoCoGo2 instance whose model contains the arm.
        spec: The manifest arm block; default: the go2_piper manifest's.
    """

    def __init__(self, go2: "MuJoCoGo2", spec: "ArmSpec | None" = None) -> None:
        if spec is None:
            from vector_os_nano.embodiments import registry

            spec = registry.load("go2_piper").arm
        self._go2 = go2
        self._spec = spec
        self._connected: bool = False

        # Cached IDs into the live go2 model (set on connect)
        self._arm_joint_qpos_adr: list[int] = []
        self._arm_joint_dof_adr: list[int] = []
        self._arm_joint_ids: list[int] = []
        self._arm_actuator_ids: list[int] = []

        # IK runs on an isolated MjModel + MjData loaded from the same scene file
        # (never the live data the physics thread steps), behind _ik_lock: fk() is
        # called by the verify oracle from another thread than the skill's IK.
        self._ik_model: Any = None
        self._ik_data: Any = None
        self._ik_lock = threading.Lock()
        self._ik_arm_qpos_adr: list[int] = []
        self._ik_arm_dof_adr: list[int] = []
        self._ik_arm_joint_ids: list[int] = []
        self._ik_ee_site_id: int = -1
        self._ik_range: np.ndarray | None = None  # (dof, 2) joint limits

        self._gt: Any = None  # GroundTruth of the parent's runtime (set on connect)

        # stop() sets it: an in-flight move_joints returns False within one
        # set-point period and writes nothing after the hold.
        self._stop_evt = threading.Event()
        self._abort_requested: Any = lambda: False

        # R2b actor-causation instrumentation. ``move_joints`` bumps these as it
        # drives the ctrl set-points toward the target. ``_ctrl_motion`` accumulates
        # the commanded |ctrl delta| actually written, so a no-op
        # move_joints(current_pose) (zero delta) never satisfies the grader.
        self._ctrl_writes: int = 0
        self._ctrl_motion: float = 0.0

    # ------------------------------------------------------------------
    # ArmProtocol properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "mujoco_piper"

    @property
    def joint_names(self) -> list[str]:
        return list(self._spec.joints)

    @property
    def dof(self) -> int:
        return self._spec.dof

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Resolve joint / actuator / site IDs against the shared model.

        Raises:
            RuntimeError: If the parent MuJoCoGo2 is not connected, or the loaded
                MJCF lacks a joint / actuator / site the manifest names.
        """
        if not getattr(self._go2, "_connected", False):
            raise RuntimeError("MuJoCoPiper: parent MuJoCoGo2 must be connected first")

        mj = _get_mujoco()
        model = self._go2._mj.model
        spec = self._spec

        joint_ids: list[int] = []
        for name in spec.joints:
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(
                    f"MuJoCoPiper: joint {name!r} not in loaded MJCF. "
                    "Ensure sim was launched with with_arm=True."
                )
            joint_ids.append(jid)
        self._arm_joint_ids = joint_ids
        self._arm_joint_qpos_adr = [int(model.jnt_qposadr[j]) for j in joint_ids]
        self._arm_joint_dof_adr = [int(model.jnt_dofadr[j]) for j in joint_ids]

        actuator_ids: list[int] = []
        for name in spec.actuators:
            aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise RuntimeError(f"MuJoCoPiper: actuator {name!r} not in loaded MJCF.")
            actuator_ids.append(aid)
        self._arm_actuator_ids = actuator_ids

        if mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, spec.ee_site) < 0:
            raise RuntimeError(
                f"MuJoCoPiper: site {spec.ee_site!r} not in loaded MJCF. "
                "Regenerate go2_piper.xml via build_go2_piper.py."
            )

        self._allocate_ik_model(mj)
        from vector_os_nano.sim.ground_truth import GroundTruth
        from vector_os_nano.vcli.cognitive.abort import is_abort_requested

        self._gt = GroundTruth(self._go2.runtime, robot_root_bid=self._go2._mj.layout.root_bid)
        self._abort_requested = is_abort_requested
        self._connected = True
        logger.info(
            "MuJoCoPiper connected: joints=%s, ee_site=%s (isolated IK model loaded)",
            list(spec.joints), spec.ee_site,
        )

    def _allocate_ik_model(self, mj: Any) -> None:
        """Load a private MjModel+MjData from the scene file the live model came from."""
        scene_path = getattr(self._go2, "_scene_xml_path", None)
        if scene_path is None:
            from vector_os_nano.hardware.sim.mujoco_go2 import _build_room_scene_xml

            scene_path = str(_build_room_scene_xml(with_arm=True))

        self._ik_model = mj.MjModel.from_xml_path(str(scene_path))
        self._ik_data = mj.MjData(self._ik_model)

        m = self._ik_model
        self._ik_arm_joint_ids = [
            mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n) for n in self._spec.joints
        ]
        self._ik_arm_qpos_adr = [int(m.jnt_qposadr[j]) for j in self._ik_arm_joint_ids]
        self._ik_arm_dof_adr = [int(m.jnt_dofadr[j]) for j in self._ik_arm_joint_ids]
        self._ik_ee_site_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, self._spec.ee_site)
        self._ik_range = np.array([m.jnt_range[j] for j in self._ik_arm_joint_ids], dtype=float)

    def disconnect(self) -> None:
        """Mark disconnected. Release the isolated IK model. Does not affect
        the parent Go2. Idempotent.
        """
        self._stop_evt.set()
        self._connected = False
        self._ik_model = None
        self._ik_data = None

    def _require_connection(self) -> None:
        if not self._connected:
            raise RuntimeError("MuJoCoPiper: not connected. Call connect() first.")

    # ------------------------------------------------------------------
    # Joint state
    # ------------------------------------------------------------------

    def get_joint_positions(self) -> list[float]:
        """Read the arm joint positions in radians."""
        self._require_connection()
        q = self._go2.runtime.snapshot().qpos
        return [float(q[adr]) for adr in self._arm_joint_qpos_adr]

    def get_object_positions(self) -> dict[str, list[float]]:
        """Free-body object world positions {body_name: [x, y, z]} from the live go2
        model — the robot's own bodies are never among them. Mirrors
        MuJoCoArm.get_object_positions so the holding_object oracle sees the SAME
        duck-typed surface on the go2 path.
        """
        self._require_connection()
        return self._gt.free_body_positions()

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def _preempted(self) -> bool:
        return self._stop_evt.is_set() or bool(self._abort_requested())

    def move_joints(self, positions: list[float], duration: float = 3.0) -> bool:
        """Interpolate the servo set-points to *positions* over *duration* SIM seconds.

        Returns True iff, after the interpolation, every joint settles within the
        manifest's ``track_tol`` of its target (within ``_SETTLE_S`` sim seconds).
        False when the arm is blocked (a collision, an unreachable limit) or when
        stop() / an abort preempts the move — the arm then holds where it is.
        """
        self._require_connection()
        if len(positions) != self.dof:
            raise ValueError(
                f"MuJoCoPiper.move_joints: expected {self.dof} positions, "
                f"got {len(positions)}"
            )
        goal = [float(g) for g in positions]
        rt = self._go2.runtime
        ids = list(self._arm_actuator_ids)
        stop = self._stop_evt
        stop.clear()
        start = rt.call(lambda _m, d: [float(d.ctrl[aid]) for aid in ids])
        # R2b: the commanded ctrl displacement (sum of |target - start|).
        commanded = sum(abs(g - s) for s, g in zip(start, goal))

        def _write(_m: Any, d: Any, v: list[float]) -> None:
            if not stop.is_set():  # checked at the step boundary: never after stop()'s hold
                d.ctrl[ids] = v

        steps = max(1, int(duration * _MOVE_UPDATE_HZ))
        dt = duration / steps
        done = 0.0
        for i in range(1, steps + 1):
            if self._preempted():
                break
            t = i / steps
            targets = [s + t * (g - s) for s, g in zip(start, goal)]
            rt.submit(lambda m, d, v=targets: _write(m, d, v))
            done = t
            rt.wait_sim(dt, cancel=stop)

        self._ctrl_writes += 1
        self._ctrl_motion += commanded * done
        if self._preempted():
            logger.info("MuJoCoPiper.move_joints: preempted at %.0f%% of the move", 100 * done)
            return False
        return self._settle(goal)

    def _settle(self, goal: list[float]) -> bool:
        """Wait (sim time) until max|q - goal| < track_tol; False if it never gets there."""
        rt = self._go2.runtime
        tol = float(self._spec.track_tol)
        deadline = rt.sim_time + _SETTLE_S
        while True:
            q = rt.snapshot().qpos
            err = max(abs(float(q[a]) - g) for a, g in zip(self._arm_joint_qpos_adr, goal))
            if err < tol:
                return True
            if (rt.sim_time >= deadline or self._preempted()
                    or not rt.wait_sim(_SETTLE_POLL_S, cancel=self._stop_evt)):
                logger.warning(
                    "MuJoCoPiper.move_joints: did not reach the target: max|q-target|=%.3f rad "
                    "(tol %.3f)", err, tol)
                return False

    def ctrl_motion(self) -> float:
        """Cumulative commanded ctrl-displacement magnitude (R2b actor-causation).

        The running sum of ``sum(|target - start|)`` written by ``move_joints`` (a
        preempted move counts only the part it wrote). The grader compares a
        baseline snapshot (before a step) with a post snapshot (after) and treats a
        delta >= ``MOTION_EPS`` as "the actor commanded arm motion". A move_joints
        to the current pose adds zero. Monotonically non-decreasing.
        """
        return float(self._ctrl_motion)

    def move_cartesian(
        self,
        target_xyz: tuple[float, float, float],
        duration: float = 3.0,
    ) -> bool:
        """Top-down IK + move_joints. Returns False if IK unreachable."""
        self._require_connection()
        q = self.ik_top_down(target_xyz)
        if q is None:
            return False
        return self.move_joints(q, duration=duration)

    def stop(self) -> None:
        """Preempt any move_joints and hold the current joint positions."""
        self._require_connection()
        self._stop_evt.set()  # BEFORE the hold, so no queued set-point lands after it
        qadr, ids = list(self._arm_joint_qpos_adr), list(self._arm_actuator_ids)
        self._go2.runtime.call(lambda _m, d: d.ctrl.__setitem__(ids, d.qpos[qadr]))

    def home(self, duration: float = 3.0) -> bool:
        """Move to the manifest home pose."""
        return self.move_joints(list(self._spec.home), duration=duration)

    # ------------------------------------------------------------------
    # Kinematics — all on the isolated IK data, base synced from live
    # ------------------------------------------------------------------

    def fk(
        self,
        joint_positions: list[float],
    ) -> tuple[list[float], list[list[float]]]:
        """Forward kinematics with the LIVE base pose and the supplied arm joints.

        Returns:
            (position_xyz_world, rotation_3x3_row_major) of the EE site.
        """
        self._require_connection()
        if len(joint_positions) != self.dof:
            raise ValueError(
                f"MuJoCoPiper.fk: expected {self.dof} joints, got {len(joint_positions)}"
            )
        mj = _get_mujoco()
        with self._ik_lock:
            self._sync_ik_base_from_live()
            self._ik_data.qpos[self._ik_arm_qpos_adr] = [float(q) for q in joint_positions]
            mj.mj_kinematics(self._ik_model, self._ik_data)
            sid = self._ik_ee_site_id
            pos = [float(v) for v in self._ik_data.site_xpos[sid]]
            rot = [[float(v) for v in row] for row in self._ik_data.site_xmat[sid].reshape(3, 3)]
        return pos, rot

    def ik(
        self,
        target_xyz: tuple[float, float, float],
        current_joints: list[float] | None = None,
    ) -> list[float] | None:
        """Position-only IK (3-DoF), Jacobian DLS, against the live base.

        Seeded from *current_joints*, else the live arm joints. Returns joint
        positions if converged within _IK_POS_TOL, else None.
        """
        self._require_connection()
        with self._ik_lock:
            live = self._sync_ik_base_from_live()
            seed = current_joints or [float(live[a]) for a in self._arm_joint_qpos_adr]
            return self._ik_iterate(target_xyz, None, seed)

    def ik_top_down(
        self,
        target_xyz: tuple[float, float, float],
        current_joints: list[float] | None = None,
    ) -> list[float] | None:
        """6-DoF IK with the tool z-axis = world -Z (top-down), against the LIVE base.

        Tries the jaw yaws {body heading, +pi, world X, +pi} and, for each, the
        seeds *current_joints*, the live arm joints, then the manifest's
        ``ik_seeds`` (clipped to the joint limits). Returns the first solution that
        converges in position (_IK_POS_TOL) and rotation angle (_IK_ROT_TOL).
        """
        self._require_connection()
        with self._ik_lock:
            live = self._sync_ik_base_from_live()
            lo = self._go2._mj.layout.quat_start
            heading = _yaw_of(live[lo:lo + 4])
            seeds: list[list[float]] = []
            if current_joints is not None:
                seeds.append([float(q) for q in current_joints])
            seeds.append([float(live[a]) for a in self._arm_joint_qpos_adr])
            seeds.extend([float(q) for q in s] for s in self._spec.ik_seeds)
            for yaw in _jaw_yaws(heading):
                target_rot = _rot_z(yaw) @ _R_TOP_DOWN
                for seed in seeds:
                    sol = self._ik_iterate(target_xyz, target_rot, seed)
                    if sol is not None:
                        return sol
        return None

    # ------------------------------------------------------------------
    # Internal: IK implementation (callers hold _ik_lock)
    # ------------------------------------------------------------------

    def _sync_ik_base_from_live(self) -> np.ndarray:
        """Copy the live floating base + legs into the IK data; return the live qpos.

        The slice comes from DofLayout (root_qpos_adr), never a literal: under the
        MjSpec.attach scene the go2 freejoint sits behind the pickables' freejoints.
        The source is the runtime snapshot — a copy, no physics pause.
        """
        go2 = self._go2
        live_qpos = go2.runtime.snapshot().qpos
        lo = go2._mj.layout.root_qpos_adr
        n = 7 + go2._mj.layout.num_actuated  # freejoint(7) + legs
        self._ik_data.qpos[lo:lo + n] = live_qpos[lo:lo + n]
        self._ik_data.qvel[:] = 0.0
        return live_qpos

    def _ik_iterate(
        self,
        target_xyz: tuple[float, float, float],
        target_rot: np.ndarray | None,
        seed_joints: list[float],
    ) -> list[float] | None:
        """Jacobian-DLS IK on the isolated data from *seed_joints* (clipped to limits)."""
        mj = _get_mujoco()
        m = self._ik_model
        d = self._ik_data
        sid = self._ik_ee_site_id
        qadr = self._ik_arm_qpos_adr
        lo, hi = self._ik_range[:, 0], self._ik_range[:, 1]
        q = np.clip(np.asarray(seed_joints, dtype=float), lo, hi)

        target_pos = np.array(target_xyz, dtype=np.float64)
        use_rot = target_rot is not None
        jacp = np.zeros((3, m.nv), dtype=np.float64)
        jacr = np.zeros((3, m.nv), dtype=np.float64) if use_rot else None
        cols = self._ik_arm_dof_adr

        for _ in range(_IK_MAX_ITER):
            d.qpos[qadr] = q
            # Kinematics + com positions are all mj_jacSite needs (no collision pass).
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            pos_err = target_pos - d.site_xpos[sid]
            if use_rot:
                dR = target_rot @ d.site_xmat[sid].reshape(3, 3).T
                if np.linalg.norm(pos_err) < _IK_POS_TOL and _rot_angle(dR) < _IK_ROT_TOL:
                    return [float(v) for v in q]
                rot_err = 0.5 * np.array(
                    [dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]
                )
                err = np.concatenate([pos_err, rot_err])
            else:
                if np.linalg.norm(pos_err) < _IK_POS_TOL:
                    return [float(v) for v in q]
                err = pos_err

            mj.mj_jacSite(m, d, jacp, jacr, sid)
            J = np.vstack([jacp[:, cols], jacr[:, cols]]) if use_rot else jacp[:, cols]
            try:
                dq = J.T @ np.linalg.solve(J @ J.T + _IK_DAMPING * np.eye(len(err)), err)
            except np.linalg.LinAlgError:
                return None
            q = np.clip(q + _IK_STEP_SIZE * dq, lo, hi)
        return None
