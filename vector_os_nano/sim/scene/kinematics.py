# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Arm kinematics on the canonical compiled model (charter section 2): forward
kinematics and damped-least-squares IK of one arm, whatever backend integrates it.

An :class:`ArmKinematics` owns one scratch ``MjData`` of ``scene.model`` behind a lock
(fk serves oracle threads while a skill solves). Every call is handed the live state
``q`` and takes from it exactly what the arm rides on: every robot joint outside the
arm's own subtree (a floating base, its legs). Everything else stays at the model's
``qpos0``, so an answer depends on the mount and the arm joints only.
"""
from __future__ import annotations

import math
import threading
from typing import Iterable, Sequence

import mujoco as mj
import numpy as np

from vector_os_nano.interfaces.physics import CompiledScene
from vector_os_nano.sim.scene import canonical

_IK_MAX_ITER = 200
_IK_POS_TOL = 2e-3           # m
_IK_ROT_TOL = 2e-2           # rad of the TRUE rotation angle (~1.1 deg)
_IK_STEP_SIZE = 0.4
_IK_DAMPING = 5e-3

# Top-down tool frame for jaw yaw 0: tool z = world -Z (fingers down), tool x = world +X.
_R_TOP_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def _rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _yaw(quat_wxyz: Iterable[float]) -> float:
    w, x, y, z = (float(v) for v in quat_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rot_angle(dR: np.ndarray) -> float:
    """Rotation angle of *dR* (log map): a 180 deg flip reads pi, not ~0."""
    return math.acos(max(-1.0, min(1.0, (float(np.trace(dR)) - 1.0) / 2.0)))


def jaw_yaws(heading: float) -> list[float]:
    """The body heading, +pi (a parallel jaw is symmetric), then world X and +pi:
    heading first, so the jaws open across the direction the body faces."""
    out: list[float] = []
    for y in (heading, heading + math.pi, 0.0, math.pi):
        if all(abs(math.remainder(y - o, 2 * math.pi)) > 1e-3 for o in out):
            out.append(y)
    return out


class ArmKinematics:
    """FK / IK of the arm *joints* (ordered) and its end-effector site *ee_site*."""

    def __init__(self, scene: CompiledScene, joints: Sequence[str], ee_site: str) -> None:
        m = self._m = scene.model
        ids = [scene.joint(n) for n in joints]
        self._site = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, ee_site)
        if self._site < 0:
            raise KeyError(f"no site {ee_site!r} in {scene.xml_path}")
        self._qadr = [scene.joint_q(j) for j in ids]
        self._cols = [scene.joint_qd(j) for j in ids]
        self.range = np.array([m.jnt_range[j] for j in ids], dtype=float)
        root = scene.robot_root
        robot = scene.subtree(root) if root >= 0 else frozenset()
        arm = scene.subtree(int(m.jnt_bodyid[ids[0]]))
        width = {int(mj.mjtJoint.mjJNT_FREE): 7, int(mj.mjtJoint.mjJNT_BALL): 4}
        self._mount = [
            a for j in range(int(m.njnt)) if int(m.jnt_bodyid[j]) in robot - arm
            for a in range(int(m.jnt_qposadr[j]), int(m.jnt_qposadr[j]) + width.get(int(m.jnt_type[j]), 1))
        ]
        free = root >= 0 and root in scene.free_bodies().values()
        self._root_quat = slice(scene.free_q(root).start + 3, scene.free_q(root).stop) if free else None
        self._heading = 0.0 if root < 0 or free else _yaw(canonical.body_poses(scene, m.qpos0)[1][root])
        self._d = mj.MjData(m)
        self._lock = threading.Lock()

    def arm_q(self, q: np.ndarray) -> list[float]:
        """The arm joints of the state *q*."""
        return [float(q[a]) for a in self._qadr]

    def _sync(self, q: np.ndarray) -> float:
        """Take the mount from the live state *q*; return the robot's heading (yaw)."""
        self._d.qpos[self._mount] = q[self._mount]
        return self._heading if self._root_quat is None else _yaw(q[self._root_quat])

    def fk(self, q: np.ndarray, joints: Sequence[float]) -> tuple[list[float], list[list[float]]]:
        """End-effector site (position, 3x3 rotation) with the mount of *q* and *joints*."""
        d = self._d
        with self._lock:
            self._sync(q)
            d.qpos[self._qadr] = [float(v) for v in joints]
            mj.mj_kinematics(self._m, d)
            pos = [float(v) for v in d.site_xpos[self._site]]
            rot = [[float(v) for v in row] for row in d.site_xmat[self._site].reshape(3, 3)]
        return pos, rot

    def ik(self, q: np.ndarray, target: Sequence[float], seed: Sequence[float]) -> list[float] | None:
        """Position-only IK from *seed*."""
        with self._lock:
            self._sync(q)
            return self._iterate(target, None, seed)

    def ik_top_down(self, q: np.ndarray, target: Sequence[float],
                    seeds: Sequence[Sequence[float]]) -> list[float] | None:
        """Tool z = world -Z: for each jaw yaw (:func:`jaw_yaws` of the robot's heading)
        and each seed in order, the first solution within both tolerances."""
        with self._lock:
            heading = self._sync(q)
            for yaw in jaw_yaws(heading):
                rot = _rot_z(yaw) @ _R_TOP_DOWN
                for seed in seeds:
                    sol = self._iterate(target, rot, seed)
                    if sol is not None:
                        return sol
        return None

    def _iterate(self, target: Sequence[float], target_rot: np.ndarray | None,
                 seed: Sequence[float]) -> list[float] | None:
        """Jacobian DLS from *seed* (clipped to the limits); callers hold the lock."""
        m, d, sid, qadr = self._m, self._d, self._site, self._qadr
        lo, hi = self.range[:, 0], self.range[:, 1]
        q = np.clip(np.asarray(seed, dtype=float), lo, hi)
        target_pos = np.array(target, dtype=np.float64)
        use_rot = target_rot is not None
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv)) if use_rot else None
        for _ in range(_IK_MAX_ITER):
            d.qpos[qadr] = q
            mj.mj_kinematics(m, d)  # + com positions: all mj_jacSite needs
            mj.mj_comPos(m, d)
            pos_err = target_pos - d.site_xpos[sid]
            if use_rot:
                dR = target_rot @ d.site_xmat[sid].reshape(3, 3).T
                if np.linalg.norm(pos_err) < _IK_POS_TOL and _rot_angle(dR) < _IK_ROT_TOL:
                    return [float(v) for v in q]
                rot_err = 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
                err = np.concatenate([pos_err, rot_err])
            else:
                if np.linalg.norm(pos_err) < _IK_POS_TOL:
                    return [float(v) for v in q]
                err = pos_err
            mj.mj_jacSite(m, d, jacp, jacr, sid)
            J = np.vstack([jacp[:, self._cols], jacr[:, self._cols]]) if use_rot else jacp[:, self._cols]
            try:
                dq = J.T @ np.linalg.solve(J @ J.T + _IK_DAMPING * np.eye(len(err)), err)
            except np.linalg.LinAlgError:
                return None
            q = np.clip(q + _IK_STEP_SIZE * dq, lo, hi)
        return None
