# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Test-side helpers: everything here reads canonical kinematics, never a backend."""
from __future__ import annotations

import math

import numpy as np

from vector_os_nano.sim.scene import canonical

SUPPORT_EPS = 0.01  # today's GroundTruth.clearance support epsilon


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qexp(axis_angle: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(axis_angle))
    if th == 0:
        return np.array([1.0, 0, 0, 0])
    return np.concatenate([[math.cos(th / 2)], math.sin(th / 2) * np.asarray(axis_angle) / th])


def qmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation angle (rad) between quaternions, row-wise; exact near zero."""
    a = np.atleast_2d(a) / np.linalg.norm(np.atleast_2d(a), axis=1, keepdims=True)
    b = np.atleast_2d(b) / np.linalg.norm(np.atleast_2d(b), axis=1, keepdims=True)
    w = np.abs(np.sum(a * b, axis=1))
    v = np.linalg.norm(b[:, :1] * a[:, 1:] - a[:, :1] * b[:, 1:] - np.cross(a[:, 1:], b[:, 1:]), axis=1)
    return 2 * np.arctan2(v, w)


def yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def clearance(scene, q: np.ndarray, body: int) -> float:
    """Today's GroundTruth.clearance over canonical shape AABBs: the object's lowest
    point minus the highest colliding non-robot shape top under its origin."""
    lo, hi, owner, colliding = canonical.shape_aabbs(scene, q)
    pos, _ = canonical.body_poses(scene, q)
    robot = list(scene.subtree(scene.robot_root)) if scene.robot_root >= 0 else []
    own = owner == body
    bottom = float(lo[own, 2].min())
    x, y = pos[body][:2]
    under = (colliding & ~own & ~np.isin(owner, robot)
             & (lo[:, 0] <= x) & (x <= hi[:, 0]) & (lo[:, 1] <= y) & (y <= hi[:, 1])
             & (hi[:, 2] <= bottom + SUPPORT_EPS))
    return bottom - float(hi[under, 2].max()) if under.any() else math.inf
