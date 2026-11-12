# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Canonical kinematics: poses and shape AABBs computed from a state ``q``.

Never asked of a backend: every function runs MuJoCo's forward kinematics on a
scratch ``MjData`` (one per thread and model), so ground truth reads the same
numbers whichever engine integrated ``q``.
"""
from __future__ import annotations

import threading

import mujoco as mj
import numpy as np

from vector_os_nano.interfaces.physics import CompiledScene

_local = threading.local()


def _kinematics(scene: CompiledScene, q: np.ndarray) -> mj.MjData:
    model = scene.model
    cached = getattr(_local, "cache", None)
    if cached is None or cached[0] is not model:  # holds the model, so `is` cannot alias
        cached = _local.cache = (model, mj.MjData(model))
    d = cached[1]
    d.qpos[:] = q
    mj.mj_kinematics(model, d)
    return d


def body_poses(scene: CompiledScene, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(pos (nbody, 3), quat (nbody, 4))`` of every body FRAME in the world."""
    d = _kinematics(scene, q)
    return d.xpos.copy(), d.xquat.copy()


def geom_poses(scene: CompiledScene, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(pos (ngeom, 3), mat (ngeom, 9) row-major)`` of every geom in the world."""
    d = _kinematics(scene, q)
    return d.geom_xpos.copy(), d.geom_xmat.copy()


def site_poses(scene: CompiledScene, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(pos (nsite, 3), mat (nsite, 3, 3))`` of every site in the world."""
    d = _kinematics(scene, q)
    return d.site_xpos.copy(), d.site_xmat.reshape(-1, 3, 3).copy()


def shape_aabbs(scene: CompiledScene, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per geom: world AABB ``lo (G, 3)``, ``hi (G, 3)`` (its local box rotated by the
    geom frame), owning ``body (G,)`` and ``colliding (G,)`` (contype or conaffinity)."""
    m = scene.model
    d = _kinematics(scene, q)
    R = d.geom_xmat.reshape(-1, 3, 3)
    c = d.geom_xpos + np.einsum("gij,gj->gi", R, m.geom_aabb[:, :3])
    h = np.einsum("gij,gj->gi", np.abs(R), m.geom_aabb[:, 3:])
    colliding = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
    return c - h, c + h, m.geom_bodyid.copy(), colliding
