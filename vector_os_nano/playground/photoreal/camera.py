# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""MuJoCo -> Blender camera transform (the co-sim no-drift core).

MuJoCo and Blender share conventions: a Z-up right-handed WORLD, and a camera
frame that looks down its local **-Z** with **+Y up / +X right**. So mapping a
MuJoCo physics camera onto the Blender render camera is the IDENTITY of the
rotation columns plus the world position — no axis swap, no handedness flip.
This module is pure (no ``bpy``, no MuJoCo import) so the transform is unit
tested deterministically and a future refactor cannot silently introduce drift.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlenderCameraPose:
    """A Blender render-camera pose derived from a MuJoCo physics camera.

    - ``matrix_world``: 4x4 homogeneous transform (camera->world). Its upper-left
      3x3 columns are the camera local x(right)/y(up)/z(-forward) axes in world
      coordinates; column 3 is the world position.
    - ``fovy_rad``: vertical field of view in radians (Blender ``sensor_fit =
      'VERTICAL'`` / ``angle_y``).
    """

    matrix_world: np.ndarray
    fovy_rad: float

    def flat16(self) -> list[float]:
        """Row-major 16-float form for JSON transport to the Blender server."""
        return [float(v) for v in np.asarray(self.matrix_world, np.float64).reshape(-1)]


def look_at_camera_matrix(pos, target, up=(0.0, 0.0, 1.0)):
    """A MuJoCo-convention camera (``cam_pos``, ``cam_mat`` flat-9) looking at a
    world ``target`` with world ``up``.

    The PRODUCTION path takes ``cam_mat`` straight from MuJoCo ``cam_xmat`` — this
    helper is for building demo/test camera poses (and placing scene cameras). The
    returned ``cam_mat`` columns are right / up / -forward (camera looks down -Z),
    matching what MuJoCo reports, so it feeds straight into
    ``mujoco_camera_to_blender``.
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    fwd = np.asarray(target, dtype=np.float64).reshape(3) - pos
    fwd /= np.linalg.norm(fwd)
    up = np.asarray(up, dtype=np.float64).reshape(3)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    up_c = np.cross(right, fwd)
    cam_mat = np.column_stack([right, up_c, -fwd])   # cols: right, up, -forward
    return pos, cam_mat.reshape(-1)


def mujoco_camera_to_blender(cam_pos, cam_mat, fovy_deg: float) -> BlenderCameraPose:
    """Convert a MuJoCo camera (``cam_xpos``/``cam_xmat``/``cam_fovy``) to Blender.

    Args:
        cam_pos: world position, array-like (3,).
        cam_mat: MuJoCo ``cam_xmat`` — flat (9,) row-major OR (3,3). Its columns
            are the camera frame axes in world coords.
        fovy_deg: vertical field of view in DEGREES (MuJoCo ``cam_fovy``).
    """
    pos = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    R = np.asarray(cam_mat, dtype=np.float64).reshape(3, 3)
    mw = np.eye(4, dtype=np.float64)
    mw[:3, :3] = R          # identity mapping — same convention both engines
    mw[:3, 3] = pos
    return BlenderCameraPose(matrix_world=mw, fovy_rad=math.radians(float(fovy_deg)))
