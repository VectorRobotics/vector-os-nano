# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Perception-driven 3D grasp point — (RGB-D + mask) -> world coordinate.

The honest core of the Go2+Piper grasp pipeline. The grasp POINT is computed
ONLY from a real rendered depth frame + an EdgeTAM mask + the proven cam->world
transform — it is NEVER a ground-truth object pose lookup. That is the whole
point of the goal: localize the target the way a real robot must (perceive it),
not by reading the simulator's omniscient state.

Pure module: no MuJoCo / torch / GPU import. Composes already-tested pieces —
``rgbd_to_pointcloud_fast`` (mask -> camera-frame points), ``remove_depth_outliers``
+ ``robust_centroid`` (the trimmed-mean reducer), and ``camera_to_world`` (the
OpenCV->MuJoCo axis flip, ground-truth-tested in test_depth_projection_groundtruth).

UNIT CONTRACT: MuJoCo ``get_depth_frame`` returns float32 METRES, so the default
``depth_scale=1.0``. The RealSense-mm default (1000.0) would divide metres by
1000 and collapse every point to ~origin — a silent, plausible-looking wrong
grasp. Always pass metres + depth_scale=1.0 (or raw mm + depth_scale=1000.0).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vector_os_nano.core.types import Pose3D
from vector_os_nano.perception._centroid import (
    remove_depth_outliers,
    robust_centroid,
    select_nearest_cluster,
)
from vector_os_nano.perception.depth_projection import camera_to_world
from vector_os_nano.perception.pointcloud import rgbd_to_pointcloud_fast

logger = logging.getLogger(__name__)

_MIN_POINTS = 4


def grasp_point_from_rgbd(
    depth: np.ndarray,
    color: np.ndarray,
    mask: np.ndarray,
    intrinsics: Any,
    cam_xpos: Any,
    cam_xmat: Any,
    *,
    depth_scale: float = 1.0,
    depth_trunc: float = 10.0,
    foreground_only: bool = True,
) -> Pose3D | None:
    """Compute the world-frame grasp point for a masked object, or None.

    Pipeline: mask + depth + intrinsics -> camera-frame point cloud (only the
    masked pixels) -> IQR depth-outlier rejection -> trimmed-mean centroid (camera
    frame) -> exact MuJoCo camera->world transform via cam_xpos/cam_xmat.

    Returns None (FAIL LOUD — never a fabricated point) when fewer than 4 valid
    depth points survive: an empty/degenerate mask or an all-zero depth patch
    means the object was not perceived, and the caller must surface that, NOT
    substitute a ground-truth pose to stay green.

    Args:
        depth: (H, W) depth image. Metres with depth_scale=1.0 (MuJoCo render),
            or raw uint16 mm with depth_scale=1000.0 (RealSense).
        color: (H, W, 3) uint8 RGB (carried for parity; not used for the point).
        mask: (H, W) binary mask; only pixels where mask > 0 are projected.
        intrinsics: CameraIntrinsics (fx, fy, cx, cy) — e.g. mujoco_intrinsics().
        cam_xpos: (3,) camera world position (MuJoCo data.cam_xpos[id]).
        cam_xmat: (9,) or (3,3) camera world rotation (MuJoCo data.cam_xmat[id]).
        depth_scale: Raw-depth-units -> metres divisor. 1.0 for metric depth.
        depth_trunc: Max depth in metres; farther points discarded.

    Returns:
        Pose3D world-frame grasp point, or None if the object was not localizable
        (or the mask, depth and intrinsics disagree on the frame size).
    """
    h, w = depth.shape[:2]
    if mask.shape[:2] != (h, w) or (int(intrinsics.height), int(intrinsics.width)) != (h, w):
        logger.error(
            "[GRASP-PT] frame mismatch: mask %s, depth %s, K %sx%s — not projecting",
            mask.shape[:2], (h, w), intrinsics.width, intrinsics.height,
        )
        return None
    points, _colors = rgbd_to_pointcloud_fast(
        depth, color, intrinsics,
        depth_scale=depth_scale, depth_trunc=depth_trunc, mask=mask,
    )
    if len(points) < _MIN_POINTS:
        return None

    # Foreground gating: keep only the nearest coherent depth cluster so a mask
    # that leaks onto same-appearance background (loose bbox / edge-bleed / a
    # same-colored object behind) localizes the FRONT object, not a midpoint.
    if foreground_only:
        points = select_nearest_cluster(points)

    points = remove_depth_outliers(points)
    if len(points) < _MIN_POINTS:
        return None

    centroid_cam = robust_centroid(points)  # camera frame (OpenCV: x=right,y=down,z=fwd)

    # The per-axis trimmed mean is taken in the camera frame and then transformed
    # (close to, not exactly, reducing the world points). It is the centroid of the
    # VISIBLE surface: an upright cylinder's axis is fitted below.
    wx, wy, wz = camera_to_world(
        centroid_cam.x, centroid_cam.y, centroid_cam.z,
        cam_xpos=cam_xpos, cam_xmat=cam_xmat,
    )
    # The visible surface of an upright cylinder sits ~2/pi r in front of its axis:
    # the axis comes from a circle fit of that surface when it is one.
    R = np.asarray(cam_xmat, dtype=np.float64).reshape(3, 3)
    world_pts = np.asarray(cam_xpos, dtype=np.float64) + points @ (R * [1.0, -1.0, -1.0]).T
    axis = upright_cylinder_axis(world_pts, np.asarray(cam_xpos, dtype=np.float64)[:2])
    if axis is not None:
        wx, wy = axis
    return Pose3D(x=float(wx), y=float(wy), z=float(wz))


# Points within this of the highest one are the top cap, not the side surface.
_CAP_BAND_M = 0.01


def upright_cylinder_axis(world_pts: np.ndarray, cam_xy: np.ndarray) -> tuple[float, float] | None:
    """(x, y) of the axis of an upright cylinder from its visible side surface, or None.

    A least-squares circle fit of the side points' xy (the top cap dropped). None
    unless the fit is a cylinder seen from *cam_xy*: its radius within 30% of half
    the visible width, its centre behind the visible surface by at most a radius.
    """
    side = world_pts[world_pts[:, 2] < world_pts[:, 2].max() - _CAP_BAND_M][:, :2]
    if len(side) < 20:
        return None
    (d, e, f), *_ = np.linalg.lstsq(np.c_[side, np.ones(len(side))], -(side ** 2).sum(1), rcond=None)
    centre = np.array([-d / 2.0, -e / 2.0])
    r2 = float(centre @ centre - f)
    mean = side.mean(0)
    view = mean - cam_xy
    if r2 <= 0.0 or not np.linalg.norm(view):
        return None
    r, view = r2 ** 0.5, view / np.linalg.norm(view)
    half_width = float(np.ptp(side @ np.array([-view[1], view[0]]))) / 2.0
    behind = float((centre - mean) @ view)
    if not (0.7 * half_width <= r <= 1.3 * half_width and 0.0 < behind <= r):
        return None
    return float(centre[0]), float(centre[1])
