# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Point cloud utilities — pure numpy, no open3d required for fast paths.

Ported from vector_ws/src/vector_perception_utils/vector_perception_utils/pointcloud_utils.py.
All ROS2 message dependencies removed. Returns vector_os types directly.

Numeric constants preserved from vector_ws:
  depth_scale = 1000.0   (RealSense uint16 mm -> metres)
  depth_trunc = 10.0     (metres)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from vector_os_nano.core.types import BBox3D, CameraIntrinsics, Pose3D


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rgbd_to_pointcloud_fast(
    depth: np.ndarray,
    color: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_scale: float = 1000.0,
    depth_trunc: float = 10.0,
    mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast numpy RGBD-to-pointcloud projection (~5x faster than Open3D).

    Vectorised numpy math — avoids Open3D Image/PointCloud objects.

    Args:
        depth: Depth image (H, W).
               - RealSense D405/D435i: uint16 in millimetres (depth_scale=1000.0)
               - ZED 2i: float32 in metres (depth_scale=1.0)
        color: RGB image (H, W, 3), values in [0, 255].
        intrinsics: CameraIntrinsics with fx, fy, cx, cy.
        depth_scale: Scale factor converting raw depth units to metres.
        depth_trunc: Maximum depth in metres; farther points are discarded.
        mask: Optional binary mask (H, W); only pixels where mask > 0 are used.

    Returns:
        Tuple of (points, colors) as float64 numpy arrays of shape (N, 3).
        colors values are in [0, 1].
    """
    fx: float = intrinsics.fx
    fy: float = intrinsics.fy
    cx: float = intrinsics.cx
    cy: float = intrinsics.cy

    if mask is not None:
        ys, xs = np.where(mask > 0)
    else:
        h, w = depth.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w]
        ys = ys.ravel()
        xs = xs.ravel()

    depths = depth[ys, xs].astype(np.float64) / depth_scale
    valid = (depths > 0.0) & (depths < depth_trunc)
    xs = xs[valid]
    ys = ys[valid]
    depths = depths[valid]

    if len(depths) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    points = np.column_stack([
        (xs - cx) * depths / fx,
        (ys - cy) * depths / fy,
        depths,
    ])
    colors = color[ys, xs].astype(np.float64) / 255.0
    return points, colors


def pointcloud_to_bbox3d_fast(points: np.ndarray) -> Optional[BBox3D]:
    """Axis-aligned bounding box from numpy (~10x faster than Open3D OBB).

    Returns None if fewer than 4 finite points (degenerate case).

    Args:
        points: Point coordinates (N, 3).

    Returns:
        BBox3D with identity orientation (axis-aligned) or None.
    """
    if len(points) < 4:
        return None

    finite_mask = np.isfinite(points).all(axis=1)
    pts = points[finite_mask]
    if len(pts) < 4:
        return None

    min_pt = pts.min(axis=0)
    max_pt = pts.max(axis=0)
    center = (min_pt + max_pt) * 0.5
    size = max_pt - min_pt

    return BBox3D(
        center=Pose3D(
            x=float(center[0]),
            y=float(center[1]),
            z=float(center[2]),
            # Identity orientation — axis-aligned
            qx=0.0, qy=0.0, qz=0.0, qw=1.0,
        ),
        size_x=float(size[0]),
        size_y=float(size[1]),
        size_z=float(size[2]),
    )
