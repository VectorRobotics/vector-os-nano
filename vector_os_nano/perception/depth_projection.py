# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Pixel + depth -> world projection through the EXACT camera pose.

Pixel -> camera frame (OpenCV: x right, y down, z forward; the intrinsics are the
frame's own ``K``), and camera -> world through the frame's ``cam_pos``/``cam_mat``
(columns right, up, -forward). There is no robot-pose approximation: a world point
needs the camera pose of its pixels.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from vector_os_nano.core.types import CameraIntrinsics


def pixel_to_camera(
    u: float,
    v: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float, float]:
    """Project a pixel (u, v) at depth_m to camera-frame 3D point.

    Camera frame: Z forward, X right, Y down (OpenCV convention).

    Returns (x_cam, y_cam, z_cam) in metres.
    """
    z = depth_m
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    return (x, y, z)


def camera_to_world(
    x_cam: float,
    y_cam: float,
    z_cam: float,
    *,
    cam_xpos: Any,
    cam_xmat: Any,
) -> tuple[float, float, float]:
    """Transform a camera-frame point (OpenCV axes) to world through the camera pose.

    ``cam_xpos`` (3,) and ``cam_xmat`` (9,) row-major are MuJoCo's ``data.cam_xpos``
    / ``data.cam_xmat`` (columns right, up, -forward) for the frame's instant.
    """
    xmat = np.asarray(cam_xmat, dtype=np.float64).reshape(3, 3)
    pos = np.asarray(cam_xpos, dtype=np.float64)
    world = pos + x_cam * xmat[:, 0] - y_cam * xmat[:, 1] - z_cam * xmat[:, 2]
    return (float(world[0]), float(world[1]), float(world[2]))
