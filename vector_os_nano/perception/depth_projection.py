# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Pixel + depth -> world projection through the EXACT camera pose.

Pinhole intrinsics from the camera's vertical FOV and the frame's actual shape
(pixels -> camera frame is pointcloud.rgbd_to_pointcloud_fast; OpenCV axes: x right,
y down, z forward), and camera -> world through MuJoCo's ``cam_xpos``/``cam_xmat`` (columns right, up, -forward). There is
no robot-pose approximation: a world point needs the camera pose of its pixels.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from vector_os_nano.core.types import CameraIntrinsics


def mujoco_intrinsics(width: int = 320, height: int = 240, vfov_deg: float = 42.0) -> CameraIntrinsics:
    """Pinhole intrinsics of a MuJoCo camera with fovy *vfov_deg* at width x height.

    Square pixels; the horizontal FOV follows from the aspect ratio.
    """
    fy = (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    return CameraIntrinsics(
        fx=fy, fy=fy, cx=width / 2.0, cy=height / 2.0, width=int(width), height=int(height),
    )


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
