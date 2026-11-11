# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Level 13: camera intrinsics and camera -> world projection (pure Python, no MuJoCo).

The MJCF-pose projection is ground-truth-tested in
test_depth_projection_groundtruth.py; the old robot-heading fallback (whose mount
geometry disagreed with the MJCF) is gone: a world point needs the exact camera pose.
"""
from __future__ import annotations

import numpy as np
import pytest

from vector_os_nano.core.types import CameraIntrinsics
from vector_os_nano.perception.depth_projection import camera_to_world, mujoco_intrinsics


def test_intrinsics_are_the_one_core_type():
    """perception:duplicate-camera-intrinsics — two CameraIntrinsics classes, duck-typed."""
    intr = mujoco_intrinsics(640, 480, vfov_deg=42.0)
    assert type(intr) is CameraIntrinsics
    assert (intr.width, intr.height, intr.cx, intr.cy) == (640, 480, 320.0, 240.0)


def test_camera_to_world_needs_the_camera_pose():
    """perception:dead-fallback-geometry — the heading+mount fallback was wrong vs MJCF."""
    with pytest.raises(TypeError):
        camera_to_world(0.0, 0.0, 2.0)  # type: ignore[call-arg]
    # MuJoCo xmat columns (right, up, -forward): camera at the origin looking down +x
    xmat = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)
    assert camera_to_world(0.0, 0.0, 2.0, cam_xpos=[1, 2, 3], cam_xmat=xmat.ravel()) == \
        pytest.approx((3.0, 2.0, 3.0))
