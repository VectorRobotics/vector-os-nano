# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Level 13: pixel -> camera -> world projection (pure Python, no MuJoCo).

The MJCF-pose projection is ground-truth-tested in
test_depth_projection_groundtruth.py; the old robot-heading fallback (whose mount
geometry disagreed with the MJCF) is gone: a world point needs the exact camera pose.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.harness.camera import pinhole
from vector_os_nano.core.types import CameraIntrinsics
from vector_os_nano.perception import depth_projection
from vector_os_nano.perception.depth_projection import camera_to_world, pixel_to_camera


class TestPixelToCamera:
    """Test pixel to camera-frame 3D projection."""

    def test_center_pixel_straight_ahead(self):
        """Center pixel at 2m depth → (0, 0, 2) in camera frame."""
        intr = pinhole(320, 240)
        x, y, z = pixel_to_camera(intr.cx, intr.cy, 2.0, intr)
        assert abs(x) < 0.001
        assert abs(y) < 0.001
        assert abs(z - 2.0) < 0.001

    def test_right_pixel_positive_x(self):
        """Pixel to the right of center → positive x_cam."""
        intr = pinhole(320, 240)
        x, y, z = pixel_to_camera(intr.cx + 50, intr.cy, 2.0, intr)
        assert x > 0
        assert abs(y) < 0.001

    def test_bottom_pixel_positive_y(self):
        """Pixel below center → positive y_cam (OpenCV: Y down)."""
        intr = pinhole(320, 240)
        x, y, z = pixel_to_camera(intr.cx, intr.cy + 50, 2.0, intr)
        assert abs(x) < 0.001
        assert y > 0

    def test_depth_preserved(self):
        intr = pinhole()
        _, _, z = pixel_to_camera(100, 100, 3.5, intr)
        assert abs(z - 3.5) < 0.001


def test_intrinsics_are_the_one_core_type():
    """perception:duplicate-camera-intrinsics — two CameraIntrinsics classes, duck-typed;
    K comes from the frame (``CameraFrame.K``), never re-derived from a fovy here."""
    assert not hasattr(depth_projection, "mujoco_intrinsics")
    intr = CameraIntrinsics.from_matrix([[415.7, 0, 320], [0, 415.7, 240], [0, 0, 1]], 640, 480)
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
