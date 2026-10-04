# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Unit tests for perception.pointcloud — pure numpy, no real camera needed."""
from __future__ import annotations

import numpy as np
import pytest

from vector_os_nano.perception.pointcloud import (
    rgbd_to_pointcloud_fast,
    pointcloud_to_bbox3d_fast,
)
from vector_os_nano.core.types import CameraIntrinsics, BBox3D


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def intrinsics() -> CameraIntrinsics:
    """Synthetic D405-like intrinsics at 640x480."""
    return CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)


@pytest.fixture()
def flat_depth() -> np.ndarray:
    """Flat depth image: all pixels at 1.0m (1000 in uint16 mm)."""
    depth = np.full((480, 640), 1000, dtype=np.uint16)
    return depth


@pytest.fixture()
def color_image() -> np.ndarray:
    """Uniform RGB image (red)."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[..., 0] = 200
    return img


# ---------------------------------------------------------------------------
# test_rgbd_to_pointcloud_shape
# ---------------------------------------------------------------------------

def test_rgbd_to_pointcloud_shape(flat_depth, color_image, intrinsics):
    """Output shape matches number of valid depth pixels (no mask)."""
    points, colors = rgbd_to_pointcloud_fast(flat_depth, color_image, intrinsics)
    h, w = flat_depth.shape
    assert points.shape == (h * w, 3), f"Expected ({h*w}, 3) got {points.shape}"
    assert colors.shape == (h * w, 3), f"Expected ({h*w}, 3) got {colors.shape}"
    # All z values should be ~1.0 m
    assert np.allclose(points[:, 2], 1.0, atol=0.01)
    # Color values normalized to [0, 1]
    assert colors.min() >= 0.0
    assert colors.max() <= 1.0


def test_rgbd_to_pointcloud_with_mask(flat_depth, color_image, intrinsics):
    """With a mask, only masked pixels are projected."""
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:200, 200:400] = 1  # 100x200 = 20000 pixels
    points, colors = rgbd_to_pointcloud_fast(flat_depth, color_image, intrinsics, mask=mask)
    assert points.shape[0] == 20000
    assert colors.shape[0] == 20000


# ---------------------------------------------------------------------------
# test_rgbd_to_pointcloud_empty_mask
# ---------------------------------------------------------------------------

def test_rgbd_to_pointcloud_empty_mask(flat_depth, color_image, intrinsics):
    """Empty mask (all zeros) returns empty pointcloud."""
    mask = np.zeros((480, 640), dtype=np.uint8)
    points, colors = rgbd_to_pointcloud_fast(flat_depth, color_image, intrinsics, mask=mask)
    assert points.shape == (0, 3)
    assert colors.shape == (0, 3)


def test_rgbd_to_pointcloud_zero_depth(color_image, intrinsics):
    """Zero depth pixels are excluded (invalid depth)."""
    depth = np.zeros((480, 640), dtype=np.uint16)
    points, colors = rgbd_to_pointcloud_fast(depth, color_image, intrinsics)
    assert points.shape == (0, 3)
    assert colors.shape == (0, 3)


# ---------------------------------------------------------------------------
# test_bbox3d_from_points
# ---------------------------------------------------------------------------

def test_bbox3d_from_points():
    """BBox3D center and size contain all input points."""
    points = np.array([
        [0.0, 0.0, 1.0],
        [0.1, 0.1, 1.1],
        [0.2, 0.2, 1.2],
        [0.3, 0.3, 1.3],
        [0.4, 0.4, 1.4],
    ])
    bbox = pointcloud_to_bbox3d_fast(points)
    assert isinstance(bbox, BBox3D)
    # Center should be at midpoint
    assert abs(bbox.center.x - 0.2) < 0.001
    assert abs(bbox.center.y - 0.2) < 0.001
    assert abs(bbox.center.z - 1.2) < 0.001
    # Size should span the full range
    assert abs(bbox.size_x - 0.4) < 0.001
    assert abs(bbox.size_y - 0.4) < 0.001
    assert abs(bbox.size_z - 0.4) < 0.001


# ---------------------------------------------------------------------------
# test_bbox3d_single_point
# ---------------------------------------------------------------------------

def test_bbox3d_single_point():
    """Single point (< 4 points) returns None (degenerate case)."""
    points = np.array([[0.1, 0.2, 0.5]])
    result = pointcloud_to_bbox3d_fast(points)
    assert result is None


def test_bbox3d_three_points():
    """Three points (< 4) returns None."""
    points = np.array([
        [0.0, 0.0, 1.0],
        [0.1, 0.0, 1.0],
        [0.0, 0.1, 1.0],
    ])
    result = pointcloud_to_bbox3d_fast(points)
    assert result is None


def test_bbox3d_four_points():
    """Exactly 4 points returns a valid BBox3D."""
    points = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 2.0],
    ])
    bbox = pointcloud_to_bbox3d_fast(points)
    assert isinstance(bbox, BBox3D)


# ---------------------------------------------------------------------------
# test_outlier_removal
# ---------------------------------------------------------------------------

def test_pointcloud_depth_truncation(color_image, intrinsics):
    """Points beyond depth_trunc are excluded."""
    # depth_trunc default is 10.0m; set pixels to 11m (11000 mm uint16)
    depth = np.full((480, 640), 11000, dtype=np.uint16)
    points, _ = rgbd_to_pointcloud_fast(depth, color_image, intrinsics, depth_trunc=10.0)
    assert points.shape[0] == 0
