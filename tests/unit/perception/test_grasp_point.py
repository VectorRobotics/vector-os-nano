# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""RED-3 — pure perception->3D-grasp-point geometry.

The honest core of the perception-driven grasp: the 3D grasp POINT must come
from real rendered depth + an EdgeTAM mask + the proven cam->world transform,
NEVER from a ground-truth object pose. These tests are pure numpy (no sim, no
GPU, no model) and pin the load-bearing geometry against the known foot-guns:
  - depth_scale: MuJoCo depth is METRES (depth_scale=1.0), not RealSense mm.
  - axis flip: the OpenCV(x=right,y=down,z=fwd) -> MuJoCo(col0,col1,-col2) map.
  - robustness: a few far-depth mask-edge outliers must not drag the centroid.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from vector_os_nano.core.types import Pose3D
from vector_os_nano.perception._centroid import select_nearest_cluster
from vector_os_nano.perception.depth_projection import camera_to_world, mujoco_intrinsics
from vector_os_nano.perception.grasp_point import grasp_point_from_rgbd

_W, _H = 64, 48
_INTR = mujoco_intrinsics(_W, _H, vfov_deg=42.0)
# Identity camera pose at the world origin (cam_xmat row-major identity).
_CAM_XPOS = np.zeros(3, dtype=np.float64)
_CAM_XMAT = np.eye(3, dtype=np.float64).reshape(9)


def _scene(depth_val: float = 1.0):
    """A centred square mask over a flat depth plane (metres)."""
    depth = np.zeros((_H, _W), dtype=np.float32)
    color = np.zeros((_H, _W, 3), dtype=np.uint8)
    mask = np.zeros((_H, _W), dtype=np.uint8)
    # 7x7 square centred exactly on (cx=32, cy=24): camera-frame x,y centroid == 0.
    us = range(29, 36)  # mean 32
    vs = range(21, 28)  # mean 24
    for v in vs:
        for u in us:
            depth[v, u] = depth_val
            mask[v, u] = 1
    return depth, color, mask


def test_intrinsics_fovy_not_500():
    """fy must derive from fovy=42 (~62.5 @H=48), never a hardcoded 500."""
    expected_fy = (_H / 2.0) / math.tan(math.radians(42.0) / 2.0)
    assert _INTR.fy == pytest.approx(expected_fy, rel=1e-6)
    assert _INTR.fx == pytest.approx(_INTR.fy)
    assert (_INTR.cx, _INTR.cy) == (_W / 2.0, _H / 2.0)


def test_grasp_point_centroid_metres():
    """A 1 m plane behind a centred mask -> world point via the proven flip.

    Identity cam pose, MuJoCo convention forward = -col2 = -Z_world, so a point
    1 m in front of the camera lands at world (0, 0, -1).
    """
    depth, color, mask = _scene(depth_val=1.0)
    p = grasp_point_from_rgbd(depth, color, mask, _INTR, _CAM_XPOS, _CAM_XMAT)
    assert isinstance(p, Pose3D)
    # Cross-check against the proven transform applied to the camera-frame centroid.
    wx, wy, wz = camera_to_world(0.0, 0.0, 1.0, 0, 0, 0, 0,
                                 cam_xpos=_CAM_XPOS, cam_xmat=_CAM_XMAT)
    assert (p.x, p.y, p.z) == pytest.approx((wx, wy, wz), abs=0.03)
    assert p.z == pytest.approx(-1.0, abs=0.03)


def test_depth_scale_default_is_metres_not_mm():
    """The default depth_scale must be 1.0 (metres). The mm default (1000) would
    collapse every point to ~origin — a silent, plausible-looking wrong grasp."""
    depth, color, mask = _scene(depth_val=1.0)
    p_default = grasp_point_from_rgbd(depth, color, mask, _INTR, _CAM_XPOS, _CAM_XMAT)
    assert abs(p_default.z) == pytest.approx(1.0, abs=0.03)  # metres, not 0.001
    # Explicit mm scale collapses it (proves the knob is real and load-bearing).
    p_mm = grasp_point_from_rgbd(depth, color, mask, _INTR, _CAM_XPOS, _CAM_XMAT,
                                 depth_scale=1000.0)
    assert abs(p_mm.z) < 0.01


def test_far_outliers_do_not_drag_centroid():
    """Mask-edge depth bleed (a few far pixels) must not move the centroid."""
    depth, color, mask = _scene(depth_val=1.0)
    # Poison 4 masked pixels with 5 m depth-bleed.
    for v, u in [(21, 29), (21, 35), (27, 29), (27, 35)]:
        depth[v, u] = 5.0
    p = grasp_point_from_rgbd(depth, color, mask, _INTR, _CAM_XPOS, _CAM_XMAT)
    assert abs(p.z) == pytest.approx(1.0, abs=0.1)  # ~1 m, not pulled toward 5 m


def test_empty_mask_returns_none():
    depth, color, _ = _scene()
    empty = np.zeros((_H, _W), dtype=np.uint8)
    assert grasp_point_from_rgbd(depth, color, empty, _INTR, _CAM_XPOS, _CAM_XMAT) is None


def test_too_few_valid_points_returns_none():
    """A mask of 2 pixels -> < 4 valid -> None (no centroid from noise)."""
    depth = np.zeros((_H, _W), dtype=np.float32)
    color = np.zeros((_H, _W, 3), dtype=np.uint8)
    mask = np.zeros((_H, _W), dtype=np.uint8)
    for v, u in [(24, 32), (24, 33)]:
        depth[v, u] = 1.0
        mask[v, u] = 1
    assert grasp_point_from_rgbd(depth, color, mask, _INTR, _CAM_XPOS, _CAM_XMAT) is None


def test_all_zero_depth_returns_none():
    """Mask present but depth all zero (sensor miss) -> None, never a fake point."""
    depth = np.zeros((_H, _W), dtype=np.float32)
    color = np.zeros((_H, _W, 3), dtype=np.uint8)
    _, _, mask = _scene()
    assert grasp_point_from_rgbd(depth, color, mask, _INTR, _CAM_XPOS, _CAM_XMAT) is None


# --- foreground gating (select_nearest_cluster) -----------------------------

def test_select_nearest_cluster_picks_front_of_bimodal():
    near = np.random.default_rng(0).normal([0, 0, 1.0], [0.02, 0.02, 0.02], (60, 3))
    far = np.random.default_rng(1).normal([0.3, 0, 3.0], [0.02, 0.02, 0.02], (60, 3))
    kept = select_nearest_cluster(np.vstack([near, far]))
    assert len(kept) >= 50
    assert kept[:, 2].max() < 1.5  # only the ~1 m cluster survives, not the 3 m one


def test_select_nearest_cluster_unimodal_is_noop():
    pts = np.random.default_rng(2).normal([0, 0, 1.0], [0.02, 0.02, 0.02], (60, 3))
    kept = select_nearest_cluster(pts)
    assert len(kept) == len(pts)


def test_select_nearest_cluster_too_few_unchanged():
    pts = np.array([[0, 0, 1.0], [0, 0, 3.0]], dtype=np.float64)
    assert len(select_nearest_cluster(pts)) == 2


def _bimodal_scene():
    """Mask covers a NEAR patch (1 m) and a FAR patch (3 m) — a leaky mask."""
    depth = np.zeros((_H, _W), dtype=np.float32)
    color = np.zeros((_H, _W, 3), dtype=np.uint8)
    mask = np.zeros((_H, _W), dtype=np.uint8)
    for v in range(30, 38):           # near patch, lower image
        for u in range(28, 36):
            depth[v, u] = 1.0; mask[v, u] = 1
    for v in range(8, 16):            # far patch (background leak), upper image
        for u in range(28, 36):
            depth[v, u] = 3.0; mask[v, u] = 1
    return depth, color, mask


def test_grasp_point_foreground_gating_picks_front_object():
    depth, color, mask = _bimodal_scene()
    p_fg = grasp_point_from_rgbd(depth, color, mask, _INTR, _CAM_XPOS, _CAM_XMAT)
    # identity cam: world z = -depth, so the front (1 m) object -> z ~ -1.0.
    assert p_fg is not None
    assert p_fg.z == pytest.approx(-1.0, abs=0.15)  # the near object, NOT the midpoint


def test_grasp_point_without_gating_is_pulled_to_midpoint():
    depth, color, mask = _bimodal_scene()
    p_raw = grasp_point_from_rgbd(depth, color, mask, _INTR, _CAM_XPOS, _CAM_XMAT,
                                  foreground_only=False)
    # Without gating the bimodal cloud's centroid sits between 1 m and 3 m.
    assert p_raw is not None
    assert p_raw.z < -1.5  # demonstrably worse — proves the gating is load-bearing


# ---------------------------------------------------------------------------
# Upright cylinder: the axis, not the visible-surface centroid
# ---------------------------------------------------------------------------

from vector_os_nano.perception.grasp_point import upright_cylinder_axis  # noqa: E402


def _visible_cylinder(axis=(1.0, 0.1), r=0.028, cam=(0.0, 0.0)):
    """The side surface a camera at *cam* sees of an upright cylinder (+ its top cap)."""
    view = np.arctan2(cam[1] - axis[1], cam[0] - axis[0])
    th = view + np.linspace(-1.3, 1.3, 40)
    side = [(axis[0] + r * np.cos(t), axis[1] + r * np.sin(t), z)
            for t in th for z in np.linspace(0.28, 0.36, 12)]
    cap = [(axis[0] + a * np.cos(t), axis[1] + a * np.sin(t), 0.36) for t in th for a in (0.0, r / 2)]
    return np.array(side + cap)


def test_cylinder_axis_removes_the_visible_surface_bias():
    pts = _visible_cylinder()
    assert np.linalg.norm(pts[:, :2].mean(0) - (1.0, 0.1)) > 0.012  # the centroid's bias
    x, y = upright_cylinder_axis(pts, np.zeros(2))
    assert math.dist((x, y), (1.0, 0.1)) < 1e-3


def test_cylinder_axis_rejects_what_is_not_a_cylinder():
    rng = np.random.default_rng(0)
    wall = np.c_[np.full(200, 1.0), rng.uniform(-0.1, 0.1, 200), rng.uniform(0.2, 0.4, 200)]
    assert upright_cylinder_axis(wall, np.zeros(2)) is None  # a flat face: no plausible circle
    flat = np.c_[rng.uniform(0.9, 1.1, 200), rng.uniform(-0.1, 0.1, 200), np.full(200, 0.3)]
    assert upright_cylinder_axis(flat, np.zeros(2)) is None  # a horizontal patch: no side surface
