# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""RED-first: the MuJoCo->Blender camera transform must NOT drift.

The whole co-sim rests on the photoreal frame being shot from the SAME camera
pose MuJoCo's physics camera has — otherwise the VLM grounds objects that are
not where the robot thinks they are. MuJoCo and Blender share conventions
(Z-up right-handed world; camera looks down local -Z with +Y up), so the
correct transform is the IDENTITY mapping of the rotation columns. These tests
pin that so a future refactor can't silently slip in an axis swap (the classic
sim-render drift bug).
"""
from __future__ import annotations

import math

import numpy as np

from vector_os_nano.playground.photoreal.camera import mujoco_camera_to_blender


# A non-trivial but valid camera pose: 90 deg about world Z, raised off origin.
# cam_xmat is MuJoCo row-major flat (9,); reshaped (3,3) its COLUMNS are the
# camera frame x(right)/y(up)/z(-forward) axes expressed in world coords.
_CAM_POS = np.array([1.5, -0.5, 0.85], dtype=np.float64)
_CAM_MAT_FLAT = np.array([0.0, -1.0, 0.0,
                          1.0, 0.0, 0.0,
                          0.0, 0.0, 1.0], dtype=np.float64)
_FOVY_DEG = 45.0


def _result():
    return mujoco_camera_to_blender(_CAM_POS, _CAM_MAT_FLAT, _FOVY_DEG)


def test_translation_matches_mujoco_camera_position():
    mw = _result().matrix_world
    np.testing.assert_allclose(mw[:3, 3], _CAM_POS, atol=1e-12)
    np.testing.assert_allclose(mw[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_camera_axes_equal_mujoco_columns_no_swap():
    """right/up/forward axes must equal MuJoCo's, with NO axis swap."""
    mw = _result().matrix_world
    cm = _CAM_MAT_FLAT.reshape(3, 3)
    right, up, back = cm[:, 0], cm[:, 1], cm[:, 2]
    np.testing.assert_allclose(mw[:3, 0], right, atol=1e-12)      # +X right
    np.testing.assert_allclose(mw[:3, 1], up, atol=1e-12)         # +Y up
    np.testing.assert_allclose(mw[:3, 2], back, atol=1e-12)       # +Z = -forward
    # forward (where the camera actually looks) = -local Z
    np.testing.assert_allclose(-mw[:3, 2], -back, atol=1e-12)


def test_rotation_is_proper_orthonormal():
    R = _result().matrix_world[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert math.isclose(float(np.linalg.det(R)), 1.0, abs_tol=1e-9)


def test_fovy_round_trips_to_radians():
    res = _result()
    assert math.isclose(res.fovy_rad, math.radians(_FOVY_DEG), abs_tol=1e-12)


def test_accepts_3x3_matrix_form():
    """cam_mat may arrive as (3,3) or flat (9,) — both must give the same pose."""
    flat = _result().matrix_world
    nested = mujoco_camera_to_blender(
        _CAM_POS, _CAM_MAT_FLAT.reshape(3, 3), _FOVY_DEG).matrix_world
    np.testing.assert_allclose(flat, nested, atol=1e-12)


def test_identity_pose_looks_down_world_minus_z():
    """Sanity anchor: identity cam_mat = MuJoCo default cam looks down world -Z."""
    res = mujoco_camera_to_blender([0.0, 0.0, 1.0], np.eye(3).reshape(-1), 60.0)
    forward = -res.matrix_world[:3, 2]
    np.testing.assert_allclose(forward, [0.0, 0.0, -1.0], atol=1e-12)
