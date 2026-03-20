"""Unit tests for perception.calibration — no hardware needed."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from vector_os.perception.calibration import Calibration


# ---------------------------------------------------------------------------
# test_solve_affine_identity
# ---------------------------------------------------------------------------

def test_solve_affine_identity():
    """Identity transform: camera and base points are the same."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.5, 0.5, size=(10, 3))
    T = Calibration.solve_affine(pts, pts)
    assert T.shape == (4, 4)
    # Apply transform — result should match original
    cal = Calibration()
    cal._matrix = T
    for p in pts:
        result = cal.camera_to_base(p)
        assert np.allclose(result, p, atol=1e-6), f"Identity failed: {result} != {p}"


# ---------------------------------------------------------------------------
# test_solve_affine_translation
# ---------------------------------------------------------------------------

def test_solve_affine_translation():
    """Pure translation: camera frame offset by (1, 2, 3)."""
    rng = np.random.default_rng(1)
    pts_cam = rng.uniform(-0.5, 0.5, size=(8, 3))
    offset = np.array([1.0, 2.0, 3.0])
    pts_base = pts_cam + offset

    T = Calibration.solve_affine(pts_cam, pts_base)
    cal = Calibration()
    cal._matrix = T
    for p, expected in zip(pts_cam, pts_base):
        result = cal.camera_to_base(p)
        assert np.allclose(result, expected, atol=1e-4), f"Translation failed: {result} != {expected}"


# ---------------------------------------------------------------------------
# test_camera_to_base_transform
# ---------------------------------------------------------------------------

def test_camera_to_base_transform():
    """Transform applied correctly for arbitrary rotation + translation."""
    pytest = __import__("pytest")
    scipy = pytest.importorskip("scipy")
    from scipy.spatial.transform import Rotation
    # Construct a known 4x4 rigid transform
    R = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    t = np.array([0.1, -0.2, 0.3])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    cal = Calibration()
    cal._matrix = T

    p_cam = np.array([1.0, 0.0, 0.0])
    p_base = cal.camera_to_base(p_cam)
    # Expected: R @ p_cam + t
    expected = R @ p_cam + t
    assert np.allclose(p_base, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# test_save_load_roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip():
    """Save then load preserves the 4x4 matrix exactly."""
    rng = np.random.default_rng(99)
    T = np.eye(4)
    T[:3, :3] = rng.uniform(-1, 1, (3, 3))  # Not a valid rotation, just test storage
    T[3, :] = [0, 0, 0, 1]

    cal = Calibration()
    cal._matrix = T

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cal.npy")
        cal.save(path)
        loaded = Calibration.load(path)

    assert np.allclose(loaded._matrix, T)


def test_calibration_default_is_identity():
    """Default Calibration with no file uses identity matrix."""
    cal = Calibration()
    p = np.array([1.0, 2.0, 3.0])
    result = cal.camera_to_base(p)
    assert np.allclose(result, p, atol=1e-9)


def test_solve_affine_minimum_points():
    """solve_affine works with exactly 4 point pairs."""
    pts_cam = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    offset = np.array([0.5, 0.5, 0.5])
    pts_base = pts_cam + offset
    T = Calibration.solve_affine(pts_cam, pts_base)
    cal = Calibration()
    cal._matrix = T
    for p, expected in zip(pts_cam, pts_base):
        result = cal.camera_to_base(p)
        assert np.allclose(result, expected, atol=1e-4)
