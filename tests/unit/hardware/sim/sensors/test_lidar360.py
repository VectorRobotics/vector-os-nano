# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""MuJoCoLivox360 unit tests (v2.4 T1).

Polar grid construction, ray casting against a known wall, range
clipping, intensity defaults, rate-limit caching, PointCloud2 byte
layout — all on the inline tiny MJCF (no Go2 model imported).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from vector_os_nano.hardware.sim.sensors import LidarSample, MuJoCoLivox360
from vector_os_nano.hardware.sim.sensors.lidar360 import _build_ray_dirs


# ---------------------------------------------------------------------------
# Pure helper — _build_ray_dirs
# ---------------------------------------------------------------------------


def test_ray_dirs_polar_grid_shape() -> None:
    dirs = _build_ray_dirs(360, 16, math.radians(-7), math.radians(52))
    assert dirs.shape == (360 * 16, 3)


def test_ray_dirs_unit_length() -> None:
    dirs = _build_ray_dirs(36, 4, math.radians(-7), math.radians(52))
    norms = np.linalg.norm(dirs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)


def test_ray_dirs_azimuth_endpoints_excluded() -> None:
    """h_resolution=4 at elevation 0 → 4 unique rays, no ±π duplicate.

    linspace(-π, π, 4, endpoint=False) yields {-π, -π/2, 0, π/2};
    in particular +π is excluded so the -π ray is not duplicated.
    """
    dirs = _build_ray_dirs(4, 1, 0.0, 0.0)
    # First ray is at azimuth -π → (-1, 0, 0)
    assert np.allclose(dirs[0], [-1.0, 0.0, 0.0], atol=1e-9)
    # Exactly 4 unique unit vectors (a +π duplicate would collapse to 3)
    unique = np.unique(np.round(dirs, 9), axis=0)
    assert unique.shape[0] == 4


def test_ray_dirs_elevation_range_within_minus7_to_52_deg() -> None:
    dirs = _build_ray_dirs(8, 5, math.radians(-7.0), math.radians(52.0))
    elevations = np.arcsin(dirs[:, 2])
    assert np.min(elevations) == pytest.approx(math.radians(-7.0), abs=1e-9)
    assert np.max(elevations) == pytest.approx(math.radians(52.0), abs=1e-9)


def test_ray_dirs_single_layer_uses_midpoint() -> None:
    dirs = _build_ray_dirs(4, 1, math.radians(-30.0), math.radians(30.0))
    # All rays should share the same elevation = 0
    elevations = np.arcsin(dirs[:, 2])
    assert np.allclose(elevations, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_unknown_body_name_raises(tiny_model_data) -> None:
    model, data = tiny_model_data
    with pytest.raises(ValueError, match="not found"):
        MuJoCoLivox360(model, data, body_name="nonexistent")


def test_zero_resolution_raises(tiny_model_data) -> None:
    model, data = tiny_model_data
    with pytest.raises(ValueError):
        MuJoCoLivox360(model, data, h_resolution=0)
    with pytest.raises(ValueError):
        MuJoCoLivox360(model, data, v_layers=0)


def test_inverted_elevation_range_raises(tiny_model_data) -> None:
    model, data = tiny_model_data
    with pytest.raises(ValueError):
        MuJoCoLivox360(model, data, v_min_deg=10.0, v_max_deg=5.0)


def test_zero_max_range_raises(tiny_model_data) -> None:
    model, data = tiny_model_data
    with pytest.raises(ValueError):
        MuJoCoLivox360(model, data, max_range=0.0)


# ---------------------------------------------------------------------------
# Ray casting against the tiny MJCF (red wall at x=3, green wall at x=-3)
# ---------------------------------------------------------------------------


def test_step_against_known_wall_returns_correct_xyz(tiny_model_data) -> None:
    """Trunk at origin; ray azimuth=0 elevation=0 → hit the front wall x≈2.9."""
    model, data = tiny_model_data
    lidar = MuJoCoLivox360(
        model, data, body_name="trunk",
        offset=(0.0, 0.0, 0.0),         # mount at body origin (z=0.5)
        h_resolution=4,                  # azimuths: -π, -π/2, 0, π/2
        v_layers=1,
        v_min_deg=0.0, v_max_deg=0.0,
    )
    sample = lidar.step(now=0.0)
    # Front wall at x=3 with size 0.1 → near face at x≈2.9
    front_hits = sample.points[
        (sample.points[:, 0] > 2.0) & (np.abs(sample.points[:, 1]) < 0.1)
    ]
    assert front_hits.shape[0] == 1
    assert front_hits[0, 0] == pytest.approx(2.9, abs=0.1)


def test_step_clamps_at_max_range(tiny_model_data) -> None:
    """Wall at x=3, max_range=2 → no front-wall hit recorded."""
    model, data = tiny_model_data
    lidar = MuJoCoLivox360(
        model, data, offset=(0.0, 0.0, 0.0),
        h_resolution=4, v_layers=1,
        v_min_deg=0.0, v_max_deg=0.0,
        max_range=2.0,
    )
    sample = lidar.step(now=0.0)
    # No point should have x > 1.9 (within max_range of origin)
    assert not np.any(sample.points[:, 0] > 1.9)


def test_intensity_field_defaults_to_one_point_zero(tiny_model_data) -> None:
    model, data = tiny_model_data
    lidar = MuJoCoLivox360(
        model, data, offset=(0.0, 0.0, 0.0),
        h_resolution=4, v_layers=1,
        v_min_deg=0.0, v_max_deg=0.0,
    )
    sample = lidar.step(now=0.0)
    if sample.points.shape[0] > 0:
        assert np.allclose(sample.points[:, 3], 1.0)


def test_step_in_world_frame_after_body_translated(
    tiny_model_data, trunk_translated_to,
) -> None:
    """Move the trunk to (5, 0, 0.5); front-facing rays now miss the wall.

    Original front wall is at x=3. With body at x=5, the ray firing in
    +X (azimuth=0) shoots away from the wall and registers no hit.
    """
    model, data = tiny_model_data
    _, move = trunk_translated_to
    move(5.0, 0.0, 0.5)
    lidar = MuJoCoLivox360(
        model, data, offset=(0.0, 0.0, 0.0),
        h_resolution=4, v_layers=1,
        v_min_deg=0.0, v_max_deg=0.0,
        max_range=10.0,
    )
    sample = lidar.step(now=0.0)
    # No hit in front (+X) because we moved past the wall
    front_hits = sample.points[
        (sample.points[:, 0] > 5.5) & (np.abs(sample.points[:, 1]) < 0.1)
    ]
    assert front_hits.shape[0] == 0


# ---------------------------------------------------------------------------
# Sample dataclass — PointCloud2 layout
# ---------------------------------------------------------------------------


def test_sample_field_layout_x_y_z_intensity_float32(tiny_model_data) -> None:
    sample = MuJoCoLivox360(
        *tiny_model_data, h_resolution=4, v_layers=1,
        v_min_deg=0.0, v_max_deg=0.0,
    ).step(now=0.0)
    layout = sample.field_layout
    names = [f[0] for f in layout]
    offsets = [f[1] for f in layout]
    types = [f[2] for f in layout]
    assert names == ["x", "y", "z", "intensity"]
    assert offsets == [0, 4, 8, 12]
    assert all(t == 7 for t in types)        # PointField.FLOAT32 == 7


def test_sample_point_step_is_sixteen(tiny_model_data) -> None:
    sample = MuJoCoLivox360(*tiny_model_data).step(now=0.0)
    assert sample.point_step == 16


def test_sample_dense_and_little_endian(tiny_model_data) -> None:
    sample = MuJoCoLivox360(*tiny_model_data).step(now=0.0)
    assert sample.is_dense is True
    assert sample.is_bigendian is False


def test_sample_frame_id_default_map(tiny_model_data) -> None:
    sample = MuJoCoLivox360(*tiny_model_data).step(now=0.0)
    assert sample.frame_id == "map"


def test_sample_frame_id_overridable(tiny_model_data) -> None:
    sample = MuJoCoLivox360(*tiny_model_data, frame_id="lidar").step(now=0.0)
    assert sample.frame_id == "lidar"


def test_pointcloud2_data_bytes_round_trip(tiny_model_data) -> None:
    """data bytes should decode back to the same float32 (N, 4) array."""
    sample = MuJoCoLivox360(
        *tiny_model_data, h_resolution=8, v_layers=2,
        v_min_deg=-7.0, v_max_deg=52.0,
    ).step(now=0.0)
    raw = sample.pointcloud2_data_bytes()
    expected = sample.points.astype(np.float32).tobytes()
    assert raw == expected


# ---------------------------------------------------------------------------
# Rate-limit caching
# ---------------------------------------------------------------------------


def test_rate_limit_returns_cached_when_called_too_fast(tiny_model_data) -> None:
    lidar = MuJoCoLivox360(
        *tiny_model_data, h_resolution=4, v_layers=1, rate_hz=10.0,
        v_min_deg=0.0, v_max_deg=0.0,
    )
    first = lidar.step(now=0.0)
    second = lidar.step(now=0.05)        # 50 ms < 100 ms
    assert second is first


def test_rate_limit_yields_after_interval(tiny_model_data) -> None:
    lidar = MuJoCoLivox360(
        *tiny_model_data, h_resolution=4, v_layers=1, rate_hz=10.0,
        v_min_deg=0.0, v_max_deg=0.0,
    )
    first = lidar.step(now=0.0)
    second = lidar.step(now=0.20)        # 200 ms > 100 ms
    assert second is not first
    assert second.stamp_seconds == pytest.approx(0.20)


def test_due_returns_true_initially(tiny_model_data) -> None:
    assert MuJoCoLivox360(*tiny_model_data, rate_hz=10.0).due(now=0.0) is True


def test_due_returns_false_within_interval(tiny_model_data) -> None:
    lidar = MuJoCoLivox360(
        *tiny_model_data, h_resolution=4, v_layers=1, rate_hz=10.0,
        v_min_deg=0.0, v_max_deg=0.0,
    )
    lidar.step(now=0.0)
    assert lidar.due(now=0.05) is False


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_num_rays_matches_grid_product(tiny_model_data) -> None:
    lidar = MuJoCoLivox360(*tiny_model_data, h_resolution=12, v_layers=4)
    assert lidar.num_rays == 48


def test_ray_dirs_body_returns_copy(tiny_model_data) -> None:
    lidar = MuJoCoLivox360(*tiny_model_data, h_resolution=4, v_layers=1)
    dirs = lidar.ray_dirs_body
    dirs[0] = 0.0       # mutate
    assert not np.allclose(lidar.ray_dirs_body, dirs)


def test_body_id_property(tiny_model_data) -> None:
    lidar = MuJoCoLivox360(*tiny_model_data)
    assert lidar.body_id > 0


def test_lidar_sample_is_frozen(tiny_model_data) -> None:
    sample = MuJoCoLivox360(*tiny_model_data).step(now=0.0)
    assert isinstance(sample, LidarSample)
    with pytest.raises(Exception):
        sample.frame_id = "nope"        # type: ignore[misc]
