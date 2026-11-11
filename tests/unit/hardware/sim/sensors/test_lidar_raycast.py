# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Equivalence tests for the shared lidar ray loop (STAGE 3a de-triplication).

These pin that the extracted :func:`raycast_lidar` (one ``world.raycast`` call) is
BYTE-IDENTICAL to the two original inline ``mj_ray`` loops it replaces.  Each test
embeds a verbatim copy of the relevant original loop body (the g1 copy from
``g1_lidar.py`` and the go2 copy from ``mujoco_go2._update_lidar``) as a golden
reference, runs it on the tiny inline MJCF at a fixed pose, and asserts the shared
loop produces identical output.  No Go2 / g1 model is imported (sensor unit tests stay light per
``feedback_no_parallel_agents``).
"""
from __future__ import annotations

import math

import numpy as np

from vector_os_nano.hardware.sim.sensors.lidar_raycast import raycast_lidar


# ----------------------------------------------------------------------------
# Golden references — VERBATIM copies of the two original inline ray loops
# ----------------------------------------------------------------------------
def _golden_g1(model, data, *, pos_lidar, heading, pelvis_bid, robot_geom_ids):
    """Verbatim copy of the original g1_lidar.py ray loop (max_range 15.0)."""
    import mujoco as _mj

    tilt_rad = math.radians(-10.0)
    cos_tilt = math.cos(tilt_rad)
    sin_tilt = math.sin(tilt_rad)
    n_azimuth = 360
    elevations = sorted(set([0] + list(range(-8, 30, 3))))
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    mid_ring_ranges: list[float] = []
    points_3d: list[tuple[float, float, float, float]] = []
    near_zero_hits = 0

    for elev_deg in elevations:
        elev_rad = math.radians(elev_deg)
        cos_elev = math.cos(elev_rad)
        sin_elev = math.sin(elev_rad)
        azimuth_step = 360.0 / n_azimuth
        for i in range(n_azimuth):
            azimuth = heading + math.radians(i * azimuth_step - 180.0)
            dx_w = cos_elev * math.cos(azimuth)
            dy_w = cos_elev * math.sin(azimuth)
            dz_w = sin_elev
            dx_b = dx_w * cos_h + dy_w * sin_h
            dy_b = -dx_w * sin_h + dy_w * cos_h
            dz_b = dz_w
            dx_bt = dx_b * cos_tilt - dz_b * sin_tilt
            dz_bt = dx_b * sin_tilt + dz_b * cos_tilt
            direction = np.array(
                [dx_bt * cos_h - dy_b * sin_h, dx_bt * sin_h + dy_b * cos_h, dz_bt],
                dtype=np.float64,
            )
            geom_id = np.zeros(1, dtype=np.int32)
            dist = _mj.mj_ray(
                model, data, pos_lidar, direction, None, 1, pelvis_bid, geom_id
            )
            is_self = int(geom_id[0]) in robot_geom_ids
            hit_valid = dist > 0 and dist < 15.0 and not is_self
            if hit_valid:
                px = pos_lidar[0] + dist * direction[0]
                py = pos_lidar[1] + dist * direction[1]
                pz = pos_lidar[2] + dist * direction[2]
                points_3d.append((float(px), float(py), float(pz), 0.0))
            if elev_deg == 0:
                if is_self and dist > 0 and dist < 0.1:
                    near_zero_hits += 1
                if hit_valid:
                    mid_ring_ranges.append(float(dist))
                else:
                    mid_ring_ranges.append(float("inf"))
    return mid_ring_ranges, points_3d, near_zero_hits


def _golden_go2(model, data, *, pos_lidar, heading, robot_body_id, robot_geom_ids):
    """Verbatim copy of the original mujoco_go2._update_lidar loop (max_range 12.0)."""
    import mujoco as mj

    tilt_rad = math.radians(-20.0)
    cos_tilt = math.cos(tilt_rad)
    sin_tilt = math.sin(tilt_rad)
    n_azimuth = 360
    elevations = list(range(-8, 53, 2))
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    mid_ring_ranges: list[float] = []
    points_3d: list[tuple[float, float, float, float]] = []

    for elev_deg in elevations:
        elev_rad = math.radians(elev_deg)
        cos_elev = math.cos(elev_rad)
        sin_elev = math.sin(elev_rad)
        azimuth_step = 360.0 / n_azimuth
        for i in range(n_azimuth):
            azimuth = heading + math.radians(i * azimuth_step - 180)
            dx_w = cos_elev * math.cos(azimuth)
            dy_w = cos_elev * math.sin(azimuth)
            dz_w = sin_elev
            dx_b = dx_w * cos_h + dy_w * sin_h
            dy_b = -dx_w * sin_h + dy_w * cos_h
            dz_b = dz_w
            dx_bt = dx_b * cos_tilt - dz_b * sin_tilt
            dz_bt = dx_b * sin_tilt + dz_b * cos_tilt
            direction = np.array(
                [dx_bt * cos_h - dy_b * sin_h, dx_bt * sin_h + dy_b * cos_h, dz_bt],
                dtype=np.float64,
            )
            geom_id = np.zeros(1, dtype=np.int32)
            dist = mj.mj_ray(
                model, data, pos_lidar, direction, None, 1, robot_body_id, geom_id
            )
            if dist > 0 and dist < 12.0 and int(geom_id[0]) not in robot_geom_ids:
                px = pos_lidar[0] + dist * direction[0]
                py = pos_lidar[1] + dist * direction[1]
                pz = pos_lidar[2] + dist * direction[2]
                points_3d.append((float(px), float(py), float(pz), 0.0))
            if elev_deg == 0:
                if dist > 0 and int(geom_id[0]) not in robot_geom_ids:
                    mid_ring_ranges.append(float(dist))
                else:
                    mid_ring_ranges.append(float("inf"))
    return mid_ring_ranges, points_3d


# ----------------------------------------------------------------------------
# g1 params (tilt -10, elevations sorted(set([0]+range(-8,30,3))), max_range 15)
# ----------------------------------------------------------------------------
_G1_ELEVATIONS = sorted(set([0] + list(range(-8, 30, 3))))


def test_raycast_lidar_matches_g1_golden(tiny_world):
    """Shared loop with g1 params == verbatim original g1 loop, byte-identical."""
    model, data = tiny_world.model, tiny_world.data
    pos_lidar = np.array([0.0, 0.0, 1.5], dtype=np.float64)
    heading = 0.3  # arbitrary non-zero yaw to exercise the rotation math

    g_mid, g_pts, g_nz = _golden_g1(
        model, data, pos_lidar=pos_lidar, heading=heading,
        pelvis_bid=-1, robot_geom_ids=set(),
    )
    result = raycast_lidar(
        tiny_world.raycast,
        pos_lidar=pos_lidar, heading=heading,
        exclude_bid=-1, robot_bodies=frozenset(),
        tilt_deg=-10.0, elevations=_G1_ELEVATIONS,
        n_azimuth=360, max_range=15.0,
        mid_ring_apply_max_range=True,
    )

    assert result.mid_ring_ranges == g_mid
    assert result.points_3d == g_pts
    assert result.near_zero_self_hits == g_nz
    # Sanity: the tiny MJCF has walls, so we expect real hits.
    assert any(math.isfinite(r) for r in result.mid_ring_ranges)
    assert len(result.points_3d) > 0


# ----------------------------------------------------------------------------
# go2 params (tilt -20, elevations range(-8,53,2), max_range 12, NO mid bound)
# ----------------------------------------------------------------------------
def test_raycast_lidar_matches_go2_golden(tiny_world):
    """Shared loop with go2 params == verbatim original go2 loop, byte-identical."""
    model, data = tiny_world.model, tiny_world.data
    pos_lidar = np.array([0.0, 0.0, 0.7], dtype=np.float64)
    heading = -0.7

    g_mid, g_pts = _golden_go2(
        model, data, pos_lidar=pos_lidar, heading=heading,
        robot_body_id=-1, robot_geom_ids=set(),
    )
    result = raycast_lidar(
        tiny_world.raycast,
        pos_lidar=pos_lidar, heading=heading,
        exclude_bid=-1, robot_bodies=frozenset(),
        tilt_deg=-20.0, elevations=list(range(-8, 53, 2)),
        n_azimuth=360, max_range=12.0,
        mid_ring_apply_max_range=False,
    )

    assert result.mid_ring_ranges == g_mid
    assert result.points_3d == g_pts
    assert any(math.isfinite(r) for r in result.mid_ring_ranges)
    assert len(result.points_3d) > 0


def test_go2_mid_ring_has_no_max_range_bound(tiny_world):
    """Pin the subtle divergence: go2 mid-ring records ranges beyond max_range.

    With ``mid_ring_apply_max_range=False`` a hit farther than ``max_range``
    still produces a finite mid-ring entry (it is only excluded from the points
    cloud).  This is the exact go2 behaviour and MUST be preserved.
    """
    pos_lidar = np.array([0.0, 0.0, 0.7], dtype=np.float64)
    heading = 0.0

    # max_range deliberately tiny so the x=3 wall (~3 m ahead) is "beyond range".
    res_bounded = raycast_lidar(
        tiny_world.raycast, pos_lidar=pos_lidar, heading=heading,
        exclude_bid=-1, robot_bodies=frozenset(),
        tilt_deg=0.0, elevations=[0], n_azimuth=360, max_range=1.0,
        mid_ring_apply_max_range=True,
    )
    res_unbounded = raycast_lidar(
        tiny_world.raycast, pos_lidar=pos_lidar, heading=heading,
        exclude_bid=-1, robot_bodies=frozenset(),
        tilt_deg=0.0, elevations=[0], n_azimuth=360, max_range=1.0,
        mid_ring_apply_max_range=False,
    )

    finite_bounded = sum(1 for r in res_bounded.mid_ring_ranges if math.isfinite(r))
    finite_unbounded = sum(1 for r in res_unbounded.mid_ring_ranges if math.isfinite(r))
    # Walls are ~3 m away; bounded (1 m) sees none, unbounded sees them.
    assert finite_bounded == 0
    assert finite_unbounded > 0


def test_self_hits_are_filtered_by_body_like_the_old_geom_set(tiny_world):
    """The self-filter is by BODY now (world.raycast returns the hit body): the robot's
    bodies filter exactly what the old geom-id set of those bodies filtered."""
    model, data = tiny_world.model, tiny_world.data
    pos_lidar = np.array([0.0, 0.0, 1.5], dtype=np.float64)
    world_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == 0}
    g_mid, g_pts, g_nz = _golden_g1(model, data, pos_lidar=pos_lidar, heading=0.3,
                                    pelvis_bid=-1, robot_geom_ids=world_geoms)
    result = raycast_lidar(
        tiny_world.raycast, pos_lidar=pos_lidar, heading=0.3, exclude_bid=-1,
        robot_bodies=frozenset({0}), tilt_deg=-10.0, elevations=_G1_ELEVATIONS,
        n_azimuth=360, max_range=15.0, mid_ring_apply_max_range=True,
    )
    assert (result.mid_ring_ranges, result.points_3d, result.near_zero_self_hits) == (g_mid, g_pts, g_nz)
    assert result.points_3d == [] and not any(math.isfinite(r) for r in result.mid_ring_ranges)
