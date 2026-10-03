# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Shared lidar polar-grid ray cast (STAGE 3a de-triplication; one mj_multiRay call).

ONE impl per capability (CLAUDE.md Rule 11). The identical multi-elevation
``mj_ray`` polar-grid loop previously lived in THREE places:

* ``sensors/g1_lidar.py`` :func:`g1_lidar_scan`        (g1 driver — live)
* ``mujoco_go2.py`` ``MuJoCoGo2._update_lidar``        (go2 driver — live)
* ``sensors/lidar360.py`` ``MuJoCoLivox360``           (PAUSED SysNav abstraction)

This module extracts the per-ray math VERBATIM from the two live copies so both
drivers share it.  Only the ray loop is shared — the LaserScan / diagnostics /
storage assembly stays per-driver, so each driver's external output remains
byte-for-byte identical to before this refactor.

The two original loops were byte-identical in the per-ray math (direction
construction, body-frame rotation, pitch tilt, ``mj_ray`` self-filter via
``robot_geom_ids`` + ``bodyexclude``) and differed ONLY in:

* ``max_range`` (g1 = 15.0, go2 = 12.0) — points-cloud upper bound;
* whether the elev==0 mid-ring applies that upper bound (g1: yes, go2: no — go2
  appends any ``dist > 0 and not self`` to the mid-ring regardless of range);
* the near-zero self-hit diagnostic counter (g1 only; go2 ignores it).

These divergences are surfaced as explicit parameters so the ONE loop reproduces
each driver's exact behaviour.
"""
from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np

__all__ = ["RaycastResult", "raycast_lidar"]


class RaycastResult(NamedTuple):
    """Raw output of the shared lidar ray loop (pre-assembly).

    Attributes:
        mid_ring_ranges: Per-azimuth range list for the elev==0 ring, in the
            same order the loop visits azimuth indices.  Hits are ``float``
            ranges; misses are ``float('inf')``.  Whether the ``max_range``
            upper bound gates a hit here depends on ``mid_ring_apply_max_range``.
        points_3d: ``(x, y, z, 0.0)`` world-frame hit points for every valid,
            non-self ray within ``max_range`` (all elevation rings).
        near_zero_self_hits: Count of elev==0 rays that hit a self geom within
            0.1 m (g1 diagnostic only; go2 leaves it unused).
    """

    mid_ring_ranges: list[float]
    points_3d: list[tuple[float, float, float, float]]
    near_zero_self_hits: int


def raycast_lidar(
    model: Any,
    data: Any,
    *,
    pos_lidar: np.ndarray,
    heading: float,
    exclude_bid: int,
    robot_geom_ids: set[int],
    tilt_deg: float,
    elevations: list[int],
    n_azimuth: int,
    max_range: float,
    mid_ring_apply_max_range: bool = True,
) -> RaycastResult:
    """Cast the multi-elevation polar grid in ONE ``mj_multiRay`` call; raw results.

    Same rays, same per-ray rules as the original per-ray ``mj_ray`` loop of the
    two live drivers (golden-tested), vectorised: the directions are built with
    numpy and every ray is cast by one C call that releases the GIL, so a scan no
    longer holds the interpreter for ~35 ms. Only the diverging knobs
    (``max_range``, ``mid_ring_apply_max_range``) are parameterised.

    Args:
        model: ``mujoco.MjModel`` for the compiled scene.
        data: ``mujoco.MjData`` for the current sim state.
        pos_lidar: ``(3,)`` float64 world position of the virtual lidar origin.
        heading: Robot yaw (radians); index 0 of the azimuth sweep is behind the
            robot (``heading - 180 deg``).
        exclude_bid: Body id passed to ``mj_multiRay`` as *bodyexclude*.
        robot_geom_ids: Geom ids that count as self-hits (filtered out).
        tilt_deg: Beam pitch tilt in degrees (g1 = -10, go2 = -20).
        elevations: Elevation rings in degrees (must include 0 for the mid ring).
        n_azimuth: Azimuth samples per ring.
        max_range: Points-cloud (and, if ``mid_ring_apply_max_range``, mid-ring)
            upper range bound in metres.
        mid_ring_apply_max_range: If True (g1), the elev==0 mid-ring hit also
            requires ``dist < max_range``; if False (go2), the mid-ring records
            any ``dist > 0 and not self`` regardless of range.

    Returns:
        :class:`RaycastResult` — the raw per-ray output, for the caller to
        assemble into its own LaserScan / diagnostics.
    """
    import mujoco as _mj  # noqa: PLC0415

    tilt = math.radians(tilt_deg)
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    elev = np.radians(np.repeat(np.asarray(elevations, dtype=np.float64), n_azimuth))
    az_idx = np.tile(np.arange(n_azimuth, dtype=np.float64), len(elevations))
    azimuth = heading + np.radians(az_idx * (360.0 / n_azimuth) - 180.0)

    # Ray direction in world frame (no tilt yet) -> body frame (rotate by
    # -heading) -> pitch tilt about body Y -> back to world (rotate by +heading).
    dx_w = np.cos(elev) * np.cos(azimuth)
    dy_w = np.cos(elev) * np.sin(azimuth)
    dz_w = np.sin(elev)
    dx_b = dx_w * cos_h + dy_w * sin_h
    dy_b = -dx_w * sin_h + dy_w * cos_h
    dx_bt = dx_b * math.cos(tilt) - dz_w * math.sin(tilt)
    dz_bt = dx_b * math.sin(tilt) + dz_w * math.cos(tilt)
    vec = np.stack([dx_bt * cos_h - dy_b * sin_h, dx_bt * sin_h + dy_b * cos_h, dz_bt], axis=1)

    n = vec.shape[0]
    geom_id = np.full(n, -1, dtype=np.int32)
    dist = np.full(n, -1.0, dtype=np.float64)
    pnt = np.asarray(pos_lidar, dtype=np.float64)
    _mj.mj_multiRay(model, data, pnt, vec.reshape(-1), None, 1, int(exclude_bid),
                    geom_id, dist, None, n, _mj.mjMAXVAL)

    is_self = np.isin(geom_id, np.fromiter(robot_geom_ids, dtype=np.int32, count=len(robot_geom_ids)))
    hit = (dist > 0) & ~is_self
    valid = hit & (dist < max_range)
    pts = pnt + dist[valid, None] * vec[valid]
    points_3d = [(float(x), float(y), float(z), 0.0) for x, y, z in pts]

    mid = elev == 0.0
    mid_hit = valid[mid] if mid_ring_apply_max_range else hit[mid]
    mid_ring_ranges = np.where(mid_hit, dist[mid], np.inf).tolist()
    near_zero_hits = int(np.count_nonzero(mid & is_self & (dist > 0) & (dist < 0.1)))

    return RaycastResult(
        mid_ring_ranges=mid_ring_ranges,
        points_3d=points_3d,
        near_zero_self_hits=near_zero_hits,
    )
