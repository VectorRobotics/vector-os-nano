# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Shared lidar polar-grid ray-casting loop (STAGE 3a de-triplication).

ONE impl per capability (CLAUDE.md Rule 11). The identical multi-elevation
``mj_ray`` polar-grid loop previously lived in two drivers:

* ``sensors/g1_lidar.py`` :func:`g1_lidar_scan`        (g1 driver — live)
* ``mujoco_go2.py`` ``MuJoCoGo2._update_lidar``        (go2 driver — live)

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
    """Run the multi-elevation polar-grid ``mj_ray`` loop, return raw results.

    The loop body is copied VERBATIM from the two live drivers; only the
    diverging knobs (``max_range``, ``mid_ring_apply_max_range``) are
    parameterised.  No LaserScan / diagnostics assembly happens here.

    Args:
        model: ``mujoco.MjModel`` for the compiled scene.
        data: ``mujoco.MjData`` for the current sim state.
        pos_lidar: ``(3,)`` float64 world position of the virtual lidar origin.
        heading: Robot yaw (radians); index 0 of the azimuth sweep is behind the
            robot (``heading - 180 deg``).
        exclude_bid: Body id passed to ``mj_ray`` as *bodyexclude*.
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

    tilt_rad = math.radians(tilt_deg)
    cos_tilt = math.cos(tilt_rad)
    sin_tilt = math.sin(tilt_rad)

    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    mid_ring_ranges: list[float] = []
    points_3d: list[tuple[float, float, float, float]] = []
    near_zero_hits: int = 0

    for elev_deg in elevations:
        elev_rad = math.radians(elev_deg)
        cos_elev = math.cos(elev_rad)
        sin_elev = math.sin(elev_rad)
        azimuth_step = 360.0 / n_azimuth
        for i in range(n_azimuth):
            azimuth = heading + math.radians(i * azimuth_step - 180.0)

            # Ray direction in world frame (no tilt yet)
            dx_w = cos_elev * math.cos(azimuth)
            dy_w = cos_elev * math.sin(azimuth)
            dz_w = sin_elev

            # World -> body frame (rotate by -heading around Z)
            dx_b = dx_w * cos_h + dy_w * sin_h    # forward
            dy_b = -dx_w * sin_h + dy_w * cos_h   # left
            dz_b = dz_w                            # up

            # Apply pitch tilt in body frame (rotate around body Y axis)
            dx_bt = dx_b * cos_tilt - dz_b * sin_tilt
            dz_bt = dx_b * sin_tilt + dz_b * cos_tilt

            # Body -> world frame (rotate by +heading)
            direction = np.array(
                [
                    dx_bt * cos_h - dy_b * sin_h,
                    dx_bt * sin_h + dy_b * cos_h,
                    dz_bt,
                ],
                dtype=np.float64,
            )

            geom_id = np.zeros(1, dtype=np.int32)
            dist = _mj.mj_ray(
                model,
                data,
                pos_lidar,
                direction,
                None,
                1,
                exclude_bid,
                geom_id,
            )

            is_self = int(geom_id[0]) in robot_geom_ids
            hit_valid = dist > 0 and dist < max_range and not is_self

            if hit_valid:
                px = pos_lidar[0] + dist * direction[0]
                py = pos_lidar[1] + dist * direction[1]
                pz = pos_lidar[2] + dist * direction[2]
                points_3d.append((float(px), float(py), float(pz), 0.0))

            if elev_deg == 0:
                if is_self and dist > 0 and dist < 0.1:
                    near_zero_hits += 1
                if mid_ring_apply_max_range:
                    if hit_valid:
                        mid_ring_ranges.append(float(dist))
                    else:
                        mid_ring_ranges.append(float("inf"))
                else:
                    if dist > 0 and not is_self:
                        mid_ring_ranges.append(float(dist))
                    else:
                        mid_ring_ranges.append(float("inf"))

    return RaycastResult(
        mid_ring_ranges=mid_ring_ranges,
        points_3d=points_3d,
        near_zero_self_hits=near_zero_hits,
    )
