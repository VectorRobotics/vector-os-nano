# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Lidar — the ONE lidar driver, built from a manifest ``sensors`` lidar entry and a
``SimRuntime``.

A scan is one ``world.raycast`` call over a polar grid (one ring per elevation, the
beams pitched by the tilt) from the mount on its body, cast off the physics lock on a
copy of one instant, and recast only when the cached one is ``period_s`` sim seconds
old. The geometry tested is the backend's own, so a lidar is never evidence.
"""
from __future__ import annotations

import math
import threading
from typing import Any, Callable, NamedTuple

import numpy as np

from vector_os_nano.schemas.robot import LidarSpec
from vector_os_nano.sim.runtime import SimRuntime
from vector_os_nano.sim.scene import canonical

_RANGE_MIN = 0.1  # m: the LaserScan's reported minimum


class RaycastResult(NamedTuple):
    """Raw output of :func:`raycast_lidar`: the elev 0 ring's ranges (inf = no return),
    ``(x, y, z, 0.0)`` world hits of every ring within range, and the elev 0 rays that
    hit the robot itself within 0.1 m (a diagnostic)."""

    mid_ring_ranges: list[float]
    points_3d: list[tuple[float, float, float, float]]
    near_zero_self_hits: int


def raycast_lidar(
    raycast: Callable[..., tuple[np.ndarray, np.ndarray]],
    *,
    pos_lidar: np.ndarray,
    heading: float,
    exclude_bid: int,
    robot_bodies: frozenset[int],
    tilt_deg: float,
    elevations: list[float],
    n_azimuth: int,
    max_range: float,
    mid_ring_apply_max_range: bool = True,
) -> RaycastResult:
    """Cast the multi-elevation polar grid in ONE ``raycast`` call.

    Azimuth index 0 points behind the robot (``heading - 180 deg``). A ray is built in
    the world, rotated into the heading frame, pitched by *tilt_deg* about its y axis
    and rotated back. Hits on *robot_bodies* are self-hits and dropped; *exclude_bid*'s
    own geometry is transparent. The elev 0 ring records every non-self hit, also beyond
    *max_range* unless *mid_ring_apply_max_range*.
    """
    tilt = math.radians(tilt_deg)
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    elev = np.radians(np.repeat(np.asarray(elevations, dtype=np.float64), n_azimuth))
    az_idx = np.tile(np.arange(n_azimuth, dtype=np.float64), len(elevations))
    azimuth = heading + np.radians(az_idx * (360.0 / n_azimuth) - 180.0)

    dx_w = np.cos(elev) * np.cos(azimuth)
    dy_w = np.cos(elev) * np.sin(azimuth)
    dz_w = np.sin(elev)
    dx_b = dx_w * cos_h + dy_w * sin_h
    dy_b = -dx_w * sin_h + dy_w * cos_h
    dx_bt = dx_b * math.cos(tilt) - dz_w * math.sin(tilt)
    dz_bt = dx_b * math.sin(tilt) + dz_w * math.cos(tilt)
    vec = np.stack([dx_bt * cos_h - dy_b * sin_h, dx_bt * sin_h + dy_b * cos_h, dz_bt], axis=1)

    pnt = np.asarray(pos_lidar, dtype=np.float64)
    dist, body = raycast(np.broadcast_to(pnt, vec.shape), vec, exclude_body=int(exclude_bid))

    is_self = np.isin(body, np.fromiter(robot_bodies, dtype=np.int64, count=len(robot_bodies)))
    hit = (dist > 0) & ~is_self
    valid = hit & (dist < max_range)
    pts = pnt + dist[valid, None] * vec[valid]
    mid = elev == 0.0
    mid_hit = valid[mid] if mid_ring_apply_max_range else hit[mid]
    return RaycastResult(
        mid_ring_ranges=np.where(mid_hit, dist[mid], np.inf).tolist(),
        points_3d=[(float(x), float(y), float(z), 0.0) for x, y, z in pts],
        near_zero_self_hits=int(np.count_nonzero(mid & is_self & (dist > 0) & (dist < 0.1))),
    )


class Lidar:
    def __init__(self, spec: LidarSpec, runtime: SimRuntime) -> None:
        scene = runtime.world.scene
        self._spec = spec
        self._rt = runtime
        self._scene = scene
        self._body = scene.body(spec.body)
        self._robot = scene.subtree(scene.robot_root)
        self._lock = threading.Lock()
        self._cache: tuple[float, Any, list] | None = None

    def scan(self) -> Any:
        """The latest 2-D ``core.types.LaserScan`` (the elev 0 ring)."""
        return self._latest()[1]

    def points(self) -> list[tuple[float, float, float, float]]:
        """The latest 3-D point cloud: ``(x, y, z, 0.0)`` world hits of every ring."""
        return self._latest()[2]

    def _latest(self) -> tuple[float, Any, list]:
        with self._lock:
            c = self._cache
            if c is not None and 0.0 <= self._rt.sim_time - c[0] < self._spec.period_s:
                return c
            self._cache = self._cast(self._rt.snapshot())
            return self._cache

    def _cast(self, snap: Any) -> tuple[float, Any, list]:
        from vector_os_nano.core.types import LaserScan

        s = self._spec
        pos, quat = canonical.body_poses(self._scene, snap.q)
        w, x, y, z = quat[self._body]
        heading = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        c, si = math.cos(heading), math.sin(heading)
        px, py, pz = s.pos
        origin = pos[self._body] + np.array([c * px - si * py, si * px + c * py, pz])
        raw = raycast_lidar(
            self._rt.world.raycast, pos_lidar=origin, heading=heading, exclude_bid=self._body,
            robot_bodies=self._robot, tilt_deg=s.tilt_deg, elevations=list(s.elevations_deg),
            n_azimuth=s.n_azimuth, max_range=s.range, mid_ring_apply_max_range=False,
        )
        scan = LaserScan(
            timestamp=snap.sim_time, angle_min=-math.pi, angle_max=math.pi,
            angle_increment=math.radians(360.0 / s.n_azimuth), range_min=_RANGE_MIN, range_max=s.range,
            ranges=tuple(raw.mid_ring_ranges),
        )
        return snap.sim_time, scan, raw.points_3d
