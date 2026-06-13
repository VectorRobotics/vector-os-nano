# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #8 R5 — a 2D occupancy grid built from lidar scans.

PURE numpy — no mujoco/GL import, so it unit-tests offline and is the single
source of truth for both autonomous exploration (frontier selection, R6) and a
real nav-stack costmap. Ray-march model: the cell under each lidar hit is
OCCUPIED; cells along the ray from the sensor to the hit are FREE; everything
else stays UNKNOWN. OCCUPIED is sticky — a later free-ray never downgrades a
known obstacle (conservative, matches a real occupancy filter's log-odds floor
without the float bookkeeping).

A FRONTIER is a FREE cell with an UNKNOWN 4-neighbour — the boundary of the
explored region, the natural target for "explore the unknown" (real
sensor-driven coverage, never hardcoded waypoints).
"""
from __future__ import annotations

import math

import numpy as np

UNKNOWN = 0
FREE = 1
OCCUPIED = 2


class OccupancyGrid:
    """Axis-aligned 2D grid over [x_min, x_max] x [y_min, y_max]."""

    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float,
                 resolution: float = 0.2) -> None:
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.res = float(resolution)
        self.nx = max(1, int(math.ceil((x_max - x_min) / self.res)))
        self.ny = max(1, int(math.ceil((y_max - y_min) / self.res)))
        self.grid = np.full((self.nx, self.ny), UNKNOWN, dtype=np.uint8)

    # -- coordinate transforms --------------------------------------------
    def world_to_cell(self, x: float, y: float) -> "tuple[int, int]":
        return (int((x - self.x_min) / self.res),
                int((y - self.y_min) / self.res))

    def cell_to_world(self, cx: int, cy: int) -> "tuple[float, float]":
        return (self.x_min + (cx + 0.5) * self.res,
                self.y_min + (cy + 0.5) * self.res)

    def in_bounds(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.nx and 0 <= cy < self.ny

    def cell(self, cx: int, cy: int) -> int:
        if not self.in_bounds(cx, cy):
            return UNKNOWN
        return int(self.grid[cx, cy])

    # -- updates -----------------------------------------------------------
    def _mark(self, cx: int, cy: int, value: int) -> None:
        if not self.in_bounds(cx, cy):
            return
        # OCCUPIED is sticky: never let a free-ray erase a known obstacle.
        if self.grid[cx, cy] == OCCUPIED and value == FREE:
            return
        self.grid[cx, cy] = value

    def _ray_free(self, c0: "tuple[int, int]", c1: "tuple[int, int]") -> None:
        """Bresenham: mark cells from c0 toward c1 FREE, EXCLUDING the endpoint
        (the endpoint is the hit cell, marked OCCUPIED by the caller)."""
        x0, y0 = c0
        x1, y1 = c1
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while (x0, y0) != (x1, y1):
            self._mark(x0, y0, FREE)
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def update_from_scan(self, sensor_xy: "tuple[float, float]",
                         points: np.ndarray) -> None:
        """Integrate one lidar scan: ``points`` is (N, >=2) world-frame XY(...);
        ``sensor_xy`` is the lidar origin in world frame."""
        if points is None or len(points) == 0:
            return
        s = self.world_to_cell(sensor_xy[0], sensor_xy[1])
        pts = np.asarray(points)
        for p in pts:
            h = self.world_to_cell(float(p[0]), float(p[1]))
            self._ray_free(s, h)
            self._mark(h[0], h[1], OCCUPIED)

    # -- queries -----------------------------------------------------------
    def coverage(self) -> float:
        """Fraction of cells that are no longer UNKNOWN."""
        known = int(np.count_nonzero(self.grid != UNKNOWN))
        return known / float(self.nx * self.ny)

    def frontiers(self) -> "list[tuple[int, int]]":
        """FREE cells with at least one UNKNOWN 4-neighbour."""
        out: list = []
        free = np.argwhere(self.grid == FREE)
        for cx, cy in free:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if self.cell(int(cx) + dx, int(cy) + dy) == UNKNOWN:
                    out.append((int(cx), int(cy)))
                    break
        return out

    def nearest_frontier_world(self, x: float, y: float
                               ) -> "tuple[float, float] | None":
        """World (x, y) of the FREE-frontier cell nearest to (x, y), or None
        when fully explored (no frontier)."""
        fr = self.frontiers()
        if not fr:
            return None
        best = min(fr, key=lambda c: (self.cell_to_world(*c)[0] - x) ** 2
                   + (self.cell_to_world(*c)[1] - y) ** 2)
        return self.cell_to_world(*best)
