# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #8 R5 — lidar occupancy grid (pure, offline, no mujoco/GL).

A 2D occupancy grid built by ray-marching lidar hits: the cell at each hit is
OCCUPIED, cells along the ray from the sensor to the hit are FREE, the rest
stay UNKNOWN. Foundation for frontier exploration (R6) and a real nav-stack
costmap. Pure numpy — unit-tests run without a sim.
"""
from __future__ import annotations

import numpy as np

from vector_os_nano.hardware.sim.occupancy import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    OccupancyGrid,
)


class TestGridBasics:
    def test_starts_all_unknown(self):
        g = OccupancyGrid(x_min=-5, y_min=-5, x_max=5, y_max=5, resolution=0.25)
        assert g.coverage() == 0.0
        cx, cy = g.world_to_cell(0.0, 0.0)
        assert g.cell(cx, cy) == UNKNOWN

    def test_world_to_cell_in_bounds(self):
        g = OccupancyGrid(x_min=0, y_min=0, x_max=10, y_max=10, resolution=1.0)
        assert g.world_to_cell(0.5, 0.5) == (0, 0)
        assert g.world_to_cell(9.5, 9.5) == (9, 9)
        assert g.in_bounds(*g.world_to_cell(5.0, 5.0)) is True
        assert g.in_bounds(999, 999) is False


class TestRayMarch:
    def test_hit_marks_occupied_and_ray_free(self):
        g = OccupancyGrid(x_min=-1, y_min=-1, x_max=5, y_max=5, resolution=0.25)
        # sensor at origin, one hit straight ahead at (3, 0)
        pts = np.array([[3.0, 0.0, 0.2, 1.0]], dtype=np.float32)
        g.update_from_scan((0.0, 0.0), pts)
        hx, hy = g.world_to_cell(3.0, 0.0)
        assert g.cell(hx, hy) == OCCUPIED
        # a cell partway along the ray is FREE
        mx, my = g.world_to_cell(1.5, 0.0)
        assert g.cell(mx, my) == FREE
        # coverage rose above zero
        assert g.coverage() > 0.0

    def test_occupied_beats_free_on_conflict(self):
        g = OccupancyGrid(x_min=-1, y_min=-1, x_max=5, y_max=5, resolution=0.5)
        pts = np.array([[2.0, 0.0, 0.2, 1.0]], dtype=np.float32)
        g.update_from_scan((0.0, 0.0), pts)
        hx, hy = g.world_to_cell(2.0, 0.0)
        # a later ray that passes THROUGH the hit cell must not downgrade it
        g.update_from_scan((0.0, 0.0), np.array([[4.0, 0.0, 0.2, 1.0]], dtype=np.float32))
        assert g.cell(hx, hy) == OCCUPIED


class TestFrontiers:
    def test_frontier_is_free_cell_touching_unknown(self):
        g = OccupancyGrid(x_min=-1, y_min=-1, x_max=5, y_max=5, resolution=0.25)
        g.update_from_scan((0.0, 0.0), np.array([[3.0, 0.0, 0.2, 1.0]], dtype=np.float32))
        fr = g.frontiers()
        assert len(fr) > 0
        # every frontier is a FREE cell with at least one UNKNOWN 4-neighbour
        for (cx, cy) in fr:
            assert g.cell(cx, cy) == FREE
            nbrs = [g.cell(cx + dx, cy + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            assert UNKNOWN in nbrs

    def test_nearest_frontier_world_returns_a_point_or_none(self):
        g = OccupancyGrid(x_min=-1, y_min=-1, x_max=5, y_max=5, resolution=0.25)
        # no scan yet → no frontier
        assert g.nearest_frontier_world(0.0, 0.0) is None
        g.update_from_scan((0.0, 0.0), np.array([[3.0, 0.0, 0.2, 1.0]], dtype=np.float32))
        f = g.nearest_frontier_world(0.0, 0.0)
        assert f is not None and len(f) == 2
