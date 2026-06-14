# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #7 R1 — visibility-graph planner (PURE geometry, no MuJoCo).

The R1 gate: a deterministic pure function. Hand-built polygon fixtures —
shortest path around a wall, +inf when the goal is inside an obstacle,
+inf when boxed in, a too-narrow gap routes the long way, straight-shot
when unobstructed. plan_path and path_length share one computation (the
verify/execution single-source invariant).
"""
from __future__ import annotations

import math

from vector_os_nano.hardware.sim import g1_vgraph as vg


def test_no_mujoco_import():
    import vector_os_nano.hardware.sim.g1_vgraph as m
    # the module must never import mujoco (pure geometry, offline-testable)
    assert "mujoco" not in dir(m)
    src = (__import__("pathlib").Path(m.__file__)).read_text()
    assert "import mujoco" not in src


class TestShapes:
    def test_box_polygon_corners(self):
        poly = vg.box_polygon(0, 0, 1, 1)
        assert set(poly) == {(-1, -1), (1, -1), (1, 1), (-1, 1)}

    def test_cylinder_is_circumscribed(self):
        poly = vg.cylinder_polygon(0, 0, 1.0, n=8)
        # every vertex is OUTSIDE the true radius (circumscribed, conservative)
        assert all(math.hypot(x, y) >= 1.0 - 1e-9 for x, y in poly)

    def test_inflate_pushes_outward(self):
        poly = vg.box_polygon(0, 0, 1, 1)
        inf = vg.inflate_polygon(poly, 0.5)
        # corners move further from centre
        for (x, y) in inf:
            assert math.hypot(x, y) > math.hypot(1, 1)


class TestPointPredicates:
    def test_point_inside_box(self):
        poly = vg.box_polygon(0, 0, 1, 1)
        assert vg.point_in_polygon((0, 0), poly) is True
        assert vg.point_in_polygon((5, 5), poly) is False

    def test_segment_clear_blocked_by_box(self):
        poly = vg.box_polygon(0, 0, 1, 1)
        assert vg.segment_clear((-3, 0), (3, 0), [poly]) is False  # through it
        assert vg.segment_clear((-3, 3), (3, 3), [poly]) is True   # above it


class TestPlanning:
    def test_straight_shot_when_unobstructed(self):
        path, length = vg.plan_path((0, 0), (5, 0), [], inflation=0.4)
        assert path == [(0, 0), (5, 0)]
        assert abs(length - 5.0) < 1e-6

    def test_routes_around_a_wall(self):
        # a vertical wall blocking the straight line from (0,0) to (4,0)
        wall = vg.box_polygon(2.0, 0.0, 0.2, 1.0)   # spans y in [-1, 1] at x=2
        path, length = vg.plan_path((0, 0), (4, 0), [wall], inflation=0.3)
        assert path is not None
        # detour is longer than the 4 m straight line
        assert length > 4.0
        # every waypoint clears the REAL obstacle (waypoints sit ON the
        # inflated margin boundary by construction — that IS the safe corner;
        # the safety property is staying out of the un-inflated wall).
        assert all(not vg._strictly_inside(p, wall) for p in path)

    def test_goal_inside_obstacle_is_unreachable(self):
        box = vg.box_polygon(0, 0, 1, 1)
        path, length = vg.plan_path((-3, 0), (0, 0), [box], inflation=0.3)
        assert path is None
        assert length == float("inf")

    def test_boxed_in_start_is_unreachable(self):
        # four walls fully enclosing the goal
        walls = [
            vg.box_polygon(0, 2, 3, 0.2),
            vg.box_polygon(0, -2, 3, 0.2),
            vg.box_polygon(2, 0, 0.2, 3),
            vg.box_polygon(-2, 0, 0.2, 3),
        ]
        path, length = vg.plan_path((10, 10), (0, 0), walls, inflation=0.3)
        assert path is None and length == float("inf")

    def test_narrow_gap_routes_the_long_way(self):
        # two boxes leave a gap too narrow for the inflation radius -> the
        # planner must NOT thread it; it routes around the whole pair.
        top = vg.box_polygon(2.0, 0.6, 0.3, 0.5)     # y in [0.1, 1.1]
        bot = vg.box_polygon(2.0, -0.6, 0.3, 0.5)    # y in [-1.1, -0.1]
        # gap is y in (-0.1, 0.1) = 0.2 m wide; inflation 0.3 closes it
        narrow, _ = vg.plan_path((0, 0), (4, 0), [top, bot], inflation=0.3)
        wide, _ = vg.plan_path((0, 0), (4, 0), [top, bot], inflation=0.05)
        assert narrow is not None and wide is not None
        # the wide-gap plan may thread closer to centre; the narrow one detours
        n_len = sum(math.hypot(narrow[i+1][0]-narrow[i][0],
                               narrow[i+1][1]-narrow[i][1])
                    for i in range(len(narrow)-1))
        assert n_len > 4.0   # had to go around, not straight through

    def test_path_length_matches_plan(self):
        wall = vg.box_polygon(2.0, 0.0, 0.2, 1.0)
        path, length = vg.plan_path((0, 0), (4, 0), [wall], inflation=0.3)
        assert vg.path_length((0, 0), (4, 0), [wall], inflation=0.3) == length
