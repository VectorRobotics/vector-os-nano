# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Visibility-graph planner (PURE geometry, no MuJoCo import).

Hand-built polygon fixtures: shortest path around a wall, +inf when the goal is
inside an obstacle (unless snapped), +inf when boxed in, a too-narrow gap routes
the long way, straight-shot when unobstructed, an exact Minkowski inflation, goal
snapping and start-in-collision escape. plan_path and path_length share one
computation.
"""
from __future__ import annotations

import math

from vector_os_nano.sim.nav import vgraph as vg


def test_no_mujoco_import():
    import vector_os_nano.sim.nav.vgraph as m
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

    def test_inflated_box_is_the_box_with_grown_half_extents(self):
        inf = vg.inflate_polygon(vg.box_polygon(1, 2, 3.0, 0.05, 0.3), 0.28)
        want = vg.box_polygon(1, 2, 3.28, 0.33, 0.3)
        assert all(math.dist(a, b) < 1e-9 for a, b in zip(inf, want))

    def test_every_inflated_edge_is_at_least_r_from_the_original(self):
        # skills-nav:vgraph-underinflation / sim-locomotion:vgraph-under-inflation:
        # the radial push gave a thin wall ~3 mm of margin instead of r.
        r = 0.28
        shapes = [
            vg.box_polygon(0, 0, 3.0, 0.05),          # thin wall
            vg.box_polygon(4, 1, 0.5, 0.5, 0.7),      # rotated square
            list(reversed(vg.box_polygon(-3, 2, 0.84, 0.45))),  # CW winding
            vg.cylinder_polygon(2, -3, 0.2),
        ]
        for poly in shapes:
            inf = vg.inflate_polygon(poly, r)
            for i in range(len(inf)):
                a, b = inf[i - 1], inf[i]
                for t in (0.0, 0.1, 0.5, 0.9, 1.0):
                    p = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
                    assert vg.distance_to_polygon(p, poly) >= r - 1e-9


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


class TestSnapAndEscape:
    def test_goal_inside_obstacle_snaps_to_nearest_free_point(self):
        table = vg.box_polygon(0, 0, 0.84, 0.45)
        path, length = vg.plan_path((-3, 0), (0, 0.2), [table], inflation=0.3, snap_goal=True)
        assert path is not None and math.isfinite(length)
        tx, ty = path[-1]
        # nearest free point is just past the top edge (0.45 + 0.3), straight above
        assert abs(tx) < 1e-6 and 0.75 <= ty <= 0.77
        assert not vg.point_in_polygon(path[-1], vg.inflate_polygon(table, 0.3))

    def test_snapped_goal_under_overlapping_polygons_is_free(self):
        a = vg.box_polygon(0, 0, 1.0, 1.0)
        b = vg.box_polygon(1.5, 0, 1.0, 1.0)
        inflated = [vg.inflate_polygon(p, 0.3) for p in (a, b)]
        snapped = vg.nearest_free((0.9, 0.0), inflated)
        assert snapped is not None and not vg.point_in_any(snapped, inflated)

    def test_start_in_collision_escapes_first(self):
        # sim-locomotion:go2-nav-bool-collapse: a start inside the inflation was a
        # hard 'unreachable'; now the path leaves straight to the nearest free point.
        box = vg.box_polygon(0, 0, 0.5, 0.5)
        start = (0.0, 0.6)  # 0.1 m from the face, inside the 0.3 m inflation
        path, length = vg.plan_path(start, (0, 4), [box], inflation=0.3)
        assert path is not None and path[0] == start
        assert not vg.point_in_any(path[1], [vg.inflate_polygon(box, 0.3)])
        assert abs(length - (4 - 0.6)) < 0.05

    def test_boxed_in_goal_stays_unreachable_when_snapped(self):
        walls = [
            vg.box_polygon(0, 2, 3, 0.2),
            vg.box_polygon(0, -2, 3, 0.2),
            vg.box_polygon(2, 0, 0.2, 3),
            vg.box_polygon(-2, 0, 0.2, 3),
        ]
        path, length = vg.plan_path((10, 10), (0, 0), walls, inflation=0.3, snap_goal=True)
        assert path is None and length == float("inf")


def test_obstacle_reader_enum_values_match_mujoco():
    import mujoco

    assert vg.GEOM_BOX == int(mujoco.mjtGeom.mjGEOM_BOX)
    assert vg.GEOM_CYLINDER == int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    assert vg.JNT_FREE == int(mujoco.mjtJoint.mjJNT_FREE)
