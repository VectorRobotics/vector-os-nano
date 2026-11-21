# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Obstacle-aware navigation on the shared planner (pure, no GL/sim window).

Tests are OFFLINE: no MuJoCo GL context, no live robot.

Coverage:
  1. plan_path routes AROUND a box blocking the straight line.
  2. plan_path returns (None, inf) when goal is inside an inflated box.
  3. obstacles_from_model with a tiny hand-built MuJoCo model:
     - group=3 box (furniture proxy) → included
     - plain collision wall → included
     - pick_table box (default contype) → included
     - freejoint pickable cylinder → excluded
     - floor plane → excluded (type=plane, skipped by gtype filter)
     - g1_pelvis box → excluded (robot_geom_ids)
     - baseboard (bb_) geom → excluded (name prefix)
  4. obstacles_from_model returns >=1 obstacle (smoke test).
  5. navigate_to signature and the NavResult contract are intact (import-only).
"""
from __future__ import annotations

import math

import pytest

from vector_os_nano.sim.nav import vgraph as vg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _euclidean(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------------------------------------------------------------------
# Tests 1–2: pure planner geometry (no mujoco)
# ---------------------------------------------------------------------------


class TestPlannerAroundObstacle:
    """plan_path routes around a box on the straight line start → goal."""

    def test_waypoint_chain_longer_than_straight_line(self) -> None:
        """A box placed dead-centre forces a detour: path length > euclidean."""
        start = (0.0, 0.0)
        goal = (4.0, 0.0)
        # Wall blocking x=2, spans y in [-1, 1]
        wall = vg.box_polygon(2.0, 0.0, 0.2, 1.0)
        path, length = vg.plan_path(start, goal, [wall], inflation=0.3)

        assert path is not None, "Planner should find a route around the wall"
        euclidean = _euclidean(start, goal)
        assert length > euclidean, (
            f"Detour path ({length:.3f} m) must exceed straight line ({euclidean:.3f} m)"
        )

    def test_mid_waypoints_exist(self) -> None:
        """Path around the wall has intermediate waypoints (not just start + goal)."""
        start = (0.0, 0.0)
        goal = (4.0, 0.0)
        wall = vg.box_polygon(2.0, 0.0, 0.2, 1.0)
        path, _ = vg.plan_path(start, goal, [wall], inflation=0.3)

        assert path is not None
        assert len(path) > 2, (
            f"Expected intermediate waypoints, got path with only {len(path)} points: {path}"
        )

    def test_path_starts_and_ends_correctly(self) -> None:
        """Path[0] == start and path[-1] == goal (within float tolerance)."""
        start = (0.0, 0.0)
        goal = (4.0, 0.0)
        wall = vg.box_polygon(2.0, 0.0, 0.2, 1.0)
        path, _ = vg.plan_path(start, goal, [wall], inflation=0.3)

        assert path is not None
        assert _euclidean(path[0], start) < 1e-6
        assert _euclidean(path[-1], goal) < 1e-6

    def test_no_waypoint_inside_uninfated_wall(self) -> None:
        """No waypoint lies strictly inside the un-inflated wall (the planner routes clear)."""
        start = (0.0, 0.0)
        goal = (4.0, 0.0)
        wall = vg.box_polygon(2.0, 0.0, 0.2, 1.0)
        path, _ = vg.plan_path(start, goal, [wall], inflation=0.3)

        assert path is not None
        for pt in path:
            assert not vg._strictly_inside(pt, wall), (
                f"Waypoint {pt} is inside the wall"
            )


class TestPlannerGoalInsideObstacle:
    """plan_path returns (None, inf) when goal is inside an inflated obstacle."""

    def test_goal_inside_box_is_unreachable(self) -> None:
        box = vg.box_polygon(3.0, 0.0, 0.8, 0.8)
        path, length = vg.plan_path((0.0, 0.0), (3.0, 0.0), [box], inflation=0.3)
        assert path is None
        assert length == float("inf")

    def test_goal_inside_inflated_margin_is_unreachable(self) -> None:
        """Goal is OUTSIDE the raw box but INSIDE the inflated margin → unreachable."""
        # box half-extents 0.5; inflation 0.4; goal at (3.0, 0.0) with box centred
        # at (2.5, 0.0) → goal is 0.5 m from centre, inside inflated polygon.
        box = vg.box_polygon(2.5, 0.0, 0.5, 0.5)
        path, length = vg.plan_path((0.0, 0.0), (2.5, 0.0), [box], inflation=0.4)
        assert path is None
        assert length == float("inf")

    def test_boxed_in_returns_none(self) -> None:
        """Four walls fully enclosing goal → unreachable."""
        walls = [
            vg.box_polygon(0, 2, 3, 0.2),
            vg.box_polygon(0, -2, 3, 0.2),
            vg.box_polygon(2, 0, 0.2, 3),
            vg.box_polygon(-2, 0, 0.2, 3),
        ]
        path, length = vg.plan_path((10, 10), (0, 0), walls, inflation=0.3)
        assert path is None
        assert length == float("inf")


# ---------------------------------------------------------------------------
# Test 3: obstacles_from_model with a tiny hand-built MuJoCo model
# ---------------------------------------------------------------------------


def _build_tiny_model():
    """Build a minimal MuJoCo model that exercises all obstacle filter paths.

    Bodies:
      - sofa_body:      group=3 box (furniture collision proxy) → INCLUDE
      - wall_south:     plain box, default contype              → INCLUDE
      - pick_table:     plain box, default contype              → INCLUDE
      - pickable_bottle: freejoint cylinder                     → EXCLUDE
      - g1_pelvis:      robot box (g1_ prefix)                  → EXCLUDE
      - baseboard:      box with name bb_south                  → EXCLUDE
    Geoms:
      - floor:          type=plane                              → EXCLUDE (gtype filter)
    """
    import mujoco

    xml = """
<mujoco model="tiny_test">
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" pos="0 0 0"/>

    <body name="sofa_body" pos="2.0 1.0 0">
      <geom type="box" size="1.1 0.45 0.45" pos="0 0 0.45"
            rgba="0 0 0 0" group="3"/>
    </body>

    <body name="wall_south" pos="10.0 0.0 0">
      <geom type="box" size="10.1 0.05 1.4" pos="0 0 1.4"/>
    </body>

    <body name="pick_table" pos="10.95 3.0 0">
      <geom type="box" size="0.15 0.25 0.14" pos="0 0 0.14"/>
    </body>

    <body name="pickable_bottle" pos="10.90 2.78 0.32">
      <freejoint/>
      <geom type="cylinder" size="0.028 0.04"/>
    </body>

    <body name="g1_pelvis" pos="10.0 3.0 0.793">
      <geom type="box" size="0.15 0.15 0.15"/>
    </body>

    <body name="bb_south_body" pos="10.0 0.06 0">
      <geom name="bb_south" type="box" size="10.0 0.01 0.04" pos="0 0 0.04"/>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class TestObstaclesFromModel:
    """obstacles_from_model correctly includes/excludes geoms."""

    @pytest.fixture(scope="class")
    def tiny_result(self):
        """Build the tiny model once and run obstacles_from_model."""
        import mujoco
        obstacles_from_model = vg.obstacles_from_model

        model, data = _build_tiny_model()
        # Build robot_geom_ids for g1_pelvis
        robot_geom_ids: set[int] = set()
        for gid in range(model.ngeom):
            bid = model.geom_bodyid[gid]
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if bname.startswith("g1_"):
                robot_geom_ids.add(gid)

        polys = obstacles_from_model(model, data.geom_xpos, data.geom_xmat, robot_geom_ids=robot_geom_ids)
        return polys, model, data

    def test_returns_list(self, tiny_result) -> None:
        polys, _, _ = tiny_result
        assert isinstance(polys, list)

    def test_at_least_one_obstacle(self, tiny_result) -> None:
        """Must include at least sofa_body, wall_south, pick_table."""
        polys, _, _ = tiny_result
        assert len(polys) >= 3, (
            f"Expected >=3 obstacle polygons, got {len(polys)}"
        )

    def test_all_polygons_are_lists_of_tuples(self, tiny_result) -> None:
        polys, _, _ = tiny_result
        for poly in polys:
            assert isinstance(poly, list), f"Polygon is not a list: {type(poly)}"
            for pt in poly:
                assert len(pt) == 2, f"Point must be 2-tuple, got {pt}"

    def test_floor_plane_excluded(self, tiny_result) -> None:
        """floor is type=plane → filtered out before the inclusion check."""
        import mujoco
        polys, model, data = tiny_result
        # The floor polygon would be centred at (0,0); if it were included it would
        # be a 4-corner box with enormous extents.  Verify by checking no polygon
        # spans more than _OBS_MAX_HALF_EXTENT (9 m).
        for poly in polys:
            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]
            span_x = max(xs) - min(xs)
            span_y = max(ys) - min(ys)
            assert span_x < 40.0 and span_y < 40.0, (
                f"Suspicious large polygon — likely floor included: span_x={span_x}, span_y={span_y}"
            )

    def test_g1_robot_geom_excluded(self, tiny_result) -> None:
        """g1_pelvis box is excluded (g1_ prefix + robot_geom_ids)."""
        # g1_pelvis is centred near (10.0, 3.0) — if included it would produce a
        # small polygon near that point.  We verify no polygon has centroid
        # extremely close to (10.0, 3.0, 0.793) AND size matching 0.15×0.15.
        import mujoco
        polys, model, data = tiny_result
        for poly in polys:
            cx = sum(pt[0] for pt in poly) / len(poly)
            cy = sum(pt[1] for pt in poly) / len(poly)
            # pelvis at (10, 3) — check no polygon has its centroid within 0.1 m
            # AND spans less than 0.5 m (the tiny pelvis box)
            if math.hypot(cx - 10.0, cy - 3.0) < 0.2:
                xs = [pt[0] for pt in poly]
                span = max(xs) - min(xs)
                # If this were the pelvis box it would be ~0.3 m span (inflated 0)
                # but larger furniture (wall_south at y=0.0, pick_table at 10.95)
                # would not land at (10, 3).  Assert span is large enough to be
                # something else (e.g. pick_table at 10.95).
                # Actually pick_table is at (10.95, 3.0) → centroid close to (10,3)?
                # No, 10.95 vs 10.0 → diff=0.95 > 0.2. So nothing should be here.
                assert False, (
                    f"Unexpected polygon near g1_pelvis at ({cx:.2f}, {cy:.2f}): {poly}"
                )

    def test_freejoint_pickable_excluded(self, tiny_result) -> None:
        """pickable_bottle has freejoint — must not appear in obstacles."""
        # pickable_bottle is at (10.90, 2.78) — check no polygon centroid near there
        polys, _, _ = tiny_result
        for poly in polys:
            cx = sum(pt[0] for pt in poly) / len(poly)
            cy = sum(pt[1] for pt in poly) / len(poly)
            assert not (math.hypot(cx - 10.90, cy - 2.78) < 0.15), (
                f"Pickable bottle seems to be included: polygon centroid at ({cx:.2f}, {cy:.2f})"
            )

    def test_baseboard_excluded(self, tiny_result) -> None:
        """bb_south baseboard (name prefix bb_) is excluded.

        We verify this by counting: bb_south_body has size=(10.0, 0.01, 0.04).
        If it were included, it would produce a polygon spanning ~20 m in x but
        only ~0.02 m in y.  Count polygons with span_y < 0.05 AND span_x > 5 m —
        there should be none (wall_south has span_y = 0.10 from size[1]=0.05).
        """
        polys, _, _ = tiny_result
        suspicious = []
        for poly in polys:
            cy_vals = [pt[1] for pt in poly]
            cx_vals = [pt[0] for pt in poly]
            span_y = max(cy_vals) - min(cy_vals)
            span_x = max(cx_vals) - min(cx_vals)
            # baseboard signature: nearly zero y-span AND very long x-span
            if span_y < 0.05 and span_x > 5.0:
                suspicious.append(poly)
        assert len(suspicious) == 0, (
            f"Baseboard-like polygon(s) found (should be excluded): {suspicious}"
        )


# ---------------------------------------------------------------------------
# Test 4: smoke test with no robot_geom_ids argument
# ---------------------------------------------------------------------------


def test_obstacles_from_model_without_robot_ids_includes_the_robot() -> None:
    """With no robot_geom_ids there is no name-prefix guess: the robot box counts."""
    model, data = _build_tiny_model()
    polys = vg.obstacles_from_model(model, data.geom_xpos, data.geom_xmat)
    assert isinstance(polys, list)
    near_pelvis = [p for p in polys
                   if math.hypot(sum(q[0] for q in p) / len(p) - 10.0,
                                 sum(q[1] for q in p) / len(p) - 3.0) < 0.2]
    assert near_pelvis, "the g1_ prefix hack is gone: only robot_geom_ids excludes the robot"


# ---------------------------------------------------------------------------
# Test 5: navigate_to signature and _G1NavResult contract (import-only)
# ---------------------------------------------------------------------------


def test_navigate_to_signature_intact() -> None:
    """navigate_to signature matches the base contract (no sim instantiation)."""
    import inspect
    from vector_os_nano.sim.drivers.base import LeggedBase

    sig = inspect.signature(LeggedBase.navigate_to)
    params = list(sig.parameters)
    assert "x" in params
    assert "y" in params
    assert "tol" in params
    assert "speed" in params
    assert "timeout" in params


def test_nav_result_bool_semantics() -> None:
    """NavResult truthiness reflects reached; dict accessors work."""
    from vector_os_nano.sim.nav.follow import NavResult

    r_reached = NavResult({"reached": True, "moved_m": 2.0, "reason": "arrived"})
    r_fell = NavResult({"reached": False, "moved_m": 0.5, "reason": "fell"})
    r_unreachable = NavResult({"reached": False, "moved_m": 0.0, "reason": "unreachable"})

    assert bool(r_reached) is True
    assert bool(r_fell) is False
    assert bool(r_unreachable) is False

    assert r_reached.get("reason") == "arrived"
    assert r_unreachable.reason == "unreachable"


def test_g1_body_radius_is_manifest_config() -> None:
    """The G1 planning radius is the manifest's nav.body_radius (Rule 11)."""
    from vector_os_nano.embodiments.config import load_embodiment_config

    assert load_embodiment_config("g1").nav["body_radius"] == pytest.approx(0.30)
