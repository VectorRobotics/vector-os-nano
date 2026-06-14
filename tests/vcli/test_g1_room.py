# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #8 R3 — g1_room scene + obstacle enumeration + obstacle-aware nav.

The room adds REAL collision geometry (walls + obstacle boxes) the gait must
physically route around; g1_room.obstacles_from_model enumerates that geometry
from the COMPILED MjModel (named obstacle_*/wall_*, NOT target_*), and
g1_vgraph.plan_path routes around it. Pure-geometry tests need no policy; the
room-build test needs the downloaded g1 assets (skipped without them).
"""
from __future__ import annotations

import math

import pytest

from vector_os_nano.hardware.sim import g1_room
from vector_os_nano.hardware.sim import g1_vgraph as vg
from vector_os_nano.hardware.sim.mujoco_g1 import g1_assets_ready

pytestmark = pytest.mark.skipif(
    not g1_assets_ready(),
    reason="g1_gait assets not installed (run scripts/setup_g1_gait.sh)")


@pytest.fixture(scope="module")
def room():
    """Compiled g1 room model + a forwarded data (geom_xpos populated)."""
    import mujoco

    from vector_os_nano.hardware.sim import g1_room
    from vector_os_nano.hardware.sim.mujoco_g1 import _ASSET_DIR
    m = g1_room.build_room_model(_ASSET_DIR)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


class TestRoomBuild:
    def test_targets_and_obstacles_present(self, room):
        import mujoco
        m, _ = room
        names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
                 for i in range(m.nbody)}
        assert {"obstacle_center", "obstacle_left", "obstacle_right"} <= names
        assert {"target_red", "target_blue", "target_green"} <= names
        assert {"wall_back", "wall_front", "wall_left", "wall_right"} <= names

    def test_target_position_lookup(self):
        from vector_os_nano.hardware.sim import g1_room
        assert g1_room.target_position("red") == (3.7, 0.0)
        assert g1_room.target_position("target_blue") == (3.7, 2.0)
        assert g1_room.target_position("nope") is None


class TestObstaclesFromModel:
    def test_enumerates_obstacles_and_walls_not_targets(self, room):
        m, d = room
        obs = g1_room.obstacles_from_model(m, d)
        # 3 obstacles + 4 walls = 7 routing polygons; targets are NOT obstacles
        assert len(obs) == 7
        # every polygon is a list of (x, y) corners
        for poly in obs:
            assert len(poly) >= 4
            assert all(len(pt) == 2 for pt in poly)

    def test_center_obstacle_polygon_matches_geometry(self, room):
        m, d = room
        obs = g1_room.obstacles_from_model(m, d, names_only={"obstacle_center"})
        assert len(obs) == 1
        xs = [p[0] for p in obs[0]]
        ys = [p[1] for p in obs[0]]
        # obstacle_center is at (1.7,0.9) half (0.25,0.5)
        assert min(xs) == pytest.approx(1.45, abs=0.02)
        assert max(xs) == pytest.approx(1.95, abs=0.02)
        assert min(ys) == pytest.approx(0.40, abs=0.02)
        assert max(ys) == pytest.approx(1.40, abs=0.02)


class TestScenarioRegistration:
    """No assets needed — registry/world contract for the g1_room scenario."""

    def test_g1_room_registered(self):
        from vector_os_nano.playground.catalog import get_scenario
        sc = get_scenario("g1_room")
        assert sc.embodiment == "g1"
        assert "target_red" in sc.object_names

    def test_g1_room_world_is_a_base_world(self):
        from vector_os_nano.playground.catalog import get_scenario
        from vector_os_nano.playground.world import PlaygroundWorld
        w = PlaygroundWorld(get_scenario("g1_room"))
        assert w.has_base() is True


class TestObstacleAwarePlanning:
    def test_route_around_center_obstacle_is_a_detour(self, room):
        m, d = room
        obs = g1_room.obstacles_from_model(
            m, d, names_only={"obstacle_center"})
        # obstacle_center is at (1.7,0.9); a line from spawn to (3.4,1.8) runs
        # straight through it → the planner must detour.
        start, goal = (0.0, 0.0), (3.4, 1.8)
        path, length = vg.plan_path(start, goal, obs, inflation=0.4)
        straight = math.hypot(goal[0] - start[0], goal[1] - start[1])
        assert path is not None                # reachable around it
        assert length > straight + 0.05        # a real detour, not the straight line
