# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #8 R6 — ExploreSkill: autonomous frontier exploration.

The skill drives a REAL sensor-built occupancy grid: observe() integrates the
lidar, nearest_frontier_world() picks the next unknown boundary, navigate_to
routes there (obstacle-aware), repeat until coverage threshold or no frontier
remains. NO hardcoded waypoints (owner red line) — every goal comes from the
live map. Tested with a fake base + scripted grid (no sim/policy needed).
"""
from __future__ import annotations

from types import SimpleNamespace

from vector_os_nano.skills.explore import ExploreSkill


class _FakeGrid:
    """A grid whose frontier/coverage follow a scripted sequence."""

    def __init__(self, frontiers, coverages):
        self._frontiers = list(frontiers)   # list of (x,y) or None per call
        self._coverages = list(coverages)
        self._ci = 0

    def nearest_frontier_world(self, x, y, min_dist=0.0):
        return self._frontiers.pop(0) if self._frontiers else None

    def coverage(self):
        c = self._coverages[min(self._ci, len(self._coverages) - 1)]
        self._ci += 1
        return c


class _FakeBase:
    def __init__(self, grid):
        self._grid = grid
        self.nav_calls = []
        self.observe_calls = 0

    def observe(self):
        self.observe_calls += 1
        return self._grid.coverage()

    def get_occupancy(self):
        return self._grid

    def get_position(self):
        return [0.0, 0.0, 0.77]

    def navigate_to(self, x, y, tol=0.45):
        self.nav_calls.append((x, y))
        return {"reached": True, "remaining": 0.2, "moved_m": 1.0,
                "pos": [x, y, 0.77], "transport": "sim_oracle"}


def _ctx(base):
    return SimpleNamespace(base=base, world_model=None, agent=None)


class TestExplore:
    def test_visits_frontiers_until_none_left(self):
        grid = _FakeGrid(frontiers=[(2.0, 0.0), (3.0, 1.0), None],
                         coverages=[0.1, 0.3, 0.5, 0.5])
        base = _FakeBase(grid)
        res = ExploreSkill().execute({"coverage_target": 0.99, "max_iters": 10}, _ctx(base))
        assert res.success is True
        # navigated to the two real frontiers, then stopped (None)
        assert base.nav_calls == [(2.0, 0.0), (3.0, 1.0)]
        assert base.observe_calls >= 2
        assert res.result_data["reason"] == "fully_explored"

    def test_stops_at_coverage_target(self):
        grid = _FakeGrid(frontiers=[(2.0, 0.0), (3.0, 1.0), (4.0, 0.0)],
                         coverages=[0.2, 0.85, 0.95])
        base = _FakeBase(grid)
        res = ExploreSkill().execute({"coverage_target": 0.8, "max_iters": 10}, _ctx(base))
        assert res.success is True
        assert res.result_data["coverage"] >= 0.8
        assert res.result_data["reason"] == "coverage_reached"

    def test_no_base_fails(self):
        res = ExploreSkill().execute({}, SimpleNamespace(base=None, world_model=None))
        assert res.success is False

    def test_base_without_occupancy_fails_loud(self):
        base = SimpleNamespace(get_occupancy=lambda: None, observe=lambda: 0.0)
        res = ExploreSkill().execute({}, _ctx(base))
        assert res.success is False
        assert "occupancy" in (res.error_message or "").lower() or \
               "explore" in (res.error_message or "").lower()
