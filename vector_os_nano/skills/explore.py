# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""ExploreSkill — autonomous frontier exploration (campaign #8 R6).

Drives a REAL sensor-built occupancy grid: observe the lidar, pick the nearest
UNKNOWN-boundary (frontier) cell from the live map, route there obstacle-aware,
repeat until a coverage target is met or no frontier remains. Every goal is
derived from the map the robot actually sensed — NEVER a hardcoded waypoint
(the owner's exploration red line). The 'autonomously explore' half of req #5;
'go to the target object's point' is NavigateToPointSkill (label/coordinate).
"""
from __future__ import annotations

import logging

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)

# The include_misses lidar already maps ~0.6 of the room from spawn, so the
# target must exceed that to force real exploration (drive to frontiers).
_DEFAULT_COVERAGE = 0.85
_DEFAULT_MAX_ITERS = 6
_FRONTIER_TOL = 0.5       # arrive loosely at a frontier — it is a viewpoint
_FRONTIER_MIN_DIST = 1.2  # only drive to frontiers farther than this, so the
                          # robot actually MOVES (a frontier within arrival tol
                          # would 'arrive' without motion → map never grows)


@skill(
    aliases=["explore", "探索", "探索房间", "自主探索", "explore the room"],
    direct=False,
)
class ExploreSkill:
    """Autonomously explore the room by visiting lidar-map frontiers."""

    name: str = "explore"
    description: str = (
        "Autonomously explore the environment: build an occupancy map from the "
        "lidar and drive to unknown-region frontiers until the room is covered. "
        "Use for 'explore / 探索房间 / 自主探索' instructions. Real sensor-driven "
        "coverage — not a fixed patrol route."
    )
    typical_duration_sec: float = 300.0   # many biped legs; floor only
    verify_hint: str = (
        "step_output('coverage') > step_output('start_coverage') — the lidar "
        "map MUST have grown (real exploration happened, never a no-op)"
    )
    parameters: dict = {
        "coverage_target": {
            "type": "number", "required": False, "default": _DEFAULT_COVERAGE,
            "description": "Stop once this fraction of the grid is known (0-1).",
        },
        "max_iters": {
            "type": "number", "required": False, "default": _DEFAULT_MAX_ITERS,
            "description": "Max frontier legs before stopping.",
        },
    }
    preconditions: list = ["base connected with an occupancy grid (room mode)"]
    postconditions: list = []
    effects: dict = {}
    failure_modes: list = ["no_base", "no_occupancy"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        base = context.base
        if base is None:
            return SkillResult(success=False, error_message="no mobile base connected",
                               result_data={"diagnosis": "no_base"})
        if not callable(getattr(base, "get_occupancy", None)) or \
                not callable(getattr(base, "observe", None)):
            return SkillResult(
                success=False,
                error_message=("base has no occupancy grid — explore needs a "
                               "room scene (--scenario g1_room) with a lidar"),
                result_data={"diagnosis": "no_occupancy"})
        grid = base.get_occupancy()
        if grid is None:
            return SkillResult(
                success=False,
                error_message=("no occupancy grid — explore needs room mode "
                               "(--scenario g1_room)"),
                result_data={"diagnosis": "no_occupancy"})

        target = float(params.get("coverage_target", _DEFAULT_COVERAGE) or _DEFAULT_COVERAGE)
        max_iters = int(params.get("max_iters", _DEFAULT_MAX_ITERS) or _DEFAULT_MAX_ITERS)

        start_coverage = base.observe()      # first look from the spawn pose
        coverage = start_coverage
        visited: list = []
        reason = "max_iters"
        for _ in range(max_iters):
            coverage = base.observe()
            if coverage >= target:
                reason = "coverage_reached"
                break
            pos = base.get_position()
            frontier = grid.nearest_frontier_world(
                float(pos[0]), float(pos[1]), min_dist=_FRONTIER_MIN_DIST)
            if frontier is None:
                reason = "fully_explored"
                break
            fx, fy = float(frontier[0]), float(frontier[1])
            out = base.navigate_to(fx, fy, _FRONTIER_TOL)
            visited.append([round(fx, 2), round(fy, 2)])
            if out.get("reason") in ("fell", "unreachable"):
                # skip an unreachable frontier; the map already marked it, so
                # nearest_frontier_world returns a different one next loop.
                continue
        coverage = base.observe()
        grew = coverage > start_coverage + 1e-6
        return SkillResult(
            success=bool(grew or reason in ("coverage_reached", "fully_explored")),
            error_message=("" if grew else "exploration did not grow the map"),
            result_data={
                "coverage": round(coverage, 3),
                "start_coverage": round(start_coverage, 3),
                "frontiers_visited": visited,
                "reason": reason,
                "transport": "sim_oracle",
            },
        )
