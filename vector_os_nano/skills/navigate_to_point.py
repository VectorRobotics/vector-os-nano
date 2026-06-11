# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""NavigateToPointSkill — shortest-path navigation to world (x, y) — M3.

Base-generic: any connected base exposing ``navigate_to(x, y, tol)`` (the
habitat kinematic base today; a future nav-stack base tomorrow) can run it.
The skill's success comes from the base's honest outcome, and the suggested
verify is the deterministic VLN criterion ``geodesic_dist(x, y) < 0.5`` —
the planner binds the SAME coordinates into params and verify, so a stuck
or unreachable navigation can never false-pass.
"""
from __future__ import annotations

import logging

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)


@skill(
    aliases=["navigate to", "go to", "走到", "导航到", "去坐标"],
    direct=False,
)
class NavigateToPointSkill:
    """Navigate the mobile base to a world coordinate along the navmesh."""

    name: str = "navigate_to"
    description: str = (
        "Navigate the mobile base to world coordinates (x, y) following the "
        "shortest navigable path. Use for any 'go to position / 走到坐标' "
        "instruction with explicit coordinates."
    )
    # Kinematic in sim, but a long path still takes wall time over the bridge.
    typical_duration_sec: float = 20.0
    # The deterministic VLN success criterion. Bind the SAME x, y here and in
    # strategy_params (the planner copies the literal coordinates).
    verify_hint: str = "geodesic_dist(x, y) < 0.5"
    parameters: dict = {
        "x": {
            "type": "number",
            "required": True,
            "description": "Target world x coordinate in metres.",
        },
        "y": {
            "type": "number",
            "required": True,
            "description": "Target world y coordinate in metres.",
        },
        "tol": {
            "type": "number",
            "required": False,
            "default": 0.2,
            "description": "Waypoint tolerance in metres.",
        },
    }
    preconditions: list = ["base connected and navigate_to-capable"]
    postconditions: list = []
    effects: dict = {}
    failure_modes: list = ["no_base", "no_navigate_support", "no_path", "nav_stuck"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        base = context.base
        if base is None:
            return SkillResult(
                success=False,
                error_message="no mobile base connected",
                result_data={"diagnosis": "no_base"},
            )
        if not callable(getattr(base, "navigate_to", None)):
            return SkillResult(
                success=False,
                error_message=(
                    f"base {type(base).__name__} has no navigate_to capability"
                ),
                result_data={"diagnosis": "no_navigate_support"},
            )
        try:
            x, y = float(params["x"]), float(params["y"])
        except (KeyError, TypeError, ValueError):
            return SkillResult(
                success=False,
                error_message="navigate_to requires numeric x and y",
                result_data={"diagnosis": "bad_params"},
            )
        tol = float(params.get("tol", 0.2) or 0.2)

        logger.info("[NAVIGATE_TO] -> (%.2f, %.2f) tol=%.2f", x, y, tol)
        out = base.navigate_to(x, y, tol)
        reached = bool(out.get("reached", False))
        remaining = float(out.get("remaining", float("inf")))
        result_data = {
            "target": [x, y],
            "reached": reached,
            "remaining_geodesic_m": remaining,
            "position": out.get("pos"),
            "diagnosis": "ok" if reached else str(out.get("reason", "nav_stuck")),
        }
        return SkillResult(
            success=reached,
            error_message="" if reached else (
                f"navigation stopped {remaining:.2f}m (geodesic) short of "
                f"({x:.2f}, {y:.2f})"
            ),
            result_data=result_data,
        )
