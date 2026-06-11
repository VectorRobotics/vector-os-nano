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
        "label": {
            "type": "string",
            "required": False,
            "description": (
                "Target OBJECT label (e.g. 'sofa'). PREFER this for any "
                "'go to the <object>' instruction: the skill resolves the "
                "object's live coordinates from the world model and FAILS "
                "LOUDLY if no such object is known — never invent x/y for "
                "an object."
            ),
        },
        "x": {
            "type": "number",
            "required": False,
            "description": "Target world x coordinate in metres (coordinate-style goals).",
        },
        "y": {
            "type": "number",
            "required": False,
            "description": "Target world y coordinate in metres (coordinate-style goals).",
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

        # Semantic goal: resolve the label against the LIVE world model — the
        # navigate analogue of named-grasp #2b. An unknown object fails LOUDLY
        # (never silently navigate to invented coordinates: that is a FALSE
        # SUCCESS — went somewhere, reported arrival at a phantom).
        label = str(params.get("label", "") or "").strip()
        if label:
            wm = getattr(context, "world_model", None)
            matches = wm.get_objects_by_label(label) if wm is not None else []
            if not matches:
                known = sorted({o.label for o in wm.get_objects()}) if wm else []
                return SkillResult(
                    success=False,
                    error_message=(
                        f"no object matching {label!r} in the live world model; "
                        f"known objects: {known or '<none>'}"
                    ),
                    result_data={"diagnosis": "object_not_found", "label": label},
                )
            best = max(matches, key=lambda o: o.confidence)
            x, y = float(best.x), float(best.y)
        else:
            try:
                x, y = float(params["x"]), float(params["y"])
            except (KeyError, TypeError, ValueError):
                return SkillResult(
                    success=False,
                    error_message="navigate_to requires a label OR numeric x and y",
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
