# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""WhereAmISkill -- report the robot's current position and room.

Reads position and heading from context.base. The room is the world layout's
floor box that contains the position (geometry, config/worlds/apartment.yaml), else
the SceneGraph's nearest room. Heading labels use the world convention the
whole agent shares: yaw 0 = +x = east, counter-clockwise positive.
"""
from __future__ import annotations

import logging
import math

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.sim.nav.coverage import room_at, room_layout
from vector_os_nano.skills.navigate import _detect_current_room

logger = logging.getLogger(__name__)

# Heading sector labels keyed by (min_deg_inclusive, max_deg_exclusive).
# The west sector spans [157.5, 202.5) which, after atan2 normalization to
# (-180, 180], includes both [157.5, 180] and (-180, -157.5).
_HEADING_SECTORS: tuple[tuple[float, float, str], ...] = (
    (-22.5,   22.5,  "east"),
    ( 22.5,   67.5,  "northeast"),
    ( 67.5,  112.5,  "north"),
    (112.5,  157.5,  "northwest"),
    (157.5,  180.1,  "west"),     # 180.1 to include exactly pi radians (180.0 deg)
    (-180.1, -157.5, "west"),
    (-157.5, -112.5, "southwest"),
    (-112.5,  -67.5, "south"),
    ( -67.5,  -22.5, "southeast"),
)


def _heading_label(radians: float) -> str:
    """Convert heading in radians to a compass label."""
    degrees = math.degrees(radians)
    for low, high, label in _HEADING_SECTORS:
        if low <= degrees < high:
            return label
    return "unknown"


@skill(
    aliases=["where am i", "where", "我在哪", "在哪", "我在哪里", "位置", "location"],
    actuates=False,
)
class WhereAmISkill:
    """Report the robot's current position and which room it is in."""

    name: str = "where_am_i"
    description: str = (
        "Report the robot's current position and which room it is in."
    )
    parameters: dict = {}
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {}
    failure_modes: list[str] = ["no_base", "position_unavailable"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        """Query position and heading, determine current room.

        Args:
            params: Unused.
            context: SkillContext with base attached.

        Returns:
            SkillResult with result_data containing room, position, and heading.
        """
        if context.base is None:
            logger.error("[WHERE_AM_I] No base connected")
            return SkillResult(
                success=False,
                error_message="No base connected",
                diagnosis_code="no_base",
            )

        try:
            pos = context.base.get_position()
        except Exception as exc:
            logger.error("[WHERE_AM_I] get_position failed: %s", exc)
            return SkillResult(
                success=False,
                error_message=f"Could not read position: {exc}",
                diagnosis_code="position_unavailable",
            )

        try:
            heading_rad: float = float(context.base.get_heading())
        except Exception as exc:
            logger.warning("[WHERE_AM_I] get_heading failed: %s", exc)
            heading_rad = 0.0

        x = float(pos[0])
        y = float(pos[1])
        z = float(pos[2]) if len(pos) > 2 else 0.0

        sg = context.services.get("spatial_memory") if context.services else None
        layout = room_layout()
        room = room_at(x, y, layout["boxes"]) or _detect_current_room(x, y, sg=sg)
        room_node = sg.get_room(room) if (sg is not None and room != "unknown") else None
        room_center = (
            (room_node.center_x, room_node.center_y) if room_node is not None
            else layout["rooms"].get(room)
        )
        compass = _heading_label(heading_rad)

        logger.info(
            "[WHERE_AM_I] room=%s pos=(%.2f, %.2f, %.2f) heading=%.2f rad (%s)",
            room, x, y, z, heading_rad, compass,
        )

        return SkillResult(
            success=True,
            result_data={
                "room": room,
                "position": [round(x, 2), round(y, 2), round(z, 2)],
                "heading_rad": round(heading_rad, 3),
                "heading": compass,
                "room_center": (
                    [round(room_center[0], 1), round(room_center[1], 1)]
                    if room_center is not None
                    else None
                ),
            },
        )
