# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Robot state context provider for LLM system prompt.

Collects real-time robot state (position, room, SceneGraph, nav state)
and formats it as an Anthropic system prompt block. Injected into
build_system_prompt() on every LLM turn.
"""
from __future__ import annotations

import math
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RobotContextProvider:
    """Collects robot state and formats as Anthropic system block."""

    def __init__(self, base: Any = None, scene_graph: Any = None, arm: Any = None) -> None:
        self._base = base
        self._sg = scene_graph
        self._arm = arm

    def get_context_block(self) -> dict[str, str]:
        """Return Anthropic system block with current robot state."""
        lines: list[str] = []
        has_hardware = (
            self._base is not None or self._sg is not None or self._arm is not None
        )

        # Position + heading from base
        if self._base is not None:
            try:
                pos = self._base.get_position()
                lines.append(f"Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.2f})")
            except Exception:
                pass
            try:
                heading = self._base.get_heading()
                deg = math.degrees(heading)
                compass = _heading_to_compass(deg)
                lines.append(f"Heading: {deg:.0f} deg ({compass})")
            except Exception:
                pass
            # Current room from SceneGraph
            if self._sg is not None:
                try:
                    pos = self._base.get_position()
                    room = self._sg.nearest_room(pos[0], pos[1])
                    lines.append(f"Current room: {room or 'unknown'}")
                except Exception:
                    pass

        # SceneGraph stats
        if self._sg is not None:
            try:
                stats = self._sg.stats()
                doors = self._sg.get_all_doors()
                lines.append(
                    f"SceneGraph: {stats['rooms']} rooms "
                    f"({stats['visited_rooms']} visited), "
                    f"{len(doors)} doors, {stats['objects']} objects"
                )
            except Exception:
                pass
            try:
                summary = self._sg.get_room_summary()
                if summary:
                    lines.append(f"Rooms: {summary[:200]}")
            except Exception:
                pass

        # Nav/explore state — only meaningful for a mobile base (Go2), not an
        # arm-only sim, so don't inject quadruped fields into an arm prompt.
        if self._base is not None:
            try:
                from vector_os_nano.skills.go2.explore import is_exploring, is_nav_stack_running
                lines.append(f"Exploring: {'yes' if is_exploring() else 'no'}")
                lines.append(f"Nav stack: {'running' if is_nav_stack_running() else 'stopped'}")
            except ImportError:
                pass

        # Arm state (an arm-only sim has no base/scene_graph; still real hardware)
        if self._arm is not None:
            name = getattr(self._arm, "name", type(self._arm).__name__)
            dof = getattr(self._arm, "dof", None)
            lines.append(f"Arm: {name}" + (f" ({dof}-DOF)" if dof else "") + " connected")
            try:
                pos = self._arm.get_joint_positions()
                lines.append("Joints: " + ", ".join(f"{v:.2f}" for v in pos))
            except Exception:
                pass

        if not has_hardware:
            return {"type": "text", "text": "[Robot State]\nNo hardware connected."}

        return {"type": "text", "text": "[Robot State]\n" + "\n".join(lines)}


def _heading_to_compass(degrees: float) -> str:
    """Compass label of a world yaw in degrees (0 = +x = east, CCW positive).

    The same convention as where_am_i (skills-nav:compass-mismatch: this used to
    call yaw 0 "N" while where_am_i called it "east").
    """
    from vector_os_nano.skills.go2.where_am_i import _heading_label

    return _heading_label(math.radians(degrees))
