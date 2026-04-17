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

    def __init__(self, base: Any = None, scene_graph: Any = None) -> None:
        self._base = base
        self._sg = scene_graph

    def get_context_block(self) -> dict[str, str]:
        """Return Anthropic system block with current robot state."""
        lines: list[str] = []
        has_hardware = self._base is not None or self._sg is not None

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

        # Nav/explore state (lightweight checks, no ROS2 dependency)
        try:
            from vector_os_nano.skills.go2.explore import is_exploring, is_nav_stack_running
            lines.append(f"Exploring: {'yes' if is_exploring() else 'no'}")
            lines.append(f"Nav stack: {'running' if is_nav_stack_running() else 'stopped'}")
        except ImportError:
            pass

        if not has_hardware:
            return {"type": "text", "text": "[Robot State]\nNo hardware connected."}

        return {"type": "text", "text": "[Robot State]\n" + "\n".join(lines)}


def _heading_to_compass(degrees: float) -> str:
    """Convert heading in degrees to compass direction."""
    # Normalize to 0-360
    d = degrees % 360
    if d < 0:
        d += 360
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                   "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(d / 22.5) % 16
    return directions[idx]
