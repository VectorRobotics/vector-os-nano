"""Navigation primitives — wrap SceneGraph + NavStackClient for CaP-X generated code.

All functions are module-level and read from the module-global _ctx.
Requires init_primitives() to be called before use.
"""
from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

from vector_os_nano.vcli.primitives import PrimitiveContext

if TYPE_CHECKING:
    pass

_ctx: PrimitiveContext | None = None


def _require_scene_graph() -> object:
    """Return _ctx.scene_graph or raise RuntimeError if unavailable."""
    if _ctx is None or _ctx.scene_graph is None:
        raise RuntimeError(
            "No SceneGraph connected. Call init_primitives() with a valid scene_graph."
        )
    return _ctx.scene_graph


# ---------------------------------------------------------------------------
# Room queries
# ---------------------------------------------------------------------------


def nearest_room() -> str | None:
    """Current room name based on robot position, or None if unknown.

    Returns:
        Room name string or None if no rooms are known.

    Raises:
        RuntimeError: If no SceneGraph is connected.
    """
    from vector_os_nano.vcli.primitives import locomotion
    sg = _require_scene_graph()
    pos = locomotion.get_position()
    return sg.nearest_room(pos[0], pos[1])


# ---------------------------------------------------------------------------
# Goal sending
# ---------------------------------------------------------------------------


def publish_goal(x: float, y: float) -> None:
    """Send a navigation goal to the planner.

    Tries nav_client first; falls back to base.navigate_to if available.

    Args:
        x: Target x coordinate in world frame (meters).
        y: Target y coordinate in world frame (meters).

    Raises:
        RuntimeError: If neither nav_client nor base is available.
    """
    if _ctx is None:
        raise RuntimeError(
            "Primitives not initialized. Call init_primitives() first."
        )
    if _ctx.nav_client is not None:
        _ctx.nav_client.navigate_to(x, y)
    elif _ctx.base is not None and hasattr(_ctx.base, "navigate_to"):
        _ctx.base.navigate_to(x, y)
    else:
        raise RuntimeError(
            "No navigation interface available. Provide nav_client or a base with navigate_to."
        )


# ---------------------------------------------------------------------------
# Blocking wait
# ---------------------------------------------------------------------------


def wait_until_near(
    x: float,
    y: float,
    tolerance: float = 0.8,
    timeout: float = 60.0,
) -> bool:
    """Block until the robot is within tolerance of the target position.

    Polls at 2 Hz. Returns immediately if already within tolerance.

    Args:
        x: Target x coordinate in world frame (meters).
        y: Target y coordinate in world frame (meters).
        tolerance: Acceptable radius in meters. Default 0.8 m.
        timeout: Maximum wait time in seconds. Default 60 s.

    Returns:
        True if position reached within timeout, False on timeout.
    """
    from vector_os_nano.vcli.primitives import locomotion
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pos = locomotion.get_position()
        dx = pos[0] - x
        dy = pos[1] - y
        if math.sqrt(dx * dx + dy * dy) <= tolerance:
            return True
        time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------


def get_door_chain(from_room: str, to_room: str) -> list[tuple[float, float, str]]:
    """Get waypoints between rooms via BFS on the SceneGraph.

    Args:
        from_room: Source room name.
        to_room: Destination room name.

    Returns:
        List of (x, y, label) tuples. Empty list if no path found.

    Raises:
        RuntimeError: If no SceneGraph is connected.
    """
    sg = _require_scene_graph()
    return list(sg.get_door_chain(from_room, to_room))


def navigate_to_room(room: str) -> bool:
    """Navigate to a named room.

    Looks up a NavigateSkill in the skill registry if available, otherwise
    falls back to publishing a goal derived from the SceneGraph.

    Args:
        room: Target room name.

    Returns:
        True if the robot arrived, False on failure or timeout.

    Raises:
        RuntimeError: If neither skill_registry nor SceneGraph is available.
    """
    if _ctx is None:
        raise RuntimeError(
            "Primitives not initialized. Call init_primitives() first."
        )

    # Attempt via skill registry
    if _ctx.skill_registry is not None:
        registry = _ctx.skill_registry
        skill = None
        # Try common names for the navigate skill
        for candidate in ("navigate", "NavigateSkill", "go_to"):
            try:
                skill = registry.get_skill(candidate)
                break
            except Exception:
                pass

        if skill is not None:
            try:
                from vector_os_nano.core.skill import SkillContext
                ctx_obj = SkillContext(
                    base=_ctx.base,
                    scene_graph=_ctx.scene_graph,
                    nav_client=_ctx.nav_client,
                )
                result = skill.execute({"room": room}, ctx_obj)
                return bool(getattr(result, "success", False))
            except Exception:
                pass  # Fall through to SceneGraph path

    # Fall back: use SceneGraph to find room center then publish goal
    sg = _require_scene_graph()
    room_node = sg.get_room(room)
    if room_node is None:
        return False

    publish_goal(room_node.center_x, room_node.center_y)
    return wait_until_near(room_node.center_x, room_node.center_y)
