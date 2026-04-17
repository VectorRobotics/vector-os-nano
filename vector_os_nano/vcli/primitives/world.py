"""World primitives — wrap SceneGraph for querying room/door/object state.

All functions are module-level and read from the module-global _ctx.
Requires init_primitives() to be called before use.
"""
from __future__ import annotations

from vector_os_nano.vcli.primitives import PrimitiveContext

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


def query_rooms() -> list[dict]:
    """Return all known rooms as a list of dicts.

    Returns:
        List of {"id": str, "x": float, "y": float, "visited": bool} dicts.

    Raises:
        RuntimeError: If no SceneGraph is connected.
    """
    sg = _require_scene_graph()
    rooms = sg.get_all_rooms()
    return [
        {
            "id": r.room_id,
            "x": float(r.center_x),
            "y": float(r.center_y),
            "visited": r.visit_count > 0,
        }
        for r in rooms
    ]


def get_visited_rooms() -> list[str]:
    """Return room names that have been visited at least once.

    Returns:
        List of room name strings.

    Raises:
        RuntimeError: If no SceneGraph is connected.
    """
    sg = _require_scene_graph()
    return list(sg.get_visited_rooms())


# ---------------------------------------------------------------------------
# Door queries
# ---------------------------------------------------------------------------


def query_doors() -> list[dict]:
    """Return all known doors as a list of dicts.

    Returns:
        List of {"rooms": [str, str], "x": float, "y": float} dicts.

    Raises:
        RuntimeError: If no SceneGraph is connected.
    """
    sg = _require_scene_graph()
    doors = sg.get_all_doors()
    return [
        {
            "rooms": list(k),
            "x": float(v[0]),
            "y": float(v[1]),
        }
        for k, v in doors.items()
    ]


# ---------------------------------------------------------------------------
# Object queries
# ---------------------------------------------------------------------------


def query_objects(room: str = "") -> list[dict]:
    """Return objects known to the world model, optionally filtered by room.

    Args:
        room: Room name filter. Empty string = all rooms.

    Returns:
        List of {"name": str, "room": str, "x": float, "y": float} dicts.

    Raises:
        RuntimeError: If no SceneGraph is connected.
    """
    sg = _require_scene_graph()
    if room:
        raw = sg.find_objects_in_room(room)
    else:
        # Aggregate all rooms
        all_rooms = sg.get_all_rooms()
        raw = []
        for r in all_rooms:
            raw.extend(sg.find_objects_in_room(r.room_id))

    return [
        {
            "name": getattr(o, "category", str(o)),
            "room": getattr(o, "room_id", ""),
            "x": float(getattr(o, "x", 0.0)),
            "y": float(getattr(o, "y", 0.0)),
        }
        for o in raw
    ]


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------


def path_between(room_a: str, room_b: str) -> list[tuple[float, float]]:
    """Return navigation waypoints between two rooms.

    Args:
        room_a: Source room name.
        room_b: Destination room name.

    Returns:
        List of (x, y) float tuples. Empty if no path found.

    Raises:
        RuntimeError: If no SceneGraph is connected.
    """
    sg = _require_scene_graph()
    chain = sg.get_door_chain(room_a, room_b)
    return [(float(wp[0]), float(wp[1])) for wp in chain]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ObjectMemory-backed queries
# ---------------------------------------------------------------------------


def last_seen(category: str = "") -> dict | None:
    """Most recent observation of category.

    Args:
        category: Object category to search for (substring match).

    Returns:
        {"room", "position", "seconds_ago", "confidence"} or None if not found.
        Delegates to ObjectMemory if available, otherwise returns None.
    """
    if _ctx is None or _ctx.object_memory is None:
        return None
    return _ctx.object_memory.last_seen(category)


def certainty(fact: str = "") -> float:
    """Time-decayed confidence of a fact like "cup在kitchen" or "cup in kitchen".

    Args:
        fact: Fact string in Chinese ("cup在kitchen") or English ("cup in kitchen") format.

    Returns:
        Effective confidence float 0.0–1.0.
        Delegates to ObjectMemory. Returns 0.0 if unavailable.
    """
    if _ctx is None or _ctx.object_memory is None:
        return 0.0
    return _ctx.object_memory.certainty(fact)


def find_object(category: str = "") -> list[dict]:
    """All known locations of category, sorted by confidence.

    Args:
        category: Object category to search for (substring match).

    Returns:
        List of {"object_id", "category", "room", "x", "y", "confidence", "seconds_ago"}.
        Delegates to ObjectMemory. Returns [] if unavailable.
    """
    if _ctx is None or _ctx.object_memory is None:
        return []
    return _ctx.object_memory.find_object(category)


def objects_in_room(room_id: str = "") -> list[dict]:
    """Objects in room with time-decayed confidence.

    Args:
        room_id: Room identifier to query.

    Returns:
        List of {"object_id", "category", "x", "y", "confidence", "seconds_ago"}.
        Delegates to ObjectMemory. Returns [] if unavailable.
    """
    if _ctx is None or _ctx.object_memory is None:
        return []
    return _ctx.object_memory.objects_in_room(room_id)


def room_coverage(room_id: str = "") -> float:
    """Exploration coverage of room (0.0~1.0).

    Args:
        room_id: Room identifier to query.

    Returns:
        Coverage fraction from SceneGraph.get_room_coverage(). Returns 0.0 on error.

    Raises:
        RuntimeError: If no SceneGraph is connected.
    """
    sg = _require_scene_graph()
    try:
        return float(sg.get_room_coverage(room_id))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def world_stats() -> dict:
    """Return high-level world model statistics.

    Returns:
        Dict with keys: "rooms" (int), "objects" (int), "visited" (int).

    Raises:
        RuntimeError: If no SceneGraph is connected.
    """
    sg = _require_scene_graph()
    raw = sg.stats()
    return {
        "rooms": int(raw.get("rooms", 0)),
        "objects": int(raw.get("objects", 0)),
        "visited": int(raw.get("visited_rooms", 0)),
    }
