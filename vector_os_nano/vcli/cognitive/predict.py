"""State prediction based on SceneGraph room topology.

Pure functions — no side effects, no hardware, no LLM.
Uses SceneGraph.get_door_chain() and connected_rooms for predictions.
"""
from __future__ import annotations

from typing import Any


def predict_navigation(
    scene_graph: Any,
    current_room: str,
    target_room: str,
) -> dict:
    """Predict the outcome of navigating to target_room.

    Args:
        scene_graph: SceneGraph instance
        current_room: Robot's current room ID
        target_room: Desired destination room ID

    Returns:
        {
            "reachable": bool,        # True if path exists in SceneGraph
            "door_count": int,        # Number of doors to traverse
            "estimated_steps": int,   # door_count + 1 (reach each door + final room)
            "rooms_on_path": [str],   # Room IDs along the path (including src and dst)
            "confidence": float,      # 1.0 if path known, 0.0 if not
        }
    """
    waypoints: list[tuple[float, float, str]] = scene_graph.get_door_chain(
        current_room, target_room
    )

    # Same room — single waypoint labelled with the room itself
    if current_room == target_room:
        if not waypoints:
            # Room not in graph at all
            return {
                "reachable": False,
                "door_count": 0,
                "estimated_steps": 1,
                "rooms_on_path": [],
                "confidence": 0.0,
            }
        return {
            "reachable": True,
            "door_count": 0,
            "estimated_steps": 1,
            "rooms_on_path": [current_room],
            "confidence": 1.0,
        }

    # Different rooms — empty list means no path
    if not waypoints:
        return {
            "reachable": False,
            "door_count": 0,
            "estimated_steps": 1,
            "rooms_on_path": [],
            "confidence": 0.0,
        }

    # Count doors: waypoints = [door1, door2, ..., dst_center]
    # The last waypoint is the destination room center (label == target_room).
    # All prior waypoints are doors (label pattern: "roomA_roomB_door").
    door_count = sum(1 for _, _, label in waypoints if label != target_room)

    # Extract room IDs along the path from door labels + endpoints.
    # Door labels: "hallway_kitchen_door" → rooms "hallway", "kitchen"
    rooms_on_path: list[str] = _extract_rooms_from_waypoints(
        current_room, target_room, waypoints
    )

    return {
        "reachable": True,
        "door_count": door_count,
        "estimated_steps": door_count + 1,
        "rooms_on_path": rooms_on_path,
        "confidence": 1.0,
    }


def _extract_rooms_from_waypoints(
    src: str,
    dst: str,
    waypoints: list[tuple[float, float, str]],
) -> list[str]:
    """Reconstruct the ordered room sequence from waypoint labels.

    Door labels use the pattern "roomA_roomB_door".  We walk them in order
    to reconstruct the chain: src → ... → dst.
    """
    rooms: list[str] = [src]
    for _, _, label in waypoints:
        if label == dst:
            # Destination room center waypoint
            if rooms[-1] != dst:
                rooms.append(dst)
        elif label.endswith("_door"):
            # "hallway_kitchen_door" → parse roomA and roomB
            core = label[: -len("_door")]
            parts = core.split("_")
            # The two room IDs were joined with "_". We know the previous room
            # in our path, so we pick the part that isn't it.
            prev = rooms[-1]
            # Attempt to find the next room: try splitting from right
            next_room = _pick_next_room(parts, prev)
            if next_room and next_room != prev and next_room not in rooms:
                rooms.append(next_room)

    if dst not in rooms:
        rooms.append(dst)
    return rooms


def _pick_next_room(parts: list[str], prev_room: str) -> str:
    """Given label parts and the previous room, identify the next room.

    Handles cases where room names contain underscores by building candidates
    from consecutive parts and checking which one differs from prev_room.

    E.g., parts=["hallway", "kitchen"], prev_room="hallway" → "kitchen"
    """
    n = len(parts)
    # Try every split point: parts[:i] as room_a and parts[i:] as room_b
    for split in range(1, n):
        room_a = "_".join(parts[:split])
        room_b = "_".join(parts[split:])
        if room_a == prev_room:
            return room_b
        if room_b == prev_room:
            return room_a
    return ""


def predict_room_after_door(
    scene_graph: Any,
    current_room: str,
    target_room: str,
) -> dict:
    """Predict which room the robot enters after passing through a door.

    Uses SceneGraph.get_door() to check if a door exists between
    current_room and target_room.

    Args:
        scene_graph: SceneGraph instance
        current_room: Robot's current room ID
        target_room: Adjacent room to check

    Returns:
        {
            "room": str,                      # target room ID, or "" if no door
            "confidence": float,              # 1.0 if door exists, 0.0 if not
            "door_position": (x, y) | None,  # Door coordinates if known
        }
    """
    door_pos = scene_graph.get_door(current_room, target_room)
    if door_pos is None:
        return {
            "room": "",
            "confidence": 0.0,
            "door_position": None,
        }
    return {
        "room": target_room,
        "confidence": 1.0,
        "door_position": door_pos,
    }


def predict_exploration_value(scene_graph: Any, room_id: str) -> dict:
    """Estimate how valuable it would be to explore a room.

    Higher value for: low coverage, many connected rooms, recently discovered.

    Args:
        scene_graph: SceneGraph instance
        room_id: Room to evaluate

    Returns:
        {
            "coverage": float,             # Current coverage 0.0~1.0
            "connected_rooms": int,        # Number of adjacent rooms
            "unexplored_neighbors": int,   # Adjacent rooms with coverage < 0.1
            "value": float,                # Heuristic score (higher = more valuable)
        }
    """
    coverage: float = scene_graph.get_room_coverage(room_id)

    room = scene_graph.get_room(room_id)
    if room is None:
        return {
            "coverage": coverage,
            "connected_rooms": 0,
            "unexplored_neighbors": 0,
            "value": (1.0 - coverage),
        }

    neighbors: tuple[str, ...] = room.connected_rooms
    connected_count = len(neighbors)

    unexplored_count = sum(
        1
        for nb in neighbors
        if scene_graph.get_room_coverage(nb) < 0.1
    )

    # Heuristic: (1 - coverage) * (1 + 0.2 * unexplored_neighbors)
    value: float = (1.0 - coverage) * (1.0 + 0.2 * unexplored_count)

    return {
        "coverage": coverage,
        "connected_rooms": connected_count,
        "unexplored_neighbors": unexplored_count,
        "value": value,
    }
