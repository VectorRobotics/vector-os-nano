# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tests for SceneGraph.nearest_room().

Level 34b — unit tests for the nearest_room() API added in T2.
Covers: empty graph, single room, multiple rooms, unvisited filter,
tie-breaking, post-visit workflow, and thread safety.
"""
from __future__ import annotations

import concurrent.futures
import math

import pytest

from vector_os_nano.core.scene_graph import SceneGraph
from vector_os_nano.core.scene_nodes import RoomNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sg_with_rooms(*rooms: tuple[str, float, float, int]) -> SceneGraph:
    """Build a SceneGraph pre-populated with RoomNodes.

    Each entry is (room_id, center_x, center_y, visit_count).
    Rooms with visit_count > 0 are treated as visited.
    """
    sg = SceneGraph()
    for room_id, cx, cy, vc in rooms:
        sg.add_room(RoomNode(room_id=room_id, center_x=cx, center_y=cy, visit_count=vc))
    return sg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_nearest_room_empty_graph() -> None:
    """nearest_room() returns None when no rooms exist."""
    sg = SceneGraph()
    assert sg.nearest_room(0.0, 0.0) is None


def test_nearest_room_single_room() -> None:
    """nearest_room() returns the only visited room."""
    sg = _sg_with_rooms(("living_room", 1.0, 2.0, 3))
    result = sg.nearest_room(0.0, 0.0)
    assert result == "living_room"


def test_nearest_room_multiple_rooms() -> None:
    """nearest_room() returns the closest visited room."""
    sg = _sg_with_rooms(
        ("kitchen", 0.0, 0.0, 2),
        ("bedroom", 10.0, 10.0, 1),
        ("bathroom", 5.0, 5.0, 1),
    )
    # Query from (0.5, 0.5) — closest to kitchen
    assert sg.nearest_room(0.5, 0.5) == "kitchen"

    # Query from (9.0, 9.0) — closest to bedroom
    assert sg.nearest_room(9.0, 9.0) == "bedroom"

    # Query from (4.5, 4.5) — closest to bathroom
    assert sg.nearest_room(4.5, 4.5) == "bathroom"


def test_nearest_room_ignores_unvisited() -> None:
    """nearest_room() skips rooms with visit_count == 0."""
    sg = _sg_with_rooms(
        ("unvisited_near", 0.1, 0.1, 0),   # very close, but visit_count=0
        ("visited_far", 5.0, 5.0, 2),
    )
    result = sg.nearest_room(0.0, 0.0)
    assert result == "visited_far"


def test_nearest_room_all_unvisited() -> None:
    """nearest_room() returns None when all rooms have visit_count == 0."""
    sg = _sg_with_rooms(
        ("room_a", 1.0, 1.0, 0),
        ("room_b", 2.0, 2.0, 0),
    )
    assert sg.nearest_room(0.0, 0.0) is None


def test_nearest_room_tie_breaking() -> None:
    """nearest_room() is deterministic when two rooms are equidistant.

    We do not mandate which room is returned, only that the call does not
    raise and returns one of the two candidates.
    """
    sg = _sg_with_rooms(
        ("room_left", -1.0, 0.0, 1),
        ("room_right", 1.0, 0.0, 1),
    )
    result = sg.nearest_room(0.0, 0.0)
    assert result in {"room_left", "room_right"}

    # Calling again must return the same value (determinism within one run)
    assert sg.nearest_room(0.0, 0.0) == result


def test_nearest_room_after_visit() -> None:
    """visit() followed by nearest_room() works correctly end-to-end."""
    sg = SceneGraph()
    sg.visit("hallway", 3.0, 4.0)
    sg.visit("garage", 10.0, 10.0)

    # Query near hallway
    result = sg.nearest_room(3.5, 4.5)
    assert result == "hallway"

    # Query near garage
    result = sg.nearest_room(9.5, 9.5)
    assert result == "garage"


def test_nearest_room_returns_exact_match() -> None:
    """nearest_room() returns the room when queried at its exact center."""
    sg = _sg_with_rooms(("center_room", 7.0, -3.0, 1))
    assert sg.nearest_room(7.0, -3.0) == "center_room"


def test_nearest_room_thread_safe() -> None:
    """Concurrent visit() and nearest_room() calls must not raise."""
    sg = SceneGraph()
    # Pre-seed with one room so nearest_room has something to return
    sg.visit("seed_room", 0.0, 0.0)

    def _visit(i: int) -> None:
        sg.visit(f"room_{i}", float(i), float(i))

    def _query(i: int) -> str | None:
        return sg.nearest_room(float(i), float(i))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = []
        for i in range(100):
            futures.append(pool.submit(_visit, i))
            futures.append(pool.submit(_query, i))

        # Collect all — will re-raise any exception from worker threads
        for f in concurrent.futures.as_completed(futures):
            f.result()  # raises if the callable raised
