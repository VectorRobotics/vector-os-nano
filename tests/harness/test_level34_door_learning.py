# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Level 34: SceneGraph Door Learning harness.

Tests verify door storage, bidirectionality, running-average positions,
BFS pathfinding (get_door_chain), and YAML persistence.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path so vector_os_nano is importable
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from vector_os_nano.core.scene_graph import SceneGraph
from vector_os_nano.core.scene_nodes import RoomNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sg(persist_path: str | None = None) -> SceneGraph:
    sg = SceneGraph(persist_path=persist_path)
    return sg


def _add_rooms(sg: SceneGraph, *room_specs: tuple) -> None:
    """Add rooms from (room_id, cx, cy) tuples."""
    for room_id, cx, cy in room_specs:
        sg.add_room(RoomNode(room_id=room_id, center_x=cx, center_y=cy, visit_count=1))


# ---------------------------------------------------------------------------
# L34: Door storage and retrieval
# ---------------------------------------------------------------------------

def test_add_door_stores_position() -> None:
    """add_door stores (x, y); get_door retrieves it."""
    sg = _make_sg()
    _add_rooms(sg, ("living", 0.0, 0.0), ("kitchen", 5.0, 0.0))
    sg.add_door("living", "kitchen", 2.5, 0.1)

    result = sg.get_door("living", "kitchen")
    assert result is not None
    x, y = result
    assert abs(x - 2.5) < 1e-9
    assert abs(y - 0.1) < 1e-9


def test_add_door_bidirectional() -> None:
    """add_door(A, B) is retrievable as get_door(B, A)."""
    sg = _make_sg()
    _add_rooms(sg, ("roomA", 0.0, 0.0), ("roomB", 4.0, 0.0))
    sg.add_door("roomA", "roomB", 2.0, 0.0)

    forward = sg.get_door("roomA", "roomB")
    reverse = sg.get_door("roomB", "roomA")
    assert forward is not None
    assert reverse is not None
    assert forward == reverse


def test_add_door_running_average() -> None:
    """Multiple add_door calls compute running average of position."""
    sg = _make_sg()
    _add_rooms(sg, ("hall", 0.0, 0.0), ("bedroom", 6.0, 0.0))

    sg.add_door("hall", "bedroom", 3.0, 0.0)   # observation 1
    sg.add_door("hall", "bedroom", 3.6, 0.6)   # observation 2
    # avg after 2: x = (3.0 + 3.6) / 2 = 3.3, y = (0.0 + 0.6) / 2 = 0.3
    sg.add_door("hall", "bedroom", 2.7, 0.3)   # observation 3
    # avg after 3: x = (3.3*2 + 2.7) / 3 = 3.1, y = (0.3*2 + 0.3) / 3 = 0.3

    result = sg.get_door("hall", "bedroom")
    assert result is not None
    x, y = result
    assert abs(x - 3.1) < 1e-6
    assert abs(y - 0.3) < 1e-6


def test_get_door_returns_none_for_unknown() -> None:
    """get_door for a non-existent pair returns None."""
    sg = _make_sg()
    _add_rooms(sg, ("roomX", 0.0, 0.0), ("roomY", 5.0, 0.0))

    assert sg.get_door("roomX", "roomY") is None
    assert sg.get_door("roomY", "roomX") is None


# ---------------------------------------------------------------------------
# L34: get_door_chain — pathfinding
# ---------------------------------------------------------------------------

def test_get_door_chain_adjacent() -> None:
    """Two directly-connected rooms: returns [door_wp, dst_center]."""
    sg = _make_sg()
    _add_rooms(sg, ("lounge", 0.0, 0.0), ("study", 5.0, 0.0))
    sg.add_door("lounge", "study", 2.5, 0.0)

    chain = sg.get_door_chain("lounge", "study")
    # Expect: door waypoint + destination center
    assert len(chain) == 2
    door_x, door_y, door_label = chain[0]
    assert abs(door_x - 2.5) < 1e-9
    assert abs(door_y - 0.0) < 1e-9
    assert "door" in door_label

    dst_x, dst_y, dst_label = chain[1]
    assert abs(dst_x - 5.0) < 1e-9
    assert abs(dst_y - 0.0) < 1e-9
    assert dst_label == "study"


def test_get_door_chain_via_hallway() -> None:
    """src -> hallway -> dst: returns [src_door, hallway_center, hallway_dst_door, dst_center]."""
    sg = _make_sg()
    _add_rooms(sg,
               ("office", 0.0, 0.0),
               ("hallway", 5.0, 0.0),
               ("bathroom", 10.0, 0.0))

    sg.add_door("office", "hallway", 2.5, 0.0)
    sg.add_door("hallway", "bathroom", 7.5, 0.0)

    chain = sg.get_door_chain("office", "bathroom")
    # Expected:
    #   (2.5, 0.0, "office_hallway_door")
    #   (7.5, 0.0, "hallway_bathroom_door")
    #   (10.0, 0.0, "bathroom")
    assert len(chain) == 3

    # First waypoint: door between office and hallway
    assert abs(chain[0][0] - 2.5) < 1e-9
    assert "door" in chain[0][2]

    # Second waypoint: door between hallway and bathroom
    assert abs(chain[1][0] - 7.5) < 1e-9
    assert "door" in chain[1][2]

    # Third waypoint: bathroom center
    assert abs(chain[2][0] - 10.0) < 1e-9
    assert chain[2][2] == "bathroom"


def test_get_door_chain_same_room() -> None:
    """src == dst returns a single-element list with the room center."""
    sg = _make_sg()
    _add_rooms(sg, ("garage", 3.0, 4.0))

    chain = sg.get_door_chain("garage", "garage")
    assert len(chain) == 1
    x, y, label = chain[0]
    assert abs(x - 3.0) < 1e-9
    assert abs(y - 4.0) < 1e-9
    assert label == "garage"


def test_get_door_chain_no_path() -> None:
    """Disconnected rooms: returns empty list."""
    sg = _make_sg()
    _add_rooms(sg, ("island_a", 0.0, 0.0), ("island_b", 100.0, 100.0))
    # No door between them — no adjacency

    chain = sg.get_door_chain("island_a", "island_b")
    assert chain == []


# ---------------------------------------------------------------------------
# L34: Persistence
# ---------------------------------------------------------------------------

def test_door_persists_across_save_load(tmp_path: Path) -> None:
    """Doors saved to YAML are correctly restored in a new SceneGraph."""
    persist_file = str(tmp_path / "scene.yaml")

    sg1 = _make_sg(persist_path=persist_file)
    _add_rooms(sg1, ("den", 1.0, 2.0), ("foyer", 6.0, 2.0))
    sg1.add_door("den", "foyer", 3.5, 2.0)
    sg1.add_door("den", "foyer", 3.7, 2.2)  # second observation — should average
    sg1.save()

    sg2 = _make_sg(persist_path=persist_file)
    sg2.load()

    result = sg2.get_door("den", "foyer")
    assert result is not None
    x, y = result
    # avg of (3.5, 2.0) and (3.7, 2.2) = (3.6, 2.1)
    assert abs(x - 3.6) < 1e-6
    assert abs(y - 2.1) < 1e-6

    # Reverse lookup still works after load
    reverse = sg2.get_door("foyer", "den")
    assert reverse is not None
    assert abs(reverse[0] - 3.6) < 1e-6


# ---------------------------------------------------------------------------
# L34: connected_rooms update
# ---------------------------------------------------------------------------

def test_add_door_updates_connected_rooms() -> None:
    """After add_door(A, B), both A and B list each other in connected_rooms."""
    sg = _make_sg()
    _add_rooms(sg, ("porch", 0.0, 0.0), ("lobby", 4.0, 0.0))
    # Initially not connected
    porch = sg.get_room("porch")
    lobby = sg.get_room("lobby")
    assert porch is not None and lobby is not None
    assert "lobby" not in porch.connected_rooms
    assert "porch" not in lobby.connected_rooms

    sg.add_door("porch", "lobby", 2.0, 0.0)

    porch_after = sg.get_room("porch")
    lobby_after = sg.get_room("lobby")
    assert porch_after is not None and lobby_after is not None
    assert "lobby" in porch_after.connected_rooms
    assert "porch" in lobby_after.connected_rooms
