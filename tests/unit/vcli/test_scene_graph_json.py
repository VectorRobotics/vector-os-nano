"""Tests for SceneGraph JSON serialization (bridge publisher format).

Validates the JSON structure published on /vector_os/scene_graph matches
the format expected by Foxglove Raw Messages panel.
"""
from __future__ import annotations

import json

import pytest

from vector_os_nano.core.scene_graph import (
    ObjectNode,
    RoomNode,
    SceneGraph,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def populated_graph() -> SceneGraph:
    """Create a SceneGraph with rooms, doors, objects for testing."""
    sg = SceneGraph()
    sg.visit("kitchen", 1.0, 2.0)
    sg.visit("living_room", 5.0, 3.0)
    sg.add_door("kitchen", "living_room", 3.0, 2.5)

    sg.add_object(ObjectNode(
        object_id="obj_1",
        category="cup",
        description="white ceramic cup",
        confidence=0.9,
        room_id="kitchen",
        x=1.5, y=2.5, z=0.8,
    ))
    sg.add_object(ObjectNode(
        object_id="obj_2",
        category="sofa",
        description="grey sofa",
        confidence=0.85,
        room_id="living_room",
        x=5.5, y=3.5, z=0.4,
    ))
    return sg


@pytest.fixture
def empty_graph() -> SceneGraph:
    return SceneGraph()


# ---------------------------------------------------------------------------
# Helper: simulate bridge serialization
# ---------------------------------------------------------------------------


def serialize_scene_graph(sg: SceneGraph) -> dict:
    """Replicate the JSON format from go2_vnav_bridge._publish_scene_graph_json."""
    rooms = sg.get_all_rooms()
    doors = sg.get_all_doors()
    objects = []
    for room in rooms:
        objects.extend(sg.find_objects_in_room(room.room_id))

    return {
        "rooms": [
            {
                "id": r.room_id,
                "center": [round(r.center_x, 2), round(r.center_y, 2)],
                "area": round(r.area, 1),
                "visits": r.visit_count,
                "description": r.representative_description[:80] if r.representative_description else "",
                "connected": list(r.connected_rooms),
            }
            for r in rooms
        ],
        "doors": [
            {
                "rooms": list(k),
                "position": [round(v[0], 2), round(v[1], 2)],
            }
            for k, v in doors.items()
        ],
        "objects": [
            {
                "category": o.category,
                "room": o.room_id,
                "position": [round(o.x, 2), round(o.y, 2)],
                "confidence": round(o.confidence, 2),
            }
            for o in objects
        ],
        "stats": sg.stats(),
    }


# ---------------------------------------------------------------------------
# Tests: JSON structure
# ---------------------------------------------------------------------------


class TestSceneGraphJson:
    def test_empty_graph_serializes(self, empty_graph: SceneGraph):
        data = serialize_scene_graph(empty_graph)
        assert data["rooms"] == []
        assert data["doors"] == []
        assert data["objects"] == []
        assert data["stats"]["rooms"] == 0

    def test_populated_graph_rooms(self, populated_graph: SceneGraph):
        data = serialize_scene_graph(populated_graph)
        assert len(data["rooms"]) == 2

        room_ids = {r["id"] for r in data["rooms"]}
        assert "kitchen" in room_ids
        assert "living_room" in room_ids

        kitchen = next(r for r in data["rooms"] if r["id"] == "kitchen")
        assert kitchen["center"] == [1.0, 2.0]
        assert kitchen["visits"] == 1
        assert "living_room" in kitchen["connected"]

    def test_populated_graph_doors(self, populated_graph: SceneGraph):
        data = serialize_scene_graph(populated_graph)
        assert len(data["doors"]) == 1

        door = data["doors"][0]
        assert set(door["rooms"]) == {"kitchen", "living_room"}
        assert door["position"] == [3.0, 2.5]

    def test_populated_graph_objects(self, populated_graph: SceneGraph):
        data = serialize_scene_graph(populated_graph)
        assert len(data["objects"]) == 2

        categories = {o["category"] for o in data["objects"]}
        assert "cup" in categories
        assert "sofa" in categories

        cup = next(o for o in data["objects"] if o["category"] == "cup")
        assert cup["room"] == "kitchen"
        assert cup["position"] == [1.5, 2.5]
        assert cup["confidence"] == 0.9

    def test_populated_graph_stats(self, populated_graph: SceneGraph):
        data = serialize_scene_graph(populated_graph)
        stats = data["stats"]
        assert stats["rooms"] == 2
        assert stats["objects"] == 2
        assert stats["visited_rooms"] == 2

    def test_json_roundtrip(self, populated_graph: SceneGraph):
        """Verify serialization produces valid JSON that can be parsed."""
        data = serialize_scene_graph(populated_graph)
        json_str = json.dumps(data, ensure_ascii=False)
        parsed = json.loads(json_str)
        assert parsed["rooms"] == data["rooms"]
        assert parsed["doors"] == data["doors"]
        assert parsed["objects"] == data["objects"]
        assert parsed["stats"] == data["stats"]

    def test_description_truncation(self, empty_graph: SceneGraph):
        """Long descriptions are truncated to 80 chars."""
        sg = empty_graph
        long_desc = "A" * 200
        sg.visit("room_a", 0.0, 0.0)
        # Manually set description
        room = sg._rooms["room_a"]
        sg._rooms["room_a"] = RoomNode(
            room_id=room.room_id,
            center_x=room.center_x,
            center_y=room.center_y,
            area=room.area,
            visit_count=room.visit_count,
            representative_description=long_desc,
        )

        data = serialize_scene_graph(sg)
        assert len(data["rooms"][0]["description"]) == 80

    def test_coordinates_rounded(self, empty_graph: SceneGraph):
        """Coordinates are rounded to 2 decimal places."""
        sg = empty_graph
        sg.visit("room_x", 1.23456789, 9.87654321)
        data = serialize_scene_graph(sg)
        center = data["rooms"][0]["center"]
        assert center == [1.23, 9.88]


# ---------------------------------------------------------------------------
# Tests: tool discovery
# ---------------------------------------------------------------------------


class TestToolDiscovery:
    def test_foxglove_tool_in_discovery(self):
        from vector_os_nano.vcli.tools import discover_all_tools

        tools = discover_all_tools()
        names = [t.name for t in tools]
        assert "open_foxglove" in names

    def test_foxglove_tool_in_system_category(self):
        from vector_os_nano.vcli.tools import _TOOL_CATEGORIES

        assert "open_foxglove" in _TOOL_CATEGORIES["system"]
