"""Tests for SceneGraphQueryTool.

TDD RED phase — all tests must fail before implementation is written.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from vector_os_nano.core.scene_graph import ObjectNode, RoomNode, SceneGraph
from vector_os_nano.vcli.tools.base import ToolContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_test_sg() -> SceneGraph:
    sg = SceneGraph()
    sg.add_room(RoomNode(room_id="kitchen", center_x=17.0, center_y=2.5, visit_count=5))
    sg.add_room(RoomNode(room_id="hallway", center_x=10.0, center_y=5.0, visit_count=3))
    sg.add_door("kitchen", "hallway", 13.5, 3.0)
    sg.add_object(ObjectNode(object_id="obj1", category="fridge", room_id="kitchen", x=18.0, y=2.0))
    return sg


def _make_context(sg: SceneGraph | None) -> MagicMock:
    ctx = MagicMock(spec=ToolContext)
    ctx.app_state = {"scene_graph": sg}
    return ctx


# ---------------------------------------------------------------------------
# Import the tool under test
# ---------------------------------------------------------------------------


@pytest.fixture()
def tool():
    from vector_os_nano.vcli.tools.scene_graph_tool import SceneGraphQueryTool
    return SceneGraphQueryTool()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSceneGraphQueryToolProtocol:
    """Verify the tool satisfies the Tool protocol requirements."""

    def test_tool_name(self, tool):
        assert tool.name == "scene_graph_query"

    def test_tool_is_read_only(self, tool):
        assert tool.is_read_only({}) is True

    def test_tool_is_concurrency_safe(self, tool):
        assert tool.is_concurrency_safe({}) is True

    def test_tool_permission_allow(self, tool):
        ctx = _make_context(_make_test_sg())
        result = tool.check_permissions({"query_type": "rooms"}, ctx)
        assert result.behavior == "allow"

    def test_input_schema_has_query_type(self, tool):
        assert "query_type" in tool.input_schema["properties"]
        assert "query_type" in tool.input_schema["required"]


class TestQueryRooms:
    def test_query_rooms_returns_list(self, tool):
        ctx = _make_context(_make_test_sg())
        result = tool.execute({"query_type": "rooms"}, ctx)
        assert not result.is_error
        data = json.loads(result.content)
        assert isinstance(data, list)
        assert len(data) == 2

    def test_query_rooms_contains_expected_fields(self, tool):
        ctx = _make_context(_make_test_sg())
        result = tool.execute({"query_type": "rooms"}, ctx)
        data = json.loads(result.content)
        room_ids = {r["room_id"] for r in data}
        assert room_ids == {"kitchen", "hallway"}
        for room in data:
            assert "center_x" in room
            assert "center_y" in room
            assert "visit_count" in room

    def test_query_rooms_correct_centers(self, tool):
        ctx = _make_context(_make_test_sg())
        result = tool.execute({"query_type": "rooms"}, ctx)
        data = json.loads(result.content)
        kitchen = next(r for r in data if r["room_id"] == "kitchen")
        assert kitchen["center_x"] == pytest.approx(17.0)
        assert kitchen["center_y"] == pytest.approx(2.5)
        assert kitchen["visit_count"] == 5


class TestQueryDoors:
    def test_query_doors_returns_dict(self, tool):
        ctx = _make_context(_make_test_sg())
        result = tool.execute({"query_type": "doors"}, ctx)
        assert not result.is_error
        data = json.loads(result.content)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_query_doors_contains_room_pairs(self, tool):
        ctx = _make_context(_make_test_sg())
        result = tool.execute({"query_type": "doors"}, ctx)
        data = json.loads(result.content)
        door = data[0]
        rooms = {door["room_a"], door["room_b"]}
        assert rooms == {"kitchen", "hallway"}
        assert door["x"] == pytest.approx(13.5)
        assert door["y"] == pytest.approx(3.0)


class TestQueryObjects:
    def test_query_objects_returns_list(self, tool):
        ctx = _make_context(_make_test_sg())
        result = tool.execute({"query_type": "objects"}, ctx)
        assert not result.is_error
        data = json.loads(result.content)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_query_objects_fields(self, tool):
        ctx = _make_context(_make_test_sg())
        result = tool.execute({"query_type": "objects"}, ctx)
        data = json.loads(result.content)
        obj = data[0]
        assert obj["category"] == "fridge"
        assert obj["room_id"] == "kitchen"
        assert "x" in obj
        assert "y" in obj


class TestQueryRoomDetail:
    def test_query_room_detail_returns_dict(self, tool):
        ctx = _make_context(_make_test_sg())
        result = tool.execute({"query_type": "room_detail", "room": "kitchen"}, ctx)
        assert not result.is_error
        data = json.loads(result.content)
        assert isinstance(data, dict)

    def test_query_room_detail_contains_room_info(self, tool):
        ctx = _make_context(_make_test_sg())
        result = tool.execute({"query_type": "room_detail", "room": "kitchen"}, ctx)
        data = json.loads(result.content)
        assert data["room_id"] == "kitchen"
        assert data["center_x"] == pytest.approx(17.0)

    def test_query_room_detail_contains_objects(self, tool):
        ctx = _make_context(_make_test_sg())
        result = tool.execute({"query_type": "room_detail", "room": "kitchen"}, ctx)
        data = json.loads(result.content)
        assert "objects" in data
        assert len(data["objects"]) == 1
        assert data["objects"][0]["category"] == "fridge"

    def test_query_room_detail_contains_viewpoints(self, tool):
        ctx = _make_context(_make_test_sg())
        result = tool.execute({"query_type": "room_detail", "room": "kitchen"}, ctx)
        data = json.loads(result.content)
        assert "viewpoints" in data
        assert isinstance(data["viewpoints"], list)

    def test_query_room_detail_unknown_room(self, tool):
        ctx = _make_context(_make_test_sg())
        result = tool.execute({"query_type": "room_detail", "room": "nonexistent"}, ctx)
        assert result.is_error

    def test_query_room_detail_missing_room_param(self, tool):
        ctx = _make_context(_make_test_sg())
        result = tool.execute({"query_type": "room_detail"}, ctx)
        assert result.is_error


class TestQueryDoorChain:
    def test_query_door_chain_returns_list(self, tool):
        ctx = _make_context(_make_test_sg())
        result = tool.execute(
            {"query_type": "door_chain", "src_room": "kitchen", "dst_room": "hallway"}, ctx
        )
        assert not result.is_error
        data = json.loads(result.content)
        assert isinstance(data, list)

    def test_query_door_chain_waypoints(self, tool):
        ctx = _make_context(_make_test_sg())
        result = tool.execute(
            {"query_type": "door_chain", "src_room": "kitchen", "dst_room": "hallway"}, ctx
        )
        data = json.loads(result.content)
        # At minimum the door + destination should appear
        assert len(data) >= 1
        for wp in data:
            assert "x" in wp
            assert "y" in wp
            assert "label" in wp

    def test_query_door_chain_same_room(self, tool):
        ctx = _make_context(_make_test_sg())
        result = tool.execute(
            {"query_type": "door_chain", "src_room": "kitchen", "dst_room": "kitchen"}, ctx
        )
        assert not result.is_error
        data = json.loads(result.content)
        assert len(data) == 1
        assert data[0]["label"] == "kitchen"

    def test_query_door_chain_missing_params(self, tool):
        ctx = _make_context(_make_test_sg())
        result = tool.execute({"query_type": "door_chain", "src_room": "kitchen"}, ctx)
        assert result.is_error


class TestQueryCoverage:
    def test_query_coverage_returns_dict(self, tool):
        ctx = _make_context(_make_test_sg())
        result = tool.execute({"query_type": "coverage"}, ctx)
        assert not result.is_error
        data = json.loads(result.content)
        assert isinstance(data, dict)

    def test_query_coverage_has_all_rooms(self, tool):
        ctx = _make_context(_make_test_sg())
        result = tool.execute({"query_type": "coverage"}, ctx)
        data = json.loads(result.content)
        assert "kitchen" in data
        assert "hallway" in data

    def test_query_coverage_values_are_floats(self, tool):
        ctx = _make_context(_make_test_sg())
        result = tool.execute({"query_type": "coverage"}, ctx)
        data = json.loads(result.content)
        for room_id, cov in data.items():
            assert isinstance(cov, float), f"{room_id} coverage should be float"
            assert 0.0 <= cov <= 1.0, f"{room_id} coverage {cov} out of range"


class TestQuerySummary:
    def test_query_summary_returns_string(self, tool):
        ctx = _make_context(_make_test_sg())
        result = tool.execute({"query_type": "summary"}, ctx)
        assert not result.is_error
        data = json.loads(result.content)
        assert isinstance(data, dict)
        assert "summary" in data
        assert isinstance(data["summary"], str)

    def test_query_summary_contains_room_names(self, tool):
        ctx = _make_context(_make_test_sg())
        result = tool.execute({"query_type": "summary"}, ctx)
        data = json.loads(result.content)
        summary = data["summary"]
        # get_room_summary() includes room names in the text
        assert "kitchen" in summary or "hallway" in summary


class TestEmptySceneGraph:
    def test_empty_sg_rooms_returns_empty_list(self, tool):
        ctx = _make_context(SceneGraph())
        result = tool.execute({"query_type": "rooms"}, ctx)
        assert not result.is_error
        data = json.loads(result.content)
        assert data == []

    def test_empty_sg_doors_returns_empty_list(self, tool):
        ctx = _make_context(SceneGraph())
        result = tool.execute({"query_type": "doors"}, ctx)
        assert not result.is_error
        data = json.loads(result.content)
        assert data == []

    def test_empty_sg_objects_returns_empty_list(self, tool):
        ctx = _make_context(SceneGraph())
        result = tool.execute({"query_type": "objects"}, ctx)
        assert not result.is_error
        data = json.loads(result.content)
        assert data == []

    def test_empty_sg_summary_returns_message(self, tool):
        ctx = _make_context(SceneGraph())
        result = tool.execute({"query_type": "summary"}, ctx)
        assert not result.is_error
        data = json.loads(result.content)
        assert "summary" in data
        # Should be informative, not an error
        assert len(data["summary"]) > 0

    def test_none_sg_returns_error(self, tool):
        ctx = _make_context(None)
        result = tool.execute({"query_type": "rooms"}, ctx)
        assert result.is_error

    def test_missing_app_state_returns_error(self, tool):
        ctx = MagicMock(spec=ToolContext)
        ctx.app_state = None
        result = tool.execute({"query_type": "rooms"}, ctx)
        assert result.is_error
