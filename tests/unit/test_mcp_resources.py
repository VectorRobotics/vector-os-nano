"""Tests for MCP resource handlers (vector_os_nano/mcp/resources.py)."""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from vector_os_nano.mcp.resources import (
    RESOURCE_DEFINITIONS,
    _read_objects,
    _read_robot_state,
    _read_world_state,
    get_resource_definitions,
    read_resource,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPECTED_URIS = {
    "world://state",
    "world://objects",
    "world://robot",
    "camera://overhead",
    "camera://front",
    "camera://side",
    "camera://live",
}

# MuJoCoArm.render() returns BGR; use all-zero array so BGR->RGB is a no-op.
_SMALL_BGR = np.zeros((2, 2, 3), dtype=np.uint8)


def _make_agent(world_dict: dict | None | bool = None) -> MagicMock:
    """Build a minimal mock Agent.

    Args:
        world_dict: Dict returned by world.to_dict(). Pass False to set
            agent.world = None (simulates missing world model).
    """
    agent = MagicMock()
    if world_dict is False:
        agent.world = None
    else:
        if world_dict is None:
            world_dict = {
                "objects": [
                    {
                        "object_id": "cup_1",
                        "label": "red cup",
                        "x": 0.1,
                        "y": 0.2,
                        "z": 0.05,
                        "confidence": 0.95,
                        "state": "on_table",
                        "last_seen": 1700000000.0,
                        "properties": {},
                    }
                ],
                "robot": {
                    "joint_positions": [0.0, -1.2, 0.5, 0.8, 0.3],
                    "gripper_state": "open",
                    "held_object": None,
                    "is_moving": False,
                    "ee_position": [0.1, 0.0, 0.15],
                },
            }
        agent.world = MagicMock()
        agent.world.to_dict.return_value = world_dict
    return agent


# ---------------------------------------------------------------------------
# Resource definitions
# ---------------------------------------------------------------------------


class TestResourceDefinitions:
    def test_resource_definitions_count(self) -> None:
        assert len(get_resource_definitions()) == 7

    def test_resource_definitions_uris(self) -> None:
        uris = {r["uri"] for r in get_resource_definitions()}
        assert uris == _EXPECTED_URIS

    def test_resource_definitions_has_required_fields(self) -> None:
        required = {"uri", "name", "description", "mimeType"}
        for defn in get_resource_definitions():
            missing = required - defn.keys()
            assert not missing, f"Resource {defn.get('uri')} missing fields: {missing}"

    def test_get_resource_definitions_returns_copy(self) -> None:
        """Mutating the returned list must not affect RESOURCE_DEFINITIONS."""
        defs = get_resource_definitions()
        defs.clear()
        assert len(RESOURCE_DEFINITIONS) == 7


# ---------------------------------------------------------------------------
# world://state
# ---------------------------------------------------------------------------


class TestReadWorldState:
    def test_read_world_state(self) -> None:
        agent = _make_agent()
        result = _read_world_state(agent)
        contents = result["contents"]
        assert len(contents) == 1
        item = contents[0]
        assert item["uri"] == "world://state"
        assert item["mimeType"] == "application/json"
        data = json.loads(item["text"])
        assert "objects" in data
        assert "robot" in data
        assert data["objects"][0]["object_id"] == "cup_1"

    def test_read_world_state_no_world(self) -> None:
        agent = _make_agent(world_dict=False)
        result = _read_world_state(agent)
        data = json.loads(result["contents"][0]["text"])
        assert data == {"objects": [], "robot": {}}


# ---------------------------------------------------------------------------
# world://objects
# ---------------------------------------------------------------------------


class TestReadObjects:
    def test_read_objects(self) -> None:
        agent = _make_agent()
        result = _read_objects(agent)
        contents = result["contents"]
        assert len(contents) == 1
        item = contents[0]
        assert item["uri"] == "world://objects"
        assert item["mimeType"] == "application/json"
        objects = json.loads(item["text"])
        assert isinstance(objects, list)
        assert objects[0]["label"] == "red cup"

    def test_read_objects_empty(self) -> None:
        agent = _make_agent(world_dict={"objects": [], "robot": {}})
        result = _read_objects(agent)
        objects = json.loads(result["contents"][0]["text"])
        assert objects == []

    def test_read_objects_no_world(self) -> None:
        agent = _make_agent(world_dict=False)
        result = _read_objects(agent)
        objects = json.loads(result["contents"][0]["text"])
        assert objects == []


# ---------------------------------------------------------------------------
# world://robot
# ---------------------------------------------------------------------------


class TestReadRobotState:
    def test_read_robot_state(self) -> None:
        agent = _make_agent()
        result = _read_robot_state(agent)
        contents = result["contents"]
        assert len(contents) == 1
        item = contents[0]
        assert item["uri"] == "world://robot"
        assert item["mimeType"] == "application/json"
        robot = json.loads(item["text"])
        assert robot["gripper_state"] == "open"
        assert robot["held_object"] is None

    def test_read_robot_state_no_world(self) -> None:
        agent = _make_agent(world_dict=False)
        result = _read_robot_state(agent)
        robot = json.loads(result["contents"][0]["text"])
        assert robot == {}


# ---------------------------------------------------------------------------
# camera://
# ---------------------------------------------------------------------------


class TestReadCamera:
    @pytest.mark.anyio
    async def test_read_camera_overhead(self) -> None:
        """read_resource('camera://overhead') returns a base64 PNG blob."""
        agent = _make_agent()
        agent._arm = MagicMock()
        agent._arm.render = MagicMock(return_value=_SMALL_BGR)

        with patch(
            "vector_os_nano.mcp.resources.asyncio.to_thread",
            new=AsyncMock(return_value=_SMALL_BGR),
        ):
            result = await read_resource(agent, "camera://overhead")

        contents = result["contents"]
        assert len(contents) == 1
        item = contents[0]
        assert item["uri"] == "camera://overhead"
        assert item["mimeType"] == "image/png"
        assert "blob" in item
        # Verify it is valid base64-encoded PNG (magic bytes: \x89PNG)
        png_bytes = base64.b64decode(item["blob"])
        assert png_bytes[:4] == b"\x89PNG"

    @pytest.mark.anyio
    async def test_read_camera_no_arm(self) -> None:
        """Agent with no arm raises ValueError."""
        agent = _make_agent()
        agent._arm = None
        with pytest.raises(ValueError, match="Camera render not available"):
            await read_resource(agent, "camera://overhead")

    @pytest.mark.anyio
    async def test_read_camera_arm_no_render(self) -> None:
        """Arm without render method raises ValueError."""
        agent = _make_agent()
        agent._arm = MagicMock(spec=[])  # no 'render' attribute
        with pytest.raises(ValueError, match="Camera render not available"):
            await read_resource(agent, "camera://overhead")

    @pytest.mark.anyio
    async def test_read_camera_returns_none(self) -> None:
        """render() returning None raises ValueError."""
        agent = _make_agent()
        agent._arm = MagicMock()
        with patch(
            "vector_os_nano.mcp.resources.asyncio.to_thread",
            new=AsyncMock(return_value=None),
        ):
            with pytest.raises(ValueError, match="returned no image"):
                await read_resource(agent, "camera://overhead")


# ---------------------------------------------------------------------------
# read_resource routing
# ---------------------------------------------------------------------------


class TestReadResourceRouting:
    @pytest.mark.anyio
    async def test_read_unknown_uri(self) -> None:
        agent = _make_agent()
        with pytest.raises(ValueError, match="Unknown resource URI"):
            await read_resource(agent, "bogus://xyz")

    @pytest.mark.anyio
    async def test_routing_world_state(self) -> None:
        agent = _make_agent()
        result = await read_resource(agent, "world://state")
        assert result["contents"][0]["uri"] == "world://state"

    @pytest.mark.anyio
    async def test_routing_world_objects(self) -> None:
        agent = _make_agent()
        result = await read_resource(agent, "world://objects")
        assert result["contents"][0]["uri"] == "world://objects"

    @pytest.mark.anyio
    async def test_routing_world_robot(self) -> None:
        agent = _make_agent()
        result = await read_resource(agent, "world://robot")
        assert result["contents"][0]["uri"] == "world://robot"

    @pytest.mark.anyio
    async def test_routing_all_camera_uris(self) -> None:
        """All three camera URIs are dispatched to _read_camera."""
        for camera in ("overhead", "front", "side"):
            agent = _make_agent()
            agent._arm = MagicMock()
            with patch(
                "vector_os_nano.mcp.resources.asyncio.to_thread",
                new=AsyncMock(return_value=_SMALL_BGR),
            ):
                result = await read_resource(agent, f"camera://{camera}")
            assert result["contents"][0]["uri"] == f"camera://{camera}"
