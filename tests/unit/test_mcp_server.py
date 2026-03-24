"""Tests for MCP server integration.

Tests cover:
- VectorMCPServer construction with a mock agent
- Tool listing (skills + natural_language meta-tool)
- Tool call routing (natural_language and direct skill)
- Resource listing
- Resource reading (world state, objects, robot, camera)
- create_sim_agent() with MuJoCo mocked out
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vector_os_nano.mcp.server import VectorMCPServer, create_sim_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_agent(skill_names: list[str] | None = None) -> MagicMock:
    """Build a minimal mock Agent with the attributes VectorMCPServer touches."""
    skill_names = skill_names or ["pick", "place", "home", "scan", "detect"]

    # Build fake skill schemas
    schemas = []
    for name in skill_names:
        schemas.append(
            {
                "name": name,
                "description": f"Execute the {name} skill",
                "parameters": {},
            }
        )

    registry = MagicMock()
    registry.to_schemas.return_value = schemas
    registry.list_skills.return_value = skill_names

    world = MagicMock()
    world.to_dict.return_value = {
        "objects": [{"id": "banana", "label": "banana", "x": 0.2, "y": 0.1, "z": 0.05}],
        "robot": {"joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0], "gripper": "open"},
    }

    arm = MagicMock()
    arm.render.return_value = None

    agent = MagicMock()
    agent._skill_registry = registry
    agent.world = world
    agent._arm = arm

    return agent


# ---------------------------------------------------------------------------
# TestVectorMCPServer
# ---------------------------------------------------------------------------


class TestVectorMCPServer:
    """Test MCP server construction and handler registration."""

    def test_server_creation(self) -> None:
        """Server creates successfully with a mock agent."""
        agent = _make_mock_agent()
        server = VectorMCPServer(agent)
        assert server._server is not None
        assert server._agent is agent

    def test_server_name(self) -> None:
        """Underlying MCP server has the correct name."""
        agent = _make_mock_agent()
        server = VectorMCPServer(agent)
        assert server._server.name == "vector-os-nano"

    def test_list_tools_includes_natural_language(self) -> None:
        """list_tools returns the natural_language meta-tool."""
        agent = _make_mock_agent([])
        server = VectorMCPServer(agent)

        # Directly call the registered list_tools handler
        tool_list = asyncio.run(_invoke_list_tools(server))
        names = [t.name for t in tool_list]
        assert "natural_language" in names

    def test_list_tools_includes_skills(self) -> None:
        """list_tools returns one tool per registered skill."""
        skills = ["pick", "place", "home"]
        agent = _make_mock_agent(skills)
        server = VectorMCPServer(agent)

        tool_list = asyncio.run(
            _invoke_list_tools(server)
        )
        names = [t.name for t in tool_list]
        for skill in skills:
            assert skill in names

    def test_list_tools_count(self) -> None:
        """Tool count equals skill count + 1 (natural_language)."""
        skills = ["pick", "place", "home", "scan", "detect"]
        agent = _make_mock_agent(skills)
        server = VectorMCPServer(agent)

        tool_list = asyncio.run(
            _invoke_list_tools(server)
        )
        # skills + natural_language + diagnostics + debug_perception + run_goal
        assert len(tool_list) == len(skills) + 4

    def test_list_resources_count(self) -> None:
        """list_resources returns 6 resources (3 world + 3 cameras)."""
        agent = _make_mock_agent()
        server = VectorMCPServer(agent)

        resource_list = asyncio.run(
            _invoke_list_resources(server)
        )
        assert len(resource_list) == 7

    def test_list_resources_uris(self) -> None:
        """list_resources URIs include world:// and camera:// schemes."""
        agent = _make_mock_agent()
        server = VectorMCPServer(agent)

        resource_list = asyncio.run(
            _invoke_list_resources(server)
        )
        uris = [str(r.uri) for r in resource_list]
        assert "world://state" in uris
        assert "world://objects" in uris
        assert "world://robot" in uris
        assert "camera://overhead" in uris
        assert "camera://front" in uris
        assert "camera://side" in uris

    def test_call_tool_natural_language(self) -> None:
        """call_tool natural_language returns a text result."""
        from vector_os_nano.core.types import ExecutionResult

        agent = _make_mock_agent()
        agent.execute = MagicMock(
            return_value=ExecutionResult(success=True, status="completed")
        )
        server = VectorMCPServer(agent)

        result = asyncio.run(
            _invoke_call_tool(server, "natural_language", {"instruction": "pick banana"})
        )
        assert len(result) == 1
        assert result[0].type == "text"
        assert "completed" in result[0].text

    def test_call_tool_direct_skill(self) -> None:
        """call_tool for a named skill routes through agent.execute_skill."""
        from vector_os_nano.core.types import ExecutionResult

        agent = _make_mock_agent(["home"])
        agent.execute_skill = MagicMock(
            return_value=ExecutionResult(success=True, status="completed")
        )
        server = VectorMCPServer(agent)

        result = asyncio.run(
            _invoke_call_tool(server, "home", {})
        )
        assert len(result) == 1
        assert result[0].type == "text"
        agent.execute_skill.assert_called_once_with("home", {})

    def test_call_tool_with_arguments(self) -> None:
        """call_tool passes structured params to agent.execute_skill."""
        from vector_os_nano.core.types import ExecutionResult

        agent = _make_mock_agent(["pick"])
        agent.execute_skill = MagicMock(
            return_value=ExecutionResult(success=True, status="completed")
        )
        server = VectorMCPServer(agent)

        asyncio.run(
            _invoke_call_tool(server, "pick", {"object_label": "mug"})
        )
        agent.execute_skill.assert_called_once_with("pick", {"object_label": "mug"})

    def test_read_resource_world_state(self) -> None:
        """read_resource world://state returns JSON text."""
        agent = _make_mock_agent()
        server = VectorMCPServer(agent)

        contents = asyncio.run(
            _invoke_read_resource(server, "world://state")
        )
        assert len(contents) == 1
        assert contents[0].mime_type == "application/json"
        assert isinstance(contents[0].content, str)
        assert "objects" in contents[0].content

    def test_read_resource_world_objects(self) -> None:
        """read_resource world://objects returns JSON list."""
        import json

        agent = _make_mock_agent()
        server = VectorMCPServer(agent)

        contents = asyncio.run(
            _invoke_read_resource(server, "world://objects")
        )
        assert len(contents) == 1
        data = json.loads(contents[0].content)
        assert isinstance(data, list)

    def test_read_resource_robot_state(self) -> None:
        """read_resource world://robot returns JSON dict."""
        import json

        agent = _make_mock_agent()
        server = VectorMCPServer(agent)

        contents = asyncio.run(
            _invoke_read_resource(server, "world://robot")
        )
        assert len(contents) == 1
        data = json.loads(contents[0].content)
        assert isinstance(data, dict)

    def test_read_resource_camera_no_arm(self) -> None:
        """read_resource camera://overhead raises ValueError when arm absent."""
        agent = _make_mock_agent()
        agent._arm = None
        server = VectorMCPServer(agent)

        with pytest.raises(ValueError, match="Camera render not available"):
            asyncio.run(
                _invoke_read_resource(server, "camera://overhead")
            )

    def test_read_resource_camera_overhead(self) -> None:
        """read_resource camera://overhead returns base64 PNG bytes."""
        import numpy as np

        agent = _make_mock_agent()
        # Create a small fake BGR image (10x10 green)
        fake_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        fake_bgr[:, :, 1] = 255
        agent._arm.render.return_value = fake_bgr
        server = VectorMCPServer(agent)

        contents = asyncio.run(
            _invoke_read_resource(server, "camera://overhead")
        )
        assert len(contents) == 1
        assert isinstance(contents[0].content, bytes)
        assert contents[0].mime_type == "image/png"

    def test_read_resource_unknown_uri(self) -> None:
        """read_resource raises ValueError for unknown URIs."""
        agent = _make_mock_agent()
        server = VectorMCPServer(agent)

        with pytest.raises(ValueError, match="Unknown resource URI"):
            asyncio.run(
                _invoke_read_resource(server, "unknown://foo")
            )


# ---------------------------------------------------------------------------
# TestCreateSimAgent
# ---------------------------------------------------------------------------


class TestCreateSimAgent:
    """Test simulation agent factory."""

    def test_create_sim_agent_returns_agent(self) -> None:
        """create_sim_agent() returns an Agent instance."""
        with _patch_mujoco():
            agent = create_sim_agent(headless=True)
        assert isinstance(agent, Agent)

    def test_create_sim_agent_headless(self) -> None:
        """create_sim_agent(headless=True) creates MuJoCoArm with gui=False."""
        with _patch_mujoco() as mock_arm_cls:
            create_sim_agent(headless=True)
        mock_arm_cls.assert_called_once_with(gui=False)

    def test_create_sim_agent_with_viewer(self) -> None:
        """create_sim_agent(headless=False) creates MuJoCoArm with gui=True."""
        with _patch_mujoco() as mock_arm_cls:
            create_sim_agent(headless=False)
        mock_arm_cls.assert_called_once_with(gui=True)

    def test_create_sim_agent_connects_arm(self) -> None:
        """create_sim_agent() calls arm.connect()."""
        with _patch_mujoco() as mock_arm_cls:
            create_sim_agent(headless=True)
        mock_arm_cls.return_value.connect.assert_called_once()

    def test_create_sim_agent_closes_gripper(self) -> None:
        """create_sim_agent() closes the gripper after creation."""
        with _patch_mujoco_full() as (mock_arm_cls, mock_gripper_cls, _):
            create_sim_agent(headless=True)
        mock_gripper_cls.return_value.close.assert_called_once()


# ---------------------------------------------------------------------------
# Async test helpers (invoke registered handlers directly)
# ---------------------------------------------------------------------------


async def _invoke_list_tools(server: VectorMCPServer) -> list[Any]:
    """Call the list_tools handler registered on the server."""
    from mcp import types as mcp_types

    handler = server._server.request_handlers.get(mcp_types.ListToolsRequest)
    if handler is None:
        pytest.fail("list_tools handler not registered")
    req = mcp_types.ListToolsRequest(method="tools/list", params=None)
    result = await handler(req)
    return result.root.tools  # type: ignore[attr-defined]


async def _invoke_list_resources(server: VectorMCPServer) -> list[Any]:
    """Call the list_resources handler registered on the server."""
    from mcp import types as mcp_types

    handler = server._server.request_handlers.get(mcp_types.ListResourcesRequest)
    if handler is None:
        pytest.fail("list_resources handler not registered")
    req = mcp_types.ListResourcesRequest(method="resources/list", params=None)
    result = await handler(req)
    return result.root.resources  # type: ignore[attr-defined]


async def _invoke_call_tool(
    server: VectorMCPServer, name: str, arguments: dict
) -> list[Any]:
    """Call the call_tool handler registered on the server."""
    from mcp import types as mcp_types

    handler = server._server.request_handlers.get(mcp_types.CallToolRequest)
    if handler is None:
        pytest.fail("call_tool handler not registered")
    params = mcp_types.CallToolRequestParams(name=name, arguments=arguments)
    req = mcp_types.CallToolRequest(method="tools/call", params=params)
    result = await handler(req)
    return result.root.content  # type: ignore[attr-defined]


async def _invoke_read_resource(server: VectorMCPServer, uri: str) -> list[Any]:
    """Call the read_resource handler and return ReadResourceContents list."""
    from mcp import types as mcp_types
    from mcp.server.lowlevel.helper_types import ReadResourceContents
    from pydantic import AnyUrl

    handler = server._server.request_handlers.get(mcp_types.ReadResourceRequest)
    if handler is None:
        pytest.fail("read_resource handler not registered")

    # The handler reads req.params.uri — wrap the string in AnyUrl
    params = mcp_types.ReadResourceRequestParams(uri=AnyUrl(uri))  # type: ignore[call-arg]
    req = mcp_types.ReadResourceRequest(method="resources/read", params=params)
    result = await handler(req)
    # result.root.contents is a list[TextResourceContents | BlobResourceContents]
    raw_contents = result.root.contents  # type: ignore[attr-defined]

    # Convert to ReadResourceContents for easy assertions
    out: list[ReadResourceContents] = []
    for item in raw_contents:
        if hasattr(item, "text"):
            out.append(ReadResourceContents(content=item.text, mime_type=item.mimeType))
        elif hasattr(item, "blob"):
            # blob is base64 encoded string in MCP
            raw_bytes = base64.b64decode(item.blob)
            out.append(ReadResourceContents(content=raw_bytes, mime_type=item.mimeType))
    return out


# ---------------------------------------------------------------------------
# MuJoCo mock context managers
# ---------------------------------------------------------------------------


def _patch_mujoco():
    """Patch MuJoCoArm for create_sim_agent tests.

    Patches the class in its source module (hardware.sim.mujoco_arm) because
    create_sim_agent() uses a local `from ... import` inside the function body.
    Returns the mock arm class as the context manager value.
    """
    mock_arm = MagicMock()
    mock_arm.get_joint_positions.return_value = [0.0] * 5
    mock_arm.get_object_positions.return_value = {}
    mock_arm._connected = True
    mock_arm.name = "mujoco_so101"

    mock_gripper = MagicMock()
    mock_perception = MagicMock()

    mock_arm_cls = MagicMock(return_value=mock_arm)
    mock_gripper_cls = MagicMock(return_value=mock_gripper)
    mock_perception_cls = MagicMock(return_value=mock_perception)

    import contextlib

    @contextlib.contextmanager
    def _ctx():
        with (
            patch("vector_os_nano.hardware.sim.mujoco_arm.MuJoCoArm", mock_arm_cls),
            patch("vector_os_nano.hardware.sim.mujoco_gripper.MuJoCoGripper", mock_gripper_cls),
            patch("vector_os_nano.hardware.sim.mujoco_perception.MuJoCoPerception", mock_perception_cls),
            patch("vector_os_nano.mcp.server.MuJoCoArm", mock_arm_cls, create=True),
            patch("vector_os_nano.mcp.server.MuJoCoGripper", mock_gripper_cls, create=True),
            patch("vector_os_nano.mcp.server.MuJoCoPerception", mock_perception_cls, create=True),
        ):
            yield mock_arm_cls

    return _ctx()


def _patch_mujoco_full():
    """Patch MuJoCoArm, MuJoCoGripper, MuJoCoPerception for fuller tests."""
    import contextlib

    mock_arm = MagicMock()
    mock_arm.get_joint_positions.return_value = [0.0] * 5
    mock_arm.get_object_positions.return_value = {}
    mock_arm._connected = True
    mock_arm.name = "mujoco_so101"

    mock_gripper = MagicMock()
    mock_perception = MagicMock()

    mock_arm_cls = MagicMock(return_value=mock_arm)
    mock_gripper_cls = MagicMock(return_value=mock_gripper)
    mock_perception_cls = MagicMock(return_value=mock_perception)

    @contextlib.contextmanager
    def _ctx():
        with (
            patch("vector_os_nano.hardware.sim.mujoco_arm.MuJoCoArm", mock_arm_cls),
            patch("vector_os_nano.hardware.sim.mujoco_gripper.MuJoCoGripper", mock_gripper_cls),
            patch("vector_os_nano.hardware.sim.mujoco_perception.MuJoCoPerception", mock_perception_cls),
            patch("vector_os_nano.mcp.server.MuJoCoArm", mock_arm_cls, create=True),
            patch("vector_os_nano.mcp.server.MuJoCoGripper", mock_gripper_cls, create=True),
            patch("vector_os_nano.mcp.server.MuJoCoPerception", mock_perception_cls, create=True),
        ):
            yield mock_arm_cls, mock_gripper_cls, mock_perception_cls

    return _ctx()


# Import Agent for isinstance check in TestCreateSimAgent
from vector_os_nano.core.agent import Agent  # noqa: E402
