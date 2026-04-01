"""Unit tests for vcli Robot Tools — TDD RED phase.

Covers:
- world_query: objects in world model, empty world, robot pose
- robot_status: arm + joints, no arm, base position/heading
- Both tools: is_read_only, is_concurrency_safe, check_permissions, __tool_name__

Uses only mock objects — no real hardware imported.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockObjectState:
    label: str
    x: float = 1.0
    y: float = 2.0
    z: float = 0.5


@dataclass(frozen=True)
class MockRobotState:
    joint_positions: tuple[float, ...] = (0.1, -0.2, 0.3)
    ee_position: tuple[float, float, float] = (0.15, 0.0, 0.25)
    position_xy: tuple[float, float] = (1.5, -0.7)
    heading: float = 1.57


class MockWorldModel:
    """Minimal world model for testing WorldQueryTool."""

    def __init__(
        self,
        objects: list[MockObjectState] | None = None,
        robot: MockRobotState | None = None,
    ) -> None:
        self._objects = objects or []
        self._robot = robot or MockRobotState()

    def get_objects(self) -> list[MockObjectState]:
        return list(self._objects)

    def get_robot(self) -> MockRobotState:
        return self._robot


class MockArm:
    """Minimal arm mock."""

    name: str = "SO101Arm"

    def get_joint_positions(self) -> list[float]:
        return [0.1, -0.2, 0.3, 0.0, 0.5]


class MockArmNoJoints:
    """Arm whose get_joint_positions raises an exception."""

    name: str = "BrokenArm"

    def get_joint_positions(self) -> list[float]:
        raise RuntimeError("joint read failed")


class MockGripper:
    name: str = "SO101Gripper"

    def get_position(self) -> float:
        return 0.75

    def is_holding(self) -> bool:
        return True


class MockBase:
    name: str = "MuJoCoGo2"

    def get_position(self) -> list[float]:
        return [1.5, -0.7, 0.0]

    def get_heading(self) -> float:
        return 1.57


class MockAgent:
    """Minimal agent mock used by both tools."""

    def __init__(
        self,
        world_model: MockWorldModel | None = None,
        arm: Any | None = None,
        gripper: Any | None = None,
        base: Any | None = None,
        perception: Any | None = None,
    ) -> None:
        self._world_model = world_model or MockWorldModel()
        self._arm = arm
        self._gripper = gripper
        self._base = base
        self._perception = perception


def _make_context(agent: Any | None = None) -> Any:
    from vector_os_nano.vcli.tools.base import ToolContext

    return ToolContext(
        agent=agent,
        cwd=Path("/tmp"),
        session=None,
        permissions=None,
        abort=threading.Event(),
    )


# ---------------------------------------------------------------------------
# WorldQueryTool
# ---------------------------------------------------------------------------


class TestWorldQueryTool:
    def _get_tool(self):
        from vector_os_nano.vcli.tools.robot import WorldQueryTool

        return WorldQueryTool()

    def test_world_query_returns_objects(self) -> None:
        """When world model has objects, they appear in output."""
        objects = [
            MockObjectState(label="red_cube", x=0.3, y=0.1, z=0.05),
            MockObjectState(label="blue_bottle", x=-0.2, y=0.4, z=0.12),
        ]
        wm = MockWorldModel(objects=objects)
        agent = MockAgent(world_model=wm)
        ctx = _make_context(agent=agent)

        tool = self._get_tool()
        result = tool.execute({}, ctx)

        assert result.is_error is False
        assert "red_cube" in result.content
        assert "blue_bottle" in result.content

    def test_world_query_empty_world(self) -> None:
        """Empty world model returns 'No objects detected' message."""
        wm = MockWorldModel(objects=[])
        agent = MockAgent(world_model=wm)
        ctx = _make_context(agent=agent)

        tool = self._get_tool()
        result = tool.execute({}, ctx)

        assert result.is_error is False
        assert "no objects" in result.content.lower()

    def test_world_query_includes_robot_pose(self) -> None:
        """Output includes robot position."""
        robot = MockRobotState(position_xy=(1.5, -0.7), heading=1.57)
        objects = [MockObjectState(label="cup", x=0.1, y=0.2, z=0.0)]
        wm = MockWorldModel(objects=objects, robot=robot)
        agent = MockAgent(world_model=wm)
        ctx = _make_context(agent=agent)

        tool = self._get_tool()
        result = tool.execute({}, ctx)

        # Position and heading values must appear somewhere in output
        assert "1.5" in result.content or "1.57" in result.content

    def test_world_query_filter_by_name(self) -> None:
        """When query param is given, only matching objects are included."""
        objects = [
            MockObjectState(label="red_cube", x=0.1, y=0.0, z=0.0),
            MockObjectState(label="blue_bottle", x=0.5, y=0.0, z=0.0),
        ]
        wm = MockWorldModel(objects=objects)
        agent = MockAgent(world_model=wm)
        ctx = _make_context(agent=agent)

        tool = self._get_tool()
        result = tool.execute({"query": "cube"}, ctx)

        assert "red_cube" in result.content
        assert "blue_bottle" not in result.content

    def test_world_query_no_agent(self) -> None:
        """When agent is None, returns an error result."""
        ctx = _make_context(agent=None)
        tool = self._get_tool()
        result = tool.execute({}, ctx)
        assert result.is_error is True

    def test_world_query_object_coordinates_in_output(self) -> None:
        """Object coordinates (x, y, z) appear in the output."""
        objects = [MockObjectState(label="marker", x=0.42, y=-0.13, z=0.07)]
        wm = MockWorldModel(objects=objects)
        agent = MockAgent(world_model=wm)
        ctx = _make_context(agent=agent)

        tool = self._get_tool()
        result = tool.execute({}, ctx)

        assert "0.42" in result.content


# ---------------------------------------------------------------------------
# RobotStatusTool
# ---------------------------------------------------------------------------


class TestRobotStatusTool:
    def _get_tool(self):
        from vector_os_nano.vcli.tools.robot import RobotStatusTool

        return RobotStatusTool()

    def test_robot_status_returns_hardware(self) -> None:
        """When arm is present, output includes arm name and joints."""
        arm = MockArm()
        agent = MockAgent(arm=arm)
        ctx = _make_context(agent=agent)

        tool = self._get_tool()
        result = tool.execute({}, ctx)

        assert result.is_error is False
        assert "SO101Arm" in result.content
        # Joint values should be in output
        assert "0.1" in result.content or "joints" in result.content.lower()

    def test_robot_status_no_hardware(self) -> None:
        """Agent with no arm or gripper returns graceful message."""
        agent = MockAgent()
        ctx = _make_context(agent=agent)

        tool = self._get_tool()
        result = tool.execute({}, ctx)

        assert result.is_error is False
        assert "not connected" in result.content.lower()

    def test_robot_status_with_base(self) -> None:
        """Agent with base includes base position and heading in output."""
        base = MockBase()
        agent = MockAgent(base=base)
        ctx = _make_context(agent=agent)

        tool = self._get_tool()
        result = tool.execute({}, ctx)

        assert result.is_error is False
        assert "MuJoCoGo2" in result.content
        assert "1.5" in result.content or "heading" in result.content.lower()

    def test_robot_status_arm_joints_unavailable(self) -> None:
        """Arm connected but joint read fails — graceful fallback message."""
        arm = MockArmNoJoints()
        agent = MockAgent(arm=arm)
        ctx = _make_context(agent=agent)

        tool = self._get_tool()
        result = tool.execute({}, ctx)

        assert result.is_error is False
        assert "BrokenArm" in result.content
        assert "unavailable" in result.content.lower() or "connected" in result.content.lower()

    def test_robot_status_no_agent(self) -> None:
        """When agent is None, returns error result."""
        ctx = _make_context(agent=None)
        tool = self._get_tool()
        result = tool.execute({}, ctx)
        assert result.is_error is True

    def test_robot_status_perception_active(self) -> None:
        """When perception is present, output shows 'active'."""
        agent = MockAgent(perception=object())
        ctx = _make_context(agent=agent)

        tool = self._get_tool()
        result = tool.execute({}, ctx)

        assert "active" in result.content.lower()

    def test_robot_status_perception_not_available(self) -> None:
        """When no perception, output shows 'not available'."""
        agent = MockAgent(perception=None)
        ctx = _make_context(agent=agent)

        tool = self._get_tool()
        result = tool.execute({}, ctx)

        assert "not available" in result.content.lower()


# ---------------------------------------------------------------------------
# Shared metadata: is_read_only, is_concurrency_safe, permissions, __tool_name__
# ---------------------------------------------------------------------------


class TestRobotToolsMetadata:
    def test_world_query_read_only(self) -> None:
        from vector_os_nano.vcli.tools.robot import WorldQueryTool

        t = WorldQueryTool()
        assert t.is_read_only({}) is True

    def test_robot_status_read_only(self) -> None:
        from vector_os_nano.vcli.tools.robot import RobotStatusTool

        t = RobotStatusTool()
        assert t.is_read_only({}) is True

    def test_world_query_concurrency_safe(self) -> None:
        from vector_os_nano.vcli.tools.robot import WorldQueryTool

        t = WorldQueryTool()
        assert t.is_concurrency_safe({}) is True

    def test_robot_status_concurrency_safe(self) -> None:
        from vector_os_nano.vcli.tools.robot import RobotStatusTool

        t = RobotStatusTool()
        assert t.is_concurrency_safe({}) is True

    def test_world_query_permission_allow(self) -> None:
        from vector_os_nano.vcli.tools.robot import WorldQueryTool

        t = WorldQueryTool()
        ctx = _make_context()
        result = t.check_permissions({}, ctx)
        assert result.behavior == "allow"

    def test_robot_status_permission_allow(self) -> None:
        from vector_os_nano.vcli.tools.robot import RobotStatusTool

        t = RobotStatusTool()
        ctx = _make_context()
        result = t.check_permissions({}, ctx)
        assert result.behavior == "allow"

    def test_world_query_tool_name(self) -> None:
        from vector_os_nano.vcli.tools.robot import WorldQueryTool

        assert WorldQueryTool.__tool_name__ == "world_query"

    def test_robot_status_tool_name(self) -> None:
        from vector_os_nano.vcli.tools.robot import RobotStatusTool

        assert RobotStatusTool.__tool_name__ == "robot_status"
