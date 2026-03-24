"""Tests for MCP run_goal tool — build_run_goal_tool, schema, list, handler.

TDD: RED phase — written before implementation.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from vector_os_nano.mcp.tools import skills_to_mcp_tools, build_run_goal_tool


class TestRunGoalMCPTool:
    def test_run_goal_in_tool_list(self):
        registry = MagicMock()
        registry.to_schemas.return_value = []
        tools = skills_to_mcp_tools(registry)
        names = [t["name"] for t in tools]
        assert "run_goal" in names

    def test_run_goal_schema(self):
        tool = build_run_goal_tool()
        assert tool["name"] == "run_goal"
        assert "goal" in tool["inputSchema"]["required"]
        assert "max_iterations" in tool["inputSchema"]["properties"]
        assert "verify" in tool["inputSchema"]["properties"]

    def test_run_goal_schema_goal_required_only(self):
        """Only 'goal' is required; max_iterations and verify are optional."""
        tool = build_run_goal_tool()
        required = tool["inputSchema"].get("required", [])
        assert "goal" in required
        assert "max_iterations" not in required
        assert "verify" not in required

    def test_run_goal_schema_types(self):
        tool = build_run_goal_tool()
        props = tool["inputSchema"]["properties"]
        assert props["goal"]["type"] == "string"
        assert props["max_iterations"]["type"] == "integer"
        assert props["verify"]["type"] == "boolean"

    def test_run_goal_schema_defaults(self):
        tool = build_run_goal_tool()
        props = tool["inputSchema"]["properties"]
        assert props["max_iterations"]["default"] == 10
        assert props["verify"]["default"] is True

    def test_format_goal_result(self):
        from vector_os_nano.mcp.tools import _format_goal_result
        from vector_os_nano.core.types import GoalResult, ActionRecord

        result = GoalResult(
            success=True,
            goal="clean",
            iterations=3,
            total_duration_sec=10.0,
            actions=[
                ActionRecord(iteration=0, action="pick", skill_success=True)
            ],
            summary="Done.",
            final_world_state={"objects": []},
        )
        raw = _format_goal_result(result)
        data = json.loads(raw)
        assert data["success"] is True
        assert data["goal"] == "clean"
        assert len(data["actions"]) == 1
        assert data["actions"][0]["action"] == "pick"

    def test_format_goal_result_failure(self):
        from vector_os_nano.mcp.tools import _format_goal_result
        from vector_os_nano.core.types import GoalResult

        result = GoalResult(
            success=False,
            goal="fail task",
            iterations=10,
            total_duration_sec=30.0,
            actions=[],
            summary="Max iterations reached.",
            final_world_state={},
        )
        raw = _format_goal_result(result)
        data = json.loads(raw)
        assert data["success"] is False
        assert data["iterations"] == 10
        assert "Max iterations" in data["summary"]

    def test_format_goal_result_non_goal_result(self):
        """Non-GoalResult input is returned as str()."""
        from vector_os_nano.mcp.tools import _format_goal_result

        assert _format_goal_result("already a string") == "already a string"
        assert _format_goal_result(42) == "42"
