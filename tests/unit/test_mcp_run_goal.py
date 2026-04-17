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

    def test_format_vgg_trace_none(self):
        from vector_os_nano.mcp.tools import _format_vgg_trace

        result = _format_vgg_trace(None)
        data = json.loads(result)
        assert data["success"] is False
