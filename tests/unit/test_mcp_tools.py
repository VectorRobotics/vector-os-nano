"""Tests for MCP tool conversion and handling."""

from __future__ import annotations

import json

import pytest

from vector_os_nano.mcp.tools import (
    skills_to_mcp_tools,
    skill_schema_to_mcp_tool,
    build_natural_language_tool,
    _build_skill_instruction,
    _format_execution_result,
)
from vector_os_nano.core.types import ExecutionResult, StepTrace


# ---------------------------------------------------------------------------
# Sample schemas — match SkillRegistry.to_schemas() output format
# ---------------------------------------------------------------------------

SAMPLE_SKILL_SCHEMA = {
    "name": "pick",
    "description": "Pick up an object from the workspace",
    "parameters": {
        "object_label": {
            "type": "string",
            "description": "Name of the object to pick",
        },
        "mode": {
            "type": "string",
            "enum": ["drop", "hold"],
            "default": "drop",
            "description": "hold to keep, drop to discard",
        },
    },
    "aliases": ["grab", "grasp"],
    "direct": False,
    "auto_steps": ["scan", "detect", "pick"],
}

DETECT_SKILL_SCHEMA = {
    "name": "detect",
    "description": "Detect objects in the workspace",
    "parameters": {
        "query": {
            "type": "string",
            "required": True,
            "description": "What to detect",
        }
    },
    "aliases": ["find"],
    "direct": False,
    "auto_steps": ["scan", "detect"],
}

HOME_SKILL_SCHEMA = {
    "name": "home",
    "description": "Move arm to home position",
    "parameters": {},
    "aliases": ["go home"],
    "direct": True,
    "auto_steps": [],
}

ALL_OPTIONAL_SCHEMA = {
    "name": "place",
    "description": "Place held object",
    "parameters": {
        "location": {
            "type": "string",
            "default": "front",
            "description": "Named location",
        },
        "x": {
            "type": "float",
            "required": False,
            "description": "Target X in metres",
        },
    },
    "aliases": ["put"],
    "direct": False,
    "auto_steps": [],
}


# ---------------------------------------------------------------------------
# 1. test_skill_schema_to_mcp_tool
# ---------------------------------------------------------------------------

class TestSkillSchemaToMcpTool:
    def test_name_and_description_preserved(self) -> None:
        tool = skill_schema_to_mcp_tool(SAMPLE_SKILL_SCHEMA)
        assert tool["name"] == "pick"
        assert tool["description"] == "Pick up an object from the workspace"

    def test_input_schema_is_object_type(self) -> None:
        tool = skill_schema_to_mcp_tool(SAMPLE_SKILL_SCHEMA)
        assert tool["inputSchema"]["type"] == "object"

    def test_properties_present(self) -> None:
        tool = skill_schema_to_mcp_tool(SAMPLE_SKILL_SCHEMA)
        props = tool["inputSchema"]["properties"]
        assert "object_label" in props
        assert "mode" in props

    def test_required_fields_detected(self) -> None:
        """Params without default and not explicitly optional are required."""
        tool = skill_schema_to_mcp_tool(SAMPLE_SKILL_SCHEMA)
        required = tool["inputSchema"].get("required", [])
        assert "object_label" in required
        assert "mode" not in required  # has default

    def test_enum_preserved_in_properties(self) -> None:
        tool = skill_schema_to_mcp_tool(SAMPLE_SKILL_SCHEMA)
        mode_prop = tool["inputSchema"]["properties"]["mode"]
        assert mode_prop["enum"] == ["drop", "hold"]

    def test_default_preserved_in_properties(self) -> None:
        tool = skill_schema_to_mcp_tool(SAMPLE_SKILL_SCHEMA)
        mode_prop = tool["inputSchema"]["properties"]["mode"]
        assert mode_prop["default"] == "drop"

    def test_internal_required_key_excluded_from_properties(self) -> None:
        """The 'required' key from skill param defs should not appear in JSON Schema properties."""
        tool = skill_schema_to_mcp_tool(DETECT_SKILL_SCHEMA)
        query_prop = tool["inputSchema"]["properties"]["query"]
        assert "required" not in query_prop

    def test_detect_query_is_required(self) -> None:
        """detect.query has no default and required=True — should be in required list."""
        tool = skill_schema_to_mcp_tool(DETECT_SKILL_SCHEMA)
        required = tool["inputSchema"].get("required", [])
        assert "query" in required


# ---------------------------------------------------------------------------
# 2. test_skill_schema_no_params
# ---------------------------------------------------------------------------

class TestSkillSchemaNoParams:
    def test_empty_parameters_produces_empty_properties(self) -> None:
        tool = skill_schema_to_mcp_tool(HOME_SKILL_SCHEMA)
        assert tool["inputSchema"]["properties"] == {}

    def test_no_required_field_when_no_params(self) -> None:
        tool = skill_schema_to_mcp_tool(HOME_SKILL_SCHEMA)
        assert "required" not in tool["inputSchema"]


# ---------------------------------------------------------------------------
# 3. test_skill_schema_all_optional
# ---------------------------------------------------------------------------

class TestSkillSchemaAllOptional:
    def test_required_list_absent_when_all_optional(self) -> None:
        """All params have default or required=False — required list should be empty/absent."""
        tool = skill_schema_to_mcp_tool(ALL_OPTIONAL_SCHEMA)
        required = tool["inputSchema"].get("required", [])
        assert required == []

    def test_properties_still_present(self) -> None:
        tool = skill_schema_to_mcp_tool(ALL_OPTIONAL_SCHEMA)
        assert "location" in tool["inputSchema"]["properties"]
        assert "x" in tool["inputSchema"]["properties"]


# ---------------------------------------------------------------------------
# 4. test_skills_to_mcp_tools_includes_natural_language
# ---------------------------------------------------------------------------

class TestSkillsToMcpTools:
    def _make_registry(self) -> "SkillRegistry":
        from vector_os_nano.core.skill import SkillRegistry
        from vector_os_nano.skills import get_default_skills

        registry = SkillRegistry()
        for s in get_default_skills():
            registry.register(s)
        return registry

    def test_natural_language_always_included(self) -> None:
        registry = self._make_registry()
        tools = skills_to_mcp_tools(registry)
        names = [t["name"] for t in tools]
        assert "natural_language" in names

    def test_count_is_skills_plus_one(self) -> None:
        registry = self._make_registry()
        skill_count = len(registry.list_skills())
        tools = skills_to_mcp_tools(registry)
        assert len(tools) == skill_count + 4  # +4 meta-tools: natural_language, diagnostics, debug_perception, run_goal

    def test_all_tools_have_name_description_input_schema(self) -> None:
        registry = self._make_registry()
        tools = skills_to_mcp_tools(registry)
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool


# ---------------------------------------------------------------------------
# 5. test_build_natural_language_tool_schema
# ---------------------------------------------------------------------------

class TestBuildNaturalLanguageTool:
    def test_name(self) -> None:
        tool = build_natural_language_tool()
        assert tool["name"] == "natural_language"

    def test_has_instruction_property(self) -> None:
        tool = build_natural_language_tool()
        props = tool["inputSchema"]["properties"]
        assert "instruction" in props
        assert props["instruction"]["type"] == "string"

    def test_instruction_is_required(self) -> None:
        tool = build_natural_language_tool()
        assert "instruction" in tool["inputSchema"]["required"]

    def test_description_non_empty(self) -> None:
        tool = build_natural_language_tool()
        assert len(tool["description"]) > 0


# ---------------------------------------------------------------------------
# 6. test_build_skill_instruction
# ---------------------------------------------------------------------------

class TestBuildSkillInstruction:
    def test_with_single_arg(self) -> None:
        result = _build_skill_instruction("pick", {"object_label": "banana"})
        assert result == "pick banana"

    def test_no_args_returns_skill_name(self) -> None:
        result = _build_skill_instruction("home", {})
        assert result == "home"

    def test_with_location_arg(self) -> None:
        result = _build_skill_instruction("place", {"location": "left"})
        assert result == "place left"

    def test_multiple_args_joined(self) -> None:
        result = _build_skill_instruction("pick", {"object_label": "mug", "mode": "hold"})
        # Both values should appear in the output
        assert "pick" in result
        assert "mug" in result
        assert "hold" in result

    def test_none_values_excluded(self) -> None:
        result = _build_skill_instruction("pick", {"object_label": None})
        assert result == "pick"

    def test_detect_with_query(self) -> None:
        result = _build_skill_instruction("detect", {"query": "red objects"})
        assert result == "detect red objects"


# ---------------------------------------------------------------------------
# 7. test_format_execution_result
# ---------------------------------------------------------------------------

class TestFormatExecutionResult:
    def _make_result(
        self,
        success: bool = True,
        status: str = "completed",
        trace_steps: list[tuple[str, str, float]] | None = None,
        failure_reason: str | None = None,
        message: str | None = None,
    ) -> ExecutionResult:
        trace = []
        if trace_steps:
            for skill_name, step_status, duration in trace_steps:
                trace.append(
                    StepTrace(
                        step_id=skill_name,
                        skill_name=skill_name,
                        status=step_status,
                        duration_sec=duration,
                    )
                )
        return ExecutionResult(
            success=success,
            status=status,
            steps_completed=len([t for t in trace if t.status == "success"]),
            steps_total=len(trace),
            trace=trace,
            failure_reason=failure_reason,
            message=message,
        )

    def test_success_result_contains_status(self) -> None:
        result = self._make_result(
            trace_steps=[("scan", "success", 1.0), ("pick", "success", 3.0)]
        )
        output = _format_execution_result("pick banana", result)
        assert "completed" in output

    def test_success_result_contains_instruction(self) -> None:
        """JSON output contains skill names from the trace."""
        result = self._make_result(
            trace_steps=[("scan", "success", 1.0), ("pick", "success", 3.0)]
        )
        data = json.loads(_format_execution_result("pick banana", result))
        skill_names = [s["skill_name"] for s in data["steps"]]
        assert "scan" in skill_names
        assert "pick" in skill_names

    def test_success_result_lists_steps(self) -> None:
        """JSON steps list has one entry per trace step with correct status."""
        result = self._make_result(
            trace_steps=[("scan", "success", 1.0), ("pick", "success", 3.0)]
        )
        data = json.loads(_format_execution_result("pick banana", result))
        assert len(data["steps"]) == 2
        assert all(s["status"] == "success" for s in data["steps"])

    def test_success_result_shows_duration(self) -> None:
        """total_duration_sec equals sum of step durations."""
        result = self._make_result(
            trace_steps=[("scan", "success", 1.0), ("pick", "success", 3.2)]
        )
        data = json.loads(_format_execution_result("pick banana", result))
        assert data["total_duration_sec"] == 4.2

    def test_failure_result_shows_status(self) -> None:
        result = self._make_result(
            success=False,
            status="failed",
            trace_steps=[("scan", "success", 1.0), ("pick", "execution_failed", 0.5)],
            failure_reason="IK solver failed",
        )
        output = _format_execution_result("pick banana", result)
        assert "failed" in output
        assert "IK solver failed" in output

    def test_failure_step_marked_failed(self) -> None:
        """Failed step has non-success status in JSON."""
        result = self._make_result(
            success=False,
            status="failed",
            trace_steps=[("pick", "execution_failed", 0.5)],
        )
        data = json.loads(_format_execution_result("pick banana", result))
        assert data["steps"][0]["skill_name"] == "pick"
        assert data["steps"][0]["status"] == "execution_failed"

    def test_failure_step_includes_error_detail(self) -> None:
        """Failed step JSON includes error field and failure_reason."""
        trace = [
            StepTrace(step_id="s1", skill_name="scan", status="success", duration_sec=1.0),
            StepTrace(step_id="s2", skill_name="pick", status="execution_failed",
                      duration_sec=0.5, error="No valid 3D position samples from perception"),
        ]
        result = ExecutionResult(
            success=False,
            status="failed",
            steps_completed=1,
            steps_total=2,
            trace=trace,
            failure_reason="Pick failed — could not determine target position",
        )
        data = json.loads(_format_execution_result("pick battery", result))
        scan_step = next(s for s in data["steps"] if s["skill_name"] == "scan")
        pick_step = next(s for s in data["steps"] if s["skill_name"] == "pick")
        assert scan_step["status"] == "success"
        assert pick_step["error"] == "No valid 3D position samples from perception"
        assert data["failure_reason"] == "Pick failed — could not determine target position"

    def test_string_passthrough(self) -> None:
        output = _format_execution_result("hello", "The robot is ready.")
        assert output == "The robot is ready."

    def test_no_trace_produces_empty_steps_list(self) -> None:
        """Empty trace produces JSON with empty steps list."""
        result = self._make_result(trace_steps=[])
        data = json.loads(_format_execution_result("home", result))
        assert data["steps"] == []

    def test_message_included_when_present(self) -> None:
        result = self._make_result(message="Task completed successfully.")
        output = _format_execution_result("home", result)
        assert "Task completed successfully." in output
