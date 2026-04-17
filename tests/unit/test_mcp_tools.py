"""Tests for MCP tool conversion and handling."""

from __future__ import annotations

import json

import pytest

from vector_os_nano.mcp.tools import (
    skills_to_mcp_tools,
    skill_schema_to_mcp_tool,
    build_natural_language_tool,
    _build_skill_instruction,
)


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


