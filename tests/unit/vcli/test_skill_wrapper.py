"""Unit tests for vcli SkillWrapperTool — TDD RED phase.

Covers:
- SkillWrapperTool name/description from skill attributes
- input_schema built from skill.parameters
- execute() delegates to skill.execute()
- ToolResult content for success
- ToolResult is_error=True for failure
- Motor skill: check_permissions returns "ask"
- Read skill: check_permissions returns "allow"
- Motor skill: is_concurrency_safe returns False
- Read skill: is_concurrency_safe returns True
- wrap_skills: returns list of SkillWrapperTool from agent registry
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from vector_os_nano.vcli.tools.base import ToolContext, ToolResult


# ---------------------------------------------------------------------------
# Mock skill helpers
# ---------------------------------------------------------------------------


def _make_context(agent: Any = None) -> ToolContext:
    return ToolContext(
        agent=agent,
        cwd=Path("/tmp"),
        session=None,
        permissions=None,
        abort=threading.Event(),
    )


class MockSkillResult:
    def __init__(self, success: bool, result_data: dict | None = None, error_message: str = "") -> None:
        self.success = success
        self.result_data = result_data
        self.error_message = error_message


class MockSkill:
    """Minimal mock skill for read-only operations (no motor keywords)."""

    name = "read_sensor"
    description = "Read a sensor value"
    parameters: dict = {
        "sensor_id": {"type": "string", "description": "ID of the sensor", "required": True},
        "timeout": {"type": "float", "description": "Timeout in seconds", "default": 1.0},
    }
    preconditions: list = []
    effects: dict = {}
    failure_modes: list = []

    def execute(self, params: dict, context: Any) -> MockSkillResult:
        return MockSkillResult(success=True, result_data={"value": 42})


class MockMotorSkill:
    """Mock skill that involves arm movement (motor keyword in preconditions)."""

    name = "move_arm"
    description = "Move the robot arm to a target pose"
    parameters: dict = {
        "target_pose": {"type": "string", "description": "Target pose name"},
    }
    preconditions: list = ["arm must be homed"]
    effects: dict = {"arm_position": "target_pose"}
    failure_modes: list = []

    def execute(self, params: dict, context: Any) -> MockSkillResult:
        return MockSkillResult(success=True, result_data={"moved_to": params.get("target_pose")})


class MockFailingSkill:
    """Mock skill that always fails."""

    name = "fail_skill"
    description = "Always fails"
    parameters: dict = {}
    preconditions: list = []
    effects: dict = {}
    failure_modes: list = []

    def execute(self, params: dict, context: Any) -> MockSkillResult:
        return MockSkillResult(success=False, error_message="Skill execution failed: hardware error")


def _make_agent(skills: list[Any] | None = None) -> MagicMock:
    """Create a mock agent with a skill registry."""
    agent = MagicMock()
    agent._build_context.return_value = MagicMock()
    agent._sync_robot_state.return_value = None

    registry = MagicMock()
    if skills:
        skill_map = {s.name: s for s in skills}
        registry.list_skills.return_value = list(skill_map.keys())
        registry.get.side_effect = lambda name: skill_map.get(name)
    else:
        registry.list_skills.return_value = []
        registry.get.return_value = None

    agent._skill_registry = registry
    return agent


# ---------------------------------------------------------------------------
# test_wrap_single_skill
# ---------------------------------------------------------------------------


class TestWrapSingleSkill:
    def test_wrap_single_skill(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        skill = MockSkill()
        agent = _make_agent([skill])
        wrapper = SkillWrapperTool(skill, agent)

        assert wrapper.name == "read_sensor"
        assert wrapper.description == "Read a sensor value"

    def test_wrap_skill_name_fallback_to_name_attr(self) -> None:
        """description falls back to skill.name when description attr is absent."""
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        class NoDescSkill:
            name = "no_desc"
            parameters: dict = {}
            preconditions: list = []
            effects: dict = {}
            failure_modes: list = []

            def execute(self, params: dict, context: Any) -> MockSkillResult:
                return MockSkillResult(success=True)

        skill = NoDescSkill()
        agent = _make_agent([skill])
        wrapper = SkillWrapperTool(skill, agent)
        assert wrapper.name == "no_desc"
        assert wrapper.description == "no_desc"


# ---------------------------------------------------------------------------
# test_wrapper_input_schema
# ---------------------------------------------------------------------------


class TestWrapperInputSchema:
    def test_wrapper_input_schema(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        skill = MockSkill()
        agent = _make_agent([skill])
        wrapper = SkillWrapperTool(skill, agent)

        schema = wrapper.input_schema
        assert schema["type"] == "object"
        assert "sensor_id" in schema["properties"]
        assert "timeout" in schema["properties"]

    def test_schema_type_mapping(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        skill = MockSkill()
        agent = _make_agent([skill])
        wrapper = SkillWrapperTool(skill, agent)

        props = wrapper.input_schema["properties"]
        assert props["sensor_id"]["type"] == "string"
        assert props["timeout"]["type"] == "number"

    def test_schema_required_excludes_optional(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        skill = MockSkill()
        agent = _make_agent([skill])
        wrapper = SkillWrapperTool(skill, agent)

        required = wrapper.input_schema.get("required", [])
        # sensor_id has required=True (default), timeout has a default so not required
        assert "sensor_id" in required
        assert "timeout" not in required

    def test_schema_empty_parameters(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        skill = MockFailingSkill()
        agent = _make_agent([skill])
        wrapper = SkillWrapperTool(skill, agent)

        schema = wrapper.input_schema
        assert schema["type"] == "object"
        assert schema["properties"] == {}
        assert schema["required"] == []

    def test_schema_description_in_property(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        skill = MockSkill()
        agent = _make_agent([skill])
        wrapper = SkillWrapperTool(skill, agent)

        props = wrapper.input_schema["properties"]
        assert props["sensor_id"].get("description") == "ID of the sensor"


# ---------------------------------------------------------------------------
# test_wrapper_execute_calls_skill
# ---------------------------------------------------------------------------


class TestWrapperExecute:
    def test_wrapper_execute_calls_skill(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        skill = MagicMock()
        skill.name = "mock_skill"
        skill.description = "A mock skill"
        skill.parameters = {}
        skill.preconditions = []
        skill.effects = {}
        skill.execute.return_value = MockSkillResult(success=True, result_data={"x": 1})

        agent = _make_agent([skill])
        wrapper = SkillWrapperTool(skill, agent)
        ctx = _make_context(agent=agent)

        result = wrapper.execute({"param": "value"}, ctx)

        skill.execute.assert_called_once()
        call_params = skill.execute.call_args[0][0]
        assert call_params == {"param": "value"}

    def test_wrapper_formats_result_success(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        skill = MockSkill()
        skill_instance = MagicMock(spec=MockSkill)
        skill_instance.name = skill.name
        skill_instance.description = skill.description
        skill_instance.parameters = skill.parameters
        skill_instance.preconditions = skill.preconditions
        skill_instance.effects = skill.effects
        skill_instance.execute.return_value = MockSkillResult(
            success=True, result_data={"value": 42}
        )

        agent = _make_agent([skill_instance])
        wrapper = SkillWrapperTool(skill_instance, agent)
        ctx = _make_context(agent=agent)

        result = wrapper.execute({}, ctx)

        assert isinstance(result, ToolResult)
        assert result.is_error is False
        assert "read_sensor" in result.content
        assert "succeeded" in result.content.lower()

    def test_wrapper_formats_result_with_data(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        skill = MagicMock()
        skill.name = "data_skill"
        skill.description = "Returns data"
        skill.parameters = {}
        skill.preconditions = []
        skill.effects = {}
        skill.execute.return_value = MockSkillResult(success=True, result_data={"key": "val"})

        agent = _make_agent([skill])
        wrapper = SkillWrapperTool(skill, agent)
        ctx = _make_context(agent=agent)

        result = wrapper.execute({}, ctx)
        assert "key" in result.content or result.metadata.get("key") == "val"

    def test_wrapper_error_result(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        skill = MockFailingSkill()
        skill_instance = MagicMock(spec=MockFailingSkill)
        skill_instance.name = skill.name
        skill_instance.description = skill.description
        skill_instance.parameters = skill.parameters
        skill_instance.preconditions = skill.preconditions
        skill_instance.effects = skill.effects
        skill_instance.execute.return_value = MockSkillResult(
            success=False, error_message="Skill execution failed: hardware error"
        )

        agent = _make_agent([skill_instance])
        wrapper = SkillWrapperTool(skill_instance, agent)
        ctx = _make_context(agent=agent)

        result = wrapper.execute({}, ctx)

        assert isinstance(result, ToolResult)
        assert result.is_error is True
        assert "hardware error" in result.content

    def test_wrapper_error_no_message_uses_fallback(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        skill = MagicMock()
        skill.name = "silent_fail"
        skill.description = "Fails silently"
        skill.parameters = {}
        skill.preconditions = []
        skill.effects = {}
        skill.execute.return_value = MockSkillResult(success=False, error_message="")

        agent = _make_agent([skill])
        wrapper = SkillWrapperTool(skill, agent)
        ctx = _make_context(agent=agent)

        result = wrapper.execute({}, ctx)
        assert result.is_error is True
        assert "silent_fail" in result.content

    def test_wrapper_syncs_robot_state_on_success(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        skill = MagicMock()
        skill.name = "sync_test"
        skill.description = "Test sync"
        skill.parameters = {}
        skill.preconditions = []
        skill.effects = {}
        skill.execute.return_value = MockSkillResult(success=True)

        agent = _make_agent([skill])
        wrapper = SkillWrapperTool(skill, agent)
        ctx = _make_context(agent=agent)

        wrapper.execute({}, ctx)
        agent._sync_robot_state.assert_called_once()


# ---------------------------------------------------------------------------
# Permission checks
# ---------------------------------------------------------------------------


class TestMotorSkillPermission:
    def test_motor_skill_permission_ask(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        skill = MockMotorSkill()
        agent = _make_agent([skill])
        wrapper = SkillWrapperTool(skill, agent)
        ctx = _make_context(agent=agent)

        result = wrapper.check_permissions({}, ctx)
        assert result.behavior == "ask"

    def test_read_skill_permission_allow(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        skill = MockSkill()
        agent = _make_agent([skill])
        wrapper = SkillWrapperTool(skill, agent)
        ctx = _make_context(agent=agent)

        result = wrapper.check_permissions({}, ctx)
        assert result.behavior == "allow"

    def test_motor_keyword_in_effects(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        class GripperSkill:
            name = "close_gripper"
            description = "Close the gripper"
            parameters: dict = {}
            preconditions: list = []
            effects: dict = {"gripper_state": "closed"}
            failure_modes: list = []

            def execute(self, params: dict, context: Any) -> MockSkillResult:
                return MockSkillResult(success=True)

        skill = GripperSkill()
        agent = _make_agent([skill])
        wrapper = SkillWrapperTool(skill, agent)
        ctx = _make_context(agent=agent)

        result = wrapper.check_permissions({}, ctx)
        assert result.behavior == "ask"

    def test_navigate_keyword_is_motor(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        class NavSkill:
            name = "go_somewhere"
            description = "Navigate to a location"
            parameters: dict = {}
            preconditions: list = ["navigate to goal"]
            effects: dict = {}
            failure_modes: list = []

            def execute(self, params: dict, context: Any) -> MockSkillResult:
                return MockSkillResult(success=True)

        skill = NavSkill()
        agent = _make_agent([skill])
        wrapper = SkillWrapperTool(skill, agent)

        assert wrapper.check_permissions({}, _make_context()).behavior == "ask"


# ---------------------------------------------------------------------------
# Concurrency safety
# ---------------------------------------------------------------------------


class TestConcurrencySafety:
    def test_motor_not_concurrency_safe(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        skill = MockMotorSkill()
        agent = _make_agent([skill])
        wrapper = SkillWrapperTool(skill, agent)

        assert wrapper.is_concurrency_safe({}) is False

    def test_read_concurrency_safe(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        skill = MockSkill()
        agent = _make_agent([skill])
        wrapper = SkillWrapperTool(skill, agent)

        assert wrapper.is_concurrency_safe({}) is True

    def test_motor_not_read_only(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        skill = MockMotorSkill()
        agent = _make_agent([skill])
        wrapper = SkillWrapperTool(skill, agent)

        assert wrapper.is_read_only({}) is False

    def test_read_skill_is_read_only(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        skill = MockSkill()
        agent = _make_agent([skill])
        wrapper = SkillWrapperTool(skill, agent)

        assert wrapper.is_read_only({}) is True


# ---------------------------------------------------------------------------
# wrap_skills
# ---------------------------------------------------------------------------


class TestWrapSkillsMultiple:
    def test_wrap_skills_multiple(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import wrap_skills

        skill_a = MockSkill()
        skill_b = MockMotorSkill()
        agent = _make_agent([skill_a, skill_b])

        tools = wrap_skills(agent)

        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "read_sensor" in names
        assert "move_arm" in names

    def test_wrap_skills_returns_wrapper_instances(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import wrap_skills, SkillWrapperTool

        skill = MockSkill()
        agent = _make_agent([skill])

        tools = wrap_skills(agent)
        assert all(isinstance(t, SkillWrapperTool) for t in tools)

    def test_wrap_skills_empty_registry(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import wrap_skills

        agent = _make_agent(skills=[])
        tools = wrap_skills(agent)
        assert tools == []

    def test_wrap_skills_skips_none(self) -> None:
        """registry.get() returning None for a skill name should be skipped."""
        from vector_os_nano.vcli.tools.skill_wrapper import wrap_skills

        agent = MagicMock()
        agent._build_context.return_value = MagicMock()
        agent._sync_robot_state.return_value = None
        registry = MagicMock()
        registry.list_skills.return_value = ["ghost_skill"]
        registry.get.return_value = None  # get() returns None for ghost_skill
        agent._skill_registry = registry

        tools = wrap_skills(agent)
        assert tools == []

    def test_wrap_skills_satisfies_tool_protocol(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import wrap_skills
        from vector_os_nano.vcli.tools.base import Tool

        skill = MockSkill()
        agent = _make_agent([skill])
        tools = wrap_skills(agent)

        for t in tools:
            assert isinstance(t, Tool)
