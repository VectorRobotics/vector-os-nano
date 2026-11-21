# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

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
    """Minimal mock skill for read-only operations (declares it does not actuate)."""

    __skill_actuates__ = False
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
    agent._build_context.return_value = MagicMock(services={})
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

    def test_wrapper_passes_the_runs_cancel_token(self) -> None:
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
        assert skill.execute.call_args.args[1].services["cancel"] is ctx.abort


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


class TestDeclaredActuation:
    """Motor-ness is the skill's declared ``@skill(actuates=...)``, never a keyword
    scan (tools-worlds:motor-keyword-misclassify, skills-nav:motor-misclassified)."""

    def test_go2_locomotion_skills_actuate(self) -> None:
        from vector_os_nano.skills import get_skills
        from vector_os_nano.embodiments.registry import load as load_robot
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        skills = {s.name: s for s in get_skills(load_robot("go2").capabilities)}
        # The keyword scan called these four non-motor: concurrent, auto-allowed on
        # real hardware, no post-state.
        for name in ("walk", "turn", "explore", "patrol", "navigate"):
            wrapper = SkillWrapperTool(skills[name], _make_agent([skills[name]]))
            assert wrapper._is_motor, name
            assert wrapper.is_concurrency_safe({}) is False, name
            assert wrapper.check_permissions({}, _make_context()).behavior == "ask", name
        for name in ("look", "where_am_i", "describe_scene"):
            assert SkillWrapperTool(skills[name], None)._is_motor is False, name

    def test_undeclared_skill_fails_closed_to_actuating(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        class Undeclared:  # no @skill, no keywords anywhere
            name = "mystery"
            description = "Does a thing"
            parameters: dict = {}

        assert SkillWrapperTool(Undeclared(), None)._is_motor is True


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
        agent._build_context.return_value = MagicMock(services={})
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


# ---------------------------------------------------------------------------
# WP14 regressions: schema passthrough, failure data, live agent + live skill
# ---------------------------------------------------------------------------


class TestSchemaPassthrough:
    """tools-worlds:schema-drops-enum-array — enum/default/array reach the model."""

    def test_enum_default_and_array_types_survive(self) -> None:
        from vector_os_nano.skills.go2.patrol import PatrolSkill
        from vector_os_nano.skills.go2.walk import WalkSkill
        from vector_os_nano.skills.place_top_down import PlaceTopDownSkill
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        walk = SkillWrapperTool(WalkSkill(), None).input_schema["properties"]["direction"]
        assert walk["enum"] == ["forward", "backward", "left", "right"]
        assert walk["default"] == "forward"
        rooms = SkillWrapperTool(PatrolSkill(), None).input_schema["properties"]["rooms"]
        assert rooms["type"] == "array" and rooms["items"] == {"type": "string"}
        xyz = SkillWrapperTool(PlaceTopDownSkill(), None).input_schema["properties"]["target_xyz"]
        assert xyz["type"] == "array"  # declared "list"

    def test_json_string_array_is_coerced(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        seen: dict = {}

        class Place:
            name = "place_at"
            parameters = {"target_xyz": {"type": "list", "required": True}}

            def execute(self, params: dict, context: Any) -> MockSkillResult:
                seen.update(params)
                return MockSkillResult(success=True)

        skill = Place()
        agent = _make_agent([skill])
        SkillWrapperTool(skill, agent).execute({"target_xyz": "[3.0, 7.5, 0.66]"}, _make_context(agent))
        assert seen["target_xyz"] == [3.0, 7.5, 0.66]


class TestFailureKeepsData:
    """tools-worlds:wrapper-drops-failure-data — the observation survives a failure."""

    def test_failure_returns_result_data_and_diagnosis_hint(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        class Nav:
            name = "nav"
            parameters: dict = {}

            def execute(self, params: dict, context: Any) -> MockSkillResult:
                return MockSkillResult(
                    success=False, error_message="no map",
                    result_data={"diagnosis": "room_not_explored", "room": "kitchen"})

        skill = Nav()
        agent = _make_agent([skill])
        res = SkillWrapperTool(skill, agent).execute({}, _make_context(agent))
        assert res.is_error
        assert res.metadata["room"] == "kitchen"
        assert "kitchen" in res.content
        assert "Suggested: Room not explored yet" in res.content

    def test_success_does_not_mutate_the_skills_result_data(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        data = {"value": 1}
        skill = MagicMock()
        skill.name, skill.parameters = "motor", {}
        skill.execute.return_value = MockSkillResult(success=True, result_data=data)
        agent = _make_agent([skill])
        agent._base.get_position.return_value = (1.0, 2.0, 0.3)
        agent._base.get_heading.return_value = 0.0
        SkillWrapperTool(skill, agent).execute({}, _make_context(agent))
        assert data == {"value": 1}


class TestLiveAgentAndSkill:
    """tools-worlds:stale-turn-agent + tools-worlds:skill-reload-stale-wrapper."""

    def test_runs_on_the_live_agent_after_an_in_turn_switch(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        old_skill, new_skill = MagicMock(), MagicMock()
        for s in (old_skill, new_skill):
            s.name, s.parameters = "walk", {}
            s.execute.return_value = MockSkillResult(success=True)
        old, new = _make_agent([old_skill]), _make_agent([new_skill])
        ctx = _make_context(agent=old)  # fixed when the turn started
        ctx.app_state = {"agent": new}  # a stop+start inside the turn rebound it
        SkillWrapperTool(new_skill, new).execute({}, ctx)
        new_skill.execute.assert_called_once()
        old_skill.execute.assert_not_called()
        new._build_context.assert_called_once()

    def test_stopped_sim_is_an_error_not_a_call_on_the_dead_agent(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        skill = MagicMock()
        skill.name, skill.parameters = "walk", {}
        agent = _make_agent([skill])
        ctx = _make_context(agent=agent)
        ctx.app_state = {"agent": None}
        res = SkillWrapperTool(skill, agent).execute({}, ctx)
        assert res.is_error and "start_simulation" in res.content
        skill.execute.assert_not_called()

    def test_reloaded_skill_instance_is_the_one_that_runs(self) -> None:
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

        old_skill, new_skill = MagicMock(), MagicMock()
        for s in (old_skill, new_skill):
            s.name, s.parameters = "walk", {}
            s.execute.return_value = MockSkillResult(success=True)
        agent = _make_agent([new_skill])  # skill_reload re-registered the name
        SkillWrapperTool(old_skill, agent).execute({}, _make_context(agent))
        new_skill.execute.assert_called_once()
        old_skill.execute.assert_not_called()
