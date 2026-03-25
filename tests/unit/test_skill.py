"""Unit tests for vector_os_nano.core.skill — TDD RED phase."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


def make_mock_skill(name: str, description: str = "A skill",
                    parameters: dict | None = None,
                    preconditions: list[str] | None = None,
                    postconditions: list[str] | None = None) -> Any:
    """Create a MagicMock that satisfies the Skill protocol."""
    from vector_os_nano.core.types import SkillResult
    skill = MagicMock()
    skill.name = name
    skill.description = description
    skill.parameters = parameters or {}
    skill.preconditions = preconditions or []
    skill.postconditions = postconditions or []
    skill.effects = {}
    skill.failure_modes = []
    skill.execute.return_value = SkillResult(success=True)
    return skill


class TestSkillRegistry:
    def test_register_and_get(self):
        from vector_os_nano.core.skill import SkillRegistry
        registry = SkillRegistry()
        skill = make_mock_skill("pick")
        registry.register(skill)
        retrieved = registry.get("pick")
        assert retrieved is skill

    def test_get_nonexistent_returns_none(self):
        from vector_os_nano.core.skill import SkillRegistry
        registry = SkillRegistry()
        assert registry.get("nonexistent") is None

    def test_register_overwrites_existing(self):
        from vector_os_nano.core.skill import SkillRegistry
        registry = SkillRegistry()
        s1 = make_mock_skill("pick", description="v1")
        s2 = make_mock_skill("pick", description="v2")
        registry.register(s1)
        registry.register(s2)
        retrieved = registry.get("pick")
        assert retrieved.description == "v2"

    def test_list_skills_empty(self):
        from vector_os_nano.core.skill import SkillRegistry
        registry = SkillRegistry()
        assert registry.list_skills() == []

    def test_list_skills_returns_names(self):
        from vector_os_nano.core.skill import SkillRegistry
        registry = SkillRegistry()
        registry.register(make_mock_skill("pick"))
        registry.register(make_mock_skill("place"))
        registry.register(make_mock_skill("home"))
        names = registry.list_skills()
        assert set(names) == {"pick", "place", "home"}

    def test_list_skills_returns_list(self):
        from vector_os_nano.core.skill import SkillRegistry
        registry = SkillRegistry()
        registry.register(make_mock_skill("pick"))
        result = registry.list_skills()
        assert isinstance(result, list)

    def test_to_schemas_empty(self):
        from vector_os_nano.core.skill import SkillRegistry
        registry = SkillRegistry()
        schemas = registry.to_schemas()
        assert schemas == []

    def test_to_schemas_one_skill(self):
        from vector_os_nano.core.skill import SkillRegistry
        registry = SkillRegistry()
        skill = make_mock_skill(
            "pick",
            description="Pick up an object",
            parameters={"object_label": {"type": "string"}},
        )
        registry.register(skill)
        schemas = registry.to_schemas()
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["name"] == "pick"
        assert schema["description"] == "Pick up an object"

    def test_to_schemas_all_skills(self):
        from vector_os_nano.core.skill import SkillRegistry
        registry = SkillRegistry()
        for name in ["pick", "place", "home", "scan", "detect"]:
            registry.register(make_mock_skill(name))
        schemas = registry.to_schemas()
        assert len(schemas) == 5

    def test_to_schemas_includes_parameters(self):
        from vector_os_nano.core.skill import SkillRegistry
        params = {
            "object_label": {"type": "string", "description": "Object to pick"},
            "speed": {"type": "number", "default": 1.0},
        }
        skill = make_mock_skill("pick", parameters=params)
        registry = SkillRegistry()
        registry.register(skill)
        schemas = registry.to_schemas()
        assert "parameters" in schemas[0]
        assert schemas[0]["parameters"] == params


class TestSkillContext:
    def test_creation_with_required_fields(self):
        from vector_os_nano.core.skill import SkillContext
        from vector_os_nano.core.world_model import WorldModel
        wm = WorldModel()
        context = SkillContext(
            arm=MagicMock(),
            gripper=MagicMock(),
            perception=None,
            world_model=wm,
            calibration=None,
        )
        assert context.world_model is wm
        assert context.perception is None
        assert context.calibration is None

    def test_creation_with_optional_fields(self):
        from vector_os_nano.core.skill import SkillContext
        from vector_os_nano.core.world_model import WorldModel
        wm = WorldModel()
        arms = {"left": MagicMock(), "right": MagicMock()}
        base = MagicMock()
        context = SkillContext(
            arm=MagicMock(),
            gripper=MagicMock(),
            perception=MagicMock(),
            world_model=wm,
            calibration=MagicMock(),
            arms=arms,
            base=base,
        )
        assert context.arms == arms
        assert context.base is base

    def test_creation_default_config(self):
        from vector_os_nano.core.skill import SkillContext
        from vector_os_nano.core.world_model import WorldModel
        context = SkillContext(
            arm=MagicMock(),
            gripper=MagicMock(),
            perception=None,
            world_model=WorldModel(),
            calibration=None,
        )
        assert isinstance(context.config, dict)
        assert len(context.config) == 0

    def test_arms_default_empty_no_base(self):
        """When no dict registries are passed, arms is empty dict and base is None."""
        from vector_os_nano.core.skill import SkillContext
        from vector_os_nano.core.world_model import WorldModel
        context = SkillContext(
            arm=MagicMock(),
            gripper=MagicMock(),
            perception=None,
            world_model=WorldModel(),
            calibration=None,
        )
        # arms dict is empty when no dict-style registry was provided
        assert isinstance(context.arms, dict)
        assert len(context.arms) == 0
        # base property returns None when no bases dict and no legacy base
        assert context.base is None


class TestSkillProtocol:
    def test_mock_skill_satisfies_protocol(self):
        from vector_os_nano.core.skill import Skill
        skill = make_mock_skill("pick")
        # runtime_checkable protocol check
        assert isinstance(skill, Skill)

    def test_skill_result_attributes(self):
        from vector_os_nano.core.types import SkillResult
        result = SkillResult(success=True, result_data={"pos": [0.2, 0.0, 0.1]})
        assert result.success is True
        assert "pos" in result.result_data

    def test_skill_result_failure(self):
        from vector_os_nano.core.types import SkillResult
        result = SkillResult(success=False, error_message="IK failed")
        assert result.success is False
        assert result.error_message == "IK failed"
