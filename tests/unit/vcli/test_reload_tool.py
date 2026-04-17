"""Tests for SkillReloadTool.

TDD RED phase — all tests must fail before implementation is written.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from vector_os_nano.vcli.tools.base import ToolContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_registry(skill_name: str = "stand") -> MagicMock:
    """Create a mock skill registry with a fake skill."""
    registry = MagicMock()

    fake_skill = MagicMock()
    fake_skill.__class__ = type(
        "StandSkill",
        (),
        {"__module__": "vector_os_nano.skills.go2.stance"},
    )
    fake_skill.name = skill_name

    def _get(name: str):
        return fake_skill if name == skill_name else None

    registry.get.side_effect = _get
    registry.list_skills.return_value = [skill_name, "walk", "sit"]
    return registry


def _make_context(registry=None, use_app_state: bool = True) -> MagicMock:
    """Build a minimal ToolContext mock that exposes the skill registry."""
    ctx = MagicMock(spec=ToolContext)
    if use_app_state:
        ctx.app_state = {"skill_registry": registry}
        ctx.agent = None
    else:
        ctx.app_state = None
        agent = MagicMock()
        agent._skill_registry = registry
        ctx.agent = agent
    return ctx


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def tool():
    from vector_os_nano.vcli.tools.reload_tool import SkillReloadTool
    return SkillReloadTool()


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------


class TestToolMetadata:
    def test_tool_name(self, tool):
        assert tool.name == "skill_reload"

    def test_not_read_only(self, tool):
        assert tool.is_read_only({}) is False

    def test_permission_ask(self, tool):
        ctx = _make_context(_make_mock_registry())
        result = tool.check_permissions({"skill_name": "stand"}, ctx)
        assert result.behavior == "ask"

    def test_input_schema_has_skill_name(self, tool):
        assert "skill_name" in tool.input_schema["properties"]
        assert "skill_name" in tool.input_schema["required"]


# ---------------------------------------------------------------------------
# Core behaviour tests
# ---------------------------------------------------------------------------


class TestReloadKnownSkill:
    """test_reload_known_skill — mock a skill in registry, reload it, verify new instance
    registered."""

    def test_reload_registers_new_instance(self, tool):
        """After a successful reload, registry.register is called with the new instance."""
        registry = _make_mock_registry("stand")
        ctx = _make_context(registry)

        # Build a fake module containing a StandSkill class
        fake_module = types.ModuleType("vector_os_nano.skills.go2.stance")
        new_instance_holder: list = []

        class StandSkill:
            name = "stand"

            def __init__(self):
                new_instance_holder.append(self)

        fake_module.StandSkill = StandSkill

        with patch.dict(sys.modules, {"vector_os_nano.skills.go2.stance": fake_module}):
            with patch("importlib.reload", return_value=fake_module):
                result = tool.execute({"skill_name": "stand"}, ctx)

        assert not result.is_error, result.content
        assert registry.register.called
        # The registered instance should be a StandSkill
        registered = registry.register.call_args[0][0]
        assert isinstance(registered, StandSkill)

    def test_reload_success_message_contains_skill_name(self, tool):
        registry = _make_mock_registry("stand")
        ctx = _make_context(registry)

        fake_module = types.ModuleType("vector_os_nano.skills.go2.stance")

        class StandSkill:
            name = "stand"

        fake_module.StandSkill = StandSkill

        with patch.dict(sys.modules, {"vector_os_nano.skills.go2.stance": fake_module}):
            with patch("importlib.reload", return_value=fake_module):
                result = tool.execute({"skill_name": "stand"}, ctx)

        assert "stand" in result.content

    def test_reload_metadata_includes_module_and_class(self, tool):
        registry = _make_mock_registry("stand")
        ctx = _make_context(registry)

        fake_module = types.ModuleType("vector_os_nano.skills.go2.stance")

        class StandSkill:
            name = "stand"

        fake_module.StandSkill = StandSkill

        with patch.dict(sys.modules, {"vector_os_nano.skills.go2.stance": fake_module}):
            with patch("importlib.reload", return_value=fake_module):
                result = tool.execute({"skill_name": "stand"}, ctx)

        assert result.metadata.get("skill_name") == "stand"
        assert result.metadata.get("module") == "vector_os_nano.skills.go2.stance"
        assert result.metadata.get("class") == "StandSkill"


class TestReloadUnknownSkill:
    """test_reload_unknown_skill — skill not in registry, verify error message."""

    def test_error_when_skill_not_found(self, tool):
        registry = _make_mock_registry("stand")  # only "stand" is registered
        ctx = _make_context(registry)

        result = tool.execute({"skill_name": "fly"}, ctx)

        assert result.is_error
        assert "fly" in result.content

    def test_error_lists_available_skills(self, tool):
        registry = _make_mock_registry("stand")
        ctx = _make_context(registry)

        result = tool.execute({"skill_name": "unknown_cmd"}, ctx)

        assert result.is_error
        # Available skills list should appear in the error message
        assert "stand" in result.content or "walk" in result.content or "sit" in result.content


class TestReloadPreservesOtherSkills:
    """test_reload_preserves_other_skills — after reloading one, others unchanged."""

    def test_only_reloaded_skill_is_re_registered(self, tool):
        """registry.register must be called exactly once (for the reloaded skill only)."""
        registry = _make_mock_registry("stand")
        ctx = _make_context(registry)

        fake_module = types.ModuleType("vector_os_nano.skills.go2.stance")

        class StandSkill:
            name = "stand"

        fake_module.StandSkill = StandSkill

        with patch.dict(sys.modules, {"vector_os_nano.skills.go2.stance": fake_module}):
            with patch("importlib.reload", return_value=fake_module):
                tool.execute({"skill_name": "stand"}, ctx)

        assert registry.register.call_count == 1


class TestReloadModuleSyntaxError:
    """test_reload_module_syntax_error — importlib.reload raises SyntaxError, verify graceful
    error."""

    def test_syntax_error_returns_error_result(self, tool):
        registry = _make_mock_registry("stand")
        ctx = _make_context(registry)

        fake_module = types.ModuleType("vector_os_nano.skills.go2.stance")

        with patch.dict(sys.modules, {"vector_os_nano.skills.go2.stance": fake_module}):
            with patch("importlib.reload", side_effect=SyntaxError("bad syntax")):
                result = tool.execute({"skill_name": "stand"}, ctx)

        assert result.is_error
        assert "syntax" in result.content.lower() or "SyntaxError" in result.content

    def test_syntax_error_does_not_call_register(self, tool):
        registry = _make_mock_registry("stand")
        ctx = _make_context(registry)

        fake_module = types.ModuleType("vector_os_nano.skills.go2.stance")

        with patch.dict(sys.modules, {"vector_os_nano.skills.go2.stance": fake_module}):
            with patch("importlib.reload", side_effect=SyntaxError("bad syntax")):
                tool.execute({"skill_name": "stand"}, ctx)

        registry.register.assert_not_called()


class TestRegistryDiscovery:
    """Tool finds the registry from app_state or agent._skill_registry."""

    def test_registry_from_agent_attribute(self, tool):
        """When app_state is None, fall back to agent._skill_registry."""
        registry = _make_mock_registry("stand")
        ctx = _make_context(registry, use_app_state=False)

        fake_module = types.ModuleType("vector_os_nano.skills.go2.stance")

        class StandSkill:
            name = "stand"

        fake_module.StandSkill = StandSkill

        with patch.dict(sys.modules, {"vector_os_nano.skills.go2.stance": fake_module}):
            with patch("importlib.reload", return_value=fake_module):
                result = tool.execute({"skill_name": "stand"}, ctx)

        assert not result.is_error

    def test_no_registry_returns_error(self, tool):
        """If no registry is reachable at all, return an error."""
        ctx = MagicMock(spec=ToolContext)
        ctx.app_state = None
        ctx.agent = None

        result = tool.execute({"skill_name": "stand"}, ctx)

        assert result.is_error
        assert "registry" in result.content.lower()
