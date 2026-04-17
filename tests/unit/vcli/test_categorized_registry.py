"""Unit tests for CategorizedToolRegistry — TDD RED phase.

Covers:
- register with category → list_categories returns correct mapping
- register without category → backward compat, goes to "default"
- disable_category → disabled category tools excluded from to_anthropic_schemas()
- enable_category → re-enabling includes tools again
- to_anthropic_schemas with mixed enabled/disabled categories
- list_categories returns {category: [tool_names]} mapping
"""
from __future__ import annotations

import pytest
from typing import Any


# ---------------------------------------------------------------------------
# Fixtures — minimal mock tools via @tool decorator
# ---------------------------------------------------------------------------


def _make_mock_read_tool():
    from vector_os_nano.vcli.tools.base import tool, ToolResult, ToolContext

    @tool(name="mock_read", description="Mock read tool", read_only=True)
    class MockReadTool:
        def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
            return ToolResult(content="ok")

    return MockReadTool()


def _make_mock_write_tool():
    from vector_os_nano.vcli.tools.base import tool, ToolResult, ToolContext

    @tool(name="mock_write", description="Mock write tool", read_only=False)
    class MockWriteTool:
        def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
            return ToolResult(content="ok")

    return MockWriteTool()


def _make_mock_robot_tool():
    from vector_os_nano.vcli.tools.base import tool, ToolResult, ToolContext

    @tool(name="mock_robot", description="Mock robot tool", read_only=False)
    class MockRobotTool:
        def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
            return ToolResult(content="ok")

    return MockRobotTool()


def _make_mock_diag_tool():
    from vector_os_nano.vcli.tools.base import tool, ToolResult, ToolContext

    @tool(name="mock_diag", description="Mock diag tool", read_only=True)
    class MockDiagTool:
        def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
            return ToolResult(content="ok")

    return MockDiagTool()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCategorizedToolRegistry:
    """Tests for CategorizedToolRegistry."""

    def test_register_with_category(self) -> None:
        """Registering a tool with a category stores it under that category."""
        from vector_os_nano.vcli.tools.base import CategorizedToolRegistry

        registry = CategorizedToolRegistry()
        registry.register(_make_mock_read_tool(), category="code")
        registry.register(_make_mock_write_tool(), category="code")

        categories = registry.list_categories()
        assert "code" in categories
        assert "mock_read" in categories["code"]
        assert "mock_write" in categories["code"]

    def test_register_without_category(self) -> None:
        """Registering without a category places the tool in 'default' category."""
        from vector_os_nano.vcli.tools.base import CategorizedToolRegistry

        registry = CategorizedToolRegistry()
        registry.register(_make_mock_read_tool())

        categories = registry.list_categories()
        assert "default" in categories
        assert "mock_read" in categories["default"]

    def test_disable_category(self) -> None:
        """Disabling a category excludes its tools from to_anthropic_schemas()."""
        from vector_os_nano.vcli.tools.base import CategorizedToolRegistry

        registry = CategorizedToolRegistry()
        registry.register(_make_mock_read_tool(), category="code")
        registry.register(_make_mock_write_tool(), category="code")
        registry.register(_make_mock_robot_tool(), category="robot")

        registry.disable_category("code")

        schemas = registry.to_anthropic_schemas()
        schema_names = {s["name"] for s in schemas}

        # "code" category tools must be absent
        assert "mock_read" not in schema_names
        assert "mock_write" not in schema_names
        # "robot" category tool must still be present
        assert "mock_robot" in schema_names

    def test_enable_category(self) -> None:
        """Re-enabling a disabled category includes its tools again."""
        from vector_os_nano.vcli.tools.base import CategorizedToolRegistry

        registry = CategorizedToolRegistry()
        registry.register(_make_mock_read_tool(), category="code")
        registry.register(_make_mock_robot_tool(), category="robot")

        registry.disable_category("code")
        registry.enable_category("code")

        schemas = registry.to_anthropic_schemas()
        schema_names = {s["name"] for s in schemas}

        assert "mock_read" in schema_names
        assert "mock_robot" in schema_names

    def test_to_anthropic_schemas_filters_disabled(self) -> None:
        """to_anthropic_schemas() returns only tools in enabled categories."""
        from vector_os_nano.vcli.tools.base import CategorizedToolRegistry

        registry = CategorizedToolRegistry()
        registry.register(_make_mock_read_tool(), category="code")
        registry.register(_make_mock_write_tool(), category="code")
        registry.register(_make_mock_robot_tool(), category="robot")
        registry.register(_make_mock_diag_tool(), category="diag")

        registry.disable_category("robot")
        registry.disable_category("diag")

        schemas = registry.to_anthropic_schemas()
        schema_names = {s["name"] for s in schemas}

        assert schema_names == {"mock_read", "mock_write"}

    def test_list_categories(self) -> None:
        """list_categories() returns correct {category: [tool_names]} mapping."""
        from vector_os_nano.vcli.tools.base import CategorizedToolRegistry

        registry = CategorizedToolRegistry()
        registry.register(_make_mock_read_tool(), category="code")
        registry.register(_make_mock_write_tool(), category="code")
        registry.register(_make_mock_robot_tool(), category="robot")
        registry.register(_make_mock_diag_tool(), category="diag")

        categories = registry.list_categories()

        assert set(categories.keys()) == {"code", "robot", "diag"}
        assert set(categories["code"]) == {"mock_read", "mock_write"}
        assert categories["robot"] == ["mock_robot"]
        assert categories["diag"] == ["mock_diag"]

    def test_backward_compat_base_registry_api(self) -> None:
        """All ToolRegistry APIs work unchanged on CategorizedToolRegistry."""
        from vector_os_nano.vcli.tools.base import CategorizedToolRegistry

        registry = CategorizedToolRegistry()
        read_tool = _make_mock_read_tool()
        registry.register(read_tool, category="code")

        # get() by name
        assert registry.get("mock_read") is read_tool

        # list_tools() includes registered tool
        assert "mock_read" in registry.list_tools()

    def test_is_category_enabled_default(self) -> None:
        """Newly registered categories are enabled by default."""
        from vector_os_nano.vcli.tools.base import CategorizedToolRegistry

        registry = CategorizedToolRegistry()
        registry.register(_make_mock_read_tool(), category="code")

        assert registry.is_category_enabled("code") is True

    def test_is_category_enabled_after_disable(self) -> None:
        """is_category_enabled() returns False after disable_category()."""
        from vector_os_nano.vcli.tools.base import CategorizedToolRegistry

        registry = CategorizedToolRegistry()
        registry.register(_make_mock_read_tool(), category="code")

        registry.disable_category("code")
        assert registry.is_category_enabled("code") is False

    def test_is_category_enabled_after_reenable(self) -> None:
        """is_category_enabled() returns True after re-enabling a disabled category."""
        from vector_os_nano.vcli.tools.base import CategorizedToolRegistry

        registry = CategorizedToolRegistry()
        registry.register(_make_mock_read_tool(), category="code")

        registry.disable_category("code")
        registry.enable_category("code")
        assert registry.is_category_enabled("code") is True

    def test_list_categories_returns_copy(self) -> None:
        """list_categories() returns a copy — mutating it does not affect the registry."""
        from vector_os_nano.vcli.tools.base import CategorizedToolRegistry

        registry = CategorizedToolRegistry()
        registry.register(_make_mock_read_tool(), category="code")

        categories = registry.list_categories()
        categories["code"].append("injected")

        # Internal state must be unchanged
        assert "injected" not in registry.list_categories()["code"]

    def test_disable_nonexistent_category_is_safe(self) -> None:
        """Disabling a category that has no tools raises no error."""
        from vector_os_nano.vcli.tools.base import CategorizedToolRegistry

        registry = CategorizedToolRegistry()
        registry.disable_category("phantom")  # should not raise

    def test_enable_never_disabled_category_is_safe(self) -> None:
        """Enabling a category that was never disabled raises no error."""
        from vector_os_nano.vcli.tools.base import CategorizedToolRegistry

        registry = CategorizedToolRegistry()
        registry.enable_category("phantom")  # should not raise
