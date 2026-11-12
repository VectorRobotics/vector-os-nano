# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Unit tests for CategorizedToolRegistry: register(category), unregister,
list_categories (a copy) — the bookkeeping native_tools._registry_tools reads."""
from __future__ import annotations

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

    def test_list_categories_returns_copy(self) -> None:
        """list_categories() returns a copy — mutating it does not affect the registry."""
        from vector_os_nano.vcli.tools.base import CategorizedToolRegistry

        registry = CategorizedToolRegistry()
        registry.register(_make_mock_read_tool(), category="code")

        categories = registry.list_categories()
        categories["code"].append("injected")

        # Internal state must be unchanged
        assert "injected" not in registry.list_categories()["code"]

    def test_unregister_drops_the_tool_from_every_category(self) -> None:
        """unregister removes the tool and its name from every category list (the
        sim-switch path re-registers skills under "robot")."""
        from vector_os_nano.vcli.tools.base import CategorizedToolRegistry

        registry = CategorizedToolRegistry()
        registry.register(_make_mock_robot_tool(), category="robot")
        registry.register(_make_mock_read_tool(), category="code")
        registry.unregister("mock_robot")
        registry.unregister("never_registered")  # absent name: a no-op

        assert registry.get("mock_robot") is None
        assert registry.list_categories() == {"robot": [], "code": ["mock_read"]}
