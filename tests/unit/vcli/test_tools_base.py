"""Unit tests for vcli tool protocol and registry — TDD RED phase.

Covers:
- @tool decorator metadata storage
- @tool without args using class defaults
- ToolRegistry register/list/get/to_anthropic_schemas
- ToolResult frozen dataclass
- ToolContext fields
- Tool Protocol runtime checkable
- is_concurrency_safe and is_read_only defaults
"""
from __future__ import annotations

import dataclasses
import threading
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_tool_result_frozen(self) -> None:
        from vector_os_nano.vcli.tools.base import ToolResult

        r = ToolResult(content="ok")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            r.content = "mutated"  # type: ignore[misc]

    def test_tool_result_defaults(self) -> None:
        from vector_os_nano.vcli.tools.base import ToolResult

        r = ToolResult(content="hello")
        assert r.is_error is False
        assert r.metadata == {}

    def test_tool_result_error_flag(self) -> None:
        from vector_os_nano.vcli.tools.base import ToolResult

        r = ToolResult(content="fail", is_error=True)
        assert r.is_error is True


# ---------------------------------------------------------------------------
# ToolContext
# ---------------------------------------------------------------------------


class TestToolContext:
    def test_tool_context_fields(self) -> None:
        from vector_os_nano.vcli.tools.base import ToolContext

        event = threading.Event()
        ctx = ToolContext(
            agent=None,
            cwd=Path("/tmp"),
            session=object(),
            permissions=None,
            abort=event,
        )
        assert ctx.agent is None
        assert ctx.cwd == Path("/tmp")
        assert ctx.abort is event

    def test_tool_context_agent_any(self) -> None:
        from vector_os_nano.vcli.tools.base import ToolContext

        sentinel = object()
        ctx = ToolContext(
            agent=sentinel,
            cwd=Path("/"),
            session=None,
            permissions=None,
            abort=threading.Event(),
        )
        assert ctx.agent is sentinel


# ---------------------------------------------------------------------------
# Tool Protocol
# ---------------------------------------------------------------------------


class TestToolProtocol:
    def test_tool_protocol_runtime_checkable(self) -> None:
        from vector_os_nano.vcli.tools.base import Tool

        # A plain object with no relevant attributes is NOT a Tool
        assert not isinstance(object(), Tool)

    def test_tool_protocol_satisfied_by_impl(self) -> None:
        from vector_os_nano.vcli.tools.base import Tool, ToolResult, ToolContext, PermissionResult

        class MyTool:
            name = "my_tool"
            description = "A test tool"
            input_schema: dict[str, Any] = {"type": "object", "properties": {}}

            def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
                return ToolResult(content="done")

            def check_permissions(
                self, params: dict[str, Any], context: ToolContext
            ) -> PermissionResult:
                return PermissionResult(behavior="allow")

            def is_read_only(self, params: dict[str, Any]) -> bool:
                return False

            def is_concurrency_safe(self, params: dict[str, Any]) -> bool:
                return False

        assert isinstance(MyTool(), Tool)


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------


class TestToolDecorator:
    def test_tool_decorator_registers_metadata(self) -> None:
        from vector_os_nano.vcli.tools.base import tool

        @tool(
            name="test_cmd",
            description="Does a test thing",
            input_schema={"type": "object", "properties": {}},
            permission="allow",
            read_only=True,
        )
        class TestCmd:
            pass

        assert TestCmd.__tool_name__ == "test_cmd"
        assert TestCmd.__tool_description__ == "Does a test thing"
        assert TestCmd.__tool_read_only__ is True
        assert TestCmd.__tool_permission__ == "allow"
        assert "type" in TestCmd.__tool_input_schema__

    def test_tool_decorator_no_args_uses_class_defaults(self) -> None:
        from vector_os_nano.vcli.tools.base import tool

        @tool()
        class NoArgsTool:
            name = "no_args_tool"
            description = "Default description"
            input_schema: dict[str, Any] = {"type": "object", "properties": {}}

        assert NoArgsTool.__tool_name__ == "no_args_tool"
        assert NoArgsTool.__tool_description__ == "Default description"

    def test_tool_decorator_default_read_only_false(self) -> None:
        from vector_os_nano.vcli.tools.base import tool

        @tool(name="cmd", description="desc")
        class Cmd:
            pass

        assert Cmd.__tool_read_only__ is False

    def test_tool_decorator_default_permission_allow(self) -> None:
        from vector_os_nano.vcli.tools.base import tool

        @tool(name="cmd2", description="desc2")
        class Cmd2:
            pass

        assert Cmd2.__tool_permission__ == "allow"

    def test_tool_is_read_only_default(self) -> None:
        """Decorated class instance.is_read_only() returns False by default."""
        from vector_os_nano.vcli.tools.base import tool, ToolResult, ToolContext

        @tool(name="ro_default", description="desc")
        class RoDefault:
            def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
                return ToolResult(content="x")

        instance = RoDefault()
        assert instance.is_read_only({}) is False

    def test_tool_is_concurrency_safe_default(self) -> None:
        """Decorated class instance.is_concurrency_safe() returns False by default."""
        from vector_os_nano.vcli.tools.base import tool, ToolResult, ToolContext

        @tool(name="cs_default", description="desc")
        class CsDefault:
            def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
                return ToolResult(content="x")

        instance = CsDefault()
        assert instance.is_concurrency_safe({}) is False


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def _make_tool_class(self, name: str, description: str = "desc"):
        from vector_os_nano.vcli.tools.base import tool, ToolResult, ToolContext

        @tool(name=name, description=description)
        class _T:
            def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
                return ToolResult(content="ok")

        _T.__name__ = name
        return _T

    def test_tool_registry_register_and_list(self) -> None:
        from vector_os_nano.vcli.tools import ToolRegistry

        registry = ToolRegistry()
        ToolA = self._make_tool_class("tool_a")
        ToolB = self._make_tool_class("tool_b")
        registry.register(ToolA())
        registry.register(ToolB())
        names = registry.list_tools()
        assert "tool_a" in names
        assert "tool_b" in names

    def test_tool_registry_get_by_name(self) -> None:
        from vector_os_nano.vcli.tools import ToolRegistry

        registry = ToolRegistry()
        ToolX = self._make_tool_class("tool_x", "X tool")
        instance = ToolX()
        registry.register(instance)
        retrieved = registry.get("tool_x")
        assert retrieved is instance

    def test_tool_registry_get_missing_returns_none(self) -> None:
        from vector_os_nano.vcli.tools import ToolRegistry

        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_tool_registry_to_anthropic_schemas(self) -> None:
        from vector_os_nano.vcli.tools import ToolRegistry
        from vector_os_nano.vcli.tools.base import tool, ToolResult, ToolContext

        @tool(
            name="schema_tool",
            description="A schema tool",
            input_schema={"type": "object", "properties": {"x": {"type": "number"}}},
        )
        class SchemaTool:
            def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
                return ToolResult(content="ok")

        registry = ToolRegistry()
        registry.register(SchemaTool())
        schemas = registry.to_anthropic_schemas()
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["name"] == "schema_tool"
        assert schema["description"] == "A schema tool"
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"

    def test_tool_registry_to_anthropic_schemas_required_keys(self) -> None:
        from vector_os_nano.vcli.tools import ToolRegistry

        registry = ToolRegistry()
        ToolC = self._make_tool_class("tool_c", "C desc")
        registry.register(ToolC())
        schemas = registry.to_anthropic_schemas()
        assert all(
            {"name", "description", "input_schema"}.issubset(s.keys()) for s in schemas
        )


# ---------------------------------------------------------------------------
# PermissionResult
# ---------------------------------------------------------------------------


class TestPermissionResult:
    def test_permission_result_frozen(self) -> None:
        from vector_os_nano.vcli.tools.base import PermissionResult

        pr = PermissionResult(behavior="allow")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            pr.behavior = "deny"  # type: ignore[misc]

    def test_permission_result_default_reason(self) -> None:
        from vector_os_nano.vcli.tools.base import PermissionResult

        pr = PermissionResult(behavior="ask")
        assert pr.reason == ""
        assert pr.behavior == "ask"


# ---------------------------------------------------------------------------
# vcli __init__ exports
# ---------------------------------------------------------------------------


class TestVcliExports:
    def test_vcli_init_exports(self) -> None:
        import vector_os_nano.vcli as vcli

        # These names must be importable from the package
        assert hasattr(vcli, "ToolRegistry")
        assert hasattr(vcli, "ToolResult")
        assert hasattr(vcli, "ToolContext")


import pytest
