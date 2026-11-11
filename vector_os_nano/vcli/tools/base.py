# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tool Protocol, decorators, and core data types for Vector CLI's agentic harness.

Public exports:
    ToolResult       — frozen dataclass for tool execution output
    PermissionResult — frozen dataclass for permission check output
    ToolContext      — mutable dataclass carrying per-call execution context
    Tool             — runtime-checkable Protocol that all tools must satisfy
    tool             — class decorator that stamps metadata and injects default helpers
    ToolRegistry     — register / look up / enumerate tools
"""
from __future__ import annotations

import dataclasses
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolResult:
    """Outcome returned by Tool.execute()."""

    content: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PermissionResult:
    """Outcome returned by Tool.check_permissions()."""

    behavior: str  # "allow" | "deny" | "ask"
    reason: str = ""


@dataclass
class ToolContext:
    """Per-call context passed to every tool method."""

    agent: Any | None
    cwd: Path
    session: Any
    permissions: Any
    abort: threading.Event
    app_state: dict[str, Any] | None = None  # mutable CLI state for runtime tools


def live_agent(context: ToolContext) -> Any:
    """The agent to act on NOW: the REPL's live ``app_state['agent']`` (a sim switch
    inside a turn rebinds it), else the ``context.agent`` the turn started with."""
    app = context.app_state or {}
    return app["agent"] if "agent" in app else context.agent


# ---------------------------------------------------------------------------
# Tool Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Tool(Protocol):
    """Structural interface that every Vector CLI tool must satisfy."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        ...

    def check_permissions(
        self, params: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        ...

    def is_read_only(self, params: dict[str, Any]) -> bool:
        ...

    def is_concurrency_safe(self, params: dict[str, Any]) -> bool:
        ...


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------

_SENTINEL = object()


def tool(
    name: str | object = _SENTINEL,
    description: str | object = _SENTINEL,
    input_schema: dict[str, Any] | None = None,
    permission: str = "allow",
    read_only: bool = False,
):
    """Class decorator that stamps tool metadata and injects default helpers.

    Usage (explicit args)::

        @tool(name="my_cmd", description="Does something", read_only=True)
        class MyCmd:
            def execute(self, params, context): ...

    Usage (rely on class attributes)::

        @tool()
        class MyCmd:
            name = "my_cmd"
            description = "Does something"
            input_schema = {...}
    """

    def decorator(cls: type) -> type:
        # Resolve name from decorator arg or class attribute
        resolved_name: str
        if name is not _SENTINEL:
            resolved_name = str(name)
        elif hasattr(cls, "name"):
            resolved_name = cls.name
        else:
            resolved_name = cls.__name__

        # Resolve description
        resolved_desc: str
        if description is not _SENTINEL:
            resolved_desc = str(description)
        elif hasattr(cls, "description"):
            resolved_desc = cls.description
        else:
            resolved_desc = ""

        # Resolve input_schema
        resolved_schema: dict[str, Any]
        if input_schema is not None:
            resolved_schema = input_schema
        elif hasattr(cls, "input_schema"):
            resolved_schema = cls.input_schema
        else:
            resolved_schema = {"type": "object", "properties": {}}

        # Stamp metadata attributes on the class
        cls.__tool_name__ = resolved_name
        cls.__tool_description__ = resolved_desc
        cls.__tool_input_schema__ = resolved_schema
        cls.__tool_permission__ = permission
        cls.__tool_read_only__ = read_only

        # Inject name / description / input_schema as class attributes when absent
        if not hasattr(cls, "name") or not isinstance(cls.__dict__.get("name"), str):
            cls.name = resolved_name  # type: ignore[attr-defined]
        if not hasattr(cls, "description") or not isinstance(
            cls.__dict__.get("description"), str
        ):
            cls.description = resolved_desc  # type: ignore[attr-defined]
        if "input_schema" not in cls.__dict__:
            cls.input_schema = resolved_schema  # type: ignore[attr-defined]

        # Inject default method implementations when not provided by the class
        _ro = read_only

        if "is_read_only" not in cls.__dict__:

            def is_read_only(self, params: dict[str, Any]) -> bool:  # noqa: D401
                return _ro

            cls.is_read_only = is_read_only  # type: ignore[attr-defined]

        if "is_concurrency_safe" not in cls.__dict__:

            def is_concurrency_safe(self, params: dict[str, Any]) -> bool:  # noqa: D401
                return False

            cls.is_concurrency_safe = is_concurrency_safe  # type: ignore[attr-defined]

        if "check_permissions" not in cls.__dict__:
            _perm = permission

            def check_permissions(  # noqa: D401
                self,
                params: dict[str, Any],
                context: ToolContext,
            ) -> PermissionResult:
                return PermissionResult(behavior=_perm)

            cls.check_permissions = check_permissions  # type: ignore[attr-defined]

        return cls

    return decorator


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Collects Tool instances and provides look-up."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool_instance: Tool) -> None:
        """Add a tool instance to the registry (keyed by tool.name)."""
        self._tools[tool_instance.name] = tool_instance

    def get(self, name: str) -> Tool | None:
        """Return the tool with *name*, or None if not found."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools.keys())


# ---------------------------------------------------------------------------
# CategorizedToolRegistry
# ---------------------------------------------------------------------------


class CategorizedToolRegistry(ToolRegistry):
    """Tool registry with category-based organization.

    Categories group tools (code, sim, robot, diag, system); the native loop picks
    its action surface by category (``native_loop._registry_tools``).
    """

    def __init__(self) -> None:
        super().__init__()
        self._categories: dict[str, list[str]] = {}

    def register(self, tool_instance: Any, category: str = "default") -> None:  # type: ignore[override]
        super().register(tool_instance)
        names = self._categories.setdefault(category, [])
        if tool_instance.name not in names:
            names.append(tool_instance.name)

    def unregister(self, name: str) -> None:
        """Remove a tool by name from the registry and every category list."""
        self._tools.pop(name, None)
        for names in self._categories.values():
            if name in names:
                names.remove(name)

    def list_categories(self) -> dict[str, list[str]]:
        """Return a copy of the {category: [tool_names]} mapping."""
        return {k: list(v) for k, v in self._categories.items()}
