"""vcli.tools — Tool registry and discovery for Vector CLI's agentic harness."""
from __future__ import annotations

from vector_os_nano.vcli.tools.base import (
    PermissionResult,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    tool,
)

__all__ = [
    "PermissionResult",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "tool",
]


def discover_all_tools(registry: ToolRegistry | None = None) -> ToolRegistry:
    """Discover and register all built-in tools, returning the registry.

    Pass an existing *registry* to extend it; otherwise a new one is created.
    Currently a placeholder — concrete tool modules are imported here as they
    are added to the harness.
    """
    if registry is None:
        registry = ToolRegistry()
    return registry
