"""vcli — Vector CLI agentic harness.

Public surface::

    from vector_os_nano.vcli import ToolRegistry, ToolResult, ToolContext, tool
"""
from __future__ import annotations

from vector_os_nano.vcli.tools.base import (
    PermissionResult,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    tool,
)
from vector_os_nano.vcli.tools import discover_all_tools

__all__ = [
    "discover_all_tools",
    "PermissionResult",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "tool",
]
