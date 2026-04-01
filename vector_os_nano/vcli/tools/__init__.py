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
    "discover_all_tools",
]


def discover_all_tools() -> list:
    """Instantiate and return all built-in tool objects.

    Each tool class is imported here to avoid circular imports at module level.
    The caller registers the returned instances into a ToolRegistry.
    """
    from vector_os_nano.vcli.tools.bash_tool import BashTool
    from vector_os_nano.vcli.tools.file_tools import FileEditTool, FileReadTool, FileWriteTool
    from vector_os_nano.vcli.tools.robot import RobotStatusTool, WorldQueryTool
    from vector_os_nano.vcli.tools.search_tools import GlobTool, GrepTool
    from vector_os_nano.vcli.tools.sim_tool import SimStartTool
    from vector_os_nano.vcli.tools.web_tool import WebFetchTool

    return [
        FileReadTool(),
        FileWriteTool(),
        FileEditTool(),
        BashTool(),
        GlobTool(),
        GrepTool(),
        WorldQueryTool(),
        RobotStatusTool(),
        SimStartTool(),
        WebFetchTool(),
    ]
