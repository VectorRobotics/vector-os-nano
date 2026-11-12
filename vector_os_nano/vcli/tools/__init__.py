# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""vcli.tools — Tool registry and discovery for Vector CLI's agentic harness."""
from __future__ import annotations

from vector_os_nano.vcli.tools.base import (
    CategorizedToolRegistry,
    PermissionResult,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    tool,
)

__all__ = [
    "CategorizedToolRegistry",
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

    Each tool class is imported here to avoid circular imports at module level. Each
    tool declares its own registry category (``@tool(category=...)``).
    """
    from vector_os_nano.vcli.tools.bash_tool import BashTool
    from vector_os_nano.vcli.tools.file_tools import FileEditTool, FileReadTool, FileWriteTool
    from vector_os_nano.vcli.tools.robot import RobotStatusTool
    from vector_os_nano.vcli.tools.search_tools import GlobTool, GrepTool
    from vector_os_nano.vcli.tools.sim_tool import SimStartTool, SimStopTool
    from vector_os_nano.vcli.tools.web_tool import WebFetchTool
    from vector_os_nano.vcli.tools.scene_graph_tool import SceneGraphQueryTool
    from vector_os_nano.vcli.tools.nav_tools import GoToRoomTool, NavStateTool, TerrainStatusTool
    from vector_os_nano.vcli.tools.reload_tool import SkillReloadTool
    from vector_os_nano.vcli.tools.viz_tool import FoxgloveTool

    return [
        # Existing tools
        FileReadTool(),
        FileWriteTool(),
        FileEditTool(),
        BashTool(),
        GlobTool(),
        GrepTool(),
        RobotStatusTool(),
        SimStartTool(),
        SimStopTool(),
        WebFetchTool(),
        # New Wave 1-2 tools
        SceneGraphQueryTool(),
        GoToRoomTool(),
        NavStateTool(),
        TerrainStatusTool(),
        SkillReloadTool(),
        FoxgloveTool(),
    ]
