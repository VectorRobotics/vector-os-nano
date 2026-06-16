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
    "discover_categorized_tools",
]


def discover_all_tools() -> list:
    """Instantiate and return all built-in tool objects (flat list, backward compat).

    Each tool class is imported here to avoid circular imports at module level.
    The caller registers the returned instances into a ToolRegistry.
    """
    from vector_os_nano.vcli.tools.bash_tool import BashTool
    from vector_os_nano.vcli.tools.file_tools import FileEditTool, FileReadTool, FileWriteTool
    from vector_os_nano.vcli.tools.robot import RobotStatusTool, WorldQueryTool
    from vector_os_nano.vcli.tools.search_tools import GlobTool, GrepTool
    from vector_os_nano.vcli.tools.sim_tool import SimStartTool, SimStopTool
    from vector_os_nano.vcli.tools.switch_tool import SwitchEmbodimentTool
    # Old MuJoCo-era SysnavSimTool (sysnav_sim_tool.py) stays unregistered —
    # superseded by the habitat runner + SysnavPerceptionTool below.
    from vector_os_nano.vcli.tools.sysnav_tool import SysnavPerceptionTool
    from vector_os_nano.vcli.tools.web_tool import WebFetchTool
    from vector_os_nano.vcli.tools.scene_graph_tool import SceneGraphQueryTool
    from vector_os_nano.vcli.tools.ros2_tools import Ros2TopicsTool, Ros2NodesTool, Ros2LogTool
    from vector_os_nano.vcli.tools.nav_tools import NavStateTool, TerrainStatusTool
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
        WorldQueryTool(),
        RobotStatusTool(),
        SimStartTool(),
        SimStopTool(),
        SwitchEmbodimentTool(),
        SysnavPerceptionTool(),
        WebFetchTool(),
        # New Wave 1-2 tools
        SceneGraphQueryTool(),
        Ros2TopicsTool(),
        Ros2NodesTool(),
        Ros2LogTool(),
        NavStateTool(),
        TerrainStatusTool(),
        SkillReloadTool(),
        FoxgloveTool(),
    ]


# Category assignments for CategorizedToolRegistry.
# "code" + "general" are the domain-general kernel toolset (work with no robot).
# "robot" / "diag" / "system" are robot-world tools (disabled when no robot agent).
_TOOL_CATEGORIES: dict[str, list[str]] = {
    "code": ["file_read", "file_write", "file_edit", "bash", "glob", "grep"],
    "general": ["web_fetch"],
    "robot": ["world_query", "scene_graph_query"],
    "diag": ["ros2_topics", "ros2_nodes", "ros2_log", "nav_state", "terrain_status"],
    # Sim lifecycle: kept enabled even in the dev (no-robot) world so the user can
    # spin up a sim conversationally ("start the arm sim", "启动habitat模拟",
    # "启动sysnav").
    "sim": ["start_simulation", "stop_simulation", "switch_embodiment", "sysnav_perception"],
    "system": ["robot_status", "skill_reload", "open_foxglove"],
}


def discover_categorized_tools() -> tuple[list, dict[str, list[str]]]:
    """Return (tools_list, categories_dict) for CategorizedToolRegistry.

    Returns:
        Tuple of (list of tool instances, dict mapping category name to tool names).
    """
    tools = discover_all_tools()
    return tools, dict(_TOOL_CATEGORIES)
