"""Navigation and terrain diagnostic tools for Vector CLI."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from vector_os_nano.vcli.tools.base import ToolContext, ToolResult, tool


# ---------------------------------------------------------------------------
# Lazy helpers — gracefully degrade when explore module is not importable
# ---------------------------------------------------------------------------


def _is_exploring() -> bool:
    try:
        from vector_os_nano.skills.go2.explore import is_exploring
        return is_exploring()
    except ImportError:
        return False


def _is_nav_stack_running() -> bool:
    try:
        from vector_os_nano.skills.go2.explore import is_nav_stack_running
        return is_nav_stack_running()
    except ImportError:
        return False


def _get_explored_rooms() -> list[str]:
    try:
        from vector_os_nano.skills.go2.explore import get_explored_rooms
        return get_explored_rooms()
    except ImportError:
        return []


def _is_process_running(name: str) -> bool:
    if not shutil.which("pgrep"):
        return False
    try:
        result = subprocess.run(
            ["pgrep", "-f", name], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# NavStateTool
# ---------------------------------------------------------------------------


@tool(
    name="nav_state",
    description=(
        "Get current navigation and exploration state: exploring, nav stack running, "
        "explored rooms, TARE/FAR process status."
    ),
    read_only=True,
    permission="allow",
)
class NavStateTool:
    """Read-only diagnostic tool for navigation and exploration state."""

    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        data: dict[str, Any] = {
            "exploring": _is_exploring(),
            "nav_stack_running": _is_nav_stack_running(),
            "nav_flag_active": os.path.exists("/tmp/vector_nav_active"),
            "explored_rooms": _get_explored_rooms(),
            "tare_running": _is_process_running("tare_planner_node"),
            "far_running": _is_process_running("far_planner"),
        }
        return ToolResult(content=json.dumps(data, indent=2))

    def is_concurrency_safe(self, params: dict[str, Any]) -> bool:
        return True


# ---------------------------------------------------------------------------
# TerrainStatusTool
# ---------------------------------------------------------------------------

_TERRAIN_PATH: str = os.path.expanduser("~/.vector_os_nano/terrain_map.npz")


@tool(
    name="terrain_status",
    description=(
        "Check terrain map persistence: file exists, size, voxel count, replay state."
    ),
    read_only=True,
    permission="allow",
)
class TerrainStatusTool:
    """Read-only diagnostic tool for terrain map file state."""

    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        path = Path(_TERRAIN_PATH)
        data: dict[str, Any] = {
            "file_exists": path.exists(),
            "file_path": str(path),
            "file_size_kb": round(path.stat().st_size / 1024, 1) if path.exists() else 0,
            "replay_triggered": os.path.exists("/tmp/vector_terrain_replay"),
        }
        if path.exists():
            try:
                import numpy as np
                npz = np.load(str(path))
                data["voxel_count"] = int(len(npz.get("ix", [])))
                npz.close()
            except Exception:
                data["voxel_count"] = -1
        else:
            data["voxel_count"] = 0
        return ToolResult(content=json.dumps(data, indent=2))

    def is_concurrency_safe(self, params: dict[str, Any]) -> bool:
        return True
