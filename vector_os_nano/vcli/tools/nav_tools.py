# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Navigation and terrain diagnostic tools for Vector CLI."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from vector_os_nano.schemas import paths
from vector_os_nano.vcli.tools.base import PermissionResult, ToolContext, ToolResult, live_agent, tool
from vector_os_nano.core.run_dir import run_path


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
    category="diag",
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
            "nav_flag_active": os.path.exists(run_path("vector_nav_active")),
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

def terrain_map(scene_graph: Any) -> Path | None:
    """The terrain map beside the running body's scene memory ($VECTOR_HOME/memory/<world>/),
    None when no body with a scene memory runs."""
    persist = getattr(scene_graph, "_persist_path", None)
    return Path(persist).with_name(paths.TERRAIN_MAP) if persist else None


@tool(
    name="terrain_status",
    category="diag",
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
        path = terrain_map(getattr(live_agent(context), "_spatial_memory", None))
        exists = path is not None and path.exists()
        data: dict[str, Any] = {
            "file_exists": exists,
            "file_path": str(path or ""),
            "file_size_kb": round(path.stat().st_size / 1024, 1) if exists else 0,
            "replay_triggered": os.path.exists(run_path("vector_terrain_replay")),
        }
        if exists:
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


# ---------------------------------------------------------------------------
# GoToRoomTool
# ---------------------------------------------------------------------------


@tool(
    name="go_to_room",
    category="robot",
    description=(
        "Walk the robot to a ROOM by name, in any language (e.g. 'kitchen', '厨房', "
        "'dining room'): a free floor point inside it from the house map, reached "
        "through the obstacle-avoiding planner. Then call verify(in_room('<room>'))."
    ),
    read_only=False,
    permission="ask",
)
class GoToRoomTool:
    """Room name -> the map's free point in it (else the SceneGraph's learned centre)
    -> the robot's own ``navigate`` skill on that coordinate."""

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "room": {"type": "string", "description": "Room name, any language (kitchen / 厨房)."},
        },
        "required": ["room"],
    }

    def check_permissions(self, params: dict[str, Any], context: ToolContext) -> PermissionResult:
        from vector_os_nano.sim.session import is_sim_agent

        return PermissionResult("allow" if is_sim_agent(live_agent(context)) else "ask")

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        from vector_os_nano.skills.navigate import _get_room_center_from_memory, _resolve_room
        from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool
        from vector_os_nano.schemas.world import world_of

        agent = live_agent(context)
        if agent is None:
            return ToolResult(content="No robot connected. Use start_simulation first.", is_error=True)
        name = str(params.get("room", "")).strip()
        world = world_of(agent)
        rooms = dict(world.rooms) if world is not None else {}
        sg = getattr(agent, "_spatial_memory", None)
        room = name if name in rooms else _resolve_room(name)
        target = rooms.get(room) if room else None
        if target is None and room and sg is not None:
            target = _get_room_center_from_memory(sg, room)
        if target is None:
            known = sorted(rooms) or (sorted(r.room_id for r in sg.get_all_rooms()) if sg is not None else [])
            return ToolResult(content=f"Unknown room {name!r}. Rooms: {', '.join(known) or 'none known'}.",
                              is_error=True)
        nav = agent._skill_registry.get("navigate")
        if nav is None:
            return ToolResult(content="This robot has no navigate skill.", is_error=True)
        res = SkillWrapperTool(nav, agent).execute({"x": target[0], "y": target[1]}, context)
        content = f"go_to_room({room}) -> ({target[0]}, {target[1]}): {res.content}"
        if not res.is_error:
            content += f"\nCall verify(in_room('{room}')) to confirm."
        return ToolResult(content=content, is_error=res.is_error, metadata=res.metadata)
