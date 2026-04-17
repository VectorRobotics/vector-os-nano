"""SceneGraphQueryTool — LLM interface for querying the robot's spatial memory.

Allows the LLM agent to inspect the SceneGraph without modifying it.
All queries are read-only and concurrency-safe.

Supported query types:
    rooms       — list all rooms with centers and visit counts
    doors       — list all doors with room pairs and positions
    objects     — list all objects with category, room, position
    room_detail — full detail for a specific room (room info + objects + viewpoints)
    door_chain  — BFS waypoint list from src_room to dst_room
    coverage    — per-room coverage percentages
    summary     — human-readable room summary text (from get_room_summary)
"""
from __future__ import annotations

import json
from typing import Any

from vector_os_nano.vcli.tools.base import (
    ToolContext,
    ToolResult,
    tool,
)


@tool(
    name="scene_graph_query",
    description=(
        "Query the robot's spatial memory (SceneGraph). "
        "Use to look up rooms, doors, objects, coverage, or navigation waypoints."
    ),
    read_only=True,
    permission="allow",
)
class SceneGraphQueryTool:
    """Read-only SceneGraph query tool for LLM agents."""

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query_type": {
                "type": "string",
                "enum": [
                    "rooms",
                    "doors",
                    "objects",
                    "room_detail",
                    "door_chain",
                    "coverage",
                    "summary",
                ],
                "description": "What to query from the SceneGraph.",
            },
            "room": {
                "type": "string",
                "description": "Room ID — required for room_detail.",
            },
            "src_room": {
                "type": "string",
                "description": "Source room ID — required for door_chain.",
            },
            "dst_room": {
                "type": "string",
                "description": "Destination room ID — required for door_chain.",
            },
        },
        "required": ["query_type"],
    }

    def is_concurrency_safe(self, params: dict[str, Any]) -> bool:
        return True

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        # Resolve SceneGraph from app_state
        if context.app_state is None:
            return ToolResult(
                content="No app_state available — SceneGraph not accessible.",
                is_error=True,
            )
        sg = context.app_state.get("scene_graph")
        if sg is None:
            return ToolResult(
                content="SceneGraph not found in app_state.",
                is_error=True,
            )

        query_type: str = params["query_type"]

        if query_type == "rooms":
            return self._query_rooms(sg)
        elif query_type == "doors":
            return self._query_doors(sg)
        elif query_type == "objects":
            return self._query_objects(sg)
        elif query_type == "room_detail":
            return self._query_room_detail(sg, params)
        elif query_type == "door_chain":
            return self._query_door_chain(sg, params)
        elif query_type == "coverage":
            return self._query_coverage(sg)
        elif query_type == "summary":
            return self._query_summary(sg)
        else:
            return ToolResult(
                content=f"Unknown query_type: {query_type!r}",
                is_error=True,
            )

    # ------------------------------------------------------------------
    # Query handlers
    # ------------------------------------------------------------------

    @staticmethod
    def _query_rooms(sg: Any) -> ToolResult:
        rooms = sg.get_all_rooms()
        data = [
            {
                "room_id": r.room_id,
                "center_x": r.center_x,
                "center_y": r.center_y,
                "visit_count": r.visit_count,
            }
            for r in rooms
        ]
        return ToolResult(content=json.dumps(data, indent=2))

    @staticmethod
    def _query_doors(sg: Any) -> ToolResult:
        doors_raw = sg.get_all_doors()
        data = [
            {
                "room_a": key[0],
                "room_b": key[1],
                "x": pos[0],
                "y": pos[1],
            }
            for key, pos in doors_raw.items()
        ]
        return ToolResult(content=json.dumps(data, indent=2))

    @staticmethod
    def _query_objects(sg: Any) -> ToolResult:
        rooms = sg.get_all_rooms()
        data = []
        for room in rooms:
            objs = sg.find_objects_in_room(room.room_id)
            for obj in objs:
                data.append(
                    {
                        "object_id": obj.object_id,
                        "category": obj.category,
                        "room_id": obj.room_id,
                        "x": obj.x,
                        "y": obj.y,
                    }
                )
        return ToolResult(content=json.dumps(data, indent=2))

    @staticmethod
    def _query_room_detail(sg: Any, params: dict[str, Any]) -> ToolResult:
        room_id = params.get("room")
        if not room_id:
            return ToolResult(
                content="room parameter is required for room_detail query.",
                is_error=True,
            )
        room = sg.get_room(room_id)
        if room is None:
            return ToolResult(
                content=f"Room {room_id!r} not found in SceneGraph.",
                is_error=True,
            )
        objects = sg.find_objects_in_room(room_id)
        viewpoints = sg.get_viewpoints_in_room(room_id)

        data = {
            "room_id": room.room_id,
            "center_x": room.center_x,
            "center_y": room.center_y,
            "visit_count": room.visit_count,
            "area": room.area,
            "representative_description": room.representative_description,
            "connected_rooms": list(room.connected_rooms),
            "objects": [
                {
                    "object_id": o.object_id,
                    "category": o.category,
                    "x": o.x,
                    "y": o.y,
                }
                for o in objects
            ],
            "viewpoints": [
                {
                    "viewpoint_id": vp.viewpoint_id,
                    "x": vp.x,
                    "y": vp.y,
                    "heading": vp.heading,
                    "scene_summary": vp.scene_summary,
                }
                for vp in viewpoints
            ],
        }
        return ToolResult(content=json.dumps(data, indent=2))

    @staticmethod
    def _query_door_chain(sg: Any, params: dict[str, Any]) -> ToolResult:
        src = params.get("src_room")
        dst = params.get("dst_room")
        if not src or not dst:
            return ToolResult(
                content="src_room and dst_room are required for door_chain query.",
                is_error=True,
            )
        chain = sg.get_door_chain(src, dst)
        data = [
            {"x": wp[0], "y": wp[1], "label": wp[2]}
            for wp in chain
        ]
        return ToolResult(content=json.dumps(data, indent=2))

    @staticmethod
    def _query_coverage(sg: Any) -> ToolResult:
        rooms = sg.get_all_rooms()
        data = {r.room_id: sg.get_room_coverage(r.room_id) for r in rooms}
        return ToolResult(content=json.dumps(data, indent=2))

    @staticmethod
    def _query_summary(sg: Any) -> ToolResult:
        summary = sg.get_room_summary()
        data: dict[str, Any] = {"summary": summary}

        # Include live explore status so LLM knows if exploration is in progress
        try:
            from vector_os_nano.skills.go2.explore import is_exploring, get_explore_status
            if is_exploring():
                status = get_explore_status()
                data["explore_in_progress"] = True
                data["explore_rooms_found"] = status["rooms_found_count"]
                data["explore_total_expected"] = status["total_expected"]
                data["explore_rooms_this_session"] = status["rooms_found"]
                data["note"] = (
                    f"Exploration is RUNNING — {status['rooms_found_count']}"
                    f"/{status['total_expected']} rooms found so far. "
                    f"SceneGraph may contain old data from previous sessions."
                )
            else:
                data["explore_in_progress"] = False
        except ImportError:
            pass

        return ToolResult(content=json.dumps(data, indent=2))
