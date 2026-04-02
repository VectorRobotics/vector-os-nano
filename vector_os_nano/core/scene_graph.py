"""Three-layer scene graph for spatial memory (SysNav-inspired).

Replaces flat SpatialMemory with a hierarchical representation:

    RoomNode  →  ViewpointNode  →  ObjectNode

Room nodes represent semantically meaningful spaces (kitchen, bedroom, etc.).
Viewpoint nodes are discrete camera positions within a room, each with a VLM
scene description and coverage area.  Object nodes are individual items
detected by the VLM, with category, position, and on-demand attributes.

Backward-compatible: implements the same public API as SpatialMemory
(visit, observe, get_visited_rooms, get_room_summary, etc.) so existing
skills work unchanged.

Reference: SysNav (arxiv 2603.06914v1) — three-layer scene representation
with VLM-guided room-level reasoning.

No ROS2 dependency.  Thread-safe via threading.Lock.
"""
from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VIEWPOINT_MIN_DISTANCE: float = 1.5   # meters — minimum distance between viewpoints
_VIEWPOINT_FOV_DEG: float = 60.0       # camera horizontal FOV (degrees)
_VIEWPOINT_RANGE: float = 3.0          # meters — max observation range
_ROOM_AREA_DEFAULT: float = 15.0       # m² — fallback room area for coverage calc
_MAX_EVENTS: int = 200


# ---------------------------------------------------------------------------
# Node dataclasses (frozen — immutable after construction)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectNode:
    """An object detected by VLM in the scene."""

    object_id: str
    category: str                       # "chair", "sofa", "fridge"
    description: str = ""               # VLM description
    confidence: float = 0.8
    room_id: str = ""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    attributes: dict = field(default_factory=dict)  # {"color": "red"} — queried on demand
    viewpoint_ids: tuple[str, ...] = ()
    first_seen: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "category": self.category,
            "description": self.description,
            "confidence": self.confidence,
            "room_id": self.room_id,
            "x": self.x, "y": self.y, "z": self.z,
            "attributes": dict(self.attributes),
            "viewpoint_ids": list(self.viewpoint_ids),
            "first_seen": self.first_seen,
        }


@dataclass(frozen=True)
class ViewpointNode:
    """A camera viewpoint within a room."""

    viewpoint_id: str
    room_id: str
    x: float
    y: float
    heading: float = 0.0                # radians
    timestamp: float = field(default_factory=time.time)
    scene_summary: str = ""             # VLM scene description
    object_ids: tuple[str, ...] = ()    # objects visible from here
    frame_b64: str = ""                 # optional cached frame (base64 jpeg)

    @property
    def coverage_area(self) -> float:
        """Estimated coverage area in m² (FOV cone approximation)."""
        half_angle = math.radians(_VIEWPOINT_FOV_DEG / 2)
        return 0.5 * _VIEWPOINT_RANGE**2 * math.sin(2 * half_angle)

    def to_dict(self) -> dict[str, Any]:
        return {
            "viewpoint_id": self.viewpoint_id,
            "room_id": self.room_id,
            "x": self.x, "y": self.y,
            "heading": self.heading,
            "timestamp": self.timestamp,
            "scene_summary": self.scene_summary,
            "object_ids": list(self.object_ids),
        }


@dataclass(frozen=True)
class RoomNode:
    """A room in the house."""

    room_id: str
    center_x: float = 0.0
    center_y: float = 0.0
    area: float = _ROOM_AREA_DEFAULT
    visit_count: int = 0
    last_visited: float = 0.0
    representative_description: str = ""
    connected_rooms: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_id": self.room_id,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "area": self.area,
            "visit_count": self.visit_count,
            "last_visited": self.last_visited,
            "representative_description": self.representative_description,
            "connected_rooms": list(self.connected_rooms),
        }


# ---------------------------------------------------------------------------
# SceneGraph
# ---------------------------------------------------------------------------


class SceneGraph:
    """Three-layer hierarchical scene graph.

    Layers:
        rooms      — dict[room_id, RoomNode]
        viewpoints — dict[viewpoint_id, ViewpointNode]
        objects    — dict[object_id, ObjectNode]

    Backward-compatible with SpatialMemory: visit(), observe(),
    get_visited_rooms(), get_room_summary(), etc.

    Thread-safe: all mutations guarded by a single Lock.
    """

    def __init__(self, persist_path: str | None = None) -> None:
        self._rooms: dict[str, RoomNode] = {}
        self._viewpoints: dict[str, ViewpointNode] = {}
        self._objects: dict[str, ObjectNode] = {}
        self._events: list[dict] = []
        self._lock = threading.RLock()
        self._persist_path = persist_path

    # ------------------------------------------------------------------
    # Room operations
    # ------------------------------------------------------------------

    def add_room(self, room: RoomNode) -> None:
        with self._lock:
            self._rooms[room.room_id] = room

    def get_room(self, room_id: str) -> RoomNode | None:
        with self._lock:
            return self._rooms.get(room_id)

    def get_all_rooms(self) -> list[RoomNode]:
        with self._lock:
            return list(self._rooms.values())

    # ------------------------------------------------------------------
    # Viewpoint operations
    # ------------------------------------------------------------------

    def add_viewpoint(self, vp: ViewpointNode) -> None:
        with self._lock:
            self._viewpoints[vp.viewpoint_id] = vp
            # Update room's visit info
            room = self._rooms.get(vp.room_id)
            if room is not None:
                self._rooms[vp.room_id] = RoomNode(
                    room_id=room.room_id,
                    center_x=room.center_x,
                    center_y=room.center_y,
                    area=room.area,
                    visit_count=room.visit_count,
                    last_visited=room.last_visited,
                    representative_description=(
                        vp.scene_summary or room.representative_description
                    ),
                    connected_rooms=room.connected_rooms,
                )

    def get_viewpoints_in_room(self, room_id: str) -> list[ViewpointNode]:
        with self._lock:
            return [
                vp for vp in self._viewpoints.values()
                if vp.room_id == room_id
            ]

    def should_add_viewpoint(
        self, room_id: str, x: float, y: float,
    ) -> bool:
        """Check if a new viewpoint should be added at (x, y).

        Returns True if no existing viewpoint in this room is within
        _VIEWPOINT_MIN_DISTANCE meters of the proposed position.
        """
        with self._lock:
            for vp in self._viewpoints.values():
                if vp.room_id != room_id:
                    continue
                dist = math.sqrt((vp.x - x)**2 + (vp.y - y)**2)
                if dist < _VIEWPOINT_MIN_DISTANCE:
                    return False
            return True

    # ------------------------------------------------------------------
    # Object operations
    # ------------------------------------------------------------------

    def add_object(self, obj: ObjectNode) -> None:
        with self._lock:
            self._objects[obj.object_id] = obj

    def find_objects_by_category(self, category: str) -> list[ObjectNode]:
        cat = category.lower().strip()
        with self._lock:
            return [
                o for o in self._objects.values()
                if cat in o.category.lower()
            ]

    def find_objects_in_room(self, room_id: str) -> list[ObjectNode]:
        with self._lock:
            return [
                o for o in self._objects.values()
                if o.room_id == room_id
            ]

    def merge_object(
        self,
        category: str,
        room_id: str,
        viewpoint_id: str,
        description: str = "",
        confidence: float = 0.8,
        x: float = 0.0,
        y: float = 0.0,
    ) -> ObjectNode:
        """Add or merge an object in a room.

        If an object with the same category already exists in this room,
        update it (add viewpoint, update description if confidence is higher).
        Otherwise create a new ObjectNode.

        Returns the resulting ObjectNode.
        """
        with self._lock:
            # Check for existing object with same category in same room
            for oid, existing in self._objects.items():
                if (existing.category.lower() == category.lower()
                        and existing.room_id == room_id):
                    # Merge: add viewpoint, keep higher confidence
                    vp_ids = set(existing.viewpoint_ids)
                    vp_ids.add(viewpoint_id)
                    merged = ObjectNode(
                        object_id=existing.object_id,
                        category=existing.category,
                        description=(
                            description if confidence > existing.confidence
                            else existing.description
                        ),
                        confidence=max(confidence, existing.confidence),
                        room_id=room_id,
                        x=x if x != 0.0 else existing.x,
                        y=y if y != 0.0 else existing.y,
                        z=existing.z,
                        attributes=existing.attributes,
                        viewpoint_ids=tuple(sorted(vp_ids)),
                        first_seen=existing.first_seen,
                    )
                    self._objects[oid] = merged
                    return merged

            # New object
            obj = ObjectNode(
                object_id=f"obj_{uuid.uuid4().hex[:8]}",
                category=category,
                description=description,
                confidence=confidence,
                room_id=room_id,
                x=x, y=y,
                viewpoint_ids=(viewpoint_id,),
            )
            self._objects[obj.object_id] = obj
            return obj

    # ------------------------------------------------------------------
    # Coverage tracking
    # ------------------------------------------------------------------

    def get_room_coverage(self, room_id: str) -> float:
        """Estimate what fraction of a room has been observed.

        Uses a simple model: each viewpoint covers coverage_area m².
        Total coverage is capped at room area. Returns 0.0-1.0.
        """
        vps = self.get_viewpoints_in_room(room_id)
        if not vps:
            return 0.0
        room = self.get_room(room_id)
        room_area = room.area if room else _ROOM_AREA_DEFAULT
        total_coverage = sum(vp.coverage_area for vp in vps)
        return min(1.0, total_coverage / room_area)

    # ------------------------------------------------------------------
    # VLM-guided room selection
    # ------------------------------------------------------------------

    def rank_rooms_for_goal(
        self, goal: str, vlm: Any,
    ) -> list[tuple[str, str]]:
        """Ask VLM which room most likely contains the goal target.

        Args:
            goal: Natural language goal (e.g. "find the red chair").
            vlm: Go2VLMPerception instance (has _call_vlm but we use
                 a text-only call via _call_vlm_text).

        Returns:
            List of (room_id, reasoning) sorted by relevance.
        """
        rooms_info = []
        with self._lock:
            for rid, room in self._rooms.items():
                objs = [
                    o.category for o in self._objects.values()
                    if o.room_id == rid
                ]
                desc = room.representative_description or "not yet explored"
                rooms_info.append(
                    f"- {rid}: {desc}. Objects: {', '.join(objs) if objs else 'none seen'}"
                )

        if not rooms_info:
            return []

        prompt = (
            f"Goal: {goal}\n\n"
            f"Known rooms and their contents:\n"
            + "\n".join(rooms_info) + "\n\n"
            "Which room most likely contains what the goal describes? "
            "Rank rooms from most to least likely. "
            'Respond in JSON: [{"room": "...", "reasoning": "..."}]'
        )

        try:
            import json
            import re
            import httpx

            # Text-only call (no image)
            headers = {
                "Authorization": f"Bearer {vlm._api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
            }
            with httpx.Client(timeout=20.0) as client:
                resp = client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload, headers=headers,
                )
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]

            # Parse JSON
            clean = text.strip()
            data = None
            try:
                data = json.loads(clean)
            except json.JSONDecodeError:
                match = re.search(r"\[.*\]", clean, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))

            if isinstance(data, list):
                return [
                    (item.get("room", ""), item.get("reasoning", ""))
                    for item in data
                    if isinstance(item, dict)
                ]
        except Exception as exc:
            logger.warning("[SceneGraph] rank_rooms_for_goal failed: %s", exc)

        return []

    # ------------------------------------------------------------------
    # Backward-compatible SpatialMemory API
    # ------------------------------------------------------------------

    def visit(self, room: str, x: float, y: float) -> None:
        """Record visiting a room. Creates RoomNode if needed."""
        with self._lock:
            existing = self._rooms.get(room)
            if existing:
                self._rooms[room] = RoomNode(
                    room_id=room,
                    center_x=existing.center_x or x,
                    center_y=existing.center_y or y,
                    area=existing.area,
                    visit_count=existing.visit_count + 1,
                    last_visited=time.time(),
                    representative_description=existing.representative_description,
                    connected_rooms=existing.connected_rooms,
                )
            else:
                self._rooms[room] = RoomNode(
                    room_id=room,
                    center_x=x,
                    center_y=y,
                    visit_count=1,
                    last_visited=time.time(),
                )
            self._append_event({
                "type": "visit", "room": room, "x": x, "y": y,
                "timestamp": time.time(),
            })

    def observe(
        self,
        room: str,
        objects: list[str],
        description: str = "",
    ) -> None:
        """Record VLM observation. Creates viewpoint + objects.

        Compatible with SpatialMemory.observe() signature.
        """
        with self._lock:
            # Ensure room exists
            if room not in self._rooms:
                self._rooms[room] = RoomNode(
                    room_id=room, visit_count=0, last_visited=time.time(),
                )

            # Create viewpoint
            room_node = self._rooms[room]
            vp_id = f"vp_{uuid.uuid4().hex[:8]}"
            vp = ViewpointNode(
                viewpoint_id=vp_id,
                room_id=room,
                x=room_node.center_x,
                y=room_node.center_y,
                scene_summary=description,
            )
            self._viewpoints[vp_id] = vp

            # Update room description
            if description:
                self._rooms[room] = RoomNode(
                    room_id=room_node.room_id,
                    center_x=room_node.center_x,
                    center_y=room_node.center_y,
                    area=room_node.area,
                    visit_count=room_node.visit_count,
                    last_visited=room_node.last_visited,
                    representative_description=description,
                    connected_rooms=room_node.connected_rooms,
                )

            # Create/merge objects
            for obj_name in objects:
                self.merge_object(
                    category=obj_name,
                    room_id=room,
                    viewpoint_id=vp_id,
                )

            self._append_event({
                "type": "observe", "room": room,
                "objects": objects, "timestamp": time.time(),
            })

    def observe_with_viewpoint(
        self,
        room: str,
        x: float,
        y: float,
        heading: float,
        objects: list[str],
        description: str = "",
    ) -> ViewpointNode | None:
        """Full viewpoint-aware observation (new API).

        Only adds a viewpoint if position is far enough from existing ones.
        Returns the ViewpointNode if created, None if skipped.
        """
        if not self.should_add_viewpoint(room, x, y):
            # Still record objects even if viewpoint not added
            with self._lock:
                for obj_name in objects:
                    # Find nearest viewpoint to associate with
                    nearest_vp = ""
                    for vp in self._viewpoints.values():
                        if vp.room_id == room:
                            nearest_vp = vp.viewpoint_id
                            break
                    if nearest_vp:
                        self.merge_object(
                            category=obj_name, room_id=room,
                            viewpoint_id=nearest_vp,
                        )
            return None

        vp_id = f"vp_{uuid.uuid4().hex[:8]}"
        vp = ViewpointNode(
            viewpoint_id=vp_id,
            room_id=room,
            x=x, y=y,
            heading=heading,
            scene_summary=description,
            object_ids=tuple(objects),
        )
        self.add_viewpoint(vp)

        # Visit room if not yet visited
        self.visit(room, x, y)

        # Add objects — no individual coords (VLM gives names, not positions).
        # The viz layer uses viewpoint (x, y, heading) to position objects.
        with self._lock:
            for obj_name in objects:
                self.merge_object(
                    category=obj_name, room_id=room,
                    viewpoint_id=vp_id,
                )

        return vp

    def get_location(self, name: str) -> Any:
        """Backward compat: return a LocationRecord-like object."""
        room = self.get_room(name)
        if room is None:
            return None
        # Return a duck-typed object with .name, .x, .y
        from vector_os_nano.core.spatial_memory import LocationRecord
        objs = self.find_objects_in_room(name)
        return LocationRecord(
            name=room.room_id,
            x=room.center_x,
            y=room.center_y,
            visit_count=room.visit_count,
            last_visited=room.last_visited,
            objects_seen=tuple(o.category for o in objs),
            description=room.representative_description,
        )

    def get_all_locations(self) -> list:
        """Backward compat: return all rooms as LocationRecord-like objects."""
        return [self.get_location(r.room_id) for r in self.get_all_rooms()]

    def remember_location(self, name: str, x: float, y: float) -> None:
        """Save a custom named location."""
        self.visit(name, x, y)

    def get_visited_rooms(self) -> list[str]:
        with self._lock:
            return [
                r.room_id for r in self._rooms.values()
                if r.visit_count > 0
            ]

    def get_unvisited_rooms(self, all_rooms: list[str]) -> list[str]:
        visited = set(self.get_visited_rooms())
        return [r for r in all_rooms if r not in visited]

    def get_room_summary(self) -> str:
        """Human-readable summary for LLM system prompt.

        Includes room descriptions, objects, and coverage.
        """
        with self._lock:
            if not self._rooms:
                return "No rooms explored yet."

            visited = [r for r in self._rooms.values() if r.visit_count > 0]
            unvisited = [r for r in self._rooms.values() if r.visit_count == 0]

            parts = []
            for r in visited:
                objs = [
                    o.category for o in self._objects.values()
                    if o.room_id == r.room_id
                ]
                vp_count = sum(
                    1 for vp in self._viewpoints.values()
                    if vp.room_id == r.room_id
                )
                coverage = self.get_room_coverage(r.room_id)
                room_str = f"{r.room_id} ({r.visit_count} visits"
                if vp_count > 0:
                    room_str += f", {vp_count} viewpoints, {coverage:.0%} coverage"
                if objs:
                    room_str += f", saw: {', '.join(objs[:8])}"
                if r.representative_description:
                    room_str += f" — {r.representative_description[:80]}"
                room_str += ")"
                parts.append(room_str)

            total = len(self._rooms)
            summary = f"Rooms explored ({len(visited)}/{total}): " + ", ".join(parts)

            if unvisited:
                summary += f"\nUnexplored: {', '.join(r.room_id for r in unvisited)}"

            return summary

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Return counts of rooms, viewpoints, and objects."""
        with self._lock:
            return {
                "rooms": len(self._rooms),
                "viewpoints": len(self._viewpoints),
                "objects": len(self._objects),
                "visited_rooms": sum(
                    1 for r in self._rooms.values() if r.visit_count > 0
                ),
            }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self) -> None:
        if not self._persist_path:
            return
        with self._lock:
            data = {
                "rooms": {k: v.to_dict() for k, v in self._rooms.items()},
                "viewpoints": {k: v.to_dict() for k, v in self._viewpoints.items()},
                "objects": {k: v.to_dict() for k, v in self._objects.items()},
            }
        with open(self._persist_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def load(self) -> None:
        if not self._persist_path:
            return
        try:
            with open(self._persist_path) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                return
            # Rooms
            for rid, rd in data.get("rooms", {}).items():
                self._rooms[rid] = RoomNode(
                    room_id=rd["room_id"],
                    center_x=float(rd.get("center_x", 0)),
                    center_y=float(rd.get("center_y", 0)),
                    area=float(rd.get("area", _ROOM_AREA_DEFAULT)),
                    visit_count=int(rd.get("visit_count", 0)),
                    last_visited=float(rd.get("last_visited", 0)),
                    representative_description=rd.get("representative_description", ""),
                    connected_rooms=tuple(rd.get("connected_rooms", ())),
                )
            # Viewpoints
            for vid, vd in data.get("viewpoints", {}).items():
                self._viewpoints[vid] = ViewpointNode(
                    viewpoint_id=vd["viewpoint_id"],
                    room_id=vd["room_id"],
                    x=float(vd.get("x", 0)),
                    y=float(vd.get("y", 0)),
                    heading=float(vd.get("heading", 0)),
                    timestamp=float(vd.get("timestamp", 0)),
                    scene_summary=vd.get("scene_summary", ""),
                    object_ids=tuple(vd.get("object_ids", ())),
                )
            # Objects
            for oid, od in data.get("objects", {}).items():
                self._objects[oid] = ObjectNode(
                    object_id=od["object_id"],
                    category=od.get("category", ""),
                    description=od.get("description", ""),
                    confidence=float(od.get("confidence", 0.8)),
                    room_id=od.get("room_id", ""),
                    x=float(od.get("x", 0)),
                    y=float(od.get("y", 0)),
                    z=float(od.get("z", 0)),
                    attributes=od.get("attributes", {}),
                    viewpoint_ids=tuple(od.get("viewpoint_ids", ())),
                    first_seen=float(od.get("first_seen", 0)),
                )
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("[SceneGraph] load failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _append_event(self, event: dict) -> None:
        """Append event to log, trimming if over limit. Caller holds lock."""
        self._events.append(event)
        if len(self._events) > _MAX_EVENTS:
            del self._events[:len(self._events) - _MAX_EVENTS]
