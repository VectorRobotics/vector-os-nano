# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

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
from typing import Any, Protocol

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TextLLM Protocol — narrow injectable adapter for text-only LLM calls.
# core/ depends ONLY on this Protocol; no vcli import needed.
# ---------------------------------------------------------------------------


class TextLLM(Protocol):
    """Minimal protocol for a text-generation adapter.

    Implementations live outside core/ (e.g. vcli/backends/text_llm_adapter.py)
    so that core/ never imports vcli.  Stubs can be used in tests.
    """

    def complete_text(self, prompt: str) -> str:
        """Send *prompt* to the language model and return the text response.

        Args:
            prompt: The full user-side prompt string.

        Returns:
            The model's text response as a string.
        """
        ...


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
        # key = tuple(sorted([room_a, room_b])) — bidirectional
        # value = (x, y, count) — running average position + observation count
        self._doors: dict[tuple[str, str], tuple[float, float, int]] = {}
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
    # Door operations
    # ------------------------------------------------------------------

    def add_door(self, room_a: str, room_b: str, x: float, y: float) -> None:
        """Record a door observation between room_a and room_b at (x, y).

        Uses a running average to maintain door position across multiple
        observations.  Also updates both rooms' connected_rooms tuples.
        """
        key: tuple[str, str] = tuple(sorted([room_a, room_b]))  # type: ignore[assignment]
        with self._lock:
            existing = self._doors.get(key)
            if existing is not None:
                old_x, old_y, n = existing
                new_x = (old_x * n + x) / (n + 1)
                new_y = (old_y * n + y) / (n + 1)
                self._doors[key] = (new_x, new_y, n + 1)
            else:
                self._doors[key] = (x, y, 1)

            # Update connected_rooms for both rooms (create rooms if missing)
            for src, dst in ((room_a, room_b), (room_b, room_a)):
                room = self._rooms.get(src)
                if room is None:
                    self._rooms[src] = RoomNode(
                        room_id=src,
                        connected_rooms=(dst,),
                    )
                elif dst not in room.connected_rooms:
                    self._rooms[src] = RoomNode(
                        room_id=room.room_id,
                        center_x=room.center_x,
                        center_y=room.center_y,
                        area=room.area,
                        visit_count=room.visit_count,
                        last_visited=room.last_visited,
                        representative_description=room.representative_description,
                        connected_rooms=room.connected_rooms + (dst,),
                    )

    def get_door(self, room_a: str, room_b: str) -> tuple[float, float] | None:
        """Return (x, y) of the door between room_a and room_b, or None."""
        key: tuple[str, str] = tuple(sorted([room_a, room_b]))  # type: ignore[assignment]
        with self._lock:
            entry = self._doors.get(key)
            if entry is None:
                return None
            return (entry[0], entry[1])

    def get_all_doors(self) -> dict[tuple[str, str], tuple[float, float]]:
        """Return a copy of all doors (without observation count) for viz/serialization."""
        with self._lock:
            return {k: (v[0], v[1]) for k, v in self._doors.items()}

    def get_door_chain(
        self,
        src_room: str,
        dst_room: str,
    ) -> list[tuple[float, float, str]]:
        """BFS over room adjacency to find a waypoint chain from src to dst.

        Returns:
            List of (x, y, label) tuples.
            - If src == dst: [(dst_center_x, dst_center_y, dst_room)]
            - Normal path: door positions between rooms + dst room center
            - No path found: []
        """
        with self._lock:
            # Same room — return destination center
            if src_room == dst_room:
                room = self._rooms.get(dst_room)
                if room is None:
                    return []
                return [(room.center_x, room.center_y, dst_room)]

            # BFS
            from collections import deque
            visited: set[str] = {src_room}
            # Each queue element: (current_room, path_so_far)
            # path_so_far is list of room ids from src (exclusive) to current (inclusive)
            queue: deque[tuple[str, list[str]]] = deque()
            queue.append((src_room, []))

            parent: dict[str, str] = {}  # child -> parent for path reconstruction
            found = False

            while queue:
                current, _ = queue.popleft()
                room_node = self._rooms.get(current)
                if room_node is None:
                    continue
                for neighbor in room_node.connected_rooms:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    parent[neighbor] = current
                    if neighbor == dst_room:
                        found = True
                        break
                    queue.append((neighbor, []))
                if found:
                    break

            if not found:
                return []

            # Reconstruct room sequence: src -> ... -> dst
            path: list[str] = []
            node = dst_room
            while node in parent:
                path.append(node)
                node = parent[node]
            path.append(src_room)
            path.reverse()  # now: [src_room, ..., dst_room]

            # Build waypoint list
            waypoints: list[tuple[float, float, str]] = []
            for i in range(len(path) - 1):
                room_from = path[i]
                room_to = path[i + 1]
                door_pos = self.get_door(room_from, room_to)
                if door_pos is None:
                    # No recorded door position — skip to avoid misleading waypoints
                    continue
                label = f"{room_from}_{room_to}_door"
                waypoints.append((door_pos[0], door_pos[1], label))

            # Add destination room center
            dst_node = self._rooms.get(dst_room)
            if dst_node is not None:
                waypoints.append((dst_node.center_x, dst_node.center_y, dst_room))

            return waypoints

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
        self, goal: str, text_llm: TextLLM,
    ) -> list[tuple[str, str]]:
        """Ask a TextLLM which room most likely contains the goal target.

        Args:
            goal: Natural language goal (e.g. "find the red chair").
            text_llm: A TextLLM adapter whose complete_text() will be called
                with a text-only prompt.  No provider, key, or model is
                hardcoded here — those details belong in the adapter.

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

            text = text_llm.complete_text(prompt)

            # Parse JSON — try direct parse first, then regex fallback
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
        """Record visiting a room. Creates RoomNode if needed.

        Room center is computed as running average of all visit positions.
        This gives a more accurate center than a single detection point
        (which is often at the doorway).
        """
        with self._lock:
            existing = self._rooms.get(room)
            if existing:
                n = existing.visit_count
                # Running average: new_center = (old_center * n + new_pos) / (n + 1)
                new_cx = (existing.center_x * n + x) / (n + 1)
                new_cy = (existing.center_y * n + y) / (n + 1)
                self._rooms[room] = RoomNode(
                    room_id=room,
                    center_x=new_cx,
                    center_y=new_cy,
                    area=existing.area,
                    visit_count=n + 1,
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
        detected_objects: list[tuple[str, float, float]] | None = None,
    ) -> ViewpointNode | None:
        """Full viewpoint-aware observation (new API).

        Only adds a viewpoint if position is far enough from existing ones.
        Returns the ViewpointNode if created, None if skipped.

        Args:
            room: Room identifier.
            x: Robot x position (world frame).
            y: Robot y position (world frame).
            heading: Robot heading in radians.
            objects: Plain list of object name strings (used when
                detected_objects is None or empty).
            description: VLM scene description.
            detected_objects: Optional list of (category, obj_x, obj_y)
                tuples carrying per-object world coordinates.
                When non-empty this overrides the plain ``objects`` list.
        """
        # Determine which object source to use.
        use_detected = bool(detected_objects)

        if not self.should_add_viewpoint(room, x, y):
            # Still record objects even if viewpoint not added.
            with self._lock:
                nearest_vp = ""
                for vp in self._viewpoints.values():
                    if vp.room_id == room:
                        nearest_vp = vp.viewpoint_id
                        break
                if nearest_vp:
                    if use_detected:
                        for category, obj_x, obj_y in detected_objects:  # type: ignore[union-attr]
                            self.merge_object(
                                category=category, room_id=room,
                                viewpoint_id=nearest_vp,
                                x=obj_x, y=obj_y,
                            )
                    else:
                        for obj_name in objects:
                            self.merge_object(
                                category=obj_name, room_id=room,
                                viewpoint_id=nearest_vp,
                            )
            return None

        # Build object_ids tuple for the ViewpointNode record.
        if use_detected:
            vp_object_ids = tuple(cat for cat, _, _ in detected_objects)  # type: ignore[union-attr]
        else:
            vp_object_ids = tuple(objects)

        vp_id = f"vp_{uuid.uuid4().hex[:8]}"
        vp = ViewpointNode(
            viewpoint_id=vp_id,
            room_id=room,
            x=x, y=y,
            heading=heading,
            scene_summary=description,
            object_ids=vp_object_ids,
        )
        self.add_viewpoint(vp)

        # Visit room if not yet visited.
        self.visit(room, x, y)

        # Merge objects with or without per-object world coordinates.
        with self._lock:
            if use_detected:
                for category, obj_x, obj_y in detected_objects:  # type: ignore[union-attr]
                    self.merge_object(
                        category=category, room_id=room,
                        viewpoint_id=vp_id, x=obj_x, y=obj_y,
                    )
            else:
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

    def nearest_room(self, x: float, y: float) -> str | None:
        """Return room_id of the nearest room center, or None if no rooms exist.

        Only considers rooms with visit_count > 0 (actually explored).
        Used to replace the hardcoded _detect_current_room() throughout the codebase.
        """
        with self._lock:
            best_id: str | None = None
            best_dist = float("inf")
            for room in self._rooms.values():
                if room.visit_count <= 0:
                    continue
                dist = math.sqrt(
                    (room.center_x - x) ** 2 + (room.center_y - y) ** 2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_id = room.room_id
            return best_id

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
    # Layout seeding (simulation)
    # ------------------------------------------------------------------

    def load_layout(self, layout_path: str) -> int:
        """Seed SceneGraph from a room layout config file.

        Loads room centers and door positions from a YAML file. Each room
        gets visit_count=10 so nearest_room() and navigate work immediately.

        SIM ONLY — for real-world, rooms are discovered via exploration.

        Args:
            layout_path: Path to config/worlds/apartment.yaml.

        Returns:
            Number of rooms loaded, or 0 on failure.
        """
        try:
            with open(layout_path) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                return 0

            count = 0
            for room_name, coords in data.get("rooms", {}).items():
                if isinstance(coords, list) and len(coords) == 2:
                    x, y = float(coords[0]), float(coords[1])
                    # High visit_count so nearest_room trusts these positions
                    with self._lock:
                        self._rooms[room_name] = RoomNode(
                            room_id=room_name,
                            center_x=x,
                            center_y=y,
                            visit_count=10,
                            last_visited=time.time(),
                        )
                    count += 1

            for door_key, coords in data.get("doors", {}).items():
                if isinstance(coords, list) and len(coords) == 2:
                    parts = door_key.split("-")
                    if len(parts) == 2:
                        self.add_door(parts[0], parts[1], float(coords[0]), float(coords[1]))

            logger.info("[SceneGraph] Loaded layout: %d rooms from %s", count, layout_path)
            return count
        except FileNotFoundError:
            return 0
        except Exception as exc:
            logger.warning("[SceneGraph] load_layout failed: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self) -> None:
        if not self._persist_path:
            return
        with self._lock:
            # Serialize doors: convert tuple keys to "room_a|room_b" strings
            doors_serialized = {
                f"{k[0]}|{k[1]}": {"x": v[0], "y": v[1], "count": v[2]}
                for k, v in self._doors.items()
            }
            data = {
                "rooms": {k: v.to_dict() for k, v in self._rooms.items()},
                "viewpoints": {k: v.to_dict() for k, v in self._viewpoints.items()},
                "objects": {k: v.to_dict() for k, v in self._objects.items()},
                "doors": doors_serialized,
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
            # Doors — key stored as "room_a|room_b" string
            for door_key, dv in data.get("doors", {}).items():
                parts = door_key.split("|", 1)
                if len(parts) != 2:
                    continue
                key: tuple[str, str] = (parts[0], parts[1])
                self._doors[key] = (
                    float(dv.get("x", 0)),
                    float(dv.get("y", 0)),
                    int(dv.get("count", 1)),
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
