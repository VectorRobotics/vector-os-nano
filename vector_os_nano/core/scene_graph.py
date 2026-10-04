# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Three-layer scene graph for spatial memory (SysNav-inspired).

Replaces flat SpatialMemory with a hierarchical representation:

    RoomNode  →  ViewpointNode  →  ObjectNode

Room nodes represent semantically meaningful spaces (kitchen, bedroom, etc.);
rooms seeded from a world manifest are FIXED (their centre never drifts) and carry
a floor box, so "which room is (x, y) in" is a membership test. Viewpoint nodes
are camera positions within a room. Object nodes are INSTANCES: two observations
of the same kind merge only within ``_MERGE_RADIUS_M`` of each other, so two
bottles a metre apart stay two. An object's position is ``None`` until it is
localized (never a (0, 0) sentinel); ``last_seen``/``n_obs`` track observations.

Persistence is atomic (temp file + ``os.replace``, ``yaml.safe_dump`` of plain
floats) and debounced after every mutation, and a file that cannot be read is
moved aside (``.bad-<time>``) with a warning, never silently overwritten.

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
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
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
_MERGE_RADIUS_M: float = 0.4           # same-kind observations closer than this are one instance
_SAVE_MIN_INTERVAL_S: float = 1.0      # debounce: at most one autosave per second


def _f(v: Any) -> float | None:
    """A plain float (numpy scalars too), or None for an unknown coordinate."""
    return None if v is None else float(v)


def _has_word(text: str, word: str) -> bool:
    """*word* in *text*: on word boundaries for ASCII, as a substring for CJK."""
    if word.isascii():
        return re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", text) is not None
    return word in text


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
    x: float | None = None              # None = not localized (never a 0.0 sentinel)
    y: float | None = None
    z: float | None = None
    attributes: dict = field(default_factory=dict)  # {"color": "red"} — queried on demand
    viewpoint_ids: tuple[str, ...] = ()
    first_seen: float = field(default_factory=time.time)
    last_seen: float = 0.0              # wall time of the latest observation (0 = first_seen)
    n_obs: int = 1

    @property
    def localized(self) -> bool:
        return self.x is not None and self.y is not None

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
            "last_seen": self.last_seen or self.first_seen,
            "n_obs": self.n_obs,
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
    fixed: bool = False                 # seeded from a world manifest: the centre never moves
    box: tuple[float, float, float, float] | None = None  # floor box [x0, y0, x1, y1]

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
            "fixed": self.fixed,
            "box": list(self.box) if self.box else None,
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

    def __init__(
        self, persist_path: str | None = None, *, resolve: Any = None,
    ) -> None:
        self._rooms: dict[str, RoomNode] = {}
        self._viewpoints: dict[str, ViewpointNode] = {}
        self._objects: dict[str, ObjectNode] = {}
        # key = tuple(sorted([room_a, room_b])) — bidirectional
        # value = (x, y, count) — running average position + observation count
        self._doors: dict[tuple[str, str], tuple[float, float, int]] = {}
        self._lock = threading.RLock()
        self._persist_path = persist_path
        # resolve(text) -> canonical object id or None (the world manifest's name
        # resolver): "绿色的瓶子" and "green bottle" are then one kind of object.
        self._resolve = resolve
        self._layout_path: str | None = None
        self._last_save = 0.0
        self._save_timer: threading.Timer | None = None
        # Never autosave over a persisted file this graph has not loaded yet.
        self._may_autosave = not (persist_path and os.path.exists(persist_path))

    def clear(self) -> None:
        """Forget everything observed, IN PLACE (every holder keeps this instance).

        The rooms and doors of a loaded world layout are the world, not memory:
        they are re-seeded.
        """
        with self._lock:
            for d in (self._rooms, self._viewpoints, self._objects, self._doors):
                d.clear()
            if self._layout_path:
                self.load_layout(self._layout_path)

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
        x, y = float(x), float(y)
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
                    self._rooms[src] = replace(room, connected_rooms=room.connected_rooms + (dst,))
        self._autosave()

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
                self._rooms[vp.room_id] = replace(
                    room,
                    representative_description=vp.scene_summary or room.representative_description,
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

    def _kind(self, text: str) -> str | None:
        """The canonical object id *text* names (the world resolver), or None."""
        if self._resolve is None:
            return None
        try:
            return self._resolve(text)
        except Exception:  # noqa: BLE001 — an unresolvable name is just unknown
            return None

    def _obj_text(self, o: ObjectNode) -> str:
        return f"{(o.attributes or {}).get('color', '')} {o.category}".strip()

    def _same_kind(self, o: ObjectNode, category: str) -> bool:
        if o.category.lower() == category.lower():
            return True
        k = self._kind(category)
        return k is not None and k == self._kind(self._obj_text(o))

    def find_objects_by_category(self, category: str) -> list[ObjectNode]:
        """Every instance *category* names ("" = all of them).

        A name the world resolves ("绿色的瓶子", "green bottle", the body name) matches
        the instances that resolve to the same object; otherwise every word of the
        query must appear as a word of the category ("bottle" matches "green
        bottle", "can" never matches "candle").
        """
        q = category.lower().strip()
        with self._lock:
            objs = list(self._objects.values())
        if not q:
            return objs
        k = self._kind(q)
        if k is not None:
            return [o for o in objs if self._kind(self._obj_text(o)) == k]
        words = re.findall(r"[a-z0-9]+|[^\x00-\x7f]+", q)
        return [o for o in objs if all(_has_word(o.category.lower(), w) for w in words)]

    def update_object(
        self, object_id: str, *, state: str | None = None,
        x: Any = None, y: Any = None, z: Any = None,
    ) -> ObjectNode | None:
        """Record an event on an instance (``state`` "held"/"placed", a new position)."""
        with self._lock:
            o = self._objects.get(object_id)
            if o is None:
                return None
            attrs = dict(o.attributes)
            if state is not None:
                attrs["state"] = state
            pos = {} if x is None or y is None else {"x": _f(x), "y": _f(y), "z": _f(z)}
            if pos:
                pos["room_id"] = self.room_at(pos["x"], pos["y"]) or o.room_id
            o = replace(o, attributes=attrs, last_seen=time.time(), **pos)
            self._objects[object_id] = o
        self._autosave()
        return o

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
        x: Any = None,
        y: Any = None,
        z: Any = None,
    ) -> ObjectNode:
        """Add an observation of *category*: merge it into its instance, or a new one.

        A localized observation (x, y given) merges into the nearest instance of the
        same kind within ``_MERGE_RADIUS_M`` (else into an unlocalized one of that
        kind in the room, which it localizes); its position replaces the old one as
        a whole (x, y and z together). An unlocalized observation only merges into
        an instance of that kind in the room and never moves it. Otherwise a new
        instance. The object's room is the room box it stands in, when known.
        """
        x, y, z, confidence = _f(x), _f(y), _f(z), float(confidence)
        if (x is None) != (y is None):
            x = y = None  # half a position is no position
        now = time.time()
        with self._lock:
            if x is not None:
                room_id = self.room_at(x, y) or room_id
            same = [o for o in self._objects.values() if self._same_kind(o, category)]
            near = [o for o in same if o.localized and x is not None
                    and math.dist((o.x, o.y), (x, y)) <= _MERGE_RADIUS_M]
            match = (
                min(near, key=lambda o: math.dist((o.x, o.y), (x, y))) if near
                else next((o for o in same if o.room_id == room_id
                           and (x is None or not o.localized)), None)
            )
            vp = (viewpoint_id,) if viewpoint_id else ()
            if match is None:
                obj = ObjectNode(
                    object_id=f"obj_{uuid.uuid4().hex[:8]}", category=category,
                    description=description, confidence=confidence, room_id=room_id,
                    x=x, y=y, z=z, viewpoint_ids=vp, first_seen=now, last_seen=now,
                )
            else:
                pos = {"x": x, "y": y, "z": z, "room_id": room_id} if x is not None else {}
                obj = replace(
                    match,
                    description=description if confidence > match.confidence else match.description,
                    confidence=max(confidence, match.confidence),
                    viewpoint_ids=tuple(sorted(set(match.viewpoint_ids) | set(vp))),
                    last_seen=now, n_obs=match.n_obs + 1, **pos,
                )
            self._objects[obj.object_id] = obj
        self._autosave()
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

        A discovered room's centre is the running average of its visit positions;
        a FIXED (manifest-seeded) room's centre never moves.
        """
        x, y = float(x), float(y)
        with self._lock:
            existing = self._rooms.get(room)
            if existing:
                n = existing.visit_count
                # Running average: new_center = (old_center * n + new_pos) / (n + 1)
                cx, cy = (
                    (existing.center_x, existing.center_y) if existing.fixed
                    else ((existing.center_x * n + x) / (n + 1), (existing.center_y * n + y) / (n + 1))
                )
                self._rooms[room] = replace(
                    existing, center_x=cx, center_y=cy, visit_count=n + 1,
                    last_visited=time.time(),
                )
            else:
                self._rooms[room] = RoomNode(
                    room_id=room,
                    center_x=x,
                    center_y=y,
                    visit_count=1,
                    last_visited=time.time(),
                )
        self._autosave()

    def observe(
        self,
        room: str,
        objects: list[str],
        description: str = "",
    ) -> None:
        """Record a names-only observation at the room centre (SpatialMemory compat).

        The names carry no position, so they are stored unlocalized (``x is None``);
        nothing downstream mistakes them for a known place.
        """
        with self._lock:
            # Ensure room exists
            if room not in self._rooms:
                self._rooms[room] = RoomNode(
                    room_id=room, visit_count=0, last_visited=time.time(),
                )
            room_node = self._rooms[room]
            vp_id = f"vp_{uuid.uuid4().hex[:8]}"
            self._viewpoints[vp_id] = ViewpointNode(
                viewpoint_id=vp_id, room_id=room,
                x=room_node.center_x, y=room_node.center_y, scene_summary=description,
            )
            if description:
                self._rooms[room] = replace(room_node, representative_description=description)
            for obj_name in objects:
                self.merge_object(category=obj_name, room_id=room, viewpoint_id=vp_id)

    def observe_with_viewpoint(
        self,
        room: str,
        x: float,
        y: float,
        heading: float,
        objects: list[str],
        description: str = "",
        detected_objects: list[tuple] | None = None,
    ) -> ViewpointNode | None:
        """Record what the camera saw from (x, y, heading).

        Adds a viewpoint only if it is far enough from the room's existing ones
        (returns it, else None; the objects are recorded either way, against the
        nearest viewpoint). ``detected_objects`` are ``(category, x, y[, z])``
        LOCALIZED objects; when given they replace the names-only ``objects``
        (stored unlocalized). It never visits the room: looking is not arriving.
        """
        x, y, heading = float(x), float(y), float(heading)
        detected = [tuple(d) for d in (detected_objects or [])]
        for d in detected:
            if not 3 <= len(d) <= 4:
                raise ValueError(f"detected_objects entries are (category, x, y[, z]), got {d!r}")
        with self._lock:
            vp: ViewpointNode | None = None
            if self.should_add_viewpoint(room, x, y):
                vp = ViewpointNode(
                    viewpoint_id=f"vp_{uuid.uuid4().hex[:8]}", room_id=room, x=x, y=y,
                    heading=heading, scene_summary=description,
                    object_ids=tuple(d[0] for d in detected) or tuple(objects),
                )
                self.add_viewpoint(vp)
                vp_id = vp.viewpoint_id
            else:
                in_room = [v for v in self._viewpoints.values() if v.room_id == room]
                vp_id = min(in_room, key=lambda v: math.dist((v.x, v.y), (x, y))).viewpoint_id
            if detected:
                for cat, ox, oy, *oz in detected:
                    self.merge_object(cat, room, vp_id, x=ox, y=oy, z=oz[0] if oz else None)
            else:
                for obj_name in objects:
                    self.merge_object(category=obj_name, room_id=room, viewpoint_id=vp_id)
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

    def room_at(self, x: Any, y: Any) -> str | None:
        """The room whose floor box contains (x, y) (the smallest on a shared edge)."""
        if x is None or y is None:
            return None
        with self._lock:
            inside = [
                ((r.box[2] - r.box[0]) * (r.box[3] - r.box[1]), r.room_id)
                for r in self._rooms.values()
                if r.box and r.box[0] <= x <= r.box[2] and r.box[1] <= y <= r.box[3]
            ]
        return min(inside)[1] if inside else None

    def nearest_room(self, x: float, y: float) -> str | None:
        """The room (x, y) is in: its floor box when rooms have boxes, else the
        nearest explored room centre (visit_count > 0); None without rooms."""
        inside = self.room_at(x, y)
        if inside is not None:
            return inside
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
        """Seed the world's rooms and doors from a world manifest (SIM ONLY).

        Each room is FIXED at its manifest centre (visits never move it) with its
        floor box, and gets visit_count=10 so nearest_room() and navigate trust it;
        a room already in memory keeps its description and connections. A door
        key "room_a-room_b" is split where both halves are rooms.

        Returns:
            Number of rooms loaded, or 0 on failure.
        """
        try:
            with open(layout_path) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                return 0
            rooms = data.get("rooms") or {}
            boxes = data.get("boxes") or {}
            with self._lock:
                for name, coords in rooms.items():
                    x, y = (float(c) for c in coords)
                    box = tuple(float(c) for c in boxes[name]) if name in boxes else None
                    old = self._rooms.get(name) or RoomNode(room_id=name)
                    self._rooms[name] = replace(
                        old, center_x=x, center_y=y, visit_count=max(old.visit_count, 10),
                        last_visited=time.time(), fixed=True, box=box,
                    )
                for key, coords in (data.get("doors") or {}).items():
                    pair = next(((key[:i], key[i + 1:]) for i, ch in enumerate(key)
                                 if ch == "-" and key[:i] in rooms and key[i + 1:] in rooms), None)
                    if pair is None:
                        logger.warning("[SceneGraph] door %r names no two layout rooms", key)
                        continue
                    self.add_door(pair[0], pair[1], float(coords[0]), float(coords[1]))
                self._layout_path = layout_path
            logger.info("[SceneGraph] Loaded layout: %d rooms from %s", len(rooms), layout_path)
            return len(rooms)
        except FileNotFoundError:
            return 0
        except Exception as exc:
            logger.warning("[SceneGraph] load_layout failed: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write the graph atomically (temp file in the same dir + os.replace)."""
        if not self._persist_path:
            return
        with self._lock:
            data = {
                "rooms": {k: v.to_dict() for k, v in self._rooms.items()},
                "viewpoints": {k: v.to_dict() for k, v in self._viewpoints.items()},
                "objects": {k: v.to_dict() for k, v in self._objects.items()},
                "doors": [
                    {"a": k[0], "b": k[1], "x": v[0], "y": v[1], "count": v[2]}
                    for k, v in self._doors.items()
                ],
            }
            self._last_save = time.monotonic()
        d = os.path.dirname(os.path.abspath(self._persist_path))
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".scene_graph.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
            os.replace(tmp, self._persist_path)
        except BaseException:
            os.unlink(tmp)
            raise

    def _autosave(self) -> None:
        """Save after a mutation, at most once per ``_SAVE_MIN_INTERVAL_S`` (the last
        change of a burst is saved by a timer)."""
        if not self._persist_path or not self._may_autosave:
            return
        with self._lock:
            wait = self._last_save + _SAVE_MIN_INTERVAL_S - time.monotonic()
            if wait > 0:
                if self._save_timer is None or not self._save_timer.is_alive():
                    self._save_timer = threading.Timer(wait, self._timed_save)
                    self._save_timer.daemon = True
                    self._save_timer.start()
                return
        self._timed_save()

    def _timed_save(self) -> None:
        try:
            self.save()
        except Exception as exc:  # noqa: BLE001 — the next save retries; never crash a skill
            logger.warning("[SceneGraph] autosave to %s failed: %s", self._persist_path, exc)

    def load(self) -> None:
        """Replace this graph's contents with the persisted file (in place).

        A missing file leaves an empty graph. A file that does not parse is moved
        aside to ``<path>.bad-<time>`` with a warning (never overwritten silently).
        """
        if not self._persist_path:
            return
        self._may_autosave = True
        try:
            with open(self._persist_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            return
        except Exception as exc:  # noqa: BLE001
            return self._quarantine(f"unreadable ({exc})")
        if data is None:
            return
        if not isinstance(data, dict):
            return self._quarantine(f"not a mapping ({type(data).__name__})")
        try:
            rooms = {
                rid: RoomNode(
                    room_id=rd["room_id"],
                    center_x=float(rd.get("center_x", 0)),
                    center_y=float(rd.get("center_y", 0)),
                    area=float(rd.get("area", _ROOM_AREA_DEFAULT)),
                    visit_count=int(rd.get("visit_count", 0)),
                    last_visited=float(rd.get("last_visited", 0)),
                    representative_description=rd.get("representative_description", ""),
                    connected_rooms=tuple(rd.get("connected_rooms", ())),
                    fixed=bool(rd.get("fixed", False)),
                    box=tuple(float(c) for c in rd["box"]) if rd.get("box") else None,
                )
                for rid, rd in (data.get("rooms") or {}).items()
            }
            viewpoints = {
                vid: ViewpointNode(
                    viewpoint_id=vd["viewpoint_id"],
                    room_id=vd["room_id"],
                    x=float(vd.get("x", 0)),
                    y=float(vd.get("y", 0)),
                    heading=float(vd.get("heading", 0)),
                    timestamp=float(vd.get("timestamp", 0)),
                    scene_summary=vd.get("scene_summary", ""),
                    object_ids=tuple(vd.get("object_ids", ())),
                )
                for vid, vd in (data.get("viewpoints") or {}).items()
            }
            objects = {
                oid: ObjectNode(
                    object_id=od["object_id"],
                    category=od.get("category", ""),
                    description=od.get("description", ""),
                    confidence=float(od.get("confidence", 0.8)),
                    room_id=od.get("room_id", ""),
                    x=_f(od.get("x")), y=_f(od.get("y")), z=_f(od.get("z")),
                    attributes=od.get("attributes") or {},
                    viewpoint_ids=tuple(od.get("viewpoint_ids", ())),
                    first_seen=float(od.get("first_seen", 0)),
                    last_seen=float(od.get("last_seen", 0)),
                    n_obs=int(od.get("n_obs", 1)),
                )
                for oid, od in (data.get("objects") or {}).items()
            }
            doors = {
                (str(dv["a"]), str(dv["b"])): (float(dv["x"]), float(dv["y"]), int(dv.get("count", 1)))
                for dv in (data.get("doors") or [])
            }
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            return self._quarantine(f"malformed ({exc!r})")
        with self._lock:
            for mine, theirs in ((self._rooms, rooms), (self._viewpoints, viewpoints),
                                 (self._objects, objects), (self._doors, doors)):
                mine.clear()
                mine.update(theirs)  # type: ignore[attr-defined]

    def _quarantine(self, why: str) -> None:
        bad = f"{self._persist_path}.bad-{int(time.time())}"
        try:
            os.replace(self._persist_path, bad)  # type: ignore[arg-type]
        except OSError:
            bad = "(could not move it aside)"
        logger.warning("[SceneGraph] %s is %s; starting empty, kept it as %s",
                       self._persist_path, why, bad)
