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
import threading
import time
import uuid
from dataclasses import replace
from typing import Any

import yaml

from vector_os_nano.core.scene_nodes import (
    _ROOM_AREA_DEFAULT,
    ObjectNode,
    RoomNode,
    ViewpointNode,
    _f,
)
from vector_os_nano.core.scene_store import SceneStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VIEWPOINT_MIN_DISTANCE: float = 1.5   # meters — minimum distance between viewpoints
_MERGE_RADIUS_M: float = 0.4           # same-kind observations closer than this are one instance


def _has_word(text: str, word: str) -> bool:
    """*word* in *text*: on word boundaries for ASCII, as a substring for CJK."""
    if word.isascii():
        return re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", text) is not None
    return word in text


# ---------------------------------------------------------------------------
# SceneGraph
# ---------------------------------------------------------------------------


class SceneGraph(SceneStore):
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
        self._obs_count = 0  # localized observations merged (monotonic; not persisted)

    @property
    def observation_count(self) -> int:
        """How many LOCALIZED observations this graph has merged (it only grows).

        The actor-causation perception channel: a step caused ``localized`` only if
        this advanced during it (loading or clearing memory never advances it)."""
        return self._obs_count

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
            if x is not None:
                self._obs_count += 1
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
