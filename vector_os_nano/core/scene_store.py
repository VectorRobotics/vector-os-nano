# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Persistence of the scene graph (core/scene_graph.py): one YAML file per world.

Atomic (temp file + ``os.replace``, ``yaml.safe_dump`` of plain floats) and debounced
after every mutation; a file that cannot be read is moved aside (``.bad-<time>``) with a
warning, never silently overwritten. ``SceneGraph`` inherits it and owns the state it
reads and writes (``_rooms``, ``_viewpoints``, ``_objects``, ``_doors``, ``_lock`` and the
``_persist_path`` / ``_last_save`` / ``_save_timer`` / ``_may_autosave`` fields).
"""
from __future__ import annotations

import logging
import os
import tempfile
import threading
import time

import yaml

from vector_os_nano.core.scene_nodes import (
    _ROOM_AREA_DEFAULT,
    ObjectNode,
    RoomNode,
    ViewpointNode,
    _f,
)

logger = logging.getLogger(__name__)

_SAVE_MIN_INTERVAL_S: float = 1.0      # debounce: at most one autosave per second


class SceneStore:
    """Save and load of a SceneGraph (the base class it inherits)."""

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
