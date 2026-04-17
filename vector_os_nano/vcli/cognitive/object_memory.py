"""ObjectMemory — time-aware object tracking with exponential confidence decay.

Wraps SceneGraph.ObjectNode with decay-based confidence so the robot can
reason about how certain it is that an object is still where it last saw it.

Decay model:
    effective_confidence = base_confidence * exp(-lambda * elapsed_seconds)

Default lambda=0.001:
    10 min (600s)  → ~0.549
    30 min (1800s) → ~0.165
    2  hr  (7200s) → ~0.000735

Thread-safe via threading.RLock (same pattern as SceneGraph).
No ROS2 dependency.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackedObject:
    """Immutable snapshot of an observed object with temporal metadata."""

    object_id: str           # UUID from SceneGraph.ObjectNode
    category: str            # e.g. "cup", "chair"
    room_id: str             # room where last observed
    x: float                 # last observed position
    y: float
    last_seen: float         # Unix timestamp of last observation
    base_confidence: float   # confidence at time of observation (0.0–1.0)
    observation_count: int   # how many times this object has been seen


# ---------------------------------------------------------------------------
# ObjectMemory
# ---------------------------------------------------------------------------


class ObjectMemory:
    """Time-aware object tracking layer.

    Maintains a dict of TrackedObject entries keyed by object_id.
    All mutations are protected by an RLock for thread safety.
    """

    def __init__(self, decay_lambda: float = 0.001) -> None:
        self._objects: dict[str, TrackedObject] = {}
        self._decay_lambda: float = decay_lambda
        self._lock: threading.RLock = threading.RLock()

    # ------------------------------------------------------------------
    # Core confidence calculation
    # ------------------------------------------------------------------

    def effective_confidence(self, obj: TrackedObject) -> float:
        """Compute time-decayed confidence.

        Returns: base_confidence * exp(-lambda * elapsed_seconds)
        """
        elapsed = time.time() - obj.last_seen
        return obj.base_confidence * math.exp(-self._decay_lambda * elapsed)

    # ------------------------------------------------------------------
    # SceneGraph sync
    # ------------------------------------------------------------------

    def sync_from_scene_graph(self, scene_graph: Any) -> int:
        """Sync all ObjectNodes from SceneGraph into ObjectMemory.

        Iterates rooms via get_all_rooms(), then find_objects_in_room()
        for each room. Updates existing entries (increments observation_count,
        refreshes position/confidence/last_seen). Adds new entries with
        observation_count=1.

        Returns:
            Count of objects synced (new + updated).
        """
        now = time.time()
        synced = 0
        rooms = scene_graph.get_all_rooms()
        with self._lock:
            for room in rooms:
                room_id: str = room.room_id
                for node in scene_graph.find_objects_in_room(room_id):
                    oid: str = node.object_id
                    existing = self._objects.get(oid)
                    if existing is not None:
                        self._objects[oid] = TrackedObject(
                            object_id=oid,
                            category=node.category,
                            room_id=room_id,
                            x=node.x,
                            y=node.y,
                            last_seen=now,
                            base_confidence=node.confidence,
                            observation_count=existing.observation_count + 1,
                        )
                    else:
                        self._objects[oid] = TrackedObject(
                            object_id=oid,
                            category=node.category,
                            room_id=room_id,
                            x=node.x,
                            y=node.y,
                            last_seen=now,
                            base_confidence=node.confidence,
                            observation_count=1,
                        )
                    synced += 1
        return synced

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def last_seen(self, category: str) -> dict | None:
        """Find the most recently observed object of the given category.

        Uses case-insensitive substring matching (same as SceneGraph).

        Returns:
            {"room": str, "position": (x, y), "seconds_ago": float, "confidence": float}
            or None if never seen.
        """
        cat_lower = category.lower().strip()
        best: TrackedObject | None = None
        best_ts: float = -1.0

        with self._lock:
            for obj in self._objects.values():
                if cat_lower not in obj.category.lower():
                    continue
                if obj.last_seen > best_ts:
                    best_ts = obj.last_seen
                    best = obj

        if best is None:
            return None

        now = time.time()
        return {
            "room": best.room_id,
            "position": (best.x, best.y),
            "seconds_ago": now - best.last_seen,
            "confidence": self.effective_confidence(best),
        }

    def certainty(self, fact: str) -> float:
        """Return effective confidence that category is in room.

        Accepted formats (case-insensitive):
            "cup在kitchen"   — Chinese format
            "cup in kitchen" — English format

        Returns 0.0 if format not recognized or object not found in that room.
        """
        if not fact:
            return 0.0

        # Try Chinese format: split on "在"
        if "在" in fact:
            parts = fact.split("在", 1)
            category = parts[0].strip()
            room_id = parts[1].strip()
        else:
            # Try English format: split on " in " (case-insensitive)
            lower = fact.lower()
            idx = lower.find(" in ")
            if idx == -1:
                return 0.0
            category = fact[:idx].strip()
            room_id = fact[idx + 4:].strip()

        if not category or not room_id:
            return 0.0

        cat_lower = category.lower()

        with self._lock:
            for obj in self._objects.values():
                if cat_lower not in obj.category.lower():
                    continue
                if obj.room_id != room_id:
                    continue
                return self.effective_confidence(obj)

        return 0.0

    def objects_in_room(self, room_id: str) -> list[dict]:
        """Return all tracked objects in room_id with effective confidence > 0.01.

        Returns:
            List of dicts: {"object_id", "category", "x", "y", "confidence", "seconds_ago"}
            Sorted by confidence descending.
        """
        now = time.time()
        results: list[dict] = []

        with self._lock:
            for obj in self._objects.values():
                if obj.room_id != room_id:
                    continue
                conf = self.effective_confidence(obj)
                if conf <= 0.01:
                    continue
                results.append({
                    "object_id": obj.object_id,
                    "category": obj.category,
                    "x": obj.x,
                    "y": obj.y,
                    "confidence": conf,
                    "seconds_ago": now - obj.last_seen,
                })

        results.sort(key=lambda d: d["confidence"], reverse=True)
        return results

    def find_object(self, category: str) -> list[dict]:
        """Find all known locations of objects matching category (substring match).

        Returns:
            List of dicts: {"object_id", "category", "room", "x", "y", "confidence", "seconds_ago"}
            Sorted by confidence descending.
        """
        cat_lower = category.lower().strip()
        now = time.time()
        results: list[dict] = []

        with self._lock:
            for obj in self._objects.values():
                if cat_lower not in obj.category.lower():
                    continue
                conf = self.effective_confidence(obj)
                results.append({
                    "object_id": obj.object_id,
                    "category": obj.category,
                    "room": obj.room_id,
                    "x": obj.x,
                    "y": obj.y,
                    "confidence": conf,
                    "seconds_ago": now - obj.last_seen,
                })

        results.sort(key=lambda d: d["confidence"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def update(
        self,
        object_id: str,
        category: str,
        room_id: str,
        x: float,
        y: float,
        confidence: float = 0.9,
    ) -> None:
        """Update or create a TrackedObject.

        Increments observation_count if the object already exists.
        Refreshes last_seen, position, and base_confidence.
        """
        now = time.time()
        with self._lock:
            existing = self._objects.get(object_id)
            count = (existing.observation_count + 1) if existing is not None else 1
            self._objects[object_id] = TrackedObject(
                object_id=object_id,
                category=category,
                room_id=room_id,
                x=x,
                y=y,
                last_seen=now,
                base_confidence=confidence,
                observation_count=count,
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> list[dict]:
        """Serialize all tracked objects for persistence."""
        with self._lock:
            return [
                {
                    "object_id": obj.object_id,
                    "category": obj.category,
                    "room_id": obj.room_id,
                    "x": obj.x,
                    "y": obj.y,
                    "last_seen": obj.last_seen,
                    "base_confidence": obj.base_confidence,
                    "observation_count": obj.observation_count,
                }
                for obj in self._objects.values()
            ]

    @classmethod
    def from_dict(cls, data: list[dict], decay_lambda: float = 0.001) -> "ObjectMemory":
        """Deserialize from a list of dicts produced by to_dict().

        Preserves last_seen timestamps exactly (does not reset to now).
        """
        mem = cls(decay_lambda=decay_lambda)
        for entry in data:
            obj = TrackedObject(
                object_id=entry["object_id"],
                category=entry["category"],
                room_id=entry["room_id"],
                x=entry["x"],
                y=entry["y"],
                last_seen=entry["last_seen"],
                base_confidence=entry["base_confidence"],
                observation_count=entry["observation_count"],
            )
            mem._objects[obj.object_id] = obj
        return mem
