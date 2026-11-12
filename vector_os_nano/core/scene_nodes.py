# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The node types of the scene graph (core/scene_graph.py): rooms, viewpoints, objects.

Frozen dataclasses; ``to_dict`` is the persisted form.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VIEWPOINT_FOV_DEG: float = 60.0       # camera horizontal FOV (degrees)
_VIEWPOINT_RANGE: float = 3.0          # meters — max observation range
_ROOM_AREA_DEFAULT: float = 15.0       # m² — fallback room area for coverage calc


def _f(v: Any) -> float | None:
    """A plain float (numpy scalars too), or None for an unknown coordinate."""
    return None if v is None else float(v)


# ---------------------------------------------------------------------------
# Node dataclasses (frozen — immutable after construction)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectNode:
    """An object instance the robot's perception observed (merged by SceneGraph.merge_object)."""

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
