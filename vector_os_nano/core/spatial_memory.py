# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""LocationRecord — the per-room snapshot ``SceneGraph.get_location`` returns.

(The flat ``SpatialMemory`` store that used to live here was superseded by
``core.scene_graph.SceneGraph``; git history keeps it.)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocationRecord:
    """Snapshot of everything the robot knows about one named location.

    Attributes:
        name:         Room or custom location name.
        x:            Position X in the robot's world frame.
        y:            Position Y in the robot's world frame.
        visit_count:  Number of recorded visits to this room.
        last_visited: Unix timestamp of the most recent visit.
        objects_seen: Tuple of object/feature names observed in this room.
        description:  Free-text VLM scene description (optional).
    """

    name: str
    x: float
    y: float
    visit_count: int = 0
    last_visited: float = 0.0
    objects_seen: tuple[str, ...] = ()
    description: str = ""
