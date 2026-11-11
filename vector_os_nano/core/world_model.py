# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Persistent world state for Vector OS Nano SDK.

Tracks the objects the SO-101 detect skill localized and the robot state the agent
syncs from its hardware, for one session (R4 replaces it with the one object store).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObjectState:
    """Immutable snapshot of a detected/tracked object's state."""

    object_id: str
    label: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    confidence: float = 1.0
    state: str = "on_table"  # on_table | grasped | placed | unknown
    last_seen: float = field(default_factory=time.time)
    properties: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "confidence": self.confidence,
            "state": self.state,
            "last_seen": self.last_seen,
            "properties": dict(self.properties),
        }

@dataclass(frozen=True)
class RobotState:
    """Immutable snapshot of the robot's current state."""

    joint_positions: tuple[float, ...] = ()
    gripper_state: str = "open"  # open | closed | holding
    held_object: str | None = None
    is_moving: bool = False
    ee_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_xy: tuple[float, float] = (0.0, 0.0)  # base position in world XY plane (meters)
    heading: float = 0.0  # radians, 0 = +X axis

    def to_dict(self) -> dict:
        return {
            "joint_positions": list(self.joint_positions),
            "gripper_state": self.gripper_state,
            "held_object": self.held_object,
            "is_moving": self.is_moving,
            "ee_position": list(self.ee_position),
            "position_xy": list(self.position_xy),
            "heading": self.heading,
        }

class WorldModel:
    """Session world state: the detected objects and the synced robot state.

    Mutable container — ObjectState and RobotState are frozen, but the
    WorldModel itself supports add/remove/update operations.
    """

    def __init__(self) -> None:
        self._objects: dict[str, ObjectState] = {}
        self._robot: RobotState = RobotState()

    # ------------------------------------------------------------------
    # Object management
    # ------------------------------------------------------------------

    def add_object(self, obj: ObjectState) -> None:
        """Add or replace an object by its object_id."""
        self._objects[obj.object_id] = obj
        logger.debug("WorldModel: added object %s (%s)", obj.object_id, obj.label)

    def remove_object(self, object_id: str) -> None:
        """Remove an object. Silently ignores unknown IDs."""
        self._objects.pop(object_id, None)
        logger.debug("WorldModel: removed object %s", object_id)

    def get_object(self, object_id: str) -> ObjectState | None:
        """Retrieve an object by ID, or None if not present."""
        return self._objects.get(object_id)

    def get_objects(self) -> list[ObjectState]:
        """Return all tracked objects as a list (order unspecified)."""
        return list(self._objects.values())

    def get_objects_by_label(self, label: str) -> list[ObjectState]:
        """Return all objects matching the given label (case-insensitive, normalised).

        Matching rules (in priority order):
        1. Exact normalised match (e.g. "protein bar" == "Protein Bar")
        2. Query is a substring of the stored label or vice versa
        """
        def _norm(s: str) -> str:
            return s.lower().strip().replace("_", " ")

        query = _norm(label)
        results: list[ObjectState] = []
        for o in self._objects.values():
            obj_label = _norm(o.label)
            if obj_label == query or query in obj_label or obj_label in query:
                results.append(o)
        return results

    # ------------------------------------------------------------------
    # Robot state
    # ------------------------------------------------------------------

    def get_robot(self) -> RobotState:
        """Return the current robot state snapshot."""
        return self._robot

    def update_robot_state(self, **kwargs: Any) -> None:
        """Update robot state fields. Creates a new frozen RobotState.

        Only keyword arguments that match RobotState fields are applied;
        all other fields are preserved from the current state.
        """
        current = self._robot
        valid_fields = {
            "joint_positions",
            "gripper_state",
            "held_object",
            "is_moving",
            "ee_position",
            "position_xy",
            "heading",
        }
        updates = {k: v for k, v in kwargs.items() if k in valid_fields}
        self._robot = RobotState(
            joint_positions=updates.get("joint_positions", current.joint_positions),
            gripper_state=updates.get("gripper_state", current.gripper_state),
            held_object=updates.get("held_object", current.held_object),
            is_moving=updates.get("is_moving", current.is_moving),
            ee_position=updates.get("ee_position", current.ee_position),
            position_xy=updates.get("position_xy", current.position_xy),
            heading=updates.get("heading", current.heading),
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize entire world state to a JSON-compatible dict.

        Suitable for sending to an LLM planner as context.
        """
        return {
            "objects": [o.to_dict() for o in self._objects.values()],
            "robot": self._robot.to_dict(),
        }
