"""Persistent world state for Vector OS Nano SDK.

Tracks objects, robot state, and spatial relations across a session.
Provides predicate evaluation for skill precondition/postcondition checking.
Thread-safe for single-threaded use (skills execute sequentially in executor).

Persistence: YAML format via PyYAML.
"""
from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Predicate name patterns (compiled once for performance)
_RE_GRIPPER_HOLDING = re.compile(r"^gripper_holding\(([^)]+)\)$")
_RE_OBJECT_VISIBLE = re.compile(r"^object_visible\(([^)]+)\)$")
_RE_OBJECT_REACHABLE = re.compile(r"^object_reachable\(([^)]+)\)$")

# Thresholds
_VISIBLE_CONFIDENCE_THRESHOLD: float = 0.5
_REACHABLE_DISTANCE_THRESHOLD: float = 0.35  # meters from origin
_SPATIAL_NEAR_THRESHOLD: float = 0.05        # 5 cm → "near"
_SPATIAL_AXIS_THRESHOLD: float = 0.02        # 2 cm → directional relations


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

    def distance_from_origin(self) -> float:
        """Euclidean distance from robot base origin in XY plane."""
        return math.sqrt(self.x ** 2 + self.y ** 2)

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

    @classmethod
    def from_dict(cls, d: dict) -> "ObjectState":
        return cls(
            object_id=d["object_id"],
            label=d["label"],
            x=float(d.get("x", 0.0)),
            y=float(d.get("y", 0.0)),
            z=float(d.get("z", 0.0)),
            confidence=float(d.get("confidence", 1.0)),
            state=str(d.get("state", "on_table")),
            last_seen=float(d.get("last_seen", time.time())),
            properties=dict(d.get("properties", {})),
        )


@dataclass(frozen=True)
class RobotState:
    """Immutable snapshot of the robot's current state."""

    joint_positions: tuple[float, ...] = ()
    gripper_state: str = "open"  # open | closed | holding
    held_object: str | None = None
    is_moving: bool = False
    ee_position: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        return {
            "joint_positions": list(self.joint_positions),
            "gripper_state": self.gripper_state,
            "held_object": self.held_object,
            "is_moving": self.is_moving,
            "ee_position": list(self.ee_position),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RobotState":
        return cls(
            joint_positions=tuple(float(v) for v in d.get("joint_positions", [])),
            gripper_state=str(d.get("gripper_state", "open")),
            held_object=d.get("held_object"),
            is_moving=bool(d.get("is_moving", False)),
            ee_position=tuple(float(v) for v in d.get("ee_position", [0.0, 0.0, 0.0])),
        )


class WorldModel:
    """Persistent world state. Tracks objects, robot, and spatial relations.

    Mutable container — ObjectState and RobotState are frozen, but the
    WorldModel itself supports add/remove/update operations.
    """

    def __init__(self) -> None:
        self._objects: dict[str, ObjectState] = {}
        self._robot: RobotState = RobotState()
        self._history: list[dict] = []

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
        }
        updates = {k: v for k, v in kwargs.items() if k in valid_fields}
        self._robot = RobotState(
            joint_positions=updates.get("joint_positions", current.joint_positions),
            gripper_state=updates.get("gripper_state", current.gripper_state),
            held_object=updates.get("held_object", current.held_object),
            is_moving=updates.get("is_moving", current.is_moving),
            ee_position=updates.get("ee_position", current.ee_position),
        )

    # ------------------------------------------------------------------
    # Predicate evaluation
    # ------------------------------------------------------------------

    def check_predicate(self, predicate: str) -> bool:
        """Evaluate a predicate string against current world state.

        Supported predicates:
            gripper_empty              → gripper_state == 'open' AND held_object is None
            gripper_holding_any        → held_object is not None
            gripper_holding(<obj_id>)  → held_object == obj_id
            object_visible(<obj_id>)   → object exists AND confidence > 0.5
            object_reachable(<obj_id>) → object distance from origin < 0.35m

        Unknown predicates return False.
        """
        p = predicate.strip()

        if p == "gripper_empty":
            # Empty = not holding anything. Closed-but-empty counts as empty.
            return self._robot.held_object is None

        if p == "gripper_holding_any":
            return self._robot.held_object is not None

        m = _RE_GRIPPER_HOLDING.match(p)
        if m:
            obj_id = m.group(1)
            return self._robot.held_object == obj_id

        m = _RE_OBJECT_VISIBLE.match(p)
        if m:
            obj_id = m.group(1)
            obj = self._objects.get(obj_id)
            return obj is not None and obj.confidence > _VISIBLE_CONFIDENCE_THRESHOLD

        m = _RE_OBJECT_REACHABLE.match(p)
        if m:
            obj_id = m.group(1)
            obj = self._objects.get(obj_id)
            if obj is None:
                return False
            return obj.distance_from_origin() < _REACHABLE_DISTANCE_THRESHOLD

        logger.debug("WorldModel: unknown predicate %r — returning False", p)
        return False

    # ------------------------------------------------------------------
    # Spatial relations
    # ------------------------------------------------------------------

    def get_spatial_relations(self, object_id: str) -> dict:
        """Compute spatial relations for one object relative to all others.

        Robot frame conventions (ROS standard):
            +X → forward (front of robot)
            +Y → left
            +Z → up

        Returns dict with keys:
            left_of, right_of, in_front_of, behind, near
        Each value is a list of object_ids satisfying that relation.
        """
        result: dict[str, list[str]] = {
            "left_of": [],
            "right_of": [],
            "in_front_of": [],
            "behind": [],
            "near": [],
        }

        obj = self._objects.get(object_id)
        if obj is None:
            return result

        for other_id, other in self._objects.items():
            if other_id == object_id:
                continue

            dx = obj.x - other.x  # positive → obj is further forward than other
            dy = obj.y - other.y  # positive → obj is further left than other (Y+ = left)

            dist_xy = math.sqrt(dx ** 2 + dy ** 2)
            if dist_xy < _SPATIAL_NEAR_THRESHOLD:
                result["near"].append(other_id)

            # dy > 0: obj is to the left of other → other is to the RIGHT of obj
            if dy > _SPATIAL_AXIS_THRESHOLD:
                result["right_of"].append(other_id)
            elif dy < -_SPATIAL_AXIS_THRESHOLD:
                result["left_of"].append(other_id)

            if dx > _SPATIAL_AXIS_THRESHOLD:
                result["in_front_of"].append(other_id)
            elif dx < -_SPATIAL_AXIS_THRESHOLD:
                result["behind"].append(other_id)

        return result

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

    def save(self, path: str) -> None:
        """Persist world state to a YAML file."""
        data = self.to_dict()
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        logger.info("WorldModel saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "WorldModel":
        """Load world state from a YAML file previously saved by save()."""
        with open(path) as f:
            data = yaml.safe_load(f)

        wm = cls()
        for obj_data in data.get("objects", []):
            wm.add_object(ObjectState.from_dict(obj_data))

        robot_data = data.get("robot", {})
        if robot_data:
            wm._robot = RobotState.from_dict(robot_data)

        logger.info("WorldModel loaded from %s (%d objects)", path, len(wm._objects))
        return wm

    # ------------------------------------------------------------------
    # Skill effects
    # ------------------------------------------------------------------

    def apply_skill_effects(self, skill_name: str, params: dict, result: Any) -> None:
        """Update world model based on skill execution results.

        Only applies effects when result.success is True. Built-in effects:
            pick  → mark object as grasped, update robot held_object
            place → mark object as placed, clear robot held_object
            home  → set gripper_state to open

        Unknown skill names are silently ignored (custom skills can call
        update_robot_state / add_object directly).
        """
        if not result.success:
            return

        _skill = skill_name.lower()

        if _skill == "pick":
            mode = params.get("mode", "drop")
            picked_id = params.get("object_id")
            if not picked_id:
                label = params.get("object_label", "")
                matches = self.get_objects_by_label(label)
                if matches:
                    picked_id = matches[0].object_id

            if mode == "hold":
                if picked_id and picked_id in self._objects:
                    old = self._objects[picked_id]
                    self._objects[picked_id] = ObjectState(
                        object_id=old.object_id,
                        label=old.label,
                        x=old.x, y=old.y, z=old.z,
                        confidence=old.confidence,
                        state="grasped",
                        last_seen=time.time(),
                        properties=old.properties,
                    )
                self.update_robot_state(held_object=picked_id, gripper_state="holding")
            else:
                # mode="drop" (default): remove the specific object, gripper returns open
                if picked_id:
                    self.remove_object(picked_id)
                self.update_robot_state(held_object=None, gripper_state="open")

        elif _skill == "place":
            obj_id = params.get("object_id") or self._robot.held_object
            if obj_id and obj_id in self._objects:
                old = self._objects[obj_id]
                self._objects[obj_id] = ObjectState(
                    object_id=old.object_id,
                    label=old.label,
                    x=float(params.get("x", old.x)),
                    y=float(params.get("y", old.y)),
                    z=float(params.get("z", old.z)),
                    confidence=old.confidence,
                    state="placed",
                    last_seen=time.time(),
                    properties=old.properties,
                )
            self.update_robot_state(held_object=None, gripper_state="open")

        elif _skill == "home":
            self.update_robot_state(gripper_state="open")

        else:
            logger.debug("WorldModel: no built-in effects for skill %r", skill_name)

    # ------------------------------------------------------------------
    # Confidence decay
    # ------------------------------------------------------------------

    def decay_confidence(self, decay_rate: float = 0.01) -> None:
        """Reduce confidence of objects based on time elapsed since last seen.

        Decay formula: new_confidence = max(0, old_confidence - decay_rate * elapsed_sec)

        Args:
            decay_rate: confidence reduction per second.
        """
        now = time.time()
        for obj_id, obj in list(self._objects.items()):
            elapsed = max(0.0, now - obj.last_seen)
            new_conf = max(0.0, obj.confidence - decay_rate * elapsed)
            if new_conf != obj.confidence:
                self._objects[obj_id] = ObjectState(
                    object_id=obj.object_id,
                    label=obj.label,
                    x=obj.x, y=obj.y, z=obj.z,
                    confidence=new_conf,
                    state=obj.state,
                    last_seen=obj.last_seen,
                    properties=obj.properties,
                )
