"""Shared data types for Vector OS Nano SDK.

All types are frozen dataclasses (immutable). Types that cross module
boundaries include to_dict() and from_dict() for serialization.

No external dependencies — stdlib only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Spatial types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pose3D:
    """6-DOF pose: position (xyz) + orientation (quaternion xyzw)."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0

    @property
    def position(self) -> tuple[float, float, float]:
        """Cartesian position as (x, y, z)."""
        return (self.x, self.y, self.z)

    @property
    def orientation(self) -> tuple[float, float, float, float]:
        """Quaternion as (qx, qy, qz, qw)."""
        return (self.qx, self.qy, self.qz, self.qw)

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "qx": self.qx,
            "qy": self.qy,
            "qz": self.qz,
            "qw": self.qw,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Pose3D:
        return cls(
            x=float(d.get("x", 0.0)),
            y=float(d.get("y", 0.0)),
            z=float(d.get("z", 0.0)),
            qx=float(d.get("qx", 0.0)),
            qy=float(d.get("qy", 0.0)),
            qz=float(d.get("qz", 0.0)),
            qw=float(d.get("qw", 1.0)),
        )


@dataclass(frozen=True)
class BBox3D:
    """3D axis-aligned bounding box defined by center pose + half-extents."""

    center: Pose3D
    size_x: float = 0.0
    size_y: float = 0.0
    size_z: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": self.center.to_dict(),
            "size_x": self.size_x,
            "size_y": self.size_y,
            "size_z": self.size_z,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BBox3D:
        return cls(
            center=Pose3D.from_dict(d.get("center", {})),
            size_x=float(d.get("size_x", 0.0)),
            size_y=float(d.get("size_y", 0.0)),
            size_z=float(d.get("size_z", 0.0)),
        )


# ---------------------------------------------------------------------------
# Camera types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CameraIntrinsics:
        return cls(
            fx=float(d["fx"]),
            fy=float(d["fy"]),
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            width=int(d["width"]),
            height=int(d["height"]),
        )


# ---------------------------------------------------------------------------
# Perception types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """Single object detection from VLM or 2D detector.

    bbox: (x1, y1, x2, y2) in pixel coordinates, top-left origin.
    """

    label: str
    bbox: tuple[float, float, float, float]
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "bbox": self.bbox,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Detection:
        bbox_raw = d["bbox"]
        if isinstance(bbox_raw, (list, tuple)):
            bbox: tuple[float, float, float, float] = (
                float(bbox_raw[0]),
                float(bbox_raw[1]),
                float(bbox_raw[2]),
                float(bbox_raw[3]),
            )
        else:
            raise ValueError(f"bbox must be a list or tuple, got {type(bbox_raw)}")
        return cls(
            label=str(d["label"]),
            bbox=bbox,
            confidence=float(d.get("confidence", 1.0)),
        )


@dataclass(frozen=True)
class TrackedObject:
    """Object being tracked across frames.

    Carries both 2D pixel bbox and optional 3D pose/bbox.
    mask is an optional numpy array — typed as Any to avoid numpy dependency.
    """

    track_id: int
    label: str
    bbox_2d: tuple[float, float, float, float]
    pose: Pose3D | None = None
    bbox_3d: BBox3D | None = None
    confidence: float = 1.0
    mask: Any = None  # np.ndarray when perception stack is available


# ---------------------------------------------------------------------------
# Skill execution types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillResult:
    """Result returned by Skill.execute().

    Frozen to prevent post-execution mutation.
    result_data carries skill-specific output (e.g., grasp position).
    diagnosis_code is a short machine-readable failure code (e.g. "no_base").
    """

    success: bool
    result_data: dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    diagnosis_code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "result_data": self.result_data,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SkillResult:
        return cls(
            success=bool(d["success"]),
            result_data=dict(d.get("result_data", {})),
            error_message=str(d.get("error_message", "")),
        )


# ---------------------------------------------------------------------------
# Task planning types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskStep:
    """One step in a task plan.

    depends_on: step_ids of steps that must complete before this one.
    preconditions: predicate expressions checked before execution.
    postconditions: predicate expressions checked after execution.
    """

    step_id: str
    skill_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "skill_name": self.skill_name,
            "parameters": self.parameters,
            "depends_on": list(self.depends_on),
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskStep:
        return cls(
            step_id=str(d["step_id"]),
            skill_name=str(d["skill_name"]),
            parameters=dict(d.get("parameters", {})),
            depends_on=list(d.get("depends_on", [])),
            preconditions=list(d.get("preconditions", [])),
            postconditions=list(d.get("postconditions", [])),
        )


@dataclass(frozen=True)
class TaskPlan:
    """A decomposed task plan produced by the LLM planner.

    steps: ordered list of TaskStep (executor respects depends_on graph).
    requires_clarification: planner couldn't resolve ambiguity.
    """

    goal: str
    steps: list[TaskStep] = field(default_factory=list)
    requires_clarification: bool = False
    clarification_question: str | None = None
    message: str | None = None  # AI message to user (from planner)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "requires_clarification": self.requires_clarification,
            "clarification_question": self.clarification_question,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskPlan:
        return cls(
            goal=str(d["goal"]),
            steps=[TaskStep.from_dict(s) for s in d.get("steps", [])],
            requires_clarification=bool(d.get("requires_clarification", False)),
            clarification_question=d.get("clarification_question"),
        )


# ---------------------------------------------------------------------------
# Execution trace types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepTrace:
    """Execution record for one TaskStep.

    status values: "success", "precondition_failed", "execution_failed",
                   "postcondition_failed", "skipped"
    """

    step_id: str
    skill_name: str
    status: str
    duration_sec: float = 0.0
    error: str = ""
    result_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "skill_name": self.skill_name,
            "status": self.status,
            "duration_sec": self.duration_sec,
            "error": self.error,
            "result_data": self.result_data,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StepTrace:
        return cls(
            step_id=str(d["step_id"]),
            skill_name=str(d["skill_name"]),
            status=str(d["status"]),
            duration_sec=float(d.get("duration_sec", 0.0)),
            error=str(d.get("error", "")),
            result_data=dict(d.get("result_data", {})),
        )


@dataclass(frozen=True)
class ExecutionResult:
    """Final result of a task execution.

    status values: "completed", "failed", "partially_completed",
                   "clarification_needed"
    """

    success: bool
    status: str
    steps_completed: int = 0
    steps_total: int = 0
    failed_step: TaskStep | None = None
    failure_reason: str | None = None
    trace: list[StepTrace] = field(default_factory=list)
    world_model_diff: dict[str, Any] = field(default_factory=dict)
    clarification_question: str | None = None
    message: str | None = None  # AI conversational response to the user

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "failed_step": self.failed_step.to_dict() if self.failed_step else None,
            "failure_reason": self.failure_reason,
            "trace": [t.to_dict() for t in self.trace],
            "world_model_diff": self.world_model_diff,
            "clarification_question": self.clarification_question,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExecutionResult:
        failed_step_data = d.get("failed_step")
        return cls(
            success=bool(d["success"]),
            status=str(d["status"]),
            steps_completed=int(d.get("steps_completed", 0)),
            steps_total=int(d.get("steps_total", 0)),
            failed_step=TaskStep.from_dict(failed_step_data) if failed_step_data else None,
            failure_reason=d.get("failure_reason"),
            trace=[StepTrace.from_dict(t) for t in d.get("trace", [])],
            world_model_diff=dict(d.get("world_model_diff", {})),
            clarification_question=d.get("clarification_question"),
        )


# ---------------------------------------------------------------------------
# Agent Loop types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionRecord:
    """Record of a single action in the agent loop."""

    iteration: int
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    skill_success: bool = False
    verified: bool = False
    reasoning: str = ""
    duration_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "action": self.action,
            "params": self.params,
            "skill_success": self.skill_success,
            "verified": self.verified,
            "reasoning": self.reasoning,
            "duration_sec": self.duration_sec,
        }


@dataclass(frozen=True)
class GoalResult:
    """Result of an iterative agent loop execution."""

    success: bool
    goal: str
    iterations: int
    total_duration_sec: float
    actions: list[ActionRecord] = field(default_factory=list)
    summary: str = ""
    final_world_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "goal": self.goal,
            "iterations": self.iterations,
            "total_duration_sec": self.total_duration_sec,
            "actions": [a.to_dict() for a in self.actions],
            "summary": self.summary,
            "final_world_state": self.final_world_state,
        }


# ---------------------------------------------------------------------------
# Navigation sensor types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Odometry:
    """Robot odometry: pose + velocity in a single snapshot."""

    timestamp: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    vyaw: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in
                ["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "vyaw"]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Odometry":
        return cls(**{k: float(d.get(k, 0.0 if k != "qw" else 1.0)) for k in
                      ["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "vyaw"]})


@dataclass(frozen=True)
class LaserScan:
    """2D laser scan (simulated or real)."""

    timestamp: float
    angle_min: float
    angle_max: float
    angle_increment: float
    range_min: float
    range_max: float
    ranges: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "angle_min": self.angle_min,
            "angle_max": self.angle_max,
            "angle_increment": self.angle_increment,
            "range_min": self.range_min,
            "range_max": self.range_max,
            "ranges": list(self.ranges),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LaserScan":
        return cls(
            timestamp=float(d["timestamp"]),
            angle_min=float(d["angle_min"]),
            angle_max=float(d["angle_max"]),
            angle_increment=float(d["angle_increment"]),
            range_min=float(d["range_min"]),
            range_max=float(d["range_max"]),
            ranges=tuple(float(r) for r in d["ranges"]),
        )
