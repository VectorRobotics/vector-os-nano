# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

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

    @classmethod
    def from_matrix(cls, K: Any, width: int, height: int) -> CameraIntrinsics:
        """From a (3, 3) pinhole matrix (``RGBDFrame.K``) and the frame's shape."""
        return cls(fx=float(K[0][0]), fy=float(K[1][1]), cx=float(K[0][2]), cy=float(K[1][2]),
                   width=int(width), height=int(height))


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
