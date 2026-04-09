"""Perception primitives — wrap camera, VLM, and lidar for CaP-X generated code.

All functions are module-level and read from the module-global _ctx.
Requires init_primitives() to be called before use.
"""
from __future__ import annotations

import math
from typing import Any

from vector_os_nano.vcli.primitives import PrimitiveContext

_ctx: PrimitiveContext | None = None


def _require_base() -> object:
    """Return _ctx.base or raise RuntimeError if unavailable."""
    if _ctx is None or _ctx.base is None:
        raise RuntimeError(
            "No hardware connected. Call init_primitives() with a valid base."
        )
    return _ctx.base


def _require_vlm() -> object:
    """Return _ctx.vlm or raise RuntimeError if unavailable."""
    if _ctx is None or _ctx.vlm is None:
        raise RuntimeError(
            "No VLM connected. Call init_primitives() with a valid vlm."
        )
    return _ctx.vlm


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


def capture_image() -> Any:
    """Capture an RGB image from the robot camera.

    Returns:
        (H, W, 3) uint8 ndarray representing the current camera frame.

    Raises:
        RuntimeError: If no hardware is connected.
    """
    base = _require_base()
    return base.get_camera_frame()


# ---------------------------------------------------------------------------
# VLM scene understanding
# ---------------------------------------------------------------------------


def describe_scene() -> str:
    """Generate a VLM text description of the current camera view.

    Returns:
        Non-empty string summarising the scene.

    Raises:
        RuntimeError: If no hardware or VLM is connected.
    """
    frame = capture_image()
    vlm = _require_vlm()
    result = vlm.describe_scene(frame)
    return str(result.summary)


def detect_objects(query: str = "") -> list[dict]:
    """Detect objects visible in the current camera frame.

    Args:
        query: Optional filter string (e.g. "cup", "chair"). Empty = all objects.

    Returns:
        List of {"name": str, "confidence": float} dicts.

    Raises:
        RuntimeError: If no hardware or VLM is connected.
    """
    frame = capture_image()
    vlm = _require_vlm()
    raw = vlm.find_objects(frame, query)
    return [{"name": str(o.name), "confidence": float(o.confidence)} for o in raw]


def identify_room() -> tuple[str, float]:
    """Use VLM to identify the room the robot is currently in.

    Returns:
        (room_name, confidence) tuple.

    Raises:
        RuntimeError: If no hardware or VLM is connected.
    """
    frame = capture_image()
    vlm = _require_vlm()
    result = vlm.identify_room(frame)
    return (str(result.room), float(result.confidence))


# ---------------------------------------------------------------------------
# Lidar
# ---------------------------------------------------------------------------


def measure_distance(angle_rad: float = 0.0) -> float:
    """Measure lidar distance at the specified angle.

    Args:
        angle_rad: Angle in radians relative to robot forward (0 = forward,
                   positive = counter-clockwise). Wrapped to [-pi, pi].

    Returns:
        Distance in meters at the nearest lidar ray to the requested angle.

    Raises:
        RuntimeError: If no hardware is connected or lidar is unavailable.
    """
    base = _require_base()
    scan = base.get_lidar_scan()
    if scan is None:
        raise RuntimeError("Lidar is not available on this hardware.")

    n = len(scan.ranges)
    if n == 0:
        raise RuntimeError("Lidar scan returned empty ranges.")

    # Map angle to index. Wrap angle to [angle_min, angle_max].
    angle_min: float = float(scan.angle_min)
    angle_increment: float = float(scan.angle_increment)

    # Clamp to valid range
    relative = angle_rad - angle_min
    idx = int(round(relative / angle_increment)) % n
    return float(scan.ranges[idx])


def scan_360() -> list[tuple[float, float]]:
    """Return the full 360-degree lidar scan as (angle_rad, distance_m) pairs.

    Returns:
        List of (angle_rad, distance_m) tuples sorted by ascending angle.

    Raises:
        RuntimeError: If no hardware is connected or lidar is unavailable.
    """
    base = _require_base()
    scan = base.get_lidar_scan()
    if scan is None:
        raise RuntimeError("Lidar is not available on this hardware.")

    n = len(scan.ranges)
    angle_min = float(scan.angle_min)
    angle_increment = float(scan.angle_increment)

    return [
        (float(angle_min + i * angle_increment), float(scan.ranges[i]))
        for i in range(n)
    ]
