"""Locomotion primitives — wrap BaseProtocol for CaP-X generated code.

All functions are module-level and read from the module-global _ctx.
Requires init_primitives() to be called before use.

No hardware connected → raises RuntimeError for all functions to keep
behavior consistent and explicit.
"""
from __future__ import annotations

import math
import time

from vector_os_nano.vcli.primitives import PrimitiveContext

_ctx: PrimitiveContext | None = None


def _require_base() -> object:
    """Return _ctx.base or raise RuntimeError if unavailable."""
    if _ctx is None or _ctx.base is None:
        raise RuntimeError(
            "No hardware connected. Call init_primitives() with a valid base."
        )
    return _ctx.base


# ---------------------------------------------------------------------------
# State queries
# ---------------------------------------------------------------------------


def get_position() -> tuple[float, float, float]:
    """Robot (x, y, z) in world frame, meters.

    Returns:
        Tuple of (x, y, z) floats in world frame.

    Raises:
        RuntimeError: If no hardware is connected.
    """
    base = _require_base()
    raw = base.get_position()
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def get_heading() -> float:
    """Robot yaw in radians. Positive = counter-clockwise.

    Returns:
        Current yaw angle in radians.

    Raises:
        RuntimeError: If no hardware is connected.
    """
    base = _require_base()
    return float(base.get_heading())


# ---------------------------------------------------------------------------
# Velocity control
# ---------------------------------------------------------------------------


def set_velocity(vx: float, vy: float, vyaw: float) -> None:
    """Set robot velocity. Non-blocking.

    Args:
        vx: Forward velocity in m/s (body frame).
        vy: Lateral velocity in m/s (body frame, positive = left).
        vyaw: Yaw rate in rad/s (positive = counter-clockwise).

    Raises:
        RuntimeError: If no hardware is connected.
    """
    base = _require_base()
    base.set_velocity(vx, vy, vyaw)


def stop() -> None:
    """Immediately stop all motion.

    Raises:
        RuntimeError: If no hardware is connected.
    """
    base = _require_base()
    base.set_velocity(0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Blocking locomotion
# ---------------------------------------------------------------------------


def walk_forward(distance_m: float, speed: float = 0.4) -> bool:
    """Walk forward the specified distance. Blocking.

    Args:
        distance_m: Distance to walk in meters (positive = forward).
        speed: Forward speed in m/s. Default 0.4 m/s.

    Returns:
        True if the distance was reached, False on timeout (30 s).

    Raises:
        RuntimeError: If no hardware is connected.
    """
    _require_base()
    start = get_position()
    set_velocity(speed, 0.0, 0.0)
    deadline = time.monotonic() + 30.0

    while time.monotonic() < deadline:
        pos = get_position()
        dx = pos[0] - start[0]
        dy = pos[1] - start[1]
        covered = math.sqrt(dx * dx + dy * dy)
        if covered >= abs(distance_m):
            stop()
            return True
        time.sleep(0.05)

    stop()
    return False


def turn(angle_rad: float, rate: float = 1.0) -> bool:
    """Turn by the given angle. Blocking.

    Args:
        angle_rad: Angle to turn in radians. Positive = counter-clockwise.
        rate: Yaw rate in rad/s. Default 1.0 rad/s.

    Returns:
        True if the turn completed, False on timeout (15 s).

    Raises:
        RuntimeError: If no hardware is connected.
    """
    _require_base()
    start_heading = get_heading()
    direction = 1.0 if angle_rad >= 0 else -1.0
    set_velocity(0.0, 0.0, direction * abs(rate))
    target = abs(angle_rad)
    deadline = time.monotonic() + 15.0

    while time.monotonic() < deadline:
        current = get_heading()
        # Compute shortest-path angular difference
        delta = abs(_angle_diff(current, start_heading))
        if delta >= target:
            stop()
            return True
        time.sleep(0.05)

    stop()
    return False


def _angle_diff(a: float, b: float) -> float:
    """Signed difference a - b, wrapped to [-pi, pi]."""
    diff = a - b
    while diff > math.pi:
        diff -= 2.0 * math.pi
    while diff < -math.pi:
        diff += 2.0 * math.pi
    return diff


# ---------------------------------------------------------------------------
# Posture control
# ---------------------------------------------------------------------------


def stand() -> bool:
    """Stand up. Returns True on success.

    Returns:
        True if the stand command succeeded.

    Raises:
        RuntimeError: If no hardware is connected.
    """
    base = _require_base()
    return bool(base.stand())


def sit() -> bool:
    """Sit down. Returns True on success.

    Returns:
        True if the sit command succeeded.

    Raises:
        RuntimeError: If no hardware is connected.
    """
    base = _require_base()
    return bool(base.sit())
