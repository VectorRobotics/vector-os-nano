# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""WalkSkill — move the Go2 quadruped in a given direction.

Maps direction + distance + speed to a timed velocity command sent via
context.base.walk(vx, vy, vyaw, duration).

Velocity clamps (conservative for Milestone 1):
  vx:  [-0.5, 0.5] m/s
  vy:  [-0.3, 0.3] m/s
"""
from __future__ import annotations

import logging
import math

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.skills.go2.turn import normalize_direction

logger = logging.getLogger(__name__)

# Velocity limits (m/s) — conservative for Milestone 1
_VX_MAX: float = 0.5
_VY_MAX: float = 0.3

# Direction → (vx_sign, vy_sign)
_DIRECTION_MAP: dict[str, tuple[float, float]] = {
    "forward":  ( 1.0,  0.0),
    "backward": (-1.0,  0.0),
    "left":     ( 0.0,  1.0),
    "right":    ( 0.0, -1.0),
}

# alias -> canonical direction; anything else fails loud (it used to walk forward)
DIRECTION_ALIASES: dict[str, str] = {
    **dict.fromkeys(("forward", "forwards", "front", "ahead", "f", "前", "前进", "往前",
                     "向前", "往前走"), "forward"),
    **dict.fromkeys(("backward", "backwards", "back", "b", "后", "后退", "往后", "向后",
                     "倒退"), "backward"),
    **dict.fromkeys(("left", "l", "左", "往左", "向左", "左走"), "left"),
    **dict.fromkeys(("right", "r", "右", "往右", "向右", "右走"), "right"),
}

# What an open-loop walk can honestly claim: under load the gait covers 40-100 % of
# a request, so a walk that moved less than _BLOCKED_FRACTION of it was blocked;
# below _MIN_CHECKABLE_M the gait's start-up transient is the whole move, so a short
# walk reports its displacement and says it cannot tell.
_BLOCKED_FRACTION: float = 0.25
_MIN_CHECKABLE_M: float = 0.3

_DEFAULT_DIRECTION: str = "forward"
_DEFAULT_DISTANCE: float = 1.0   # metres
_DEFAULT_SPEED: float = 0.3      # m/s


@skill(
    aliases=["walk", "go", "move", "走", "走路", "往前走", "前进", "后退"],
)
class WalkSkill:
    """Move the Go2 quadruped in a given direction for a given distance."""

    name: str = "walk"
    description: str = (
        "Walk the quadruped forward, backward, left, or right by a given distance."
    )
    # Typical REAL-TIME duration for a default 1m walk at 0.3 m/s: ~3s command +
    # ROS2/network overhead + stabilisation ≈ 10–15s; 30s gives headroom for longer
    # distances. GoalExecutor floors the step timeout at this value (R2-2).
    typical_duration_sec: float = 30.0
    parameters: dict = {
        "direction": {
            "type": "string",
            "required": False,
            "default": _DEFAULT_DIRECTION,
            "enum": ["forward", "backward", "left", "right"],
            "description": "Direction of travel",
        },
        "distance": {
            "type": "number",
            "required": False,
            "default": _DEFAULT_DISTANCE,
            "description": "Distance to travel in metres",
        },
        "speed": {
            "type": "number",
            "required": False,
            "default": _DEFAULT_SPEED,
            "description": "Translational speed in m/s (clamped to 0.5 m/s)",
        },
    }
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {"is_moving": False}
    failure_modes: list[str] = ["no_base", "walk_failed", "walk_blocked", "bad_direction", "bad_distance"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        """Execute the walk command.

        Args:
            params: direction, distance, speed.
            context: SkillContext with base (Go2) attached.

        Returns:
            SkillResult(success=True) when the walk ran and was not blocked, with the
            displacement it measured; failure when the base fell or barely moved.
        """
        if context.base is None:
            logger.error("[WALK] No base connected")
            return SkillResult(
                success=False,
                error_message="No base connected",
                diagnosis_code="no_base",
            )

        direction = normalize_direction(params.get("direction", _DEFAULT_DIRECTION),
                                        DIRECTION_ALIASES)
        if direction is None:
            return SkillResult(
                success=False, diagnosis_code="bad_direction",
                error_message=(
                    f"Unknown walk direction {params.get('direction')!r}; valid: "
                    f"{', '.join(_DIRECTION_MAP)}"
                ),
            )
        try:
            distance = float(params.get("distance", _DEFAULT_DISTANCE))
            speed = float(params.get("speed", _DEFAULT_SPEED))
        except (TypeError, ValueError):
            return SkillResult(success=False, diagnosis_code="bad_distance",
                               error_message="Walk distance and speed must be numbers")
        # A negative distance made a negative duration: the gait ran with no stop
        # (skills-nav:walk-runaway-on-exception). Direction carries the sign.
        if not (math.isfinite(distance) and distance > 0 and math.isfinite(speed) and speed > 0):
            return SkillResult(
                success=False, diagnosis_code="bad_distance",
                error_message=(
                    f"Walk needs a positive distance and speed (got distance={distance}, "
                    f"speed={speed}); use direction for backward/left/right"
                ),
            )

        vx_sign, vy_sign = _DIRECTION_MAP[direction]

        # Apply speed and clamp
        if vy_sign != 0.0:
            # Lateral movement
            vy: float = max(-_VY_MAX, min(_VY_MAX, vy_sign * abs(speed)))
            vx: float = 0.0
            actual_speed = abs(vy)
        else:
            # Forward/backward movement
            vx = max(-_VX_MAX, min(_VX_MAX, vx_sign * abs(speed)))
            vy = 0.0
            actual_speed = abs(vx)

        duration: float = distance / actual_speed if actual_speed > 0.0 else 0.0

        logger.info(
            "[WALK] direction=%s vx=%.2f vy=%.2f vyaw=%.2f duration=%.1fs",
            direction, vx, vy, 0.0, duration,
        )

        start = _position(context.base)
        ok = context.base.walk(vx, vy, 0.0, duration)
        end = _position(context.base)
        moved = None if start is None or end is None else math.hypot(end[0] - start[0], end[1] - start[1])
        data = {
            "direction": direction,
            "distance": distance,
            "moved_m": None if moved is None else round(moved, 3),
            "position": end,
        }
        if not ok:
            return SkillResult(success=False, error_message="Walk command failed",
                               diagnosis_code="walk_failed", result_data=data)
        if moved is None or distance < _MIN_CHECKABLE_M:
            data["note"] = "the displacement cannot confirm a walk this short (or was not measured)"
        elif moved < _BLOCKED_FRACTION * distance:
            return SkillResult(
                success=False, diagnosis_code="walk_blocked", result_data=data,
                error_message=f"Walk blocked: moved {moved:.2f} m of the {distance:.2f} m asked",
            )
        return SkillResult(success=True, result_data=data)


def _position(base) -> list[float] | None:
    """The base's position, or None when it cannot report one."""
    try:
        p = [float(v) for v in base.get_position()]
        return p if len(p) >= 2 else None
    except Exception:  # noqa: BLE001 — a base without a pose cannot be measured
        return None
