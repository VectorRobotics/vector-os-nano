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

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult

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

_DEFAULT_DIRECTION: str = "forward"
_DEFAULT_DISTANCE: float = 1.0   # metres
_DEFAULT_SPEED: float = 0.3      # m/s


@skill(
    aliases=["walk", "go", "move", "走", "走路", "往前走", "前进", "后退"],
    direct=False,
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
    # R7: REAL evidence predicate — verifies measured displacement, not the
    # command echo (step_output is the step's own result_data).
    verify_template: str = "step_output('moved_m') > 0.05"
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
    failure_modes: list[str] = ["no_base", "walk_failed"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        """Execute the walk command.

        Args:
            params: direction, distance, speed.
            context: SkillContext with base (Go2) attached.

        Returns:
            SkillResult(success=True) when walk completes.
            SkillResult(success=False, diagnosis_code="no_base") if base missing.
        """
        if context.base is None:
            logger.error("[WALK] No base connected")
            return SkillResult(
                success=False,
                error_message="No base connected",
                diagnosis_code="no_base",
            )

        direction: str = params.get("direction", _DEFAULT_DIRECTION)
        distance: float = float(params.get("distance", _DEFAULT_DISTANCE))
        speed: float = float(params.get("speed", _DEFAULT_SPEED))
        if distance <= 0.0:
            # R7 (design review #6): a zero/negative distance silently
            # produced a 0-duration no-op PASS.
            return SkillResult(
                success=False,
                error_message=f"walk distance must be > 0 (got {distance})",
                result_data={"diagnosis": "bad_params"},
            )

        vx_sign, vy_sign = _DIRECTION_MAP.get(direction, (1.0, 0.0))

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

        def _pos():
            if hasattr(context.base, "get_position"):
                try:
                    return list(context.base.get_position())
                except Exception:  # noqa: BLE001
                    return None
            return None

        import math as _math
        import time as _time

        start = _pos()
        t0 = _time.monotonic()
        ok = context.base.walk(vx, vy, 0.0, duration)
        elapsed = _time.monotonic() - t0

        if not ok:
            return SkillResult(
                success=False,
                error_message="Walk command failed",
                diagnosis_code="walk_failed",
            )

        position = _pos()
        # Motion evidence (R7, design review #6): MEASURE, never echo the
        # command. moved_m=None when the base has no odometry (fail-open,
        # explicitly unmeasured — never fabricated).
        moved = None
        if start is not None and position is not None:
            moved = round(_math.hypot(position[0] - start[0],
                                      position[1] - start[1]), 3)
        result_data = {
            "direction": direction,
            "requested_m": distance,
            "moved_m": moved,
            "start": start,
            "position": position,
            "duration_s": round(elapsed, 3),
        }
        if moved is not None and moved < 0.3 * distance:
            # Commanded a metre, moved (almost) nothing: wall-blocked, or the
            # base silently dropped the axis (habitat vy, #7). Loud, typed.
            result_data["diagnosis"] = "moved_short"
            return SkillResult(
                success=False,
                error_message=(
                    f"walk moved {moved:.2f}m of the requested "
                    f"{distance:.2f}m — blocked or unsupported direction"
                ),
                result_data=result_data,
            )
        return SkillResult(success=True, result_data=result_data)
