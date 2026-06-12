# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""TurnSkill — rotate the Go2 quadruped by a given angle.

Maps direction + angle to a timed yaw velocity command via
context.base.walk(0, 0, vyaw, duration).
"""
from __future__ import annotations

import logging
import math

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)

_VYAW_SPEED: float = 0.5  # rad/s

_DEFAULT_DIRECTION: str = "left"
_DEFAULT_ANGLE: float = 90.0  # degrees


@skill(
    aliases=["turn", "rotate", "转", "转弯", "转向", "左转", "右转"],
    direct=False,
)
class TurnSkill:
    """Rotate the Go2 quadruped left or right by a given angle."""

    name: str = "turn"
    description: str = "Rotate the quadruped left or right by a given angle in degrees."
    parameters: dict = {
        "direction": {
            "type": "string",
            "required": False,
            "default": _DEFAULT_DIRECTION,
            "enum": ["left", "right"],
            "description": "Turn direction: left (counter-clockwise) or right (clockwise)",
        },
        "angle": {
            "type": "number",
            "required": False,
            "default": _DEFAULT_ANGLE,
            "description": "Rotation angle in degrees",
        },
    }
    # Typical REAL-TIME duration for a 90-degree turn at 0.5 rad/s: ~3s; 10s gives
    # headroom for larger angles and ROS2 overhead. GoalExecutor floors the step
    # timeout at this value (R2-2).
    typical_duration_sec: float = 10.0
    # R7: REAL evidence predicate — measured rotation, not the command echo.
    verify_template: str = "abs(step_output('turned_rad')) > 0.1"
    is_motor: bool = True
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {"is_moving": False}
    failure_modes: list[str] = ["no_base", "turn_failed"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        """Execute the turn command.

        Args:
            params: direction, angle.
            context: SkillContext with base (Go2) attached.

        Returns:
            SkillResult(success=True) when rotation completes.
            SkillResult(success=False, diagnosis_code="no_base") if base missing.
        """
        if context.base is None:
            logger.error("[TURN] No base connected")
            return SkillResult(
                success=False,
                error_message="No base connected",
                diagnosis_code="no_base",
            )

        direction: str = params.get("direction", _DEFAULT_DIRECTION)
        angle_deg: float = float(params.get("angle", _DEFAULT_ANGLE))
        if angle_deg == 0.0:
            return SkillResult(
                success=False,
                error_message="turn angle must be non-zero",
                result_data={"diagnosis": "bad_params"},
            )
        angle_rad: float = math.radians(abs(angle_deg))

        # Left = positive yaw (counter-clockwise), right = negative yaw (clockwise)
        sign: float = 1.0 if direction == "left" else -1.0
        vyaw: float = sign * _VYAW_SPEED
        duration: float = angle_rad / _VYAW_SPEED

        logger.info(
            "[TURN] direction=%s angle=%.1fdeg vyaw=%.2f duration=%.2fs",
            direction, angle_deg, vyaw, duration,
        )

        def _heading():
            if hasattr(context.base, "get_heading"):
                try:
                    return float(context.base.get_heading())
                except Exception:  # noqa: BLE001
                    return None
            return None

        import time as _time

        h0 = _heading()
        t0 = _time.monotonic()
        ok = context.base.walk(0.0, 0.0, vyaw, duration)
        elapsed = _time.monotonic() - t0

        if not ok:
            return SkillResult(
                success=False,
                error_message="Turn command failed",
                diagnosis_code="turn_failed",
            )

        h1 = _heading()
        turned = None
        if h0 is not None and h1 is not None:
            # wrap-aware smallest signed delta
            turned = math.atan2(math.sin(h1 - h0), math.cos(h1 - h0))
            turned = round(turned, 4)
        result_data = {
            "direction": direction,
            "angle_deg": angle_deg,
            "requested_rad": round(angle_rad, 4),
            "turned_rad": turned,
            "duration_s": round(elapsed, 3),
        }
        if turned is not None and abs(turned) < 0.3 * angle_rad:
            result_data["diagnosis"] = "turned_short"
            return SkillResult(
                success=False,
                error_message=(
                    f"turn rotated {abs(turned):.2f}rad of the requested "
                    f"{angle_rad:.2f}rad"
                ),
                result_data=result_data,
            )
        return SkillResult(success=True, result_data=result_data)
