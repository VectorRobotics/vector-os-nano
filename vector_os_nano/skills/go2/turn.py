# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""TurnSkill — rotate the base in place by a given angle, closed-loop on heading.

The direction is normalised from its aliases (左/逆时针/ccw -> left, 右/顺时针/cw ->
right); anything else fails loud with the valid set (it used to turn RIGHT). The
turn is one timed yaw walk plus short corrections on the measured heading.
"""
from __future__ import annotations

import logging
import math

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.sim.nav.follow import turn_by

logger = logging.getLogger(__name__)

_VYAW_SPEED: float = 0.5  # rad/s
_HEADING_TOL_RAD: float = math.radians(5.0)

_DEFAULT_DIRECTION: str = "left"
_DEFAULT_ANGLE: float = 90.0  # degrees

# alias -> canonical direction (lower-cased lookup)
DIRECTION_ALIASES: dict[str, str] = {
    **dict.fromkeys(("left", "l", "ccw", "counterclockwise", "counter-clockwise",
                     "anticlockwise", "左", "左转", "向左", "向左转", "往左", "往左转",
                     "逆时针"), "left"),
    **dict.fromkeys(("right", "r", "cw", "clockwise", "右", "右转", "向右", "向右转",
                     "往右", "往右转", "顺时针"), "right"),
}


def normalize_direction(value: object, aliases: dict[str, str]) -> str | None:
    """The canonical direction for *value* (case/whitespace-insensitive), or None."""
    return aliases.get(str(value).strip().lower())


@skill(
    aliases=["turn", "rotate", "转", "转弯", "转向", "左转", "右转"],
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
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {"is_moving": False}
    failure_modes: list[str] = ["no_base", "turn_failed", "bad_direction", "bad_angle"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        """Turn by *angle* degrees toward *direction*; success = heading within 5 deg."""
        if context.base is None:
            logger.error("[TURN] No base connected")
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
                    f"Unknown turn direction {params.get('direction')!r}; valid: left, right "
                    "(or 左/右, 逆时针/顺时针, ccw/cw)"
                ),
            )
        try:
            angle_deg = float(params.get("angle", _DEFAULT_ANGLE))
        except (TypeError, ValueError):
            return SkillResult(success=False, diagnosis_code="bad_angle",
                               error_message=f"Turn angle must be a number, got {params.get('angle')!r}")
        if not math.isfinite(angle_deg):
            return SkillResult(success=False, diagnosis_code="bad_angle",
                               error_message=f"Turn angle must be finite, got {angle_deg}")
        angle_rad = math.radians(abs(angle_deg))
        # Left = positive yaw (counter-clockwise), right = negative yaw (clockwise)
        signed = angle_rad if direction == "left" else -angle_rad

        logger.info("[TURN] direction=%s angle=%.1fdeg", direction, angle_deg)
        ok, residual = turn_by(context.base, signed, vyaw=_VYAW_SPEED, tol=_HEADING_TOL_RAD)
        data = {
            "direction": direction,
            "angle_deg": abs(angle_deg),
            "angle_rad": round(angle_rad, 4),
            "heading_error_deg": round(math.degrees(residual), 1),
        }
        if not ok:
            return SkillResult(
                success=False,
                error_message=f"Turn ended {abs(math.degrees(residual)):.0f} deg off the target heading",
                diagnosis_code="turn_failed",
                result_data=data,
            )
        return SkillResult(success=True, result_data=data)
