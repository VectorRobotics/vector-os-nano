"""PlaceSkill — place a held object at a target position.

Full port of skill_node_v2._execute_place(). The target position is
specified via parameters (x, y, z in metres in the base frame).

Algorithm:
  1. Build target and above-target positions
  2. IK for above-target
  3. Move above-target
  4. IK for target (warm-started from above)
  5. Descend to target
  6. Open gripper
  7. Lift back above target

No ROS2 imports.
"""
from __future__ import annotations

import logging

import numpy as np

from vector_os_nano.core.skill import Skill, SkillContext, skill
from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)

# Named location map — from robot's perspective
# x+ = forward (away from base), y+ = left, y- = right
_LOCATION_MAP: dict[str, tuple[float, float]] = {
    "front":        (0.30, 0.00),
    "front_left":   (0.30, 0.12),
    "front_right":  (0.30, -0.12),
    "center":       (0.22, 0.00),
    "left":         (0.22, 0.12),
    "right":        (0.22, -0.12),
    "back":         (0.12, 0.00),
    "back_left":    (0.12, 0.12),
    "back_right":   (0.12, -0.12),
}

_DEFAULT_PLACE_Z: float = 0.04
_DEFAULT_PRE_GRASP_HEIGHT: float = 0.06
_APPROACH_DURATION: float = 3.0
_DESCEND_DURATION: float = 2.0
_LIFT_DURATION: float = 2.0
_HOME_DURATION: float = 3.0
_DEFAULT_HOME_JOINTS: list[float] = [-0.014, -1.238, 0.562, 0.858, 0.311]


@skill(
    aliases=["put", "put down", "放", "放下", "放到", "放置", "放在"],
    direct=False,
)
class PlaceSkill:
    """Place a held object at a named location or coordinates.

    Accepts either a named location (front, left, back_right, etc.)
    or explicit x, y, z coordinates in metres.

    Parameters:
        location (str, optional): Named position — front, front_left, front_right,
            center, left, right, back, back_left, back_right.
        x, y, z (float, optional): Explicit coordinates (override location).
    """

    name: str = "place"
    description: str = "Place held object at a location: front, left, right, center, back, front_left, front_right, back_left, back_right"
    parameters: dict = {
        "location": {
            "type": "string",
            "required": False,
            "default": "front",
            "description": "Named position: front, front_left, front_right, center, left, right, back, back_left, back_right",
        },
        "x": {
            "type": "float",
            "required": False,
            "description": "Target X in metres (overrides location)",
        },
        "y": {
            "type": "float",
            "required": False,
            "description": "Target Y in metres (overrides location)",
        },
        "z": {
            "type": "float",
            "required": False,
            "description": "Target Z in metres",
        },
    }
    preconditions: list[str] = ["gripper_holding_any"]
    postconditions: list[str] = ["gripper_empty"]
    effects: dict = {
        "gripper_state": "open",
        "held_object": None,
    }

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        """Execute place sequence.

        Falls back to context.config["skills"]["place"] defaults if params
        are missing; then falls back to the module-level defaults.

        Args:
            params: optional x, y, z keys.
            context: SkillContext providing arm and gripper access.

        Returns:
            SkillResult(success=True, result_data={"placed_at": [x, y, z]}) on success.
            SkillResult(success=False, error_message=...) on failure.
        """
        if context.arm is None:
            return SkillResult(success=False, error_message="No arm connected")

        cfg_place = context.config.get("skills", {}).get("place", {})
        pre_grasp_h: float = (
            context.config.get("skills", {})
            .get("pick", {})
            .get("pre_grasp_height", _DEFAULT_PRE_GRASP_HEIGHT)
        )
        home_joints: list[float] = (
            context.config.get("skills", {}).get("home", {}).get(
                "joint_values", _DEFAULT_HOME_JOINTS
            )
        )

        # Resolve target coordinates from location name or explicit params
        if "x" in params and "y" in params:
            tx = float(params["x"])
            ty = float(params["y"])
        else:
            location = params.get("location", "front")
            loc_xy = _LOCATION_MAP.get(location, _LOCATION_MAP["front"])
            tx, ty = loc_xy
            logger.info("[PLACE] Location '%s' → (%.3f, %.3f)", location, tx, ty)
        tz = float(params.get("z", cfg_place.get("z", _DEFAULT_PLACE_Z)))

        logger.info("[PLACE] Target: (%.3f, %.3f, %.3f) m", tx, ty, tz)

        place_pos = np.array([tx, ty, tz], dtype=float)
        above_pos = place_pos.copy()
        above_pos[2] += pre_grasp_h

        current_joints = context.arm.get_joint_positions()

        # IK for above-place position
        q_above_result = context.arm.ik(
            (above_pos[0], above_pos[1], above_pos[2]),
            current_joints,
        )
        if q_above_result is None:
            return SkillResult(
                success=False,
                error_message="IK failed for above-place position",
            )
        q_above = list(q_above_result)

        # Move above target
        logger.info("[PLACE] Moving above target ...")
        if not context.arm.move_joints(q_above, duration=_APPROACH_DURATION):
            return SkillResult(success=False, error_message="Move to above-place failed")

        # IK for place position (warm-started from above)
        q_place_result = context.arm.ik(
            (place_pos[0], place_pos[1], place_pos[2]),
            q_above,
        )
        if q_place_result is None:
            return SkillResult(
                success=False,
                error_message="IK failed for place position",
            )
        q_place = list(q_place_result)

        # Descend to place position
        logger.info("[PLACE] Descending ...")
        if not context.arm.move_joints(q_place, duration=_DESCEND_DURATION):
            return SkillResult(success=False, error_message="Place descent failed")

        # Open gripper to release object
        logger.info("[PLACE] Opening gripper ...")
        if context.gripper is not None:
            context.gripper.open()

        # Lift back to above position
        logger.info("[PLACE] Lifting ...")
        if not context.arm.move_joints(q_above, duration=_LIFT_DURATION):
            return SkillResult(success=False, error_message="Place lift failed")

        # Close gripper and return home
        if context.gripper is not None:
            context.gripper.close()

        logger.info("[PLACE] Returning home ...")
        context.arm.move_joints(home_joints, duration=_HOME_DURATION)

        logger.info("[PLACE] Place complete!")
        return SkillResult(
            success=True,
            result_data={"placed_at": [round(tx, 4), round(ty, 4), round(tz, 4)]},
        )
