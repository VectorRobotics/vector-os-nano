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

from vector_os.core.skill import Skill, SkillContext
from vector_os.core.types import SkillResult

logger = logging.getLogger(__name__)

# Default place target (metres, base frame) — from skill_node_v2 parameters
_DEFAULT_PLACE_X: float = 0.25
_DEFAULT_PLACE_Y: float = 0.0
_DEFAULT_PLACE_Z: float = 0.05

# Pre-grasp height above the place target (matches skill_node_v2 pre_grasp_height)
_DEFAULT_PRE_GRASP_HEIGHT: float = 0.06  # 6 cm

# Motion durations (seconds)
_APPROACH_DURATION: float = 3.0
_DESCEND_DURATION: float = 2.0
_LIFT_DURATION: float = 2.0


class PlaceSkill:
    """Place a held object at a specified position in the workspace.

    The skill moves above the target, descends, opens the gripper, and lifts.
    Target position is specified in base_link frame metres.

    Parameters:
        x (float, optional): target X in metres (default 0.25).
        y (float, optional): target Y in metres (default 0.0).
        z (float, optional): target Z in metres (default 0.05).
    """

    name: str = "place"
    description: str = "Place a held object at a target position"
    parameters: dict = {
        "x": {
            "type": "float",
            "required": False,
            "default": _DEFAULT_PLACE_X,
            "description": "Target X in metres (base frame)",
        },
        "y": {
            "type": "float",
            "required": False,
            "default": _DEFAULT_PLACE_Y,
            "description": "Target Y in metres (base frame)",
        },
        "z": {
            "type": "float",
            "required": False,
            "default": _DEFAULT_PLACE_Z,
            "description": "Target Z in metres (base frame)",
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
            .get("pick", {})  # shared with pick config
            .get("pre_grasp_height", _DEFAULT_PRE_GRASP_HEIGHT)
        )

        tx = float(params.get("x", cfg_place.get("x", _DEFAULT_PLACE_X)))
        ty = float(params.get("y", cfg_place.get("y", _DEFAULT_PLACE_Y)))
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

        logger.info("[PLACE] Place complete!")
        return SkillResult(
            success=True,
            result_data={"placed_at": [round(tx, 4), round(ty, 4), round(tz, 4)]},
        )
