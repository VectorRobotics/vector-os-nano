# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""HandoverSkill — hand an object to the user.

Rotates the arm 90 degrees toward the user and releases the gripper.
Used when the user says "给我" (give it to me) instead of placing
on the table.

Algorithm:
  1. From home position (holding object), rotate shoulder_pan +90deg
  2. Open gripper to release
  3. Close gripper
  4. Return to home

No ROS2 imports.
"""
from __future__ import annotations

import logging
import time

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)

_HOME_DURATION: float = 3.0
_DEFAULT_HOME_JOINTS: list[float] = [-0.014, -1.238, 0.562, 0.858, 0.311]


@skill(
    aliases=["give", "hand over", "给我", "递给我", "给", "拿给我", "交给我"],
)
class HandoverSkill:
    """Hand a held object to the user by rotating and releasing.

    Rotates shoulder_pan ~90 degrees from home position, opens gripper
    to release the object, then returns home. Similar to pick's "drop"
    mode but intended as a deliberate handover to the user.
    """

    name: str = "handover"
    description: str = "Hand the held object to the user. Rotates arm toward user and releases. Use when user says 'give me' or '给我'."
    parameters: dict = {
        "direction": {
            "type": "string",
            "required": False,
            "default": "right",
            "enum": ["left", "right"],
            "description": "Which side the user is on: 'right' (default) or 'left'",
        },
    }
    preconditions: list[str] = ["gripper_holding_any"]
    postconditions: list[str] = ["gripper_empty"]
    effects: dict = {"gripper_state": "open", "held_object": None}
    failure_modes: list[str] = ["no_arm", "move_failed"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        if context.arm is None:
            return SkillResult(success=False, error_message="No arm connected",
                               result_data={"diagnosis": "no_arm"})

        home_joints: list[float] = (
            context.config.get("skills", {})
            .get("home", {})
            .get("joint_values", _DEFAULT_HOME_JOINTS)
        )

        direction = params.get("direction", "right")
        # Rotate shoulder_pan: +90deg for right, -90deg for left
        rotation = 1.57 if direction == "right" else -1.57

        # Step 1: Move to handover position (rotate from home)
        handover_joints = list(home_joints)
        handover_joints[0] = handover_joints[0] + rotation

        logger.info("[HANDOVER] Rotating %s (%.2f rad) to hand over...", direction, rotation)
        if not context.arm.move_joints(handover_joints, duration=_HOME_DURATION):
            return SkillResult(success=False, error_message="Move to handover position failed",
                               result_data={"diagnosis": "move_failed", "phase": "rotate"})

        # Step 2: Open gripper to release
        logger.info("[HANDOVER] Opening gripper...")
        if context.gripper is not None:
            context.gripper.open()
            time.sleep(0.5)
            context.gripper.close()

        # Step 3: Return home
        logger.info("[HANDOVER] Returning home...")
        if not context.arm.move_joints(home_joints, duration=_HOME_DURATION):
            return SkillResult(success=False, error_message="Return home failed",
                               result_data={"diagnosis": "move_failed", "phase": "home"})

        logger.info("[HANDOVER] Done!")
        return SkillResult(
            success=True,
            result_data={"diagnosis": "ok", "direction": direction},
        )
