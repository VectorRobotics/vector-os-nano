"""HomeSkill — move arm to home position and open gripper.

Ported from skill_node_v2._execute_home(). No ROS2 imports.
"""
from __future__ import annotations

import logging

from vector_os_nano.core.skill import Skill, SkillContext, skill
from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)

_DEFAULT_HOME_JOINTS: list[float] = [-0.014, -1.238, 0.562, 0.858, 0.311]
_HOME_DURATION: float = 3.0


@skill(
    aliases=["go home", "reset", "回家", "归位", "复位", "回到初始位置"],
    direct=True,
)
class HomeSkill:
    """Move arm to home position and open gripper.

    Always executable — no preconditions required. After execution the
    gripper is open and the arm is in the home configuration.
    """

    name: str = "home"
    description: str = "Move arm to home position and open gripper"
    parameters: dict = {}
    preconditions: list[str] = []
    postconditions: list[str] = ["gripper_empty"]
    effects: dict = {
        "gripper_state": "open",
        "held_object": None,
        "is_moving": False,
    }

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        """Move to home joint configuration, then open gripper.

        Home joint values are read from context.config["skills"]["home"]["joint_values"]
        if present; otherwise the hard-coded default is used.

        Args:
            params: ignored (HomeSkill takes no parameters).
            context: SkillContext providing arm and gripper access.

        Returns:
            SkillResult(success=True) when arm reaches home and gripper opens.
            SkillResult(success=False) if the arm move fails.
        """
        home_joints: list[float] = (
            context.config
            .get("skills", {})
            .get("home", {})
            .get("joint_values", _DEFAULT_HOME_JOINTS)
        )

        if context.arm is None:
            return SkillResult(success=False, error_message="No arm connected")

        logger.info("[HOME] Moving to home pose: %s", home_joints)
        success = context.arm.move_joints(home_joints, duration=_HOME_DURATION)

        if not success:
            logger.error("[HOME] Arm move failed")
            return SkillResult(success=False, error_message="Arm move to home failed")

        if context.gripper is not None:
            logger.info("[HOME] Opening gripper")
            context.gripper.open()

        logger.info("[HOME] Done")
        return SkillResult(
            success=True,
            result_data={"joint_values": list(home_joints)},
        )
