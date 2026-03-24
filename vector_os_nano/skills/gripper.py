"""Gripper skills — open and close as proper @skill classes.

Replaces hard-coded gripper routing in Agent._try_direct().
"""
from __future__ import annotations

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult


@skill(
    aliases=[
        "open", "open grip", "open gripper", "open claw",
        "release", "let go",
        "张开", "松开", "打开",
    ],
    direct=True,
)
class GripperOpenSkill:
    """Open the gripper / release held object."""

    name: str = "gripper_open"
    description: str = "Open the gripper to release any held object"
    parameters: dict = {}
    preconditions: list[str] = []
    postconditions: list[str] = ["gripper_empty"]
    effects: dict = {"gripper_state": "open", "held_object": None}
    failure_modes: list[str] = ["no_arm"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        if context.gripper is None:
            return SkillResult(
                success=False,
                error_message="No gripper connected",
                result_data={"diagnosis": "no_arm"},
            )
        context.gripper.open()
        context.world_model.update_robot_state(gripper_state="open", held_object=None)
        return SkillResult(success=True, result_data={"diagnosis": "ok"})


@skill(
    aliases=[
        "close", "close grip", "close gripper", "close claw",
        "grip", "clench",
        "夹紧", "合上", "关闭",
    ],
    direct=True,
)
class GripperCloseSkill:
    """Close the gripper."""

    name: str = "gripper_close"
    description: str = "Close the gripper"
    parameters: dict = {}
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {"gripper_state": "closed"}
    failure_modes: list[str] = ["no_arm"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        if context.gripper is None:
            return SkillResult(
                success=False,
                error_message="No gripper connected",
                result_data={"diagnosis": "no_arm"},
            )
        context.gripper.close()
        context.world_model.update_robot_state(gripper_state="closed")
        return SkillResult(success=True, result_data={"diagnosis": "ok"})
