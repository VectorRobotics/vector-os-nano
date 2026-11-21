# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Home through the arm's configured controller; preserve a held load."""
from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.perception.base import failure_cause
import threading


@skill(aliases=["go home", "reset", "回家", "归位", "复位", "回到初始位置"])
class HomeSkill:
    name = "home"
    description = "Move the empty arm to its configured home pose. Place a held object first."
    verify_hint = "arm_at_home()"
    parameters = {}
    preconditions = []
    postconditions = []
    effects = {"is_moving": False}
    failure_modes = ["no_arm", "holding_object", "move_failed", "gripper_failed", "cancelled", "device_failed"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        cancel = context.services.get("cancel") or threading.Event()
        def fail(code, message):
            return SkillResult(False, {"diagnosis": code}, message, diagnosis_code=code)
        if context.arm is None:
            return fail("no_arm", "No arm connected")
        try:
            if context.gripper is not None and context.gripper.is_holding():
                return fail("holding_object", "Place the held object before home; keeping its current hold.")
            if cancel.is_set():
                return fail("cancelled", "Home cancelled")
            moved = context.arm.home()
            if cancel.is_set():
                return fail("cancelled", "Home cancelled")
            if not moved:
                return fail("move_failed", "Arm did not reach its configured home pose")
            if context.gripper is not None and not context.gripper.open():
                return fail("gripper_failed", "Arm reached home but the gripper did not open")
            return SkillResult(True, {"diagnosis": "ok", "joint_values": context.arm.get_joint_positions()})
        except Exception as exc:
            return fail("device_failed", f"Home device failed: {failure_cause(exc)}")
