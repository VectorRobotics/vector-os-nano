"""WaveSkill — wave the arm to greet the user.

Raises the arm to an upright position, oscillates the base joint
left and right several times, then returns home.
"""
from __future__ import annotations

import logging
import time

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)

# Raised position: shoulder up, elbow extended
_RAISED: list[float] = [0.0, -0.6, 0.3, 0.3, 0.0]

# Wave positions: rotate base left/right from raised pose
_WAVE_LEFT: list[float] = [-0.4, -0.6, 0.3, 0.3, 0.0]
_WAVE_RIGHT: list[float] = [0.4, -0.6, 0.3, 0.3, 0.0]

_WAVE_CYCLES: int = 3
_RAISE_DURATION: float = 2.0
_WAVE_DURATION: float = 0.8
_PAUSE: float = 0.15


@skill(
    aliases=["wave", "hello", "hi", "greet", "招手", "打招呼", "挥手", "你好"],
    direct=True,
)
class WaveSkill:
    """Wave the robot arm to greet the user."""

    name: str = "wave"
    description: str = "Wave the arm to greet the user"
    parameters: dict = {}
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {"is_moving": False}
    failure_modes: list[str] = ["no_arm", "move_failed"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        if context.arm is None:
            return SkillResult(
                success=False,
                error_message="No arm connected",
                result_data={"diagnosis": "no_arm"},
            )

        # 1. Raise arm
        logger.info("[WAVE] Raising arm")
        if not context.arm.move_joints(_RAISED, duration=_RAISE_DURATION):
            return SkillResult(
                success=False,
                error_message="Failed to raise arm",
                result_data={"diagnosis": "move_failed"},
            )

        # 2. Open gripper (open hand)
        if context.gripper is not None:
            context.gripper.open()

        time.sleep(_PAUSE)

        # 3. Wave left-right
        for i in range(_WAVE_CYCLES):
            logger.info("[WAVE] Cycle %d/%d", i + 1, _WAVE_CYCLES)
            if not context.arm.move_joints(_WAVE_LEFT, duration=_WAVE_DURATION):
                break
            time.sleep(_PAUSE)
            if not context.arm.move_joints(_WAVE_RIGHT, duration=_WAVE_DURATION):
                break
            time.sleep(_PAUSE)

        # 4. Return to center, then home
        context.arm.move_joints(_RAISED, duration=_WAVE_DURATION)

        home_joints: list[float] = (
            context.config
            .get("skills", {})
            .get("home", {})
            .get("joint_values", [-0.014, -1.238, 0.562, 0.858, 0.311])
        )
        context.arm.move_joints(home_joints, duration=_RAISE_DURATION)

        logger.info("[WAVE] Done")
        return SkillResult(success=True, result_data={"diagnosis": "ok"})
