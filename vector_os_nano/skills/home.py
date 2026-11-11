# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""HomeSkill — move arm to home position and open gripper.

Ported from skill_node_v2._execute_home(). No ROS2 imports.
"""
from __future__ import annotations

import logging

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)

_DEFAULT_HOME_JOINTS: list[float] = [-0.014, -1.238, 0.562, 0.858, 0.311]
_HOME_DURATION: float = 3.0


@skill(
    aliases=["go home", "reset", "回家", "归位", "复位", "回到初始位置"],
)
class HomeSkill:
    """Move arm to home position and open gripper.

    Always executable — no preconditions required. After execution the
    gripper is open and the arm is in the home configuration.
    """

    name: str = "home"
    description: str = "Move arm to home position and open gripper"
    # Typical REAL-TIME (viewer-synced) duration: 3s arm move + gripper + overhead.
    # GoalExecutor floors the step timeout at this value (R2-2) so a fast-emitted plan
    # (e.g. timeout_sec=5) does not falsely mark home as timed-out under a live viewer.
    typical_duration_sec: float = 12.0
    # Success predicate this skill is verified against (single-source for the planner).
    verify_hint: str = "arm_at_home()"
    parameters: dict = {}
    preconditions: list[str] = []
    postconditions: list[str] = ["gripper_empty"]
    effects: dict = {
        "gripper_state": "open",
        "held_object": None,
        "is_moving": False,
    }
    failure_modes: list[str] = ["no_arm", "move_failed"]

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
            return SkillResult(
                success=False,
                error_message="No arm connected",
                result_data={"diagnosis": "no_arm"},
            )

        # DoF-aware (Rule 11): the default home pose is the 5-DoF SO-101 pose. A
        # different arm (e.g. the 6-DoF Piper) has a different DoF, so a fixed-length
        # pose is rejected by move_joints ("expected 6 positions, got 5") — the bug
        # that blocked the bare-`vector-cli` + NL fetch path through the planner/
        # executor. If the configured pose length doesn't match THIS arm's DoF, fall
        # back to the URDF-zero neutral home (`[0.0]*dof`, exactly what the arm's own
        # home() uses, e.g. MuJoCoPiper.home()). SO-101 (5==5) is byte-identical —
        # this branch never fires for it.
        arm_dof = getattr(context.arm, "dof", None)
        if arm_dof is None:
            try:
                arm_dof = len(context.arm.get_joint_positions())
            except Exception:  # noqa: BLE001
                arm_dof = len(home_joints)
        if len(home_joints) != arm_dof:
            home_joints = [0.0] * int(arm_dof)
            logger.info(
                "[HOME] configured home pose len != arm DoF (%s); "
                "using URDF-zero neutral home",
                arm_dof,
            )

        logger.info("[HOME] Moving to home pose: %s", home_joints)
        success = context.arm.move_joints(home_joints, duration=_HOME_DURATION)

        if not success:
            logger.error("[HOME] Arm move failed")
            return SkillResult(
                success=False,
                error_message="Arm move to home failed",
                result_data={"diagnosis": "move_failed"},
            )

        if context.gripper is not None:
            logger.info("[HOME] Opening gripper")
            context.gripper.open()

        logger.info("[HOME] Done")
        return SkillResult(
            success=True,
            result_data={"joint_values": list(home_joints), "diagnosis": "ok"},
        )
