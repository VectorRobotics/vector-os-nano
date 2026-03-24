"""ScanSkill — move arm to scan position.

Ported from skill_node_v2._execute_scan(). For SO-101, scan pose == home
pose because the camera already provides a good workspace view at home.
The scan pose is configurable via context.config.

No ROS2 imports.
"""
from __future__ import annotations

import logging

from vector_os_nano.core.skill import Skill, SkillContext, skill
from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)

_DEFAULT_SCAN_JOINTS: list[float] = [-0.014, -1.238, 0.562, 0.858, 0.311]
_SCAN_DURATION: float = 3.0


@skill(
    aliases=["look", "observe", "看看", "扫描", "看一下"],
    direct=True,
)
class ScanSkill:
    """Move arm to scan position for workspace observation.

    For SO-101, scan pose is identical to home pose — the camera already
    provides a complete workspace view from this configuration.  A different
    scan pose can be provided via context.config["skills"]["scan"]["joint_values"].

    No preconditions; always executable.
    """

    name: str = "scan"
    description: str = "Move arm to scan position for workspace observation"
    parameters: dict = {}
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {"is_moving": False}
    failure_modes: list[str] = ["no_arm", "move_failed"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        """Move to scan joint configuration.

        Scan joint values are read from context.config["skills"]["scan"]["joint_values"]
        if present; falls back to the default (== home pose).

        Args:
            params: ignored (ScanSkill takes no parameters).
            context: SkillContext providing arm access.

        Returns:
            SkillResult(success=True) when arm reaches scan pose.
            SkillResult(success=False) if the arm move fails.
        """
        scan_joints: list[float] = (
            context.config
            .get("skills", {})
            .get("scan", {})
            .get("joint_values", _DEFAULT_SCAN_JOINTS)
        )

        if context.arm is None:
            return SkillResult(
                success=False,
                error_message="No arm connected",
                result_data={"diagnosis": "no_arm"},
            )

        logger.info("[SCAN] Moving to scan pose: %s", scan_joints)
        success = context.arm.move_joints(scan_joints, duration=_SCAN_DURATION)

        if not success:
            logger.error("[SCAN] Arm move failed")
            return SkillResult(
                success=False,
                error_message="Arm move to scan pose failed",
                result_data={"diagnosis": "move_failed"},
            )

        logger.info("[SCAN] Done")
        return SkillResult(
            success=True,
            result_data={"joint_values": list(scan_joints), "diagnosis": "ok"},
        )
