"""ScanSkill — move arm to scan position.

Ported from skill_node_v2._execute_scan(). For SO-101, scan pose == home
pose because the camera already provides a good workspace view at home.
The scan pose is configurable via context.config.

No ROS2 imports.
"""
from __future__ import annotations

import logging

from vector_os.core.skill import Skill, SkillContext
from vector_os.core.types import SkillResult

logger = logging.getLogger(__name__)

# Scan pose = home pose (matches ARM_SCAN_VALUES in skill_node_v2.py)
_DEFAULT_SCAN_JOINTS: list[float] = [-0.014, -1.238, 0.562, 0.858, 0.311]

# Duration for the scan joint move (seconds)
_SCAN_DURATION: float = 3.0


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
            return SkillResult(success=False, error_message="No arm connected")

        logger.info("[SCAN] Moving to scan pose: %s", scan_joints)
        success = context.arm.move_joints(scan_joints, duration=_SCAN_DURATION)

        if not success:
            logger.error("[SCAN] Arm move failed")
            return SkillResult(success=False, error_message="Arm move to scan pose failed")

        logger.info("[SCAN] Done")
        return SkillResult(
            success=True,
            result_data={"joint_values": list(scan_joints)},
        )
