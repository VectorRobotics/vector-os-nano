# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""ScanSkill — move arm to scan position.

Ported from skill_node_v2._execute_scan(). For SO-101, scan pose == home
pose because the camera already provides a good workspace view at home.
The scan pose is configurable via context.config.

No ROS2 imports.
"""
from __future__ import annotations

import logging

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)

_DEFAULT_SCAN_JOINTS: list[float] = [-0.014, -1.238, 0.562, 0.858, 0.311]
_SCAN_DURATION: float = 3.0


@skill(
    aliases=["look", "observe", "看看", "扫描", "看一下"],
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
    # No meaningful state predicate — the always-safe truthy literal.
    verify_hint: str = "True"
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

        # DoF-aware (Rule 11): the default scan pose is the 5-DoF SO-101 pose. A
        # different arm (e.g. the 6-DoF Piper) has a different DoF, so a fixed-length
        # pose is rejected by move_joints ("expected 6 positions, got 5"). If the pose
        # length doesn't match THIS arm's DoF, hold the arm's CURRENT pose (a safe
        # no-op move): for go2+Piper the workspace view comes from the go2 head camera,
        # not the arm wrist, so the scan arm pose is non-critical. SO-101 (5==5) is
        # byte-identical — this branch never fires for it.
        arm_dof = getattr(context.arm, "dof", None)
        if arm_dof is None:
            try:
                arm_dof = len(context.arm.get_joint_positions())
            except Exception:  # noqa: BLE001
                arm_dof = len(scan_joints)
        if len(scan_joints) != arm_dof:
            try:
                scan_joints = list(context.arm.get_joint_positions())
            except Exception:  # noqa: BLE001
                scan_joints = [0.0] * int(arm_dof)
            logger.info(
                "[SCAN] configured scan pose len != arm DoF (%s); holding current pose",
                arm_dof,
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


def scan_first(context: SkillContext, caller: str) -> SkillResult | None:
    """The scan prerequisite of a perceiving skill: the arm moves out of the camera's
    view before it looks. None when the arm reached the scan pose (or there is no arm),
    else the caller's own failure naming the phase."""
    if context.arm is None:
        return None
    scan = ScanSkill().execute({}, context)
    if scan.success:
        return None
    return SkillResult(
        success=False,
        error_message=f"{caller}: moving to the scan pose failed: {scan.error_message}",
        result_data={**scan.result_data, "phase": "scan"},
    )
