# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Stance skills for the Go2 quadruped: stand, sit, lie down.

All three are direct skills (no LLM planning needed). Each validates that
context.base is present before issuing the command.
"""
from __future__ import annotations

import logging

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)


@skill(
    aliases=["stand", "站", "站起来", "起立"],
)
class StandSkill:
    """Command the Go2 to stand up from any posture."""

    name: str = "stand"
    description: str = "Command the quadruped to stand upright."
    parameters: dict = {}
    preconditions: list[str] = []
    postconditions: list[str] = ["base_standing"]
    effects: dict = {"base_stance": "stand"}
    failure_modes: list[str] = ["no_base", "stand_failed"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        """Issue stand command to base.

        Args:
            params: unused.
            context: SkillContext with base (Go2).

        Returns:
            SkillResult(success=True) when command accepted.
            SkillResult(success=False) if base missing or command fails.
        """
        if context.base is None:
            logger.error("[STAND] No base connected")
            return SkillResult(
                success=False,
                error_message="No base connected",
                diagnosis_code="no_base",
            )

        logger.info("[STAND] Commanding stand")
        ok = context.base.stand()

        if not ok:
            return SkillResult(
                success=False,
                error_message="Stand command failed",
                diagnosis_code="stand_failed",
            )

        return SkillResult(success=True, result_data={"stance": "stand"})


@skill(
    aliases=["sit", "坐", "坐下"],
)
class SitSkill:
    """Command the Go2 to sit down."""

    name: str = "sit"
    description: str = "Command the quadruped to sit down."
    parameters: dict = {}
    preconditions: list[str] = []
    postconditions: list[str] = ["base_sitting"]
    effects: dict = {"base_stance": "sit"}
    failure_modes: list[str] = ["no_base", "sit_failed"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        """Issue sit command to base.

        Args:
            params: unused.
            context: SkillContext with base (Go2).

        Returns:
            SkillResult(success=True) when command accepted.
            SkillResult(success=False) if base missing or command fails.
        """
        if context.base is None:
            logger.error("[SIT] No base connected")
            return SkillResult(
                success=False,
                error_message="No base connected",
                diagnosis_code="no_base",
            )

        logger.info("[SIT] Commanding sit")
        ok = context.base.sit()

        if not ok:
            return SkillResult(
                success=False,
                error_message="Sit command failed",
                diagnosis_code="sit_failed",
            )

        return SkillResult(success=True, result_data={"stance": "sit"})


@skill(
    aliases=["lie down", "lie", "趴", "趴下", "躺下"],
)
class LieDownSkill:
    """Command the Go2 to lie down (prone posture)."""

    name: str = "lie_down"
    description: str = "Command the quadruped to lie down in prone posture."
    parameters: dict = {}
    preconditions: list[str] = []
    postconditions: list[str] = ["base_lying"]
    effects: dict = {"base_stance": "lie_down"}
    failure_modes: list[str] = ["no_base", "lie_down_failed"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        """Issue lie_down command to base.

        Args:
            params: unused.
            context: SkillContext with base (Go2).

        Returns:
            SkillResult(success=True) when command accepted.
            SkillResult(success=False) if base missing or command fails.
        """
        if context.base is None:
            logger.error("[LIE_DOWN] No base connected")
            return SkillResult(
                success=False,
                error_message="No base connected",
                diagnosis_code="no_base",
            )

        lie_down = getattr(context.base, "lie_down", None)
        if not callable(lie_down):  # e.g. the ROS2 proxy base (skills-nav:missed5)
            return SkillResult(
                success=False,
                error_message=f"{type(context.base).__name__} cannot lie down (no lie_down)",
                diagnosis_code="lie_down_failed",
            )
        logger.info("[LIE_DOWN] Commanding lie down")
        ok = lie_down()

        if not ok:
            return SkillResult(
                success=False,
                error_message="Lie down command failed",
                diagnosis_code="lie_down_failed",
            )

        return SkillResult(success=True, result_data={"stance": "lie_down"})
