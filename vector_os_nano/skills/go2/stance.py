# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Stance skills for the Go2 quadruped: stand, sit, lie down.

Each commands the base's posture and reports what it can measure: the base height
before and after. The base exposes no posture read-back, so a skill never claims
the posture was reached; it says the command ran and what the height did.
"""
from __future__ import annotations

import logging

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)


def _height(base) -> float | None:
    try:
        return round(float(base.get_position()[2]), 3)
    except Exception:  # noqa: BLE001 — a base without a pose cannot be measured
        return None


def _command_posture(context: SkillContext, posture: str) -> SkillResult:
    """Run base.<posture>() and report the measured base height around it."""
    base = context.base
    if base is None:
        return SkillResult(success=False, error_message="No base connected", diagnosis_code="no_base")
    command = getattr(base, posture, None)
    if not callable(command):
        return SkillResult(
            success=False, diagnosis_code=f"{posture}_failed",
            error_message=f"{type(base).__name__} cannot {posture.replace('_', ' ')} (no {posture})",
        )
    logger.info("[STANCE] Commanding %s", posture)
    before = _height(base)
    ok = command()
    data = {
        "stance": posture,
        "height_before_m": before,
        "height_after_m": _height(base),
        "posture_verified": False,
        "note": "the base reports no posture: the command ran; the height change is all it can tell",
    }
    if not ok:
        return SkillResult(success=False, error_message=f"{posture} command failed",
                           diagnosis_code=f"{posture}_failed", result_data=data)
    return SkillResult(success=True, result_data=data)


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
        return _command_posture(context, "stand")


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
        return _command_posture(context, "sit")


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
        return _command_posture(context, "lie_down")
