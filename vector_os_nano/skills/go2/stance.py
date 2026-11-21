# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Stance skills for a legged base: stand, sit, lie down.

Each commands the base's posture (``BaseProtocol.set_posture``: held until the next
velocity command) and reports the controller result and base height before and after.
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
    """Run base.set_posture(posture) and report the measured base height around it."""
    base = context.base
    if base is None:
        return SkillResult(success=False, error_message="No base connected", diagnosis_code="no_base")
    if not callable(getattr(base, "set_posture", None)):
        return SkillResult(
            success=False, diagnosis_code=f"{posture}_failed",
            error_message=f"{type(base).__name__} cannot {posture.replace('_', ' ')} (no set_posture)",
        )
    logger.info("[STANCE] Commanding %s", posture)
    before = _height(base)
    try:
        ok = base.set_posture(posture)
    except KeyError as exc:  # a body without this posture: the valid ones are named
        return SkillResult(success=False, error_message=str(exc), diagnosis_code=f"{posture}_failed")
    data = {
        "stance": posture,
        "height_before_m": before,
        "height_after_m": _height(base),
        "controller_reached_posture": bool(ok),
        "posture_verified": False,
        "note": "controller result only; use the world's posture oracle for independent verification",
    }
    if not ok:
        return SkillResult(success=False, error_message=f"{posture}: the controller did not reach the posture",
                           diagnosis_code=f"{posture}_failed", result_data=data)
    return SkillResult(success=True, result_data=data)


@skill(
    aliases=["stand", "站", "站起来", "起立"],
)
class StandSkill:
    """Command the base's standing posture."""

    name: str = "stand"
    description: str = "Command the base to stand upright."
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
    """Command the sitting posture when the body declares it."""

    name: str = "sit"
    description: str = "Command the base to sit down, if its manifest supports it."
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
    """Command the prone posture when the body declares it."""

    name: str = "lie_down"
    description: str = "Command the base to lie down, if its manifest supports it."
    parameters: dict = {}
    preconditions: list[str] = []
    postconditions: list[str] = ["base_lying"]
    effects: dict = {"base_stance": "lie_down"}
    failure_modes: list[str] = ["no_base", "lie_down_failed"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        return _command_posture(context, "lie_down")
