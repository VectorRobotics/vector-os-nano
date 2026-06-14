# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""ExploreAndSeekSkill — the grand VLN capstone (campaign #8 R11).

One command does the full req #5 loop: AUTONOMOUSLY EXPLORE to cover the area,
then VISUALLY FIND & APPROACH the named colour target. It composes the proven
ExploreSkill + VisionSeekSkill DETERMINISTICALLY in one step, so the behaviour
never depends on the LLM splitting the goal correctly (a 3-way split dropped
the colour param and broke the chain). Honest (rule 5): if the target is never
recognised the seek fails and so does the whole skill — no GT fallback; the
kernel's at_position(<target>, tol) verify is the ground-truth judge.
"""
from __future__ import annotations

import logging

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.skills.explore import ExploreSkill
from vector_os_nano.skills.vision_seek import VisionSeekSkill

logger = logging.getLogger(__name__)


@skill(
    aliases=["explore and find", "探索并找到", "探索房间找到", "探索找到",
             "explore find go"],
    direct=False,
)
class ExploreAndSeekSkill:
    """Explore the room, then visually find & walk to a colour target."""

    name: str = "explore_and_seek"
    description: str = (
        "Autonomously explore the room AND then visually find & walk to a "
        "coloured object — for '探索房间找到X物体并走过去 / explore and find the "
        "X object'. One step: explore coverage, then camera recognition + "
        "approach (NOT GT coordinates)."
    )
    typical_duration_sec: float = 300.0
    verify_hint: str = (
        "at_position(x, y, 1.6) with the named target's coordinates — ground "
        "truth that the robot reached the object it explored for and recognised"
    )
    parameters: dict = {
        "label": {
            "type": "string", "required": True,
            "description": "Target colour/object (e.g. 'red' / '红色物体').",
        },
    }
    preconditions: list = ["base connected with lidar + camera (room mode)"]
    postconditions: list = []
    effects: dict = {}
    failure_modes: list = ["no_base", "no_occupancy", "no_camera", "not_found"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        label = str(params.get("label", "") or "")
        # 1) explore to cover the area (real lidar-map frontier exploration).
        exp = ExploreSkill().execute({}, context)
        # explore failing (e.g. tiny coverage) is non-fatal to the goal — the
        # target may already be in view; press on to the visual seek.
        # 2) visually find & approach the target by recognition.
        seek = VisionSeekSkill().execute({"label": label}, context)
        ok = bool(seek.success)
        return SkillResult(
            success=ok,
            error_message="" if ok else (seek.error_message
                                         or "could not reach the target"),
            result_data={
                "explore": exp.result_data if exp is not None else None,
                "seek": seek.result_data,
                "label": label,
                "transport": "vision",
            },
        )
