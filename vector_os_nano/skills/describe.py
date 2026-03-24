"""DescribeSkill — capture current frame and ask VLM to describe the scene.

Unlike DetectSkill (which returns bounding boxes for a specific query),
DescribeSkill does open-ended scene understanding: "what objects are on
the table?", "describe what you see", etc.

Uses VLM caption() for general descriptions and visual_query() for
specific questions.

No ROS2 imports.
"""
from __future__ import annotations

import logging

from vector_os_nano.core.skill import Skill, SkillContext, skill
from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)


@skill(
    aliases=["describe", "what do you see", "看到什么", "描述", "有什么", "桌上有什么", "识别一下"],
    direct=False,
    auto_steps=["scan", "describe"],
)
class DescribeSkill:
    """Capture camera frame and use VLM to describe the scene.

    Two modes:
    - No question (or generic): VLM caption — "A table with a battery
      and a banana."
    - Specific question: VLM visual_query — "Is there a red cup?"
    """

    name: str = "describe"
    description: str = "Describe what the camera sees using VLM scene understanding"
    parameters: dict = {
        "question": {
            "type": "string",
            "required": False,
            "description": (
                "Optional question about the scene. If omitted, returns a "
                "general description of all visible objects."
            ),
        }
    }
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {}
    failure_modes: list[str] = ["no_perception", "vlm_error"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        if context.perception is None:
            logger.warning("[DESCRIBE] No perception backend available")
            return SkillResult(
                success=False,
                error_message="No perception backend available",
                result_data={"diagnosis": "no_perception"},
            )

        question: str = params.get("question", "").strip()

        try:
            if question:
                logger.info("[DESCRIBE] Asking VLM: %r", question)
                answer = context.perception.visual_query(question)
                return SkillResult(
                    success=True,
                    result_data={
                        "mode": "query",
                        "question": question,
                        "answer": answer,
                        "diagnosis": "ok",
                    },
                )
            else:
                logger.info("[DESCRIBE] Generating scene caption")
                caption = context.perception.caption(length="long")
                return SkillResult(
                    success=True,
                    result_data={
                        "mode": "caption",
                        "description": caption,
                        "diagnosis": "ok",
                    },
                )
        except Exception as exc:
            logger.error("[DESCRIBE] VLM error: %s", exc)
            return SkillResult(
                success=False,
                error_message=f"VLM error: {exc}",
                result_data={"diagnosis": "vlm_error", "error_detail": str(exc)},
            )
