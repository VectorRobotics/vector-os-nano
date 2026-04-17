"""VLM-powered look skills for the Go2 quadruped robot.

Two skills:
- LookSkill: capture frame, call describe_scene + identify_room, record to SpatialMemory.
- DescribeSceneSkill: detailed VLM scene description with optional query.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np  # noqa: F401 — used by callers for frame type

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LookSkill
# ---------------------------------------------------------------------------


@skill(
    aliases=[
        "look",
        "看",
        "看看",
        "看一下",
        "看一看",
        "what do you see",
        "describe",
    ],
    direct=False,
)
class LookSkill:
    """Look around and describe what the robot sees using VLM."""

    name: str = "look"
    description: str = "Look around and describe what the robot sees using VLM."
    parameters: dict = {}
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {"scene_observed": True}
    failure_modes: list[str] = [
        "no_base",
        "no_vlm",
        "camera_failed",
        "vlm_failed",
    ]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        """Capture a camera frame, run VLM scene description and room ID."""
        if context.base is None:
            logger.error("[LOOK] No base connected")
            return SkillResult(
                success=False,
                error_message="No base connected",
                diagnosis_code="no_base",
            )

        vlm = context.services.get("vlm")
        if vlm is None:
            logger.error("[LOOK] VLM service not available")
            return SkillResult(
                success=False,
                error_message="VLM service not available",
                diagnosis_code="no_vlm",
            )

        # Capture frame and pose
        try:
            frame: np.ndarray = context.base.get_camera_frame()
        except Exception as exc:
            logger.error("[LOOK] get_camera_frame failed: %s", exc)
            return SkillResult(
                success=False,
                error_message=f"Camera capture failed: {exc}",
                diagnosis_code="camera_failed",
            )

        pos = context.base.get_position()
        heading = context.base.get_heading()

        # Run VLM
        try:
            scene = vlm.describe_scene(frame)
            room_id = vlm.identify_room(frame)
        except Exception as exc:
            logger.error("[LOOK] VLM call failed: %s", exc)
            return SkillResult(
                success=False,
                error_message=f"VLM inference failed: {exc}",
                diagnosis_code="vlm_failed",
            )

        room: str = room_id.room if room_id.room != "unknown" else _fallback_room(context)

        # Record room visit to spatial memory with detected objects.
        objects = list(scene.objects)
        spatial_memory = context.services.get("spatial_memory")
        if spatial_memory is not None:
            try:
                if hasattr(spatial_memory, "observe_with_viewpoint"):
                    spatial_memory.observe_with_viewpoint(
                        room, float(pos[0]), float(pos[1]),
                        float(heading), objects, scene.summary,
                    )
                else:
                    spatial_memory.visit(room, float(pos[0]), float(pos[1]))
            except Exception as exc:
                logger.warning("[LOOK] spatial_memory update failed: %s", exc)

        logger.info(
            "[LOOK] room=%s confidence=%.2f objects=%d summary=%s",
            room, room_id.confidence, len(objects), scene.summary,
        )

        return SkillResult(
            success=True,
            result_data={
                "room": room,
                "summary": scene.summary,
                "details": scene.details,
                "room_confidence": room_id.confidence,
                "objects": objects,
            },
        )


# ---------------------------------------------------------------------------
# DescribeSceneSkill
# ---------------------------------------------------------------------------


@skill(
    aliases=[
        "describe scene",
        "描述场景",
        "描述环境",
    ],
    direct=False,
)
class DescribeSceneSkill:
    """Get a detailed VLM description of the current scene."""

    name: str = "describe_scene"
    description: str = "Get a detailed VLM description of the current scene."
    parameters: dict = {
        "query": {
            "type": "string",
            "required": False,
            "description": "Optional: what to look for",
        }
    }
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {"scene_observed": True}
    failure_modes: list[str] = [
        "no_base",
        "no_vlm",
        "camera_failed",
        "vlm_failed",
    ]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        """Capture a camera frame and run a detailed VLM scene analysis.

        When ``params["query"]`` is provided, delegates to
        ``vlm.find_objects(frame, query)`` and returns matching objects.
        Otherwise runs the full describe_scene + identify_room pipeline.

        Args:
            params: May contain an optional ``query`` string.
            context: SkillContext with base and vlm service attached.

        Returns:
            SkillResult with result_data containing scene analysis fields.
        """
        if context.base is None:
            logger.error("[DESCRIBE_SCENE] No base connected")
            return SkillResult(
                success=False,
                error_message="No base connected",
                diagnosis_code="no_base",
            )

        vlm = context.services.get("vlm")
        if vlm is None:
            logger.error("[DESCRIBE_SCENE] VLM service not available")
            return SkillResult(
                success=False,
                error_message="VLM service not available",
                diagnosis_code="no_vlm",
            )

        # Capture frame.
        try:
            frame: np.ndarray = context.base.get_camera_frame()
        except Exception as exc:
            logger.error("[DESCRIBE_SCENE] get_camera_frame failed: %s", exc)
            return SkillResult(
                success=False,
                error_message=f"Camera capture failed: {exc}",
                diagnosis_code="camera_failed",
            )

        query: str | None = params.get("query") or None

        if query is not None:
            # Query mode — use find_objects for targeted search.
            return self._run_find_objects(frame, query, context, vlm)

        # Full description mode.
        return self._run_full_description(frame, context, vlm)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_find_objects(
        self,
        frame: np.ndarray,
        query: str,
        context: SkillContext,
        vlm: Any,
    ) -> SkillResult:
        """Run find_objects for a targeted query and return matching objects."""
        try:
            found = vlm.find_objects(frame, query)
        except Exception as exc:
            logger.error("[DESCRIBE_SCENE] find_objects failed: %s", exc)
            return SkillResult(
                success=False,
                error_message=f"VLM inference failed: {exc}",
                diagnosis_code="vlm_failed",
            )

        objects_data: list[dict[str, Any]] = [
            {
                "name": obj.name,
                "description": obj.description,
                "confidence": obj.confidence,
            }
            for obj in found
        ]

        # Record to SpatialMemory if available.
        room: str = _fallback_room(context)
        spatial_memory = context.services.get("spatial_memory")
        if spatial_memory is not None and objects_data:
            object_names: list[str] = [obj.name for obj in found]
            try:
                spatial_memory.observe(room, object_names, f"query: {query}")
            except Exception as exc:
                logger.warning(
                    "[DESCRIBE_SCENE] spatial_memory.observe failed: %s", exc
                )

        logger.info(
            "[DESCRIBE_SCENE] query=%r found=%d objects",
            query,
            len(found),
        )

        return SkillResult(
            success=True,
            result_data={
                "query": query,
                "objects": objects_data,
                "count": len(found),
            },
        )

    def _run_full_description(
        self,
        frame: np.ndarray,
        context: SkillContext,
        vlm: Any,
    ) -> SkillResult:
        """Run full scene description + room identification."""
        try:
            scene = vlm.describe_scene(frame)
            room_id = vlm.identify_room(frame)
        except Exception as exc:
            logger.error("[DESCRIBE_SCENE] VLM call failed: %s", exc)
            return SkillResult(
                success=False,
                error_message=f"VLM inference failed: {exc}",
                diagnosis_code="vlm_failed",
            )

        room: str = room_id.room if room_id.room != "unknown" else _fallback_room(context)

        # Record to SpatialMemory if available.
        spatial_memory = context.services.get("spatial_memory")
        if spatial_memory is not None:
            object_names: list[str] = [obj.name for obj in scene.objects]
            try:
                spatial_memory.observe(room, object_names, scene.summary)
            except Exception as exc:
                logger.warning(
                    "[DESCRIBE_SCENE] spatial_memory.observe failed: %s", exc
                )

        objects_data: list[dict[str, Any]] = [
            {
                "name": obj.name,
                "description": obj.description,
                "confidence": obj.confidence,
            }
            for obj in scene.objects
        ]

        logger.info(
            "[DESCRIBE_SCENE] room=%s confidence=%.2f objects=%d",
            room,
            room_id.confidence,
            len(scene.objects),
        )

        return SkillResult(
            success=True,
            result_data={
                "room": room,
                "summary": scene.summary,
                "objects": objects_data,
                "details": scene.details,
                "room_confidence": room_id.confidence,
                "room_reasoning": room_id.reasoning,
            },
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _fallback_room(context: SkillContext) -> str:
    """Best-effort room name from positional data when VLM returns 'unknown'.

    Uses the robot's current XY position and the SceneGraph nearest_room
    heuristic.  Returns "unknown" if position or SceneGraph is unavailable.

    Args:
        context: SkillContext with optional base and spatial_memory service.

    Returns:
        Room name string.
    """
    if context.base is None:
        return "unknown"
    try:
        from vector_os_nano.skills.navigate import _detect_current_room
        pos = context.base.get_position()
        x = float(pos[0])
        y = float(pos[1])
        sg = context.services.get("spatial_memory") if context.services else None
        return _detect_current_room(x, y, sg=sg)
    except Exception as exc:
        logger.debug("[look] _fallback_room position unavailable: %s", exc)
        return "unknown"
