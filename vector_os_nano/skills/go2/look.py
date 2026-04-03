"""VLM-powered look skills for the Go2 quadruped robot.

Two skills:
- LookSkill: capture frame, call describe_scene + identify_room, optionally
  record to SpatialMemory.
- DescribeSceneSkill: detailed VLM scene description with optional query
  (delegates to find_objects when a query is provided).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.skills.navigate import _detect_current_room

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
        """Capture a camera frame, run VLM scene description and room ID.

        VLM provides: room identification (identify_room) and scene summary
        (describe_scene).  GroundingDINO detector (if available in services)
        provides per-object world coordinates via depth projection.

        Args:
            params: Unused (LookSkill has no parameters).
            context: SkillContext with base and vlm service attached.
                     Optional: context.services["detector"] callable with
                     signature (rgb, depth, RobotPose) -> list[Detection].

        Returns:
            SkillResult with result_data containing room, summary, objects,
            details, and room_confidence.  When detector is active, each
            object entry also has world_x, world_y, depth_m.
        """
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

        # Capture RGB frame from robot camera (sim: MuJoCo, real: D435).
        try:
            frame: np.ndarray = context.base.get_camera_frame()
        except Exception as exc:
            logger.error("[LOOK] get_camera_frame failed: %s", exc)
            return SkillResult(
                success=False,
                error_message=f"Camera capture failed: {exc}",
                diagnosis_code="camera_failed",
            )

        # Run VLM calls — scene description (summary + object names) and room ID.
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

        # Prefer VLM room over positional heuristic; fall back if "unknown".
        room: str = room_id.room if room_id.room != "unknown" else _fallback_room(context)

        # Get robot pose for viewpoint recording and depth projection.
        pos = context.base.get_position()
        heading = context.base.get_heading()

        # ------------------------------------------------------------------
        # Object detection with world positioning (GroundingDINO + depth).
        # Detector provides per-object (x, y, z) in world frame.
        # VLM still owns room_id and scene summary.
        # ------------------------------------------------------------------
        detected_objects: list[Any] = []
        detector = context.services.get("detector")
        if detector is not None and hasattr(context.base, "get_depth_frame"):
            try:
                from vector_os_nano.perception.object_detector import RobotPose
                depth_frame: np.ndarray = context.base.get_depth_frame()
                # Get camera world pose for exact projection (sim only)
                cam_xpos, cam_xmat = None, None
                if hasattr(context.base, "get_camera_pose"):
                    try:
                        cam_xpos, cam_xmat = context.base.get_camera_pose()
                    except Exception:
                        pass
                pose = RobotPose(
                    x=float(pos[0]),
                    y=float(pos[1]),
                    z=float(pos[2]),
                    heading=float(heading),
                    cam_xpos=cam_xpos,
                    cam_xmat=cam_xmat,
                )
                detected_objects = detector(frame, depth_frame, pose)
            except Exception as exc:
                logger.warning("[LOOK] Detector failed: %s", exc)
                detected_objects = []

        # ------------------------------------------------------------------
        # Build objects_data: prefer detector results (have world coords),
        # fall back to VLM-only names when detector unavailable or empty.
        # ------------------------------------------------------------------
        if detected_objects:
            objects_data: list[dict[str, Any]] = [
                {
                    "name": det.label,
                    "description": "",
                    "confidence": det.confidence,
                    "world_x": det.world_x,
                    "world_y": det.world_y,
                    "world_z": det.world_z,
                    "depth_m": det.depth_m,
                }
                for det in detected_objects
            ]
        else:
            objects_data = [
                {
                    "name": obj.name,
                    "description": obj.description,
                    "confidence": obj.confidence,
                }
                for obj in scene.objects
            ]

        # ------------------------------------------------------------------
        # Record to spatial memory / scene graph.
        # When detector found objects: record each with its world coords.
        # When VLM-only: use standard observe_with_viewpoint.
        # ------------------------------------------------------------------
        spatial_memory = context.services.get("spatial_memory")
        if spatial_memory is not None:
            try:
                if detected_objects and hasattr(spatial_memory, "merge_object"):
                    # SceneGraph path: viewpoint first, then per-object world coords.
                    vp_id: str = ""
                    if hasattr(spatial_memory, "observe_with_viewpoint"):
                        vp = spatial_memory.observe_with_viewpoint(
                            room, float(pos[0]), float(pos[1]),
                            float(heading),
                            [det.label for det in detected_objects],
                            scene.summary,
                        )
                        vp_id = vp.viewpoint_id if vp is not None else ""
                        if not vp_id and hasattr(spatial_memory, "_viewpoints"):
                            # Nearest viewpoint was reused — find it
                            for existing_vp in spatial_memory._viewpoints.values():
                                if existing_vp.room_id == room:
                                    vp_id = existing_vp.viewpoint_id
                                    break

                    # Merge each detected object with its world coordinates.
                    for det in detected_objects:
                        spatial_memory.merge_object(
                            category=det.label,
                            room_id=room,
                            viewpoint_id=vp_id,
                            confidence=det.confidence,
                            x=det.world_x,
                            y=det.world_y,
                        )
                elif hasattr(spatial_memory, "observe_with_viewpoint"):
                    object_names: list[str] = [obj.name for obj in scene.objects]
                    spatial_memory.observe_with_viewpoint(
                        room, float(pos[0]), float(pos[1]),
                        float(heading), object_names, scene.summary,
                    )
                else:
                    object_names = [obj.name for obj in scene.objects]
                    spatial_memory.visit(room, float(pos[0]), float(pos[1]))
                    spatial_memory.observe(room, object_names, scene.summary)
            except Exception as exc:
                logger.warning("[LOOK] spatial_memory update failed: %s", exc)

        logger.info(
            "[LOOK] room=%s confidence=%.2f summary=%s objects=%d "
            "(detector=%s)",
            room,
            room_id.confidence,
            scene.summary,
            len(objects_data),
            "yes" if detected_objects else "no",
        )

        return SkillResult(
            success=True,
            result_data={
                "room": room,
                "summary": scene.summary,
                "objects": objects_data,
                "details": scene.details,
                "room_confidence": room_id.confidence,
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

    Uses the robot's current XY position and the room-boundary heuristic from
    the navigate module.  Returns "unknown" if position is unavailable.

    Args:
        context: SkillContext with optional base.

    Returns:
        Room name string.
    """
    if context.base is None:
        return "unknown"
    try:
        pos = context.base.get_position()
        x = float(pos[0])
        y = float(pos[1])
        return _detect_current_room(x, y)
    except Exception as exc:
        logger.debug("[look] _fallback_room position unavailable: %s", exc)
        return "unknown"
