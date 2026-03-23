"""DetectSkill — detect objects in workspace using perception backend.

Uses VLM for 2D detection, then tracker + depth for 3D position.
Stores objects with 3D positions in the world model.

No ROS2 imports.
"""
from __future__ import annotations

import logging
import time

from vector_os_nano.core.skill import Skill, SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.core.world_model import ObjectState

logger = logging.getLogger(__name__)


@skill(
    aliases=["find", "search", "检测", "识别", "找一下"],
    direct=False,
    auto_steps=["scan", "detect"],
)
class DetectSkill:
    """Detect objects using VLM + depth for 3D positions.

    Pipeline (mirrors vector_ws track_3d.py):
    1. VLM detect(query) → 2D bounding boxes
    2. tracker.init_track(image, bboxes) → masks
    3. RGBD + mask → pointcloud → 3D centroid in camera frame
    4. calibration → base frame position
    5. Store in world model with 3D position
    """

    name: str = "detect"
    description: str = "Detect objects in the workspace using VLM"
    parameters: dict = {
        "query": {
            "type": "string",
            "required": True,
            "description": "What to detect (e.g. 'red cup', 'all objects')",
        }
    }
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {}

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        if context.perception is None:
            logger.warning("[DETECT] No perception backend available")
            return SkillResult(
                success=False,
                error_message="No perception backend available",
            )

        query: str = params.get("query", "all objects")
        logger.info("[DETECT] Running detection for query: %r", query)

        # Step 1: VLM 2D detection
        try:
            detections = context.perception.detect(query)
        except Exception as exc:
            logger.error("[DETECT] Perception error: %s", exc)
            return SkillResult(
                success=False,
                error_message=f"Perception error: {exc}",
            )

        if not detections:
            logger.info("[DETECT] No objects detected")
            return SkillResult(
                success=True,
                result_data={"objects": [], "count": 0},
            )

        logger.info("[DETECT] VLM found %d object(s), getting 3D positions...", len(detections))

        # Step 2: Track to get 3D positions (tracker + depth → pointcloud → centroid)
        tracked_objects = []
        try:
            tracked_objects = context.perception.track(detections)
        except Exception as exc:
            logger.warning("[DETECT] Tracking failed, storing 2D-only: %s", exc)

        # Step 3: Store in world model
        now = time.time()
        object_summaries: list[dict] = []

        for idx, det in enumerate(detections):
            # If VLM returned the query as label (e.g., "all objects" for every detection),
            # give each object a unique name like "object_0", "object_1", etc.
            label = det.label
            if label.lower() in ("all objects", "all", "objects", "everything"):
                label = f"object_{idx}"
            safe_label = label.replace(" ", "_").lower()
            obj_id = f"{safe_label}_{idx}"

            # Try to get 3D position from tracked object
            x, y, z = 0.0, 0.0, 0.0
            has_3d = False

            if idx < len(tracked_objects) and tracked_objects[idx].pose is not None:
                pose = tracked_objects[idx].pose
                cam_pos = [pose.x, pose.y, pose.z]

                # Apply calibration if available
                if context.calibration is not None:
                    try:
                        import numpy as np
                        base_pos = context.calibration.camera_to_base(
                            np.array(cam_pos, dtype=float)
                        )
                        x, y, z = float(base_pos[0]), float(base_pos[1]), float(base_pos[2])
                        has_3d = True
                        logger.info(
                            "[DETECT] %s: camera(%.3f,%.3f,%.3f) -> base(%.1f,%.1f,%.1f)cm",
                            det.label, cam_pos[0], cam_pos[1], cam_pos[2],
                            x * 100, y * 100, z * 100,
                        )
                    except Exception as exc:
                        logger.warning("[DETECT] Calibration failed for %s: %s", det.label, exc)
                        x, y, z = cam_pos[0], cam_pos[1], cam_pos[2]
                        has_3d = True
                else:
                    # No calibration — store camera frame coords
                    x, y, z = cam_pos[0], cam_pos[1], cam_pos[2]
                    has_3d = True
                    logger.info("[DETECT] %s: camera(%.3f,%.3f,%.3f) (no calibration)", det.label, x, y, z)

            obj = ObjectState(
                object_id=obj_id,
                label=label,
                x=x,
                y=y,
                z=z,
                confidence=det.confidence,
                state="on_table",
                last_seen=now,
            )
            context.world_model.add_object(obj)

            summary = {
                "object_id": obj_id,
                "label": det.label,
                "confidence": round(det.confidence, 4),
                "has_3d": has_3d,
            }
            if has_3d:
                summary["position_cm"] = [round(x * 100, 1), round(y * 100, 1), round(z * 100, 1)]
            object_summaries.append(summary)

        logger.info("[DETECT] Detected %d object(s), %d with 3D positions",
                    len(detections), sum(1 for s in object_summaries if s.get("has_3d")))

        return SkillResult(
            success=True,
            result_data={
                "objects": object_summaries,
                "count": len(object_summaries),
            },
        )
