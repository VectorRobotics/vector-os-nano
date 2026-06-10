# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""DetectSkill — detect objects in workspace using perception backend.

Uses VLM for 2D detection, then tracker + depth for 3D position.
Stores objects with 3D positions in the world model.

No ROS2 imports.
"""
from __future__ import annotations

import logging
import time

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.core.world_model import ObjectState

logger = logging.getLogger(__name__)

# Generic quantifier terms that the LLM normalises to when the user asks for
# "everything" rather than a specific named target.  A detect call with one of
# these queries is asking "tell me what's there at all" — an empty result is a
# valid answer (empty scene).  A detect call for a SPECIFIC target (e.g. "apple",
# "banana") that finds nothing means the target is absent; that is a failure, not a
# success.  Lifted to a module constant so the label-rename logic below can reuse it
# without duplicating the set.
_GENERIC_QUERIES: frozenset[str] = frozenset({"all objects", "all", "objects", "everything"})


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
    description: str = "Detect objects in the workspace using VLM. The query is a natural-language noun/phrase in ANY language (e.g. 'banana' / '香蕉' / 'red cup' / '红色杯子'), or 'all objects' to detect everything."
    # Success predicate this skill is verified against (single-source for the
    # planner). Backlog #3 (Rule 4): verify THIS step's own alias-aware output
    # via the kernel-injected step_output() — never a separate query-less
    # oracle call, which FALSE-PASSES when the requested target is absent but
    # the scene holds other objects.
    verify_hint: str = "len(step_output('objects')) > 0"
    parameters: dict = {
        "query": {
            "type": "string",
            "required": True,
            "description": "Target to detect, as a natural-language noun/phrase in ANY language (e.g. 'banana' / '香蕉' / 'red cup' / '红色杯子'), or 'all objects' to detect everything. Copy the object named in the task here.",
        }
    }
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {}
    failure_modes: list[str] = ["no_perception", "no_detections", "track_failed", "calibration_error"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        if context.perception is None:
            logger.warning("[DETECT] No perception backend available")
            return SkillResult(
                success=False,
                error_message="No perception backend available",
                result_data={"diagnosis": "no_perception"},
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
                result_data={"diagnosis": "no_perception", "error_detail": str(exc)},
            )

        if not detections:
            # Decide success based on query intent (R2-7 honest-success fix):
            # - Generic "all objects" / "everything" queries: an empty scene is a
            #   valid answer — the robot looked and found nothing; success=True so a
            #   downstream foreach can no-op cleanly over the empty list.
            # - Specific target queries (e.g. "apple", "苹果"): no detections means
            #   the target is absent; the step genuinely failed — success=False so
            #   the verify moat catches the miss instead of false-passing.
            q = (params.get("query", "") or "").strip().lower()
            generic = q in _GENERIC_QUERIES
            logger.info(
                "[DETECT] No objects detected (query=%r, generic=%s -> success=%s)",
                query, generic, generic,
            )
            result_data: dict = {
                "objects": [],
                "count": 0,
                "diagnosis": "no_detections",
                "query": query,
            }
            if not generic:
                return SkillResult(
                    success=False,
                    error_message=f"No detections found for {query!r} (target not in scene?)",
                    result_data=result_data,
                )
            return SkillResult(success=True, result_data=result_data)

        logger.info("[DETECT] VLM found %d object(s), getting 3D positions...", len(detections))

        # Step 2: Track to get 3D positions (tracker + depth → pointcloud → centroid)
        tracked_objects = []
        track_warning: str | None = None
        try:
            tracked_objects = context.perception.track(detections)
        except Exception as exc:
            logger.warning("[DETECT] Tracking failed, storing 2D-only: %s", exc)
            track_warning = str(exc)

        # Step 3: Store in world model
        now = time.time()
        object_summaries: list[dict] = []
        merged_count = 0

        for idx, det in enumerate(detections):
            # If VLM returned the query as label (e.g., "all objects" for every detection),
            # give each object a unique name like "object_0", "object_1", etc.
            label = det.label
            if label.lower() in _GENERIC_QUERIES:  # reuse the module constant
                label = f"object_{idx}"
            safe_label = label.replace(" ", "_").lower()

            # Merge with existing world model objects by label
            existing = context.world_model.get_objects_by_label(label)
            if existing:
                obj_id = existing[0].object_id  # Reuse existing ID
                merged_count += 1
            else:
                # Generate unique ID avoiding collisions
                existing_ids = {o.object_id for o in context.world_model.get_objects()}
                counter = 0
                obj_id = f"{safe_label}_{counter}"
                while obj_id in existing_ids:
                    counter += 1
                    obj_id = f"{safe_label}_{counter}"

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
                # Expose BOTH "name" and "label" (same value): a downstream foreach
                # binds a per-item field (e.g. "${item.name}") into a skill param, and
                # the decompose example / the playground detect producer use "name"
                # while this skill historically used "label". Carrying both keeps the
                # producer->foreach->skill contract robust to whichever the planner emits.
                "name": det.label,
                "label": det.label,
                "confidence": round(det.confidence, 4),
                "has_3d": has_3d,
            }
            if has_3d:
                summary["position_cm"] = [round(x * 100, 1), round(y * 100, 1), round(z * 100, 1)]
            object_summaries.append(summary)

        logger.info("[DETECT] Detected %d object(s), %d with 3D positions, %d merged",
                    len(detections), sum(1 for s in object_summaries if s.get("has_3d")), merged_count)

        result_data: dict = {
            "objects": object_summaries,
            "count": len(object_summaries),
            "diagnosis": "ok",
            "merged_count": merged_count,
        }
        if track_warning is not None:
            result_data["track_warning"] = track_warning

        return SkillResult(
            success=True,
            result_data=result_data,
        )
