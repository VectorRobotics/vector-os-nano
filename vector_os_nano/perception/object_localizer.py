# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Reusable 3D object localizer — depth+mask centroid lifted to world frame.

Routes object name queries through the existing (correct) depth→world pipeline:

    grounding-dino detect → EdgeTAM segment → grasp_point_from_rgbd → Pose3D

This is the SysNav-style masked-cloud centroid approach: detect each object
by name, segment the bounding box to get a tight mask, project the masked
depth pixels through the proven camera→world transform, and return the
world-frame centroid.  The pipeline reuses grasp_point_from_rgbd exactly —
no re-implementation of projection.

Used by the look skill to fill accurate (x, y) coordinates in the scene
graph instead of the 0.0, 0.0 defaults that placed every object at room
centre.

GUARD contract: if perception is None or lacks any required method, returns
[].  Per-object failures are silently skipped (debug-logged) so one bad
detection can never kill the sweep.  This function NEVER raises.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Methods the perception object must expose for localization to run.
_REQUIRED_METHODS = (
    "detect",
    "segment",
    "get_color_frame",
    "get_depth_frame",
    "get_intrinsics",
    "get_camera_pose",
)


def _perception_is_usable(perception: Any) -> bool:
    """Return True only if *perception* exposes every required method."""
    if perception is None:
        return False
    for m in _REQUIRED_METHODS:
        if not callable(getattr(perception, m, None)):
            return False
    return True


def capture_rgbd(perception: Any) -> tuple[Any, Any, Any, Any, Any]:
    """``(rgb, depth, intrinsics, cam_xpos, cam_xmat)`` of ONE camera instant.

    Uses the backend's atomic ``capture()`` (an RGBDFrame: pixels, depth and pose
    of one sim time, intrinsics from the actual shape) when it has one; else the
    four separate getters.
    """
    cap = getattr(perception, "capture", None)
    if callable(cap):
        f = cap()
        return f.rgb, f.depth, f.K, f.cam_pos, f.cam_mat
    cam_xpos, cam_xmat = perception.get_camera_pose()[:2]
    return (perception.get_color_frame(), perception.get_depth_frame(),
            perception.get_intrinsics(), cam_xpos, cam_xmat)


def detect_on(perception: Any, rgb: Any, query: str) -> Any:
    """Detections on *rgb* itself when the backend can, so box and depth share a frame."""
    fn = getattr(perception, "detect_in", None)
    return fn(rgb, query) if callable(fn) else perception.detect(query)


def localize_objects_3d(
    perception: Any,
    queries: list[str],
) -> list[tuple[str, float, float, float]]:
    """Localize each query string to a world-frame (x, y, z) position.

    For each query:
      1. Run the open-vocab detector (grounding-dino) on the live RGB frame.
      2. For the highest-confidence detection, segment the bbox (EdgeTAM /
         box-rect fallback) to get a binary mask.
      3. Feed depth + mask + intrinsics + camera pose through
         grasp_point_from_rgbd to obtain a world-frame Pose3D.
      4. Emit (label, float(x), float(y), float(z)).

    Results are keyed by the input QUERY string (the caller's handle), not the
    detector's returned label — grounding-dino returns labels like "a green
    bottle" for the query "green bottle", and callers map their requested name
    back to a position.  Each distinct query is localized once, using the
    highest-confidence detection that yields a valid world point.

    Args:
        perception: A Go2GraspPerception-like object with the required API.
            Pass None (or an object missing a method) to get [] with no error.
        queries: List of natural-language object name strings to detect.

    Returns:
        List of (label, x, y, z) for successfully localised objects.  May be
        shorter than *queries* (failed detections are silently dropped).
        Returns [] if perception is unusable or all detections fail.
    """
    if not _perception_is_usable(perception):
        logger.debug(
            "[OBJ-LOC] perception not usable (None or missing method) — returning []"
        )
        return []

    from vector_os_nano.perception.grasp_point import grasp_point_from_rgbd

    # Capture ONE camera instant (reused across all queries): the boxes, depth and
    # pose all come from it, so the back-projection uses the pose of its pixels.
    try:
        color, depth, intrinsics, cam_xpos, cam_xmat = capture_rgbd(perception)
    except Exception as exc:
        logger.debug("[OBJ-LOC] failed to capture frames/pose: %s", exc)
        return []

    results: dict[str, tuple[str, float, float, float]] = {}

    for query in queries:
        # Key results by the QUERY (the caller's handle), NOT the detector's
        # returned label.  grounding-dino returns strings like "a green bottle"
        # for the query "green bottle", and callers (e.g. the look skill) map
        # their requested NAME -> position.  Keying by det.label silently breaks
        # that mapping and stores objects at (0, 0) — the real-sim bug this
        # contract guards against.  Each distinct query is localized once.
        if query in results:
            continue
        try:
            detections = detect_on(perception, color, query)
        except Exception as exc:
            logger.debug("[OBJ-LOC] detect(%r) failed: %s", query, exc)
            continue

        if not detections:
            logger.debug("[OBJ-LOC] no detections for %r", query)
            continue

        # detections are sorted by confidence descending (grounding-dino contract);
        # take the highest-confidence detection that yields a valid world point.
        for det in detections:
            try:
                mask = perception.segment(color, det.bbox)
                if mask is None:
                    logger.debug("[OBJ-LOC] segment returned None for %r", query)
                    continue

                world = grasp_point_from_rgbd(
                    depth, color, mask, intrinsics, cam_xpos, cam_xmat
                )
                if world is None:
                    logger.debug(
                        "[OBJ-LOC] grasp_point_from_rgbd returned None for %r "
                        "(too few depth points in mask)",
                        query,
                    )
                    continue

                results[query] = (query, float(world.x), float(world.y), float(world.z))
                logger.debug(
                    "[OBJ-LOC] localised %r → (%.3f, %.3f, %.3f)",
                    query, world.x, world.y, world.z,
                )
                # Found a good point for this query; move to the next.
                break

            except Exception as exc:
                logger.debug(
                    "[OBJ-LOC] per-object localization failed for %r: %s", query, exc
                )
                continue

    return list(results.values())
