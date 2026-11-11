# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Object localizer — named objects on ONE RGB-D snapshot, lifted to world points.

    snapshot (RGBDFrame) -> grounding-dino on THAT frame -> segment the box ->
    nearest depth cluster (grasp_point.object_points) -> plausibility gates ->
    world point; results of different queries on one spot are deduplicated.

Every query is detected on the same captured frame, so its box, depth and camera
pose are of one instant (a box from a later re-render is never projected through
an earlier pose). A detection becomes a position only if it is plausibly a
graspable object THERE: a confident box of sane shape, at a sane range and height,
the size of an object (not a wall or a sofa), standing in front of what surrounds
it, and of the colour the query names. Anything else is dropped, never stored.

GUARD contract: an unusable perception (None or missing a method) gives [];
per-query failures are skipped with a WARNING naming the cause (a missing model, an
ungroundable name). This function NEVER raises.
"""
from __future__ import annotations

import logging
from typing import Any, NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

_REQUIRED_METHODS = ("capture", "detect_in", "segment")

# --- plausibility gates (a graspable object seen by the head camera) ----------
_MIN_SCORE = {"default": 0.30}          # detector score floor, per vocabulary noun
_MAX_EXTENT_M = {"default": 0.35}       # object size (p5-p95 span of its points), per noun
_DEPTH_RANGE_M = (0.2, 5.0)             # median depth of the object's points
_Z_BAND_M = (-0.05, 1.8)                # world height of the object's points
_ASPECT = (0.15, 6.7)                   # box width / height
_BEHIND_M, _MIN_BEHIND_FRAC = 0.05, 0.2  # >=20% of the box's surround lies >=5 cm behind it
_DEDUPE_M = 0.06                        # two queries on one spot: the higher score wins


class Localized(NamedTuple):
    label: str           # the caller's query (its handle), not the detector's label
    x: float
    y: float
    z: float
    score: float
    segmenter: str | None


def _perception_is_usable(perception: Any) -> bool:
    return perception is not None and all(
        callable(getattr(perception, m, None)) for m in _REQUIRED_METHODS
    )


def capture_rgbd(perception: Any) -> tuple[Any, Any, Any, Any, Any]:
    """``(rgb, depth, intrinsics, cam_xpos, cam_xmat)`` of ONE camera instant."""
    f = perception.capture()
    return f.rgb, f.depth, f.K, f.cam_pos, f.cam_mat


def detect_on(perception: Any, rgb: Any, query: str) -> Any:
    """Detections on *rgb* itself, so box and depth share a frame."""
    return perception.detect_in(rgb, query)


def _behind_fraction(depth: np.ndarray, bbox: Any, d: float) -> float | None:
    """Fraction of the valid depth in a ring around *bbox* lying >= _BEHIND_M behind *d*."""
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    px, py = max(4.0, 0.25 * (x2 - x1)), max(4.0, 0.25 * (y2 - y1))
    X1, Y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    X2, Y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    ring = np.zeros((h, w), dtype=bool)
    ring[Y1:Y2, X1:X2] = True
    # the box itself, 1 px generous (its edge pixels are the object's own)
    ring[max(0, int(y1) - 1):int(y2) + 2, max(0, int(x1) - 1):int(x2) + 2] = False
    vals = depth[ring & (depth > 0)]
    return None if vals.size < 10 else float(np.mean(vals > d + _BEHIND_M))


def implausible(
    det: Any, mask: np.ndarray, cam_pts: np.ndarray, world_pts: np.ndarray,
    depth: np.ndarray, rgb: np.ndarray, *, noun: str | None, colour: str | None,
) -> str | None:
    """Why *det* is not a graspable object where it projects, or None if it is."""
    from vector_os_nano.perception.front_object import colour_matches

    floor = _MIN_SCORE.get(noun or "", _MIN_SCORE["default"])
    if float(det.confidence) < floor:
        return f"score {float(det.confidence):.2f} < {floor:.2f}"
    x1, y1, x2, y2 = (float(v) for v in det.bbox)
    if x2 <= x1 or y2 <= y1 or not _ASPECT[0] <= (x2 - x1) / (y2 - y1) <= _ASPECT[1]:
        return f"box {det.bbox} is a sliver"
    d = float(np.median(cam_pts[:, 2]))
    if not _DEPTH_RANGE_M[0] <= d <= _DEPTH_RANGE_M[1]:
        return f"depth {d:.2f} m outside {_DEPTH_RANGE_M}"
    z = float(np.median(world_pts[:, 2]))
    if not _Z_BAND_M[0] <= z <= _Z_BAND_M[1]:
        return f"height {z:.2f} m outside {_Z_BAND_M}"
    span = float(np.max(np.percentile(world_pts, 95, axis=0) - np.percentile(world_pts, 5, axis=0)))
    cap = _MAX_EXTENT_M.get(noun or "", _MAX_EXTENT_M["default"])
    if span > cap:
        return f"extent {span:.2f} m > {cap:.2f} m (not an object of this size)"
    behind = _behind_fraction(depth, det.bbox, d)
    if behind is not None and behind < _MIN_BEHIND_FRAC:
        return f"flush with its surroundings ({behind:.0%} behind): a surface, not an object"
    if colour and colour_matches(rgb, mask, colour) is False:
        return f"not {colour}"
    return None


def localize_objects_3d(
    perception: Any,
    queries: list[str],
    frame: Any = None,
) -> list[Localized]:
    """Localize each query on ONE snapshot (*frame*, else a fresh ``capture()``).

    Per query: the highest-scoring detection whose mask yields a plausible object
    point (see :func:`implausible`). Results are keyed by the QUERY (the caller's
    handle: grounding-dino labels "a green bottle" for "green bottle"); two
    queries that land on one spot keep only the higher-scoring one.
    """
    if not _perception_is_usable(perception):
        logger.debug("[OBJ-LOC] perception not usable (None or missing a method)")
        return []

    from vector_os_nano.perception.grasp_point import grasp_point_from_points, object_points
    from vector_os_nano.perception.vocab import vocab

    try:
        f = frame if frame is not None else perception.capture()
        rgb, depth, K, cam_pos, cam_mat = f.rgb, f.depth, f.K, f.cam_pos, f.cam_mat
    except Exception as exc:  # noqa: BLE001
        logger.warning("[OBJ-LOC] capture failed: %s: %s", type(exc).__name__, exc)
        return []

    found: list[Localized] = []
    for query in dict.fromkeys(queries):  # each distinct query once, in order
        try:
            detections = perception.detect_in(rgb, query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[OBJ-LOC] detect(%r) failed: %s: %s", query, type(exc).__name__, exc)
            continue
        noun, colour = vocab().noun(query), vocab().colour(query)
        for det in detections or []:
            try:
                mask = perception.segment(rgb, det.bbox)
                pts = None if mask is None else object_points(depth, rgb, mask, K, cam_pos, cam_mat)
                if pts is None:
                    continue
                why = implausible(det, mask, *pts, depth, rgb, noun=noun, colour=colour)
                if why:
                    logger.info("[OBJ-LOC] %r box %s rejected: %s", query,
                                [round(float(v)) for v in det.bbox], why)
                    continue
                gp = grasp_point_from_points(*pts, cam_pos, cam_mat)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[OBJ-LOC] %r: localization failed: %s: %s", query, type(exc).__name__, exc)
                continue
            found.append(Localized(query, gp.x, gp.y, gp.z, float(det.confidence),
                                   getattr(perception, "last_segmenter", None)))
            break

    kept: list[Localized] = []
    for r in sorted(found, key=lambda r: -r.score):
        if all(np.hypot(r.x - k.x, r.y - k.y) > _DEDUPE_M for k in kept):
            kept.append(r)
        else:
            logger.info("[OBJ-LOC] %r dropped: the same spot as a better-scoring query", r.label)
    return [r for r in found if r in kept]
