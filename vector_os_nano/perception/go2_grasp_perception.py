# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Go2GraspPerception — REAL RGB-D + detector + segmenter on the Go2 d435 camera.

The perception backend the go2+piper grasp and look/explore consume. It uses ONLY
what a real robot has, and only through ONE camera instant:

  - :meth:`capture` -> an RGBDFrame (MuJoCoGo2.get_rgbd_frame): RGB, metric depth
    with the robot's own arm zeroed out, intrinsics from the MJCF fovy and the
    frame's ACTUAL shape, and the exact camera->world pose, all of one sim time;
  - :meth:`detect_in` runs grounding-dino-tiny (the shared singleton) on a GIVEN
    frame, so a box always belongs to the pose it is projected with;
  - :meth:`segment` masks a box with EdgeTAM (a segmenter that cannot load or run
    raises), or the box rectangle when EdgeTAM finds nothing in the box;
    ``last_segmenter`` says which ("edgetam"/"box_rect"/"hsv").

It exposes NO ground-truth pose — the 3D point comes from depth + mask downstream
(grasp_point.py). Detector + EdgeTAM load LAZILY, so importing needs no torch.
"""
from __future__ import annotations

import dataclasses
import logging
import threading
from typing import Any

import numpy as np

from vector_os_nano.core.types import Detection

logger = logging.getLogger(__name__)

_warned_box_rect = False  # the box-rect fallback is announced once per process


class Go2GraspPerception:
    """Real RGB-D + detector + EdgeTAM perception over a MuJoCoGo2 base.

    Args:
        base: A connected MuJoCoGo2 (``get_rgbd_frame``; ``self_mask`` with an arm).
        vlm: Optional pre-built detector (else the shared grounding-dino).
        tracker: Optional pre-built EdgeTAMTracker (else lazily constructed).
        width/height: Render resolution (the embodiment manifest's camera size).
    """

    def __init__(
        self,
        base: Any,
        *,
        vlm: Any = None,
        tracker: Any = None,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self._base = base
        self._vlm = vlm
        self._tracker = tracker
        self._w = int(width)
        self._h = int(height)
        # One detector/segmenter call at a time: the EdgeTAM session and the lazy
        # model loads are not thread-safe (explore auto-observe vs skill threads).
        self._lock = threading.Lock()
        self.last_segmenter: str | None = None

    @property
    def has_self_mask(self) -> bool:
        return callable(getattr(self._base, "self_mask", None))

    def capture(self) -> Any:
        """ONE camera instant as an RGBDFrame: rgb, self-filtered depth, K, pose.

        Every field shares the frame's ``sim_time`` (a single render of a
        step-boundary copy); ``K`` is derived from the frame's actual shape. The
        robot's own arm pixels are zeroed in depth (D30 self-occlusion: the arm
        occludes the table and must never be picked as the target; depth 0 is
        invalid downstream).
        """
        kinds = ("rgb", "depth", "seg") if self.has_self_mask else ("rgb", "depth")
        f = self._base.get_rgbd_frame(self._w, self._h, kinds)
        depth = f.depth
        if self.has_self_mask and f.seg is not None:
            depth = depth.copy()
            depth[self._base.self_mask(f.seg)] = 0.0
        return dataclasses.replace(f, depth=depth)

    # --- detector (grounding-dino, lazy) + EdgeTAM (lazy) --------------------

    def _ensure_detector(self) -> None:
        if self._vlm is None:
            from vector_os_nano.perception.grounding_dino import get_shared_detector
            logger.info("[GO2-PERCEPT] using shared grounding-dino detector (lazy)")
            self._vlm = get_shared_detector()

    def _ensure_tracker(self) -> None:
        if self._tracker is None:
            from vector_os_nano.perception.tracker import EdgeTAMTracker
            logger.info("[GO2-PERCEPT] loading EdgeTAM tracker (lazy)")
            self._tracker = EdgeTAMTracker()

    def detect_in(self, rgb: np.ndarray, query: str) -> list[Detection]:
        """Open-vocab detections of *query* on the GIVEN frame (score-sorted boxes).

        Routed through the SHARED grounding-dino-tiny detector (the same instance
        the registered ``detect`` capability uses). It reads ONLY the pixels + the
        text query — never a ground-truth pose.
        """
        with self._lock:
            self._ensure_detector()
            return self._vlm.detect(rgb, query)

    def segment(self, image: np.ndarray, bbox: Any) -> np.ndarray | None:
        """Binary mask for a single bbox on *image*, or None.

        EdgeTAM for a tight mask. A segmenter that cannot load or run is an error
        the caller sees (e.g. ``WeightsUnavailable`` naming the pre-fetch command),
        never a silent substitute. When EdgeTAM runs but finds nothing in the box,
        the box-interior rectangle (grasp_point's nearest-depth-cluster gate then
        trims the background it covers, D18); ``last_segmenter`` records which one
        made the mask, and the first box-rect mask of the process is a WARNING:
        a coarse mask drags the point toward the support (tricky-bugs Case 6).
        """
        global _warned_box_rect
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
        self.last_segmenter = None
        with self._lock:
            self._ensure_tracker()
            results = self._tracker.init_track(image, bboxes=[(x1, y1, x2, y2)])
        mask = results[0].get("mask") if results else None
        if mask is not None and int(np.count_nonzero(mask)) > 0:
            self.last_segmenter = "edgetam"
            return mask
        if not _warned_box_rect:
            _warned_box_rect = True
            logger.warning("[GO2-PERCEPT] EdgeTAM returned an empty mask — box-rect mask (coarse)")
        self.last_segmenter = "box_rect"
        return self._box_rect_mask(image, (x1, y1, x2, y2))

    @staticmethod
    def _box_rect_mask(
        image: np.ndarray, box: tuple[int, int, int, int]
    ) -> np.ndarray | None:
        """A binary rect mask filling the (clamped) bbox interior, or None if empty."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        x1, x2 = sorted((max(0, min(x1, w)), max(0, min(x2, w))))
        y1, y2 = sorted((max(0, min(y1, h)), max(0, min(y2, h))))
        if x2 <= x1 or y2 <= y1:
            return None
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        return mask

    def front_object_mask(
        self,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        *,
        color: str | None = None,
    ) -> np.ndarray | None:
        """Mask of the salient object in front (deictic '前面的东西'), or None.

        No detector/segmenter — the salient blob of the frame's RGB + depth; with
        *color*, the salient blob of THAT colour (None if none — FAIL LOUD).
        """
        from vector_os_nano.perception.front_object import front_object_mask
        self.last_segmenter = "hsv"
        return front_object_mask(rgb, depth, color=color)

    # --- lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        """Release the EdgeTAM inference state (frees GPU between runs)."""
        if self._tracker is not None:
            try:
                self._tracker.stop()
            except Exception as exc:  # noqa: BLE001
                logger.debug("[GO2-PERCEPT] tracker stop failed: %s", exc)
