# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""CameraPerception — RGB-D + detector + segmenter over one body camera.

The perception backend the grasp and look/explore consume, on any body with a camera
(``interfaces.CameraProtocol``). It uses ONLY what a real robot has, and only through
ONE camera instant:

  - :meth:`capture` -> a CameraFrame: RGB, metric depth with the robot's own pixels
    zeroed out (its self mask), intrinsics from the camera fovy and the frame's
    ACTUAL shape, and the exact camera->world pose, all of one sim time;
  - :meth:`detect_in` runs the selected detector on a GIVEN
    frame, so a box always belongs to the pose it is projected with;
  - :meth:`segment` masks a box with EdgeTAM (a segmenter that cannot load or run
    raises), or the box rectangle when EdgeTAM finds nothing in the box;
    ``last_segmenter`` says which ("edgetam"/"box_rect"/"hsv").

It exposes NO ground-truth pose — the 3D point comes from depth + mask downstream
(grasp_point.py). Detector + EdgeTAM load LAZILY, so importing needs no torch.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from vector_os_nano.interfaces.perception import Detection, Detector, Segmenter
from vector_os_nano.interfaces.devices import CameraFrame, CameraProtocol
from vector_os_nano.perception.models import build
from vector_os_nano.schemas.user import DETECTORS, SEGMENTERS

logger = logging.getLogger(__name__)

_warned_box_rect = False  # the box-rect fallback is announced once per process


class CameraPerception:
    """RGB-D + selected detector and segmenter over one camera.

    Args:
        camera: the body's camera (renders at its manifest size).
        vlm: Optional pre-built detector (else the shared grounding-dino).
        tracker: Optional pre-built segmenter (else lazily constructed).
        detector_name, segmenter_name: rows of the public user-config model tables.
    """

    def __init__(self, camera: CameraProtocol, *, vlm: Detector | None = None, tracker: Segmenter | None = None,
                 cameras: dict[str, CameraProtocol] | None = None,
                 detector_name: str = 'grounding-dino', segmenter_name: str = 'edgetam') -> None:
        for table, name in ((DETECTORS, detector_name), (SEGMENTERS, segmenter_name)):
            if name not in table:
                raise ValueError(f'unknown perception model {name!r}; valid: {list(table)}')
        self.detector_name, self.segmenter_name = detector_name, segmenter_name
        self._camera = camera
        self._cameras = dict(cameras or {})
        self._vlm = vlm
        self._tracker = tracker
        # One detector/segmenter call at a time: the EdgeTAM session and the lazy
        # model loads are not thread-safe (explore auto-observe vs skill threads).
        self._lock = threading.Lock()
        self.last_segmenter: str | None = None

    def capture(self, *, camera: str | None = None) -> CameraFrame:
        """ONE camera instant: rgb, self-filtered depth, K, pose, of one sim time.

        The robot's own pixels are zeroed in depth (D30 self-occlusion: the arm
        occludes the table and must never be picked as the target; depth 0 is
        invalid downstream).
        """
        if camera is not None and camera not in self._cameras:
            raise ValueError(f'unknown camera {camera!r}; available: {sorted(self._cameras)}')
        f = (self._camera if camera is None else self._cameras[camera]).capture()
        if f.depth is not None and f.self_mask is not None:
            f.depth[f.self_mask] = 0.0
        return f

    # --- detector (grounding-dino, lazy) + EdgeTAM (lazy) --------------------

    def _ensure_detector(self) -> None:
        if self._vlm is None:
            self._vlm = build(DETECTORS, self.detector_name)

    def _ensure_tracker(self) -> None:
        if self._tracker is None:
            self._tracker = build(SEGMENTERS, self.segmenter_name)

    def detect_in(self, rgb: np.ndarray, query: str) -> list[Detection]:
        """Open-vocab detections of *query* on the GIVEN frame (score-sorted boxes).

        GroundingDINO uses its shared singleton. Each detector reads ONLY pixels and the
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
            self.last_segmenter = self.segmenter_name
            return mask
        if not _warned_box_rect:
            _warned_box_rect = True
            logger.warning("[GO2-PERCEPT] %s returned an empty mask — box-rect mask (coarse)", self.segmenter_name)
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
