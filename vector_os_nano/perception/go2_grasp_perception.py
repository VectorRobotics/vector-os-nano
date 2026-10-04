# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Go2GraspPerception — REAL RGB-D + detector + segmenter on the Go2 d435 camera.

The perception backend PerceptionGraspSkill consumes on the go2+piper sim. Unlike
the legacy MuJoCoPerception (which returns ground-truth poses + a zero depth
stub), this backend uses ONLY what a real robot has:

  - ONE head-camera instant (MuJoCoGo2.get_rgbd_frame -> RGBDFrame): RGB, metric
    depth (float32 metres), the self-segmentation and the exact camera->world
    pose (cam_xpos/cam_xmat), all rendered from one step-boundary copy of the sim;
  - intrinsics from the MJCF camera fovy and the frame's ACTUAL shape;
  - grounding-dino-tiny (a learned open-vocab DETECTOR, the shared singleton) for
    NAMED-object detection, EdgeTAM for segmentation with a box-rect fallback.

It exposes NO get_object_positions / ground-truth pose — the 3D grasp point is
derived from depth + mask downstream (grasp_point.py). Detector + EdgeTAM are
loaded LAZILY on first use so importing this module needs no GPU/torch.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from vector_os_nano.core.types import Detection
from vector_os_nano.perception.depth_projection import mujoco_intrinsics

logger = logging.getLogger(__name__)


class Go2GraspPerception:
    """Real RGB-D + VLM + EdgeTAM perception over a MuJoCoGo2 base.

    Args:
        base: A connected MuJoCoGo2 (provides get_camera_frame/get_depth_frame/
            get_camera_pose). Real depth is in METRES.
        vlm: Optional pre-built VLMDetector (else lazily constructed).
        tracker: Optional pre-built EdgeTAMTracker (else lazily constructed).
        width/height: Render resolution; intrinsics derive from fovy + this.
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

    # --- frames / geometry ---------------------------------------------------

    def capture(self) -> Any:
        """ONE camera instant as an RGBDFrame: rgb, self-filtered depth, K, pose.

        Every field shares the frame's ``sim_time`` (a single render of a
        step-boundary copy); ``K`` is derived from the frame's actual shape. The
        robot's own arm pixels are zeroed in depth (see :meth:`get_depth_frame`).
        """
        import dataclasses

        has_mask = callable(getattr(self._base, "self_mask", None))
        kinds = ("rgb", "depth", "seg") if has_mask else ("rgb", "depth")
        f = self._base.get_rgbd_frame(self._w, self._h, kinds)
        depth = f.depth
        if has_mask and f.seg is not None:
            depth = depth.copy()
            depth[self._base.self_mask(f.seg)] = 0.0
        return dataclasses.replace(f, depth=depth)

    def get_color_frame(self) -> np.ndarray:
        return self._base.get_camera_frame(self._w, self._h)

    def get_depth_frame(self) -> np.ndarray:
        """(H, W) float32 depth in METRES (use depth_scale=1.0 downstream).

        SELF-FILTER: zero out pixels showing the robot's OWN Piper arm (segmentation
        self-mask) so the arm — which occludes the table in the head camera — is never
        picked as the grasp target (D30 self-occlusion). depth==0 is already treated as
        invalid by front_object_mask + the pointcloud, so this cleanly removes the arm.
        """
        depth = self._base.get_depth_frame(self._w, self._h)
        get_self = getattr(self._base, "get_self_mask", None)
        if get_self is not None:
            try:
                self_mask = get_self(self._w, self._h)
                if self_mask is not None and self_mask.shape == depth.shape:
                    depth = depth.copy()
                    depth[self_mask] = 0.0
            except Exception as exc:  # noqa: BLE001 — self-filter is best-effort
                logger.debug("[GO2-PERCEPT] self-mask failed: %s", exc)
        return depth

    def get_intrinsics(self):
        return mujoco_intrinsics(self._w, self._h, vfov_deg=42.0)

    def get_camera_pose(self) -> tuple:
        """(cam_xpos (3,), cam_xmat (9,)) for the d435 camera, world frame."""
        return self._base.get_camera_pose()

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

    def detect(self, query: str) -> list[Detection]:
        """Run the learned open-vocab detector on the live RGB frame; pixel bboxes.

        Routed through the SHARED grounding-dino-tiny detector (the same instance
        the registered ``detect`` capability uses — the model loads at most once
        per process). Returns boxes sorted by score (the named-query branch of
        perception_grasp consumes ``list[Detection]`` unchanged). It reads ONLY the
        rendered RGB + the text query — never a ground-truth pose.
        """
        return self.detect_in(self.get_color_frame(), query)

    def detect_in(self, rgb: np.ndarray, query: str) -> list[Detection]:
        """:meth:`detect` on a GIVEN frame (the box then belongs to that frame's pose)."""
        with self._lock:
            self._ensure_detector()
            return self._vlm.detect(rgb, query)

    def segment(self, image: np.ndarray, bbox: Any) -> np.ndarray | None:
        """Binary mask for a single bbox on *image*, or None.

        Tries EdgeTAM for a tight mask; if EdgeTAM is unavailable (timm is
        network-blocked) or yields nothing, FALLS BACK to a box-interior rect mask.
        The box-rect mask is sufficient because grasp_point_from_rgbd
        (foreground_only=True) already gates a coarse mask leaking onto background
        down to the nearest depth cluster (D18). NEVER fabricates beyond the box.
        """
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
        try:
            with self._lock:
                self._ensure_tracker()
                results = self._tracker.init_track(image, bboxes=[(x1, y1, x2, y2)])
            mask = results[0].get("mask") if results else None
            if mask is not None and int(np.count_nonzero(mask)) > 0:
                return mask
            logger.info("[GO2-PERCEPT] EdgeTAM empty mask — box-rect fallback")
        except Exception as exc:  # noqa: BLE001 — segmenter optional; box-rect fallback
            logger.info("[GO2-PERCEPT] EdgeTAM unavailable (%s) — box-rect fallback", exc)
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
        rgb: np.ndarray | None = None,
        depth: np.ndarray | None = None,
        *,
        color: str | None = None,
    ) -> np.ndarray | None:
        """Mask of the salient object in front (deictic '前面的东西'), or None.

        Needs no VLM/EdgeTAM — resolves the front object from the rendered RGB
        saliency + depth. Renders fresh frames if not supplied. When *color* is
        given ('red'/'green'/'blue', D47), selects the salient blob of THAT colour
        instead of the front-most one (None if no such blob — FAIL LOUD).
        """
        from vector_os_nano.perception.front_object import front_object_mask
        if rgb is None:
            rgb = self.get_color_frame()
        if depth is None:
            depth = self.get_depth_frame()
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
