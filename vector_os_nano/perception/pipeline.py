"""PerceptionPipeline — orchestrates camera, VLM, tracker, and 3D.

Ported from:
  vector_ws/src/track_anything/track_anything/track_3d.py (rgbd_callback logic)

The ROS2 pub/sub loop is replaced by direct method calls.

Pipeline steps (mirrors track_3d.py rgbd_callback):
  1. Acquire aligned RGBD frames from camera
  2. On detect() call: run VLM -> get bboxes -> init_track
  3. On track() call: process_image -> get masks -> project to 3D

Performance constants from track_3d.py:
  depth_scale  = 1000.0  (mm -> metres)
  depth_trunc  = 10.0    (metres)
  bbox_max_points = 2000 (sampling for fast bbox3d)
"""
from __future__ import annotations

import logging
import threading
import time
from threading import Lock
from typing import Optional

import numpy as np

from vector_os_nano.core.types import (
    BBox3D,
    CameraIntrinsics,
    Detection,
    Pose3D,
    TrackedObject,
)
from vector_os_nano.perception.base import PerceptionProtocol
from vector_os_nano.perception.pointcloud import (
    pointcloud_to_bbox3d_fast,
    rgbd_to_pointcloud_fast,
)

logger = logging.getLogger(__name__)

# Performance constants from track_3d.py
_DEPTH_SCALE = 1000.0
_DEPTH_TRUNC = 10.0
_BBOX_MAX_POINTS = 2000
_TRACKING_TIMEOUT_S = 20.0
_MASK_OPEN_KERNEL_SIZE = 5
_MASK_ERODE_KERNEL_SIZE = 7

# Background tracking loop: process every Nth frame for 3D to reduce GPU load.
# At 20fps camera rate, stride=4 gives 5Hz 3D updates (plenty for pick & place).
_TRACKING_FRAME_STRIDE = 4


class PerceptionPipeline:
    """Full perception pipeline: camera -> VLM -> tracker -> 3D.

    Satisfies PerceptionProtocol structurally.

    Usage (with real hardware):
        cam = RealSenseCamera()
        cam.connect()
        vlm = VLMDetector()
        tracker = EdgeTAMTracker()
        pipeline = PerceptionPipeline(camera=cam, vlm=vlm, tracker=tracker)

        detections = pipeline.detect("red cup")
        tracked = pipeline.track(detections)

    Usage (without hardware — synthetic frames):
        pipeline = PerceptionPipeline()
        pipeline.set_synthetic_frames(color, depth, intrinsics)
        ...
    """

    def __init__(
        self,
        camera: object | None = None,
        vlm: object | None = None,
        tracker: object | None = None,
        depth_scale: float = _DEPTH_SCALE,
        depth_trunc: float = _DEPTH_TRUNC,
        bbox_max_points: int = _BBOX_MAX_POINTS,
    ) -> None:
        self._camera = camera
        self._vlm = vlm
        self._tracker = tracker

        # Use hardware depth_scale if camera provides it (D405: 0.0001, D435i: 0.001)
        # The constructor's depth_scale is a fallback for synthetic/test use.
        if camera is not None and hasattr(camera, "get_depth_scale"):
            hw_scale = camera.get_depth_scale()
            # Convert to our convention: depth_scale = 1/hw_scale
            # rgbd_to_pointcloud_fast does: depth_raw / depth_scale → metres
            # So depth_scale = 1 / hw_scale
            self._depth_scale = 1.0 / hw_scale
            logger.info("Depth scale from hardware: raw / %.1f = metres (hw_scale=%g)", self._depth_scale, hw_scale)
        else:
            self._depth_scale = depth_scale
        self._depth_trunc = depth_trunc
        self._bbox_max_points = bbox_max_points

        self._lock = Lock()
        self._tracked_objects: list[TrackedObject] = []
        self._tracking_loss_time: float | None = None

        # Exposed for camera viewer overlay (read by run.py camera thread)
        self._last_detections: list[Detection] = []
        self._last_tracked: list[TrackedObject] = []

        # Background tracking thread state
        self._tracking_thread: threading.Thread | None = None
        self._stop_tracking: threading.Event | None = None

        # Synthetic frame support (for testing without hardware)
        self._synthetic_color: np.ndarray | None = None
        self._synthetic_depth: np.ndarray | None = None
        self._synthetic_intrinsics: CameraIntrinsics | None = None

    # ------------------------------------------------------------------
    # Synthetic frame injection (for unit testing)
    # ------------------------------------------------------------------

    def set_synthetic_frames(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> None:
        """Inject synthetic frames for testing without a real camera."""
        self._synthetic_color = color
        self._synthetic_depth = depth
        self._synthetic_intrinsics = intrinsics

    # ------------------------------------------------------------------
    # PerceptionProtocol implementation
    # ------------------------------------------------------------------

    def get_color_frame(self) -> np.ndarray:
        """Return latest RGB color frame."""
        if self._synthetic_color is not None:
            return self._synthetic_color
        if self._camera is None:
            raise RuntimeError("No camera configured and no synthetic frames set")
        return self._camera.get_color_frame()  # type: ignore[union-attr]

    def get_depth_frame(self) -> np.ndarray:
        """Return latest depth frame (uint16 mm)."""
        if self._synthetic_depth is not None:
            return self._synthetic_depth
        if self._camera is None:
            raise RuntimeError("No camera configured and no synthetic frames set")
        return self._camera.get_depth_frame()  # type: ignore[union-attr]

    def get_intrinsics(self) -> CameraIntrinsics:
        """Return camera intrinsics."""
        if self._synthetic_intrinsics is not None:
            return self._synthetic_intrinsics
        if self._camera is None:
            raise RuntimeError("No camera configured and no synthetic intrinsics set")
        return self._camera.get_intrinsics()  # type: ignore[union-attr]

    def get_point_cloud(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Project depth to 3D point cloud using current frame.

        Args:
            mask: Optional (H, W) binary mask.

        Returns:
            (N, 3) float64 array of 3D points in camera frame.
        """
        depth = self.get_depth_frame()
        color = self.get_color_frame()
        intrinsics = self.get_intrinsics()
        points, _ = rgbd_to_pointcloud_fast(
            depth, color, intrinsics,
            depth_scale=self._depth_scale,
            depth_trunc=self._depth_trunc,
            mask=mask,
        )
        return points

    def caption(self, length: str = "normal") -> str:
        """Generate a natural language caption describing the current frame.

        Args:
            length: Caption length — "short", "normal", or "long".

        Returns:
            Caption string.

        Raises:
            RuntimeError: If no VLM is configured.
        """
        if self._vlm is None:
            raise RuntimeError("No VLM configured")
        color = self.get_color_frame()
        caption_text: str = self._vlm.caption(color, length=length) if hasattr(self._vlm, "caption") else ""
        if isinstance(caption_text, dict):
            caption_text = caption_text.get("caption", str(caption_text))
        logger.info("VLM caption(%s) -> %s", length, caption_text[:80])
        return str(caption_text)

    def visual_query(self, question: str) -> str:
        """Answer a free-form question about the current camera frame.

        Args:
            question: Natural language question about the scene.

        Returns:
            Answer string.

        Raises:
            RuntimeError: If no VLM is configured.
        """
        if self._vlm is None:
            raise RuntimeError("No VLM configured")
        color = self.get_color_frame()
        answer: str = self._vlm.query(color, question) if hasattr(self._vlm, "query") else ""
        if isinstance(answer, dict):
            answer = answer.get("answer", str(answer))
        logger.info("VLM query(%r) -> %s", question, str(answer)[:80])
        return str(answer)

    def detect(self, query: str) -> list[Detection]:
        """Run VLM detection on the current frame.

        Args:
            query: Natural language object description.

        Returns:
            List of Detection with pixel-space bboxes.

        Raises:
            RuntimeError: If no VLM is configured.
        """
        if self._vlm is None:
            raise RuntimeError("No VLM configured. Pass vlm=VLMDetector() to PerceptionPipeline")
        color = self.get_color_frame()
        detections: list[Detection] = self._vlm.detect(color, query)  # type: ignore[union-attr]
        self._last_detections = detections
        logger.info("VLM detect('%s') -> %d detections", query, len(detections))
        return detections

    def track(self, detections: list[Detection]) -> list[TrackedObject]:
        """Initialize tracker from detections and return TrackedObject list.

        Ported from track_3d.py track_init_callback + rgbd_callback flow:
          1. Convert Detection bboxes to tracker format
          2. init_track() with validated bboxes
          3. For each tracked object: project mask to 3D bbox

        Args:
            detections: List of Detection from detect().

        Returns:
            List of TrackedObject with 2D bbox, optional 3D bbox, mask.

        Raises:
            RuntimeError: If no tracker is configured.
        """
        if self._tracker is None:
            raise RuntimeError("No tracker configured. Pass tracker=EdgeTAMTracker() to PerceptionPipeline")

        color = self.get_color_frame()
        depth = self.get_depth_frame()
        intrinsics = self.get_intrinsics()
        h, w = color.shape[:2]

        # Convert Detection bboxes to tracker format (x1, y1, x2, y2) ints
        bboxes = []
        labels = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1 = max(0, min(int(round(x1)), w - 1))
            y1 = max(0, min(int(round(y1)), h - 1))
            x2 = max(0, min(int(round(x2)), w - 1))
            y2 = max(0, min(int(round(y2)), h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            bboxes.append((x1, y1, x2, y2))
            labels.append(det.label)

        if not bboxes:
            self._tracked_objects = []
            return []

        raw_tracks = self._tracker.init_track(color, bboxes=bboxes)  # type: ignore[union-attr]
        tracked_objects = self._build_tracked_objects(
            raw_tracks, labels, color, depth, intrinsics
        )
        with self._lock:
            self._tracked_objects = tracked_objects
            self._last_tracked = tracked_objects

        # Mirror track_3d.py: start continuous background loop so the camera
        # viewer overlay updates at ~20fps without explicit update() calls.
        self.start_continuous_tracking()

        return tracked_objects

    def get_tracked_objects(self) -> list[TrackedObject]:
        """Return the last set of tracked objects (from the most recent track() call)."""
        with self._lock:
            return list(self._tracked_objects)

    def update(self) -> list[TrackedObject]:
        """Process the current frame and update tracking.

        Call this each frame after an initial track() to propagate masks.
        Ported from track_3d.py rgbd_callback (process_image path).

        Returns:
            Updated TrackedObject list. Empty if tracking is lost.
        """
        if self._tracker is None:
            return []

        color = self.get_color_frame()
        depth = self.get_depth_frame()
        intrinsics = self.get_intrinsics()

        raw_tracks = self._tracker.process_image(color)  # type: ignore[union-attr]

        if not raw_tracks:
            if self._tracking_loss_time is None:
                self._tracking_loss_time = time.monotonic()
                logger.warning("Tracking lost, starting timeout")
            elapsed = time.monotonic() - self._tracking_loss_time
            if elapsed > _TRACKING_TIMEOUT_S:
                logger.warning("Tracking lost for %.0fs, stopping", elapsed)
                self._tracker.stop()  # type: ignore[union-attr]
                self._tracking_loss_time = None
                with self._lock:
                    self._tracked_objects = []
            return []

        self._tracking_loss_time = None
        # Preserve labels from last known tracked objects
        label_map = {t.track_id: t.label for t in self._tracked_objects}
        labels = [label_map.get(r.get("track_id", 0), "object") for r in raw_tracks]

        tracked_objects = self._build_tracked_objects(
            raw_tracks, labels, color, depth, intrinsics
        )
        with self._lock:
            self._tracked_objects = tracked_objects
            self._last_tracked = tracked_objects
        return tracked_objects

    # ------------------------------------------------------------------
    # Continuous background tracking (mirrors track_3d.py _color_callback loop)
    # ------------------------------------------------------------------

    def start_continuous_tracking(self) -> None:
        """Start background thread that continuously processes camera frames through EdgeTAM.

        Updates _last_tracked with fresh masks/bboxes at ~15-20fps.
        Call after tracker has been initialized via track().

        The thread is daemon=True so it dies automatically with the main process.
        Calling this method when a thread is already running is a no-op.
        """
        if self._tracking_thread is not None and self._tracking_thread.is_alive():
            return  # already running
        self._stop_tracking = threading.Event()
        self._tracking_thread = threading.Thread(
            target=self._tracking_loop, daemon=True, name="bg-edgetam-tracking"
        )
        self._tracking_thread.start()
        logger.info("Background tracking thread started")

    def stop_continuous_tracking(self) -> None:
        """Stop the background tracking thread and wait for it to exit (max 2s)."""
        if self._stop_tracking is not None:
            self._stop_tracking.set()
        if self._tracking_thread is not None:
            self._tracking_thread.join(timeout=2.0)
            self._tracking_thread = None
        self._stop_tracking = None
        logger.info("Background tracking thread stopped")

    def _tracking_loop(self) -> None:
        """Background thread: process frames through EdgeTAM, update tracked objects.

        Design mirrors track_3d.py _color_callback / rgbd_callback:
          - Every color frame triggers process_image() on EdgeTAM.
          - 3D projection runs every _TRACKING_FRAME_STRIDE frames to reduce GPU load.
          - _last_tracked is updated atomically under _lock after each batch.
        """
        logger.info("Background tracking loop running")
        frame_counter = 0

        while not self._stop_tracking.is_set():  # type: ignore[union-attr]
            try:
                if self._tracker is None or not self._tracker.is_tracking():  # type: ignore[union-attr]
                    time.sleep(0.1)
                    continue

                color = self.get_color_frame()
                depth = self.get_depth_frame()
                if color is None or depth is None:
                    time.sleep(0.05)
                    continue

                # Process through EdgeTAM — propagates masks to current frame
                raw_tracks = self._tracker.process_image(color)  # type: ignore[union-attr]
                if not raw_tracks:
                    time.sleep(0.05)
                    continue

                frame_counter += 1
                do_3d = (frame_counter % _TRACKING_FRAME_STRIDE == 0) or (frame_counter == 1)

                # Preserve labels from current tracked objects
                with self._lock:
                    label_map = {t.track_id: t.label for t in self._tracked_objects}

                labels = [label_map.get(r.get("track_id", 0), "object") for r in raw_tracks]
                intrinsics = self.get_intrinsics()

                if do_3d:
                    tracked = self._build_tracked_objects(
                        raw_tracks, labels, color, depth, intrinsics
                    )
                else:
                    # Lightweight update: refresh 2D bbox/mask without 3D projection.
                    # Reuses existing pose/bbox_3d from the last full 3D frame.
                    with self._lock:
                        old_map = {t.track_id: t for t in self._tracked_objects}
                    tracked = self._build_tracked_objects_2d(
                        raw_tracks, labels, old_map
                    )

                with self._lock:
                    self._tracked_objects = tracked
                    self._last_tracked = tracked

            except Exception as exc:
                logger.debug("Tracking loop error: %s", exc)
                time.sleep(0.1)

        logger.info("Background tracking loop exited")

    def _build_tracked_objects_2d(
        self,
        raw_tracks: list[dict],
        labels: list[str],
        old_map: dict[int, TrackedObject],
    ) -> list[TrackedObject]:
        """Lightweight frame update: refresh mask/bbox_2d, reuse existing 3D data.

        Used on non-stride frames to avoid repeated GPU pointcloud projection.
        The existing pose and bbox_3d from the previous full frame are carried forward.
        """
        result: list[TrackedObject] = []
        for i, raw in enumerate(raw_tracks):
            track_id = int(raw.get("track_id", i + 1))
            mask = raw.get("mask")
            bbox_2d_raw = raw.get("bbox", [0, 0, 0, 0])
            score = float(raw.get("score", 1.0))
            label = labels[i] if i < len(labels) else "object"

            bbox_2d: tuple[float, float, float, float] = (
                float(bbox_2d_raw[0]),
                float(bbox_2d_raw[1]),
                float(bbox_2d_raw[2]),
                float(bbox_2d_raw[3]),
            )

            refined_mask: np.ndarray | None = None
            if mask is not None:
                refined_mask = self._refine_mask(mask)

            # Carry forward 3D data from previous frame for this track_id
            old = old_map.get(track_id)
            pose = old.pose if old is not None else None
            bbox_3d = old.bbox_3d if old is not None else None

            result.append(TrackedObject(
                track_id=track_id,
                label=label,
                bbox_2d=bbox_2d,
                pose=pose,
                bbox_3d=bbox_3d,
                confidence=score,
                mask=refined_mask,
            ))
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_tracked_objects(
        self,
        raw_tracks: list[dict],
        labels: list[str],
        color: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> list[TrackedObject]:
        """Convert raw tracker output dicts to TrackedObject list with 3D data."""
        result: list[TrackedObject] = []
        for i, raw in enumerate(raw_tracks):
            track_id = int(raw.get("track_id", i + 1))
            mask = raw.get("mask")
            bbox_2d_raw = raw.get("bbox", [0, 0, 0, 0])
            score = float(raw.get("score", 1.0))
            label = labels[i] if i < len(labels) else "object"

            bbox_2d: tuple[float, float, float, float] = (
                float(bbox_2d_raw[0]),
                float(bbox_2d_raw[1]),
                float(bbox_2d_raw[2]),
                float(bbox_2d_raw[3]),
            )

            # Refine mask morphologically (same as track_3d.py _refine_mask)
            refined_mask: np.ndarray | None = None
            if mask is not None:
                refined_mask = self._refine_mask(mask)

            # Project mask to 3D (mirrors track_3d.py rgbd_to_pointcloud_fast call)
            bbox_3d: BBox3D | None = None
            pose: Pose3D | None = None
            if refined_mask is not None and np.any(refined_mask):
                points, _ = rgbd_to_pointcloud_fast(
                    depth,
                    color,
                    intrinsics,
                    depth_scale=self._depth_scale,
                    depth_trunc=self._depth_trunc,
                    mask=refined_mask,
                )
                if len(points) >= 4:
                    # 1. Remove statistical outliers (depth noise, edge bleeding)
                    clean = self._remove_depth_outliers(points)
                    if len(clean) < 4:
                        clean = points

                    sampled = self._sample_points(clean, self._bbox_max_points)
                    bbox_3d = pointcloud_to_bbox3d_fast(sampled)

                    # 2. Compute robust centroid using trimmed mean
                    # Trim 10% extremes on each axis to remove edge artifacts
                    pose = self._robust_centroid(sampled)

            result.append(TrackedObject(
                track_id=track_id,
                label=label,
                bbox_2d=bbox_2d,
                pose=pose,
                bbox_3d=bbox_3d,
                confidence=score,
                mask=refined_mask,
            ))

        return result

    @staticmethod
    def _refine_mask(mask: np.ndarray) -> np.ndarray:
        """Morphological mask refinement (ported from track_3d.py _refine_mask).

        Opening removes noise; erosion shrinks edges to filter boundary noise.
        Falls back gracefully if cv2 not installed.
        """
        try:
            import cv2
        except ImportError:
            return (mask > 0).astype(np.uint8)

        mask_u8 = mask.astype(np.uint8) * 255
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (_MASK_OPEN_KERNEL_SIZE, _MASK_OPEN_KERNEL_SIZE),
        )
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel)
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (_MASK_ERODE_KERNEL_SIZE, _MASK_ERODE_KERNEL_SIZE),
        )
        mask_u8 = cv2.erode(mask_u8, erode_kernel, iterations=2)  # 2x erode for cleaner edges
        return (mask_u8 > 0).astype(np.uint8)

    @staticmethod
    def _sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
        """Uniformly subsample points for performance (same as track_3d.py)."""
        if max_points <= 0 or len(points) <= max_points:
            return points
        idx = np.linspace(0, len(points) - 1, num=max_points, dtype=np.int32)
        return points[idx]

    @staticmethod
    def _remove_depth_outliers(points: np.ndarray) -> np.ndarray:
        """Remove depth outliers using IQR on Z axis.

        Mask edges produce depth bleed — points that are far behind or in front
        of the actual object surface. IQR filtering removes these.
        """
        if len(points) < 10:
            return points
        z = points[:, 2]
        q1, q3 = np.percentile(z, [25, 75])
        iqr = q3 - q1
        if iqr < 1e-6:
            return points
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (z >= lower) & (z <= upper)
        filtered = points[mask]
        return filtered if len(filtered) >= 4 else points

    @staticmethod
    def _robust_centroid(points: np.ndarray) -> Pose3D:
        """Compute centroid using trimmed mean (10% trim on each axis).

        More robust than median or mean:
        - Mean is sensitive to outliers
        - Median ignores point density
        - Trimmed mean balances both: removes 10% extremes, averages the rest
        """
        if len(points) < 10:
            c = np.median(points, axis=0)
            return Pose3D(x=float(c[0]), y=float(c[1]), z=float(c[2]))

        trim_frac = 0.10
        n = len(points)
        trim_n = max(1, int(n * trim_frac))

        cx, cy, cz = 0.0, 0.0, 0.0
        for axis in range(3):
            sorted_vals = np.sort(points[:, axis])
            trimmed = sorted_vals[trim_n: n - trim_n]
            if len(trimmed) == 0:
                trimmed = sorted_vals
            val = float(np.mean(trimmed))
            if axis == 0:
                cx = val
            elif axis == 1:
                cy = val
            else:
                cz = val

        return Pose3D(x=cx, y=cy, z=cz)
