"""Intel RealSense D405 camera driver — pure Python, no ROS2.

Ported from:
  vector_ws/src/so101_perception/launch/camera.launch.py (config constants)
  vector_ws/src/so101_perception/scripts/calibrate_hand_eye.py (connect pattern)

D405 config preserved from camera.launch.py:
  - depth stream:  640x480 @ 30fps  (depth_module.depth_profile)
  - color stream:  640x480 @ 30fps  (depth_module.color_profile on D405)
  - align_depth to color: enabled
  - serial: '335122270413' (default; override via constructor)

pyrealsense2 is imported lazily at connect() time so this module can be
imported on machines without a RealSense SDK installed.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from vector_os_nano.core.types import CameraIntrinsics

logger = logging.getLogger(__name__)

# D405 defaults from camera.launch.py
_DEFAULT_SERIAL = "335122270413"
_DEFAULT_WIDTH = 640
_DEFAULT_HEIGHT = 480
_DEFAULT_FPS = 30


class RealSenseCamera:
    """Intel RealSense D405 driver.

    Usage:
        cam = RealSenseCamera()
        cam.connect()
        color = cam.get_color_frame()   # (H, W, 3) uint8 RGB
        depth = cam.get_depth_frame()   # (H, W)    uint16 mm
        cam.disconnect()

    Context manager:
        with RealSenseCamera() as cam:
            color, depth = cam.get_aligned_frames()
    """

    def __init__(
        self,
        serial: str = _DEFAULT_SERIAL,
        resolution: tuple[int, int] = (_DEFAULT_WIDTH, _DEFAULT_HEIGHT),
        fps: int = _DEFAULT_FPS,
        warmup_frames: int = 30,
    ) -> None:
        self._serial = serial
        self._width, self._height = resolution
        self._fps = fps
        self._warmup_frames = warmup_frames

        self._pipeline: object | None = None
        self._align: object | None = None
        self._intrinsics: CameraIntrinsics | None = None
        self._depth_scale: float = 0.001  # default: mm → m; overwritten by hardware value

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Start the RealSense pipeline and warm up auto-exposure.

        Raises:
            ImportError: If pyrealsense2 is not installed.
            RuntimeError: If device with given serial not found.
        """
        try:
            import pyrealsense2 as rs  # lazy import
        except ImportError as exc:
            raise ImportError(
                "pyrealsense2 is required for RealSenseCamera. "
                "Install with: pip install pyrealsense2"
            ) from exc

        config = rs.config()
        if self._serial:
            config.enable_device(self._serial)

        # D405: both depth and color go through the depth module
        config.enable_stream(
            rs.stream.depth, self._width, self._height, rs.format.z16, self._fps
        )
        config.enable_stream(
            rs.stream.color, self._width, self._height, rs.format.rgb8, self._fps
        )

        self._pipeline = rs.pipeline()
        try:
            profile = self._pipeline.start(config)
        except Exception as exc:
            self._pipeline = None
            raise RuntimeError(
                f"Failed to start RealSense pipeline (serial={self._serial!r}): {exc}"
            ) from exc

        # Read actual depth scale from hardware (D405 uses 0.0001, not 0.001)
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())
        logger.info("RealSense depth_scale: %g (raw * scale = metres)", self._depth_scale)

        # align depth to color (as in camera.launch.py: align_depth.enable=True)
        self._align = rs.align(rs.stream.color)

        # Cache intrinsics from the color stream profile
        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()
        self._intrinsics = CameraIntrinsics(
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.ppx),
            cy=float(intr.ppy),
            width=int(intr.width),
            height=int(intr.height),
        )

        # Warm up auto-exposure (same as calibrate_hand_eye.py)
        for _ in range(self._warmup_frames):
            self._pipeline.wait_for_frames()

        logger.info(
            "RealSense connected: serial=%s, %dx%d@%dfps, fx=%.1f fy=%.1f",
            self._serial, self._width, self._height, self._fps,
            self._intrinsics.fx, self._intrinsics.fy,
        )

    def disconnect(self) -> None:
        """Stop the RealSense pipeline and release resources."""
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
            self._align = None
            logger.info("RealSense disconnected")

    def __enter__(self) -> "RealSenseCamera":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------

    def get_aligned_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """Acquire one frame pair with depth aligned to color.

        Returns:
            (color_image, depth_image) where:
              color_image: (H, W, 3) uint8 RGB
              depth_image: (H, W)    uint16 mm

        Raises:
            RuntimeError: If camera is not connected.
        """
        self._require_connected()
        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("Invalid frames from RealSense pipeline")

        color = np.asanyarray(color_frame.get_data())  # (H, W, 3) RGB uint8
        depth = np.asanyarray(depth_frame.get_data())  # (H, W) uint16 mm
        return color, depth

    def get_color_frame(self) -> np.ndarray:
        """Return latest RGB color frame as (H, W, 3) uint8."""
        color, _ = self.get_aligned_frames()
        return color

    def get_depth_frame(self) -> np.ndarray:
        """Return latest depth frame as (H, W) uint16 (mm units)."""
        _, depth = self.get_aligned_frames()
        return depth

    def get_depth_scale(self) -> float:
        """Return depth scale: multiply raw uint16 values by this to get metres.

        D405 typically returns 0.0001 (units of 0.1mm).
        D435i typically returns 0.001 (units of 1mm).
        """
        return self._depth_scale

    def get_intrinsics(self) -> CameraIntrinsics:
        """Return cached camera intrinsics (available after connect()).

        Raises:
            RuntimeError: If called before connect().
        """
        self._require_connected()
        assert self._intrinsics is not None
        return self._intrinsics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if self._pipeline is None:
            raise RuntimeError(
                "RealSenseCamera is not connected. Call connect() first."
            )
