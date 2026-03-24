"""PerceptionProtocol — structural interface for all perception backends.

No ROS2 imports. Pure Python typing.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from vector_os_nano.core.types import CameraIntrinsics, Detection, TrackedObject


@runtime_checkable
class PerceptionProtocol(Protocol):
    """Structural protocol for perception backends.

    Any object implementing these six methods satisfies the protocol,
    enabling duck-typed composition in PerceptionPipeline.
    """

    def get_color_frame(self) -> np.ndarray:
        """Return latest RGB image as (H, W, 3) uint8 array."""
        ...

    def get_depth_frame(self) -> np.ndarray:
        """Return latest depth image as (H, W) uint16 array (mm units)."""
        ...

    def get_intrinsics(self) -> CameraIntrinsics:
        """Return pinhole camera intrinsic parameters."""
        ...

    def caption(self, length: str = "normal") -> str:
        """Generate a natural language caption for the current frame."""
        ...

    def visual_query(self, question: str) -> str:
        """Answer a free-form question about the current frame."""
        ...

    def detect(self, query: str) -> list[Detection]:
        """Run VLM detection for objects matching query.

        Returns list of Detection with pixel-space bboxes.
        """
        ...

    def track(self, detections: list[Detection]) -> list[TrackedObject]:
        """Initialize or update tracker from Detection list.

        Returns TrackedObject list with masks and optional 3D data.
        """
        ...

    def get_point_cloud(self, mask: np.ndarray | None = None) -> np.ndarray:
        """Project depth to 3D point cloud.

        Args:
            mask: Optional (H, W) binary mask — only masked pixels projected.

        Returns:
            (N, 3) float64 array of 3D points in camera frame.
        """
        ...
