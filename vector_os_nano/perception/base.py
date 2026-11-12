# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""PerceptionProtocol — structural interface for all perception backends.

No ROS2 imports. Pure Python typing.
"""
from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

import numpy as np

from vector_os_nano.core.types import CameraIntrinsics, Detection, TrackedObject

_GPU = re.compile(r"\bGPU (\d+)\b")
# an absolute or home-relative path (two components or more): a machine detail
_PATH = re.compile(r"(?<![\w.~/])~?/(?:[\w.@+-]+/)+[\w.@+-]*")


def failure_cause(exc: BaseException) -> str:
    """Why a camera, detector or segmenter could not run, as a tool result may say it.

    Out of GPU memory is named as that, with its device (another process on the GPU
    is the usual cause); a missing model keeps its sentence (it names the pre-fetch
    command); anything else is ``Type: message``. No machine path: the caller logs
    the exception itself.
    """
    from vector_os_nano.perception.weights import WeightsUnavailable

    text = str(exc)
    if type(exc).__name__ == "OutOfMemoryError" or "out of memory" in text.lower():
        gpu = _GPU.search(text)
        return f"out of GPU memory on {f'cuda:{gpu[1]}' if gpu else 'the GPU'} (another process may hold it)"
    text = _PATH.sub("<path>", text)
    return text if isinstance(exc, WeightsUnavailable) else f"{type(exc).__name__}: {text}"


def detect_query(perception: object, query: str) -> list[Detection]:
    """2D boxes for *query*: ``detect(query)`` on a streaming pipeline, else
    ``detect_in`` on ONE ``capture()`` of a snapshot perception (CameraPerception)."""
    detect = getattr(perception, "detect", None)
    if detect is not None:
        return detect(query)
    return perception.detect_in(perception.capture().rgb, query)  # type: ignore[attr-defined]


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
