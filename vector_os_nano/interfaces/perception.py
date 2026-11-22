# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Public pixel-only detector and segmentation contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Detection:
    """Single object detection from VLM or 2D detector.

    bbox: (x1, y1, x2, y2) in pixel coordinates, top-left origin.
    """

    label: str
    bbox: tuple[float, float, float, float]
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "bbox": self.bbox,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Detection:
        bbox_raw = d["bbox"]
        if isinstance(bbox_raw, (list, tuple)):
            bbox: tuple[float, float, float, float] = (
                float(bbox_raw[0]),
                float(bbox_raw[1]),
                float(bbox_raw[2]),
                float(bbox_raw[3]),
            )
        else:
            raise ValueError(f"bbox must be a list or tuple, got {type(bbox_raw)}")
        return cls(
            label=str(d["label"]),
            bbox=bbox,
            confidence=float(d.get("confidence", 1.0)),
        )

@runtime_checkable
class Detector(Protocol):
    def detect(self, rgb: np.ndarray, query: str) -> list[Detection]:
        """Pixel-space xyxy detections; an empty result differs from a model error."""
        ...


@runtime_checkable
class Segmenter(Protocol):
    def init_track(self, image: np.ndarray,
                   bboxes: list[tuple[int, int, int, int]] | None = None,
                   points: list[tuple[int, int]] | None = None) -> list[dict[str, Any]]:
        """Initialize masks from pixel prompts; each result holds an image-sized mask."""
        ...

    def stop(self) -> None:
        """Release the current segmentation session."""
        ...
