"""Open-vocabulary object detection for accurate world positioning.

Uses GroundingDINO (via HuggingFace transformers) for detection with
D435 depth for 3D projection. Sim-to-real compatible.

Pipeline:
    RGB frame → GroundingDINO → per-object bbox (u1,v1,u2,v2)
    Depth frame → depth at each bbox center
    Camera intrinsics + robot pose → world (x, y, z) per object

Usage::

    detector = ObjectDetector()  # loads model on first call
    detections = detector.detect(rgb_frame, depth_frame, robot_pose)
    # → [Detection(label="sofa", x=4.2, y=1.8, z=0.3, confidence=0.87), ...]

Requires: torch, transformers (pip install torch transformers)
Falls back gracefully when not installed — detect() returns empty list.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded model (heavy — only load when first called)
_model: Any = None
_processor: Any = None
_model_loaded: bool = False
_model_load_failed: bool = False

# Default detection prompt — common indoor furniture and objects
_DEFAULT_PROMPT: str = (
    "sofa . chair . table . desk . bed . fridge . counter . lamp . "
    "tv . bookshelf . plant . door . wardrobe . nightstand . "
    "bathtub . toilet . sink . stool . dresser . rug"
)

_CONFIDENCE_THRESHOLD: float = 0.25
_MODEL_ID: str = "IDEA-Research/grounding-dino-tiny"


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """A detected object with world coordinates."""

    label: str
    confidence: float
    # Bounding box in image pixels
    bbox_u1: float
    bbox_v1: float
    bbox_u2: float
    bbox_v2: float
    # World coordinates (from depth projection)
    world_x: float = 0.0
    world_y: float = 0.0
    world_z: float = 0.0
    # Depth at bbox center (metres)
    depth_m: float = 0.0


@dataclass(frozen=True)
class RobotPose:
    """Robot pose for depth-to-world projection."""

    x: float
    y: float
    z: float
    heading: float  # yaw in radians


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _ensure_model() -> bool:
    """Lazy-load GroundingDINO model. Returns True if available."""
    global _model, _processor, _model_loaded, _model_load_failed

    if _model_loaded:
        return True
    if _model_load_failed:
        return False

    try:
        import torch  # noqa: F401
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        logger.info("[ObjectDetector] Loading %s...", _MODEL_ID)
        _processor = AutoProcessor.from_pretrained(_MODEL_ID)
        _model = AutoModelForZeroShotObjectDetection.from_pretrained(_MODEL_ID)

        # Move to GPU if available
        if torch.cuda.is_available():
            _model = _model.to("cuda")
            logger.info("[ObjectDetector] Model loaded on CUDA")
        else:
            logger.info("[ObjectDetector] Model loaded on CPU")

        _model_loaded = True
        return True

    except ImportError as exc:
        logger.warning(
            "[ObjectDetector] torch/transformers not installed: %s. "
            "Install with: pip install torch transformers",
            exc,
        )
        _model_load_failed = True
        return False
    except Exception as exc:
        logger.error("[ObjectDetector] Model load failed: %s", exc)
        _model_load_failed = True
        return False


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_objects(
    rgb: np.ndarray,
    prompt: str | None = None,
    confidence_threshold: float = _CONFIDENCE_THRESHOLD,
) -> list[dict]:
    """Run GroundingDINO on an RGB frame.

    Args:
        rgb: (H, W, 3) uint8 RGB array.
        prompt: Object categories to detect, separated by " . ".
                Defaults to common indoor furniture.
        confidence_threshold: Minimum confidence to keep.

    Returns:
        List of dicts: [{"label": str, "confidence": float,
                         "bbox": (u1,v1,u2,v2)}]
        Returns empty list if model unavailable.
    """
    if not _ensure_model():
        return []

    import torch
    from PIL import Image

    text = prompt or _DEFAULT_PROMPT
    pil_image = Image.fromarray(rgb)

    inputs = _processor(images=pil_image, text=text, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)

    results = _processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=confidence_threshold,
        text_threshold=confidence_threshold,
        target_sizes=[pil_image.size[::-1]],  # (H, W)
    )[0]

    detections = []
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        u1, v1, u2, v2 = box.cpu().tolist()
        detections.append({
            "label": label.strip(),
            "confidence": float(score.cpu()),
            "bbox": (u1, v1, u2, v2),
        })

    return detections


def detect_and_project(
    rgb: np.ndarray,
    depth: np.ndarray,
    pose: RobotPose,
    prompt: str | None = None,
    confidence_threshold: float = _CONFIDENCE_THRESHOLD,
) -> list[Detection]:
    """Detect objects and project each to world coordinates using depth.

    Full pipeline: RGB → GroundingDINO → bbox → depth at center → world coords.

    Args:
        rgb: (H, W, 3) uint8 RGB array.
        depth: (H, W) float32 depth in metres (aligned to RGB).
        pose: Robot pose for world projection.
        prompt: Detection categories (default: indoor furniture).
        confidence_threshold: Minimum confidence.

    Returns:
        List of Detection with world coordinates filled in.
        Returns empty list if model unavailable.
    """
    from vector_os_nano.perception.depth_projection import (
        d435_intrinsics,
        depth_to_world,
    )

    raw = detect_objects(rgb, prompt, confidence_threshold)
    if not raw:
        return []

    h, w = depth.shape[:2]
    intrinsics = d435_intrinsics(w, h)
    results: list[Detection] = []

    for det in raw:
        u1, v1, u2, v2 = det["bbox"]
        # Bbox center
        cu = (u1 + u2) / 2.0
        cv = (v1 + v2) / 2.0

        # Sample depth in a small region around bbox center (robust to noise)
        r = 5  # pixel radius
        cu_i, cv_i = int(cu), int(cv)
        patch = depth[
            max(0, cv_i - r): min(h, cv_i + r),
            max(0, cu_i - r): min(w, cu_i + r),
        ]
        valid = patch[(patch > 0.1) & (patch < 10.0)]
        d_m = float(np.median(valid)) if len(valid) > 0 else 0.0

        # Project to world
        wx, wy, wz = 0.0, 0.0, 0.0
        if d_m > 0.0:
            world_pt = depth_to_world(
                depth, cu, cv, intrinsics,
                pose.x, pose.y, pose.z, pose.heading,
            )
            if world_pt is not None:
                wx, wy, wz = world_pt

        results.append(Detection(
            label=det["label"],
            confidence=det["confidence"],
            bbox_u1=u1, bbox_v1=v1,
            bbox_u2=u2, bbox_v2=v2,
            world_x=wx, world_y=wy, world_z=wz,
            depth_m=d_m,
        ))

    return results
