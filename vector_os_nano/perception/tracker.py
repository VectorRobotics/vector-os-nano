# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""EdgeTAM video segmentation tracker — pure Python, no ROS2.

Ported from:
  vector_ws/src/track_anything/track_anything/edge_tam.py  (EdgeTAMProcessor)

Key changes from vector_ws:
  - Removed ROS2 subscribers/publishers — pure function calls
  - Model repo from edge_tam.py: "yonigozlan/EdgeTAM-hf", pinned (WEIGHTS)
  - GPU memory management: torch.cuda.empty_cache() every 10 frames
  - torch/transformers imported lazily at __init__ time

Usage:
    tracker = EdgeTAMTracker()
    results = tracker.init_track(image, bboxes=[(x1, y1, x2, y2)])
    for frame in video:
        results = tracker.process_image(frame)
        # results: list of {"track_id": int, "mask": np.ndarray, "bbox": list, "score": float}
    tracker.stop()
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from vector_os_nano.perception.weights import Weights, local_dir

logger = logging.getLogger(__name__)


class EdgeTAMTracker:
    """EdgeTAM video segmentation tracker.

    Loads :attr:`WEIGHTS` from the Hugging Face cache (perception/weights.py:
    fetched on first use with one notice unless HF_HUB_OFFLINE=1). Absent weights
    fail loud.

    Args:
        device: "cuda", "cpu", or None (auto-detect).
        buffer_size: Max vision feature cache size (sliding window).
    """

    WEIGHTS = Weights("yonigozlan/EdgeTAM-hf", "c266ce53b3fc00f0f495b583f6a116c4e57f53bb")

    def __init__(
        self,
        device: Optional[str] = None,
        buffer_size: int = 5,
    ) -> None:
        self._device_str = device
        self._buffer_size = max(1, int(buffer_size))

        # Lazy-loaded — set in _load_model()
        self._processor: object | None = None
        self._model: object | None = None
        self._device: object | None = None
        self._dtype: object | None = None
        self._loaded = False

        # Tracking state
        self._inference_state: object | None = None
        self._frame_count: int = 0
        self._tracking: bool = False

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load EdgeTAM model and processor (lazy — imports torch/transformers)."""
        if self._loaded:
            return

        try:
            import torch
            from transformers import EdgeTamVideoModel, Sam2VideoProcessor
        except ImportError as exc:
            raise ImportError(
                "EdgeTAMTracker requires torch and transformers. "
                "Install with: pip install torch transformers"
            ) from exc

        if self._device_str is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self._device_str)

        self._dtype = self._select_dtype(self._device, torch)

        path = local_dir(self.WEIGHTS)  # raises WeightsUnavailable with the pre-fetch command
        logger.info("Loading EdgeTAM %s on %s", self.WEIGHTS, self._device)
        self._processor = Sam2VideoProcessor.from_pretrained(path)
        self._model = EdgeTamVideoModel.from_pretrained(path).to(self._device, dtype=self._dtype)
        self._model.eval()
        self._loaded = True
        logger.info("EdgeTAM loaded")

    @staticmethod
    def _select_dtype(device, torch) -> object:
        """Select best dtype for device (bf16 > fp16 > fp32)."""
        if device.type == "cuda":
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    # ------------------------------------------------------------------
    # Tracking API
    # ------------------------------------------------------------------

    def init_track(
        self,
        image: np.ndarray,
        bboxes: Optional[list[tuple[int, int, int, int]]] = None,
        points: Optional[list[tuple[int, int]]] = None,
    ) -> list[dict[str, Any]]:
        """Initialize tracking with bboxes or point prompts.

        Args:
            image: RGB image (H, W, 3) uint8.
            bboxes: List of (x1, y1, x2, y2) bounding boxes, one per object.
            points: List of (x, y) point prompts, one per object.

        Returns:
            List of detection dicts with keys:
                - track_id: int
                - mask: (H, W) uint8 binary mask
                - bbox: [x1, y1, x2, y2]
                - score: float confidence

        Raises:
            ValueError: If neither bboxes nor points provided.
        """
        if not bboxes and not points:
            raise ValueError("Either bboxes or points must be provided")

        self._load_model()

        if self._inference_state is not None:
            self.stop()

        self._inference_state = self._processor.init_video_session(  # type: ignore[union-attr]
            inference_device=self._device,
            max_vision_features_cache_size=self._buffer_size,
            dtype=self._dtype,
        )
        first_frame, original_size = self._prepare_frame(image)
        self._tracking = True
        self._frame_count = 0

        next_obj_id = 1

        if points:
            point_obj_ids = list(range(next_obj_id, next_obj_id + len(points)))
            input_points = [[[[ float(x), float(y)] ] for x, y in points]]
            input_labels = [[[1] for _ in points]]
            self._processor.add_inputs_to_inference_session(  # type: ignore[union-attr]
                inference_session=self._inference_state,
                frame_idx=0,
                obj_ids=point_obj_ids,
                input_points=input_points,
                input_labels=input_labels,
                original_size=original_size,
            )
            next_obj_id += len(points)

        if bboxes:
            bbox_obj_ids = list(range(next_obj_id, next_obj_id + len(bboxes)))
            input_boxes = [[
                [float(x1), float(y1), float(x2), float(y2)]
                for x1, y1, x2, y2 in bboxes
            ]]
            self._processor.add_inputs_to_inference_session(  # type: ignore[union-attr]
                inference_session=self._inference_state,
                frame_idx=0,
                obj_ids=bbox_obj_ids,
                input_boxes=input_boxes,
                original_size=original_size,
            )

        output = self._model(  # type: ignore[operator]
            inference_session=self._inference_state, frame=first_frame
        )
        return self._process_results(output, original_size)

    def process_image(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Process a video frame and propagate tracking.

        Args:
            image: RGB image (H, W, 3) uint8.

        Returns:
            Same format as init_track(). Empty list if not tracking.
        """
        if not self._tracking or self._inference_state is None:
            return []

        self._frame_count += 1
        next_frame, original_size = self._prepare_frame(image)
        output = self._model(  # type: ignore[operator]
            inference_session=self._inference_state, frame=next_frame
        )
        results = self._process_results(output, original_size)

        # Free intermediate GPU tensors every 10 frames (same as edge_tam.py)
        if self._frame_count % 10 == 0:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        return results

    def stop(self) -> None:
        """Stop tracking and release session resources."""
        self._tracking = False
        self._inference_state = None
        self._frame_count = 0

    def is_tracking(self) -> bool:
        """Return True if tracker is active."""
        return self._tracking

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_frame(self, image: np.ndarray) -> tuple:
        """Preprocess image for the model, return (pixel_values, original_size)."""
        import torch

        model_inputs = self._processor(  # type: ignore[operator]
            images=image, device=self._device, return_tensors="pt"
        )
        pixel_values = model_inputs.pixel_values[0]
        original_size = model_inputs.original_sizes[0]
        if torch.is_tensor(original_size):
            original_size_tuple = (
                int(original_size[0].item()),
                int(original_size[1].item()),
            )
        else:
            original_size_tuple = (int(original_size[0]), int(original_size[1]))
        return pixel_values, original_size_tuple

    def _process_results(
        self,
        model_output: object,
        original_size: tuple[int, int],
    ) -> list[dict[str, Any]]:
        """Convert model output to standardised detection dicts."""
        import torch

        detections: list[dict[str, Any]] = []
        mask_logits = getattr(model_output, "pred_masks", None)
        if mask_logits is None:
            return detections

        masks = self._processor.post_process_masks(  # type: ignore[union-attr]
            [mask_logits],
            original_sizes=[original_size],
            binarize=True,
        )[0]

        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()

        if masks.ndim == 4:
            masks = masks[:, 0, :, :]
        elif masks.ndim != 3:
            return detections

        obj_ids = getattr(model_output, "object_ids", None)
        if obj_ids is None:
            obj_ids = list(range(1, masks.shape[0] + 1))

        scores = None
        score_logits = getattr(model_output, "object_score_logits", None)
        if score_logits is not None:
            # transformers >=5.x returns object_score_logits shaped (N, 1) (a
            # per-object logit), so the raw numpy array's row ``scores[i]`` is a
            # (1,)-array, not a scalar — float() on it raises "only 0-dimensional
            # arrays can be converted to Python scalars" and the whole segment()
            # call falls back to a coarse box-rect mask (the z-mislocalization
            # bug). Flatten to 1-D so ``scores[i]`` is always a Python-coercible
            # scalar regardless of the transformers version's logit shape.
            scores = (
                torch.sigmoid(score_logits.float())
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

        for i, obj_id in enumerate(obj_ids):
            if i >= masks.shape[0]:
                break
            mask = (masks[i] > 0).astype(np.uint8)
            if not mask.any():
                continue

            y_indices, x_indices = np.where(mask > 0)
            if len(x_indices) == 0:
                continue

            x_min, y_min = int(np.min(x_indices)), int(np.min(y_indices))
            x_max, y_max = int(np.max(x_indices)), int(np.max(y_indices))

            detections.append({
                "track_id": int(obj_id),
                "mask": mask,
                "bbox": [x_min, y_min, x_max, y_max],
                "score": (
                    float(scores[i])
                    if scores is not None and i < len(scores)
                    else 1.0
                ),
            })

        return detections

    def __del__(self) -> None:
        """Cleanup on deletion."""
        if hasattr(self, "_inference_state") and self._inference_state is not None:
            self.stop()
