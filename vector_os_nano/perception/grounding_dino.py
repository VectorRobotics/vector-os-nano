# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""GroundingDinoDetector — open-vocabulary 2D detector (grounding-dino-tiny).

The first *learned* perception model the runtime routes to as a second model
family (alongside the chat LLM): a zero-shot open-vocabulary detector that
localizes a NAMED object from the rendered RGB + a text query alone. It reads
ONLY (rgb, query) — never a ground-truth pose — so the grasp it feeds stays
honestly perception-driven and the verify spine stays the sole grader.

Backend: ``IDEA-Research/grounding-dino-tiny`` via transformers'
``AutoModelForZeroShotObjectDetection`` + ``AutoProcessor``, loaded from the pinned
snapshot in the Hugging Face cache (perception/weights.py: fetched on first use with
one notice unless HF_HUB_OFFLINE=1). torch/transformers are imported LAZILY on first ``detect`` — importing
this module must never pull GPU libs (keeps dev / go2-only worlds and fast unit
runs torch-free). CUDA is used when present, else CPU (the tiny model is fine on CPU).

Prompt format that WORKED in the go/no-go probe: lowercase, period-separated noun
phrases ("a can. a bottle. a cylinder."). ``query_to_prompt`` maps an NL query to
that form through the shared world vocabulary. Boxes are returned as ``core.types.Detection`` (label, bbox xyxy in
pixels, confidence=score), sorted by score descending, above a score threshold.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from vector_os_nano.core.types import Detection
from vector_os_nano.perception.weights import Weights, loading

logger = logging.getLogger(__name__)

# Box-score floor: the probe localized the three cylinders cleanly at ~0.2-0.25.
_BOX_THRESHOLD = 0.25
# Text-token score floor handed to the post-processor (looser than the box floor;
# the box floor above is the real gate on what we return).
_TEXT_THRESHOLD = 0.20

# NL → generic open-vocab prompt phrases when the query is deictic/generic
# (no concrete noun). The scene's three graspable cylinders are can/bottle, so a
# generic "拿起那个罐子"/"the can" probes the common graspable shapes. lowercase,
# period-separated — the form grounding-dino expects.
_GENERIC_PROMPT = "a can. a bottle. a cylinder."


class UngroundableQuery(ValueError):
    """The query names something the world vocabulary cannot put into English."""


def query_to_prompt(query: str | None) -> str:
    """Map an NL query to a lowercase, period-terminated grounding-dino prompt.

    The noun phrase comes from the shared world vocabulary (perception/vocab.py):
    "绿色的瓶子" -> "a green bottle.", "pick up the red can" -> "a red can.", "红色的" ->
    "a red object." (grounding-dino is colour-conditioned, so the colour is kept;
    articles and request verbs are dropped). An empty, deictic or generic query gets
    the generic graspable-shapes prompt. CJK text the vocabulary cannot translate
    raises :class:`UngroundableQuery`: the model is English-text grounded and turns
    untranslated CJK into ``[UNK]`` tokens that still return confident boxes.
    """
    from vector_os_nano.perception.vocab import vocab

    phrase = vocab().english(query)
    if phrase is None:
        raise UngroundableQuery(
            f"cannot ground {query!r}: no English name for it in the world vocabulary; "
            "name the object in English (e.g. 'green bottle')"
        )
    return f"a {phrase}." if phrase else _GENERIC_PROMPT


class GroundingDinoDetector:
    """Zero-shot open-vocabulary detector over grounding-dino-tiny (lazy)."""

    WEIGHTS = (Weights("IDEA-Research/grounding-dino-tiny", "a2bb814dd30d776dcf7e30523b00659f4f141c71"),)

    def __init__(
        self,
        *,
        box_threshold: float = _BOX_THRESHOLD,
        text_threshold: float = _TEXT_THRESHOLD,
    ) -> None:
        self._box_threshold = float(box_threshold)
        self._text_threshold = float(text_threshold)
        self._processor: Any = None
        self._model: Any = None
        self._device: str = "cpu"
        self._torch: Any = None
        # The shared instance serves the detect capability AND the grasp/explore
        # perception from different threads: one load + inference at a time.
        self._lock = threading.Lock()

    # --- lazy load -----------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
        except ImportError as exc:  # noqa: F841 — re-raised with guidance
            raise ImportError(
                "GroundingDinoDetector requires torch and transformers. "
                "Install with: pip install torch transformers"
            ) from exc

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # raises WeightsUnavailable with the pre-fetch command
        with loading(self.WEIGHTS) as (path,):
            logger.info("[GDINO] loading %s on %s", self.WEIGHTS[0], self._device)
            self._processor = AutoProcessor.from_pretrained(path)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(path).to(self._device).eval()
        logger.info("[GDINO] loaded on %s", self._device)

    # --- detection -----------------------------------------------------------

    def detect(self, rgb: np.ndarray, query: str) -> list[Detection]:
        """Detect objects matching *query* in *rgb*; pixel-space xyxy boxes.

        Serialized (thread-safe lazy load + inference); see :meth:`_detect`.
        """
        with self._lock:
            return self._detect(rgb, query)

    def _detect(self, rgb: np.ndarray, query: str) -> list[Detection]:
        """Detect objects matching *query* in *rgb*; pixel-space xyxy boxes.

        Args:
            rgb: (H, W, 3) uint8 RGB image (the rendered camera frame ONLY — never
                a ground-truth pose; the detector's whole input is pixels + text).
            query: natural-language target ("拿起那个罐子" / "the can"); mapped to a
                lowercase period-separated grounding-dino prompt.

        Returns:
            ``Detection`` list (label, bbox=(x1,y1,x2,y2) px, confidence=score),
            sorted by score descending, filtered to score >= box_threshold.
        """
        if rgb is None or not isinstance(rgb, np.ndarray) or rgb.ndim != 3:
            raise ValueError("GroundingDinoDetector.detect needs an (H,W,3) RGB array")
        prompt = query_to_prompt(query)  # an ungroundable query fails before any load
        self._ensure_loaded()
        from PIL import Image as PILImage  # lazy

        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        pil = PILImage.fromarray(rgb.astype(np.uint8))

        torch = self._torch
        inputs = self._processor(images=pil, text=prompt, return_tensors="pt").to(
            self._device
        )
        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            target_sizes=[(height, width)],
        )
        if not results:
            return []
        result = results[0]
        boxes = result.get("boxes")
        scores = result.get("scores")
        # transformers >=4.51 renames "labels"->"text_labels"; accept either.
        labels = result.get("text_labels", result.get("labels"))

        detections: list[Detection] = []
        n = 0 if boxes is None else len(boxes)
        for i in range(n):
            box = [float(v) for v in boxes[i].tolist()]
            score = float(scores[i]) if scores is not None else 0.0
            if score < self._box_threshold:
                continue
            label = self._coerce_label(labels[i] if labels is not None else query, query)
            if "[UNK]" in label:
                continue  # the text tokens it grounded were unknown to the model
            x1, y1, x2, y2 = box
            detections.append(
                Detection(
                    label=label,
                    bbox=(x1, y1, x2, y2),
                    confidence=score,
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    @staticmethod
    def _coerce_label(raw: Any, query: str) -> str:
        """Coerce a model label (str or token-id tensor) to a clean string."""
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        return (query or "object").strip() or "object"


# --- shared lazy singleton ----------------------------------------------------
# ONE detector instance shared by the registered DetectorCapability AND the grasp
# route's Go2GraspPerception.detect, so the model loads at most once per process.

_SHARED: GroundingDinoDetector | None = None
_SHARED_LOCK = threading.Lock()


def get_shared_detector() -> GroundingDinoDetector:
    """Return the process-wide shared GroundingDinoDetector (constructed on demand).

    Construction is cheap (no model load — that is lazy on first ``detect``), so
    importing/constructing here never pulls torch.
    """
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = GroundingDinoDetector()
        return _SHARED
