# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""GroundingDinoDetector — open-vocabulary 2D detector (grounding-dino-tiny), OFFLINE.

The first *learned* perception model the runtime routes to as a second model
family (alongside the chat LLM): a zero-shot open-vocabulary detector that
localizes a NAMED object from the rendered RGB + a text query alone. It reads
ONLY (rgb, query) — never a ground-truth pose — so the grasp it feeds stays
honestly perception-driven and the verify spine stays the sole grader.

Backend: ``IDEA-Research/grounding-dino-tiny`` via transformers'
``AutoModelForZeroShotObjectDetection`` + ``AutoProcessor``, loaded with
``local_files_only=True`` (HF_HUB_OFFLINE pinned in-process) so it never reaches
the network. torch/transformers are imported LAZILY on first ``detect`` — importing
this module must never pull GPU libs (keeps dev / go2-only worlds and fast unit
runs torch-free). CUDA is used when present, else CPU (the tiny model is fine on CPU).

Prompt format that WORKED in the go/no-go probe: lowercase, period-separated noun
phrases ("a can. a bottle. a cylinder."). ``query_to_prompt`` maps an NL query to
that form. Boxes are returned as ``core.types.Detection`` (label, bbox xyxy in
pixels, confidence=score), sorted by score descending, above a score threshold.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np

from vector_os_nano.core.types import Detection

logger = logging.getLogger(__name__)

_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
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

# zh → en object-noun map so a Chinese named query becomes an English phrase the
# model grounds well (the model is English-text grounded). Substring match.
_ZH_NOUN_EN: dict[str, str] = {
    "罐子": "can",
    "罐": "can",
    "瓶子": "bottle",
    "瓶": "bottle",
    "杯子": "cup",
    "杯": "cup",
    "盒子": "box",
    "盒": "box",
    "球": "ball",
    "碗": "bowl",
}

# zh → en colour-adjective map. grounding-dino IS colour-conditioned (the probe
# boxed "a red can./a green bottle./a blue bottle." with distinct scores), so a
# zh colour word must be PRESERVED into the prompt — dropping it (the old
# behaviour) collapsed all three cylinders to "a bottle." and made colour
# selection staging-dependent. Substring match on the lowercased query; longest
# alias first so "红色" beats the bare "红" prefix. Mirrors front_object's
# _COLOR_ALIASES (the HSV path) so both routes speak the same colour vocabulary.
_ZH_COLOR_EN: tuple[tuple[str, str], ...] = (
    ("红色", "red"), ("红", "red"),
    ("绿色", "green"), ("绿", "green"),
    ("蓝色", "blue"), ("蓝", "blue"),
)
# English colour adjectives kept as-is when an English query already names one
# (we never re-translate; this is only used by the colour-only fallback below).
_EN_COLORS: tuple[str, ...] = ("red", "green", "blue")


def _zh_color(q: str) -> str | None:
    """Return the English colour named in zh in *q* (lowercased), else None."""
    for zh, en in _ZH_COLOR_EN:
        if zh in q:
            return en
    return None


def query_to_prompt(query: str | None) -> str:
    """Map an NL query to a lowercase, period-terminated grounding-dino prompt.

    grounding-dino is colour-conditioned, so the colour adjective is PRESERVED:

    - A zh colour+noun query → "a <colour> <english-noun>." (绿色的瓶子→"a green
      bottle.", 红色的罐子→"a red can.", 蓝色的瓶子→"a blue bottle.").
    - A zh noun without a colour → "a <english-noun>." (罐子→"a can.").
    - A query that already names a thing in English keeps its adjective verbatim →
      "a <thing>." (lowercased, punctuation stripped; trailing period guaranteed;
      "red cup"→"a red cup.").
    - A colour-WITHOUT-a-noun (zh "红色的" or en "red") → "a <colour> object."
      rather than dropping the colour. (The SKILL routes a pure-colour query to the
      classical HSV resolver, NOT here — this is only a best-effort fallback so the
      detector is still colour-aware if ever handed a colour-only query directly.)
    - A deictic / empty query → the generic graspable-shapes prompt.
    """
    q = (query or "").strip().lower()
    if not q:
        return _GENERIC_PROMPT
    color = _zh_color(q)
    for zh, en in _ZH_NOUN_EN.items():
        if zh in q:
            return f"a {color} {en}." if color else f"a {en}."
    # No zh noun matched. If a zh colour was named with no recognised noun, build a
    # colour-only prompt rather than dropping the colour.
    if color:
        return f"a {color} object."
    # English (or already-romanized) query: strip to a bare noun phrase, adjective
    # (incl. colour) preserved verbatim.
    cleaned = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in q)
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return _GENERIC_PROMPT
    # A bare English colour word with no noun → "a <colour> object." (parallel to
    # the zh colour-only case above) so the colour is never dropped.
    if cleaned in _EN_COLORS:
        return f"a {cleaned} object."
    return f"a {cleaned}."


class GroundingDinoDetector:
    """Zero-shot open-vocabulary detector over grounding-dino-tiny (offline, lazy)."""

    def __init__(
        self,
        *,
        model_id: str = _MODEL_ID,
        box_threshold: float = _BOX_THRESHOLD,
        text_threshold: float = _TEXT_THRESHOLD,
    ) -> None:
        self._model_id = model_id
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
        # Pin offline BEFORE importing transformers so the hub never tries the net.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
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
        logger.info(
            "[GDINO] loading %s on %s (offline)", self._model_id, self._device
        )
        self._processor = AutoProcessor.from_pretrained(
            self._model_id, local_files_only=True
        )
        self._model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                self._model_id, local_files_only=True
            )
            .to(self._device)
            .eval()
        )
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
        self._ensure_loaded()
        from PIL import Image as PILImage  # lazy

        prompt = query_to_prompt(query)
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
