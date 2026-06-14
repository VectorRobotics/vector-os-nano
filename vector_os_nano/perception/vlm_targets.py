# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #9 R1 — VLM semantic target detection.

The track-A upgrade of color_targets: instead of segmenting saturated red/blue/
green blobs, ask a REAL vision-language model (Qwen-VL via OpenRouter) to GROUND
a semantic class — "chair", "potted plant", "table" — and return its bounding
box. The box is adapted to the SAME detection contract the vision-seek state
machine already consumes:

    ``[{label, x_norm, y_norm, area_frac}]``

so swapping the detector needs no change to the seek loop. x_norm/y_norm are the
box centre in [-1, 1] (0 = image centre; x_norm>0 = right); area_frac is the
box's fraction of the frame (a coarse range proxy: bigger = closer).

Honesty (CLAUDE.md rule 5): the VLM is REALLY called. The detector NEVER consults
ground-truth coordinates; if the model does not localise the target it returns
[] and the seek loop keeps searching / fails. Deterministic tests inject a fake
``vlm_call``; the live Qwen-VL call is exercised only in GUI acceptance.

GPU/network libraries are imported lazily — this module imports on a CPU-only box
without a key (the default client is built on first real detect, not at import).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np

# Reuse the robust JSON extractor (handles ```json fences / bare JSON / garbage).
from vector_os_nano.perception.vlm_go2 import _parse_json_response

logger = logging.getLogger(__name__)

# A coarse area floor mirroring color_targets — reject sub-promille specks but
# register a small/far object. Tuned alongside the colour detector's 0.0025.
_DEFAULT_MIN_AREA_FRAC = 0.0025

# Grounding prompt: ask for normalised [0,1] boxes. Qwen-VL also emits [0,1000];
# both are auto-detected on parse. JSON-only keeps parsing deterministic.
_GROUNDING_PROMPT = (
    "You are a robot's first-person camera vision system inside a room. "
    "Locate every instance of the following target object in this image: "
    "{query}. For each instance give a tight bounding box. "
    'Respond ONLY in JSON: {{"objects": [{{"label": "<object name>", '
    '"box": [x0, y0, x1, y1]}}]}} where the box is the top-left and '
    "bottom-right corners as fractions of the image width/height in [0,1] "
    "(x0,x1 horizontal; y0,y1 vertical). If the target is not visible, "
    'respond {{"objects": []}}. Do not invent objects that are not present.'
)

# Caller signature: (rgb_frame, prompt) -> raw model text.
VlmCall = Callable[[np.ndarray, str], str]


def _default_vlm_call() -> VlmCall:
    """Build the production caller: Qwen-VL via OpenRouter (Go2VLMPerception).

    Constructed lazily so importing this module never needs a key or network.
    The underlying ``_call_vlm(frame, prompt) -> str`` issues the real request;
    the API key is read from the environment by Go2VLMPerception itself.
    """
    from vector_os_nano.perception.vlm_go2 import Go2VLMPerception

    client = Go2VLMPerception()

    def call(frame: np.ndarray, prompt: str) -> str:
        return client._call_vlm(frame, prompt)  # noqa: SLF001 — intended raw hook

    return call


def _normalise_box(box: Any) -> "tuple[float, float, float, float] | None":
    """Coerce a model box to (x0, y0, x1, y1) in [0,1], or None if unusable.

    Accepts [0,1] fractions or [0,1000]-style coords (auto-scaled). Orders the
    corners so x0<=x1, y0<=y1. Rejects non-4-tuples and degenerate/off-frame."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        vals = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    # Auto-scale: any coordinate clearly above 1 means a non-fraction encoding
    # (Qwen's 0-1000 grid, or raw pixels on a small frame); divide by 1000.
    if max(abs(v) for v in vals) > 1.5:
        vals = [v / 1000.0 for v in vals]
    x0, y0, x1, y1 = vals
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    # Clamp into frame; reject fully-degenerate.
    x0, y0, x1, y1 = (max(0.0, min(1.0, v)) for v in (x0, y0, x1, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _label_matches(label: str, query: str) -> bool:
    """True when a detected label semantically matches the queried class.

    The grounding prompt already scopes the VLM to ``query``; this is a guard
    against a chatty model that lists extra objects. Case-insensitive: matches
    on substring either way (so 'plant' matches 'potted plant', 'chair' matches
    'chair desk') or any shared word token."""
    lo_l, lo_q = label.strip().lower(), query.strip().lower()
    if not lo_q:
        return True
    if lo_q in lo_l or lo_l in lo_q:
        return True
    return bool(set(lo_l.split()) & set(lo_q.split()))


class VlmTargetDetector:
    """Detect a semantic target class in a camera frame via a real VLM.

    Args:
        vlm_call: callable (frame, prompt) -> raw model text. Defaults to the
            production Qwen-VL/OpenRouter caller (built lazily on first detect).
            Tests inject a deterministic fake.
    """

    def __init__(self, vlm_call: Optional[VlmCall] = None) -> None:
        self._vlm_call = vlm_call
        self._lazy = vlm_call is None

    def _caller(self) -> VlmCall:
        if self._vlm_call is None:
            self._vlm_call = _default_vlm_call()
        return self._vlm_call

    def detect_targets(
        self,
        rgb: np.ndarray,
        query: str,
        min_area_frac: float = _DEFAULT_MIN_AREA_FRAC,
    ) -> "list[dict]":
        """Ground ``query`` in ``rgb`` with the VLM; return contract detections.

        Returns one dict per localised instance ``{label, x_norm, y_norm,
        area_frac}``, sorted by descending area_frac. Returns [] when the model
        sees nothing, returns garbage, or the call fails — NEVER a GT fallback.
        """
        if rgb is None or getattr(rgb, "ndim", 0) != 3:
            return []
        prompt = _GROUNDING_PROMPT.format(query=query)
        try:
            raw = self._caller()(rgb, prompt)
        except Exception as exc:  # noqa: BLE001 — a flaky VLM never crashes seek
            logger.warning("VLM grounding call failed for %r: %s", query, exc)
            return []
        data = _parse_json_response(raw) if isinstance(raw, str) else {}
        objects = data.get("objects") if isinstance(data, dict) else None
        if not isinstance(objects, list):
            return []

        out: list = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            label = str(obj.get("label", query))
            if not _label_matches(label, query):
                continue
            nb = _normalise_box(obj.get("box"))
            if nb is None:
                continue
            x0, y0, x1, y1 = nb
            area = (x1 - x0) * (y1 - y0)
            if area < min_area_frac:
                continue
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            out.append({
                "label": label,
                "x_norm": cx * 2.0 - 1.0,
                "y_norm": cy * 2.0 - 1.0,
                "area_frac": area,
            })
        out.sort(key=lambda d: d["area_frac"], reverse=True)
        return out
