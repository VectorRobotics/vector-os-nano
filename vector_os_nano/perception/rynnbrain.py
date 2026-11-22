# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""RynnBrain — an external embodied VLM service as a second detector backend.

``POST {url}/infer`` with ``{"image": <base64 JPEG>, "text": prompt, "think": false}``
returns ``{"reply": "<object> (x1, y1), (x2, y2) </object>"}``, coordinates in
[0, 1000] of the image sent. Slower than grounding-dino (~1 s) and unscored, but a
language model: it reads zh and relations ("the bottle left of the red can").

The service belongs to another project: this client only CALLS it (never starts or
stops it) and fails with a clear message when it is down. ``VECTOR_RYNNBRAIN_URL``
overrides the default ``http://127.0.0.1:8786``.
"""
from __future__ import annotations

import base64
import io
import os
import re

import numpy as np

from vector_os_nano.interfaces.perception import Detection

DEFAULT_URL = "http://127.0.0.1:8786"
_PROMPT = (
    "Find {query}.\nGenerate coordinates for one object bounding box. Constraints: "
    "x1,y1,x2,y2 ∈ [0,1000]. Response must be in the format: <object> (x1, y1), (x2, y2) </object>"
)
_POINT = re.compile(r"\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)")


class RynnBrainUnavailable(RuntimeError):
    """The RynnBrain service did not answer."""


class RynnBrainDetector:
    """One bounding box per query from the RynnBrain service (``detect(rgb, query)``)."""

    def __init__(self, url: str | None = None, *, timeout: float = 20.0) -> None:
        self.url = (url or os.environ.get("VECTOR_RYNNBRAIN_URL") or DEFAULT_URL).rstrip("/")
        self.timeout = float(timeout)

    def available(self, timeout: float = 0.5) -> bool:
        """True when ``GET /health`` answers ok (never raises)."""
        import httpx

        try:
            return bool(httpx.get(f"{self.url}/health", timeout=timeout).json().get("ok"))
        except Exception:  # noqa: BLE001 — a probe: down is just False
            return False

    def detect(self, rgb: np.ndarray, query: str) -> list[Detection]:
        """The box RynnBrain gives for *query* on *rgb* (pixels), or [] if it names none."""
        import httpx
        from PIL import Image

        h, w = rgb.shape[:2]
        buf = io.BytesIO()
        Image.fromarray(rgb.astype(np.uint8)).save(buf, format="JPEG", quality=90)
        payload = {"image": base64.b64encode(buf.getvalue()).decode("ascii"),
                   "text": _PROMPT.format(query=query), "think": False}
        try:
            r = httpx.post(f"{self.url}/infer", json=payload, timeout=self.timeout)
            r.raise_for_status()
            reply = str(r.json().get("reply", ""))
        except Exception as exc:  # noqa: BLE001
            raise RynnBrainUnavailable(
                f"RynnBrain at {self.url} is not reachable ({type(exc).__name__}: {exc}); it is an "
                "external service owned by its own project (start it there), or use "
                "backend='grounding-dino'"
            ) from exc
        return parse_reply(reply, w, h, query)


def parse_reply(reply: str, width: int, height: int, query: str) -> list[Detection]:
    """The first ``(x1, y1), (x2, y2)`` box of a reply, scaled from [0, 1000] to pixels."""
    pts = [(float(a), float(b)) for a, b in _POINT.findall(reply)]
    if len(pts) < 2:
        return []
    (x1, y1), (x2, y2) = pts[0], pts[1]
    sx, sy = width / 1000.0, height / 1000.0
    box = (min(x1, x2) * sx, min(y1, y2) * sy, max(x1, x2) * sx, max(y1, y2) * sy)
    if box[2] - box[0] < 1.0 or box[3] - box[1] < 1.0:
        return []
    # RynnBrain gives no score: 1.0 means "the box it named", not a calibrated confidence.
    return [Detection(label=query, bbox=box, confidence=1.0)]
