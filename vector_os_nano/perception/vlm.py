"""Vision Language Model detector — pure Python, no ROS2.

Ported from:
  vector_ws/src/vlm/vlm/models/moondream.py  (MoondreamVlm)
  vector_ws/src/vlm/vlm/models/base.py        (VlmModelBase)
  vector_ws/src/vlm/vlm/models/types.py       (VlmDetection)

Supports three backends (selected automatically):
  1. Local transformers inference (fastest, offline):
       Set model="vikhyatk/moondream2" (or any HF model ID)
  2. Moondream Station (local HTTP server):
       Start `moondream` daemon; no model download needed
  3. Moondream Cloud API:
       Set MOONDREAM_API_KEY env var

GPU libraries (torch, transformers) are imported lazily — this module
can be imported on CPU-only machines without those packages installed.
"""
from __future__ import annotations

import base64
import contextlib
import logging
import os
import socket
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

import numpy as np

from vector_os_nano.core.types import Detection

logger = logging.getLogger(__name__)

_STATION_ENDPOINT = "http://localhost:2020/v1"


@dataclass(frozen=True)
class VLMConfig:
    """Configuration for VLMDetector."""

    provider: str = "moondream"
    model: str | None = None          # HF model ID for local inference
    api_key: str | None = None        # Cloud API key
    bgr_input: bool = False           # True if input images are BGR (e.g. from OpenCV)
    caption_length: str = "normal"    # "short" | "normal" | "long"
    timeout_s: float | None = 8.0    # Network timeout for non-local modes
    max_detections: int = 0           # 0 = unlimited


class VLMDetector:
    """Vision Language Model for object detection.

    Auto-selects backend:
      - model kwarg or MOONDREAM_MODEL env var -> local transformers
      - api_key kwarg or MOONDREAM_API_KEY env var -> cloud API
      - neither -> Moondream Station on localhost:2020
    """

    def __init__(
        self,
        provider: str = "moondream",
        model: str | None = None,
        config: VLMConfig | None = None,
    ) -> None:
        if config is not None:
            self._cfg = config
        else:
            self._cfg = VLMConfig(provider=provider, model=model)

        self._local = False
        self._use_station = False
        self._client: object | None = None

        self._setup()

    def _setup(self) -> None:
        """Initialize the chosen backend (lazy GPU import)."""
        model_id = self._cfg.model or os.getenv("MOONDREAM_MODEL")
        api_key = self._cfg.api_key or os.getenv("MOONDREAM_API_KEY")

        if model_id:
            self._setup_local(model_id)
        elif api_key:
            self._setup_api(api_key)
        else:
            self._setup_station()

    def _setup_local(self, model_id: str) -> None:
        """Load model locally via transformers (lazy import)."""
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError(
                "Local VLM inference requires torch and transformers. "
                "Install with: pip install torch transformers"
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Moondream locally from %s on %s", model_id, device)

        # Load model and move to device. Don't force dtype — let the model
        # use its native format (bf16 on Blackwell/RTX 50xx, fp32 on CPU).
        self._client = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True,
        ).to(device)
        self._local = True
        logger.info("Moondream loaded locally on %s", device)

    def _setup_api(self, api_key: str) -> None:
        """Connect to Moondream Cloud API."""
        try:
            import moondream as md  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "moondream package required for API mode. "
                "Install with: pip install moondream"
            ) from exc

        self._client = md.vl(api_key=api_key)
        logger.info("Using Moondream Cloud API")

    def _setup_station(self) -> None:
        """Connect to Moondream Station on localhost."""
        self._use_station = True
        logger.info("Using Moondream Station at %s", _STATION_ENDPOINT)
        try:
            import moondream as md  # type: ignore[import]
            self._client = md.vl(api_key="local", endpoint=_STATION_ENDPOINT)
        except Exception:
            self._client = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray, query: str) -> list[Detection]:
        """Detect objects matching query in image.

        Args:
            image: (H, W, 3) uint8 RGB image.
            query: Natural language description, e.g. "red cup".

        Returns:
            List of Detection with pixel-space bboxes (x1, y1, x2, y2).
        """
        from PIL import Image as PILImage  # lazy — not available everywhere

        pil_image = self._to_pil(image)
        height, width = image.shape[:2]

        if self._use_station:
            objects = self._station_detect(pil_image, query)
        elif self._local:
            result = self._client.detect(pil_image, query)  # type: ignore[union-attr]
            if isinstance(result, dict):
                objects = result.get("objects", [])
            elif isinstance(result, list):
                objects = result
            else:
                objects = []
        else:
            with self._timeout():
                result = self._client.detect(pil_image, query)  # type: ignore[union-attr]
            if isinstance(result, dict):
                objects = result.get("objects", [])
            elif isinstance(result, list):
                objects = result
            else:
                objects = []

        detections: list[Detection] = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            label = obj.get("label", query)
            x1 = float(obj.get("x_min", obj.get("x1", obj.get("left", 0.0))))
            y1 = float(obj.get("y_min", obj.get("y1", obj.get("top", 0.0))))
            x2 = float(obj.get("x_max", obj.get("x2", obj.get("right", 0.0))))
            y2 = float(obj.get("y_max", obj.get("y2", obj.get("bottom", 0.0))))
            # If values are normalized (0-1 range), scale to pixels
            if x2 <= 1.0 and y2 <= 1.0:
                x1, x2 = x1 * width, x2 * width
                y1, y2 = y1 * height, y2 * height
            detections.append(Detection(label=str(label), bbox=(x1, y1, x2, y2)))

        if self._cfg.max_detections > 0:
            detections = detections[: self._cfg.max_detections]

        return detections

    def caption(self, image: np.ndarray, length: str | None = None) -> str:
        """Generate a natural language caption for the image.

        Args:
            image: (H, W, 3) uint8 RGB image.
            length: Caption length override ("short", "normal", "long").
                    Falls back to config caption_length if not provided.

        Returns:
            Caption string.
        """
        pil_image = self._to_pil(image)
        length = length or self._cfg.caption_length
        if self._local:
            result = self._client.caption(pil_image, length=length)  # type: ignore[union-attr]
        else:
            with self._timeout():
                result = self._client.caption(pil_image, length=length)  # type: ignore[union-attr]
        if isinstance(result, dict):
            return result.get("caption", str(result))
        return str(result)

    def query(self, image: np.ndarray, prompt: str) -> str:
        """Answer a free-form question about the image.

        Args:
            image: (H, W, 3) uint8 RGB image.
            prompt: Question string.

        Returns:
            Answer string.
        """
        pil_image = self._to_pil(image)
        if self._local:
            result = self._client.query(pil_image, question=prompt)  # type: ignore[union-attr]
            if isinstance(result, dict):
                return result.get("answer", str(result))
            return str(result)
        with self._timeout():
            result = self._client.query(pil_image, prompt)  # type: ignore[union-attr]
        if isinstance(result, dict):
            return result.get("answer", str(result))
        return str(result)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_pil(self, image: np.ndarray):
        """Convert numpy image to PIL Image, handling BGR->RGB if needed."""
        from PIL import Image as PILImage  # lazy
        img = image
        if self._cfg.bgr_input:
            img = image[..., ::-1]
        return PILImage.fromarray(img)

    def _station_detect(self, pil_image, query: str) -> list[dict]:
        """Call Moondream Station detect API directly via HTTP.

        Bypasses the SDK detect() which has known bugs (same pattern as
        moondream.py _station_detect).
        """
        try:
            import requests  # type: ignore[import]
        except ImportError:
            logger.error("requests package required for Station mode. Install: pip install requests")
            return []

        from io import BytesIO

        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        try:
            resp = requests.post(
                f"{_STATION_ENDPOINT}/detect",
                json={
                    "image_url": f"data:image/png;base64,{img_b64}",
                    "object": query,
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                return (
                    data.get("objects")
                    or data.get("detections")
                    or data.get("result")
                    or data.get("data")
                    or []
                )
            if isinstance(data, list):
                return data
        except Exception as exc:
            logger.error("Moondream Station detect failed: %s", exc)
        return []

    @contextlib.contextmanager
    def _timeout(self):
        """Set socket default timeout for non-local API calls."""
        prev = socket.getdefaulttimeout()
        if self._cfg.timeout_s is not None:
            socket.setdefaulttimeout(self._cfg.timeout_s)
        try:
            yield
        finally:
            socket.setdefaulttimeout(prev)
