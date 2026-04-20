"""Qwen2.5-VL grounded object detection via OpenRouter.

Calls ``qwen/qwen2.5-vl-72b-instruct`` to detect objects in a camera frame
and returns pixel-space bounding boxes as ``list[Detection]``.

Typical usage::

    from vector_os_nano.perception.vlm_qwen import QwenVLMDetector

    det = QwenVLMDetector()                          # key from env
    det = QwenVLMDetector(config={"api_key": "…"})  # explicit key

    detections = det.detect(rgb_frame, "blue bottle")

Env overrides:
    VECTOR_VLM_URL   — local endpoint (no auth header, no API key required)
    VECTOR_VLM_MODEL — override the model name
    OPENROUTER_API_KEY — remote API key (required when VECTOR_VLM_URL not set)
"""
from __future__ import annotations

import base64
import logging
import os
import threading
import time
from io import BytesIO
from typing import Any

import httpx
from PIL import Image

from vector_os_nano.core.types import Detection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL: str = "qwen/qwen2.5-vl-72b-instruct"
_TIMEOUT_S: float = 30.0
_MAX_RETRIES: int = 2
_JPEG_QUALITY: int = 50
_VLM_IMAGE_MAX_DIM: int = 160    # remote: keep base64 < 10 KB
_VLM_IMAGE_MAX_DIM_LOCAL: int = 512  # local: higher resolution OK

# Qwen2.5-VL cost model (OpenRouter, as of 2026-04)
_COST_PER_INPUT_TOKEN: float = 0.40e-6   # USD per input token
_COST_PER_OUTPUT_TOKEN: float = 1.20e-6  # USD per output token

_DETECT_PROMPT_TEMPLATE: str = (
    "You are a vision grounding model. Find all instances of the requested "
    "object in this image. Return ONLY a JSON array — no prose, no markdown "
    "fences. Use pixel coordinates (0..width, 0..height).\n\n"
    'Schema: [{{"label": string, "bbox": [x1, y1, x2, y2], "confidence": 0.0..1.0}}]\n'
    "Return [] if no matching object is found.\n"
    "Query: {query}"
)


# ---------------------------------------------------------------------------
# QwenVLMDetector
# ---------------------------------------------------------------------------


class QwenVLMDetector:
    """Vision-Language detector using Qwen2.5-VL-72B via OpenRouter.

    Thread-safe.  Cumulative cost tracked under a lock.

    Args:
        config: Optional dict with key ``api_key`` (str).
                Falls back to ``OPENROUTER_API_KEY`` environment variable.
                Pass nothing when ``VECTOR_VLM_URL`` is set (local mode).

    Raises:
        ValueError: If neither an API key nor ``VECTOR_VLM_URL`` is present.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        local_url: str | None = os.environ.get("VECTOR_VLM_URL")

        if local_url:
            self._base_url: str = local_url
            self._api_key: str = ""  # not sent
            self._model: str = os.environ.get("VECTOR_VLM_MODEL") or _DEFAULT_MODEL
            self._local: bool = True
            self._timeout: float = 45.0
            logger.info(
                "[vlm_qwen] local mode: %s model=%s", self._base_url, self._model
            )
        else:
            api_key: str | None = cfg.get("api_key") or os.environ.get(
                "OPENROUTER_API_KEY"
            )
            if not api_key:
                raise ValueError(
                    "QwenVLMDetector: API key not found. "
                    "Pass config={'api_key': '...'} or set OPENROUTER_API_KEY, "
                    "or set VECTOR_VLM_URL for a local endpoint."
                )
            self._api_key = api_key
            self._base_url = _OPENROUTER_BASE_URL
            self._model = os.environ.get("VECTOR_VLM_MODEL") or _DEFAULT_MODEL
            self._local = False
            self._timeout = _TIMEOUT_S

        self._cost_usd: float = 0.0
        self._cost_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: Any, query: str) -> list[Detection]:
        """Detect all instances of *query* in *image*.

        Args:
            image: (H, W, 3) uint8 RGB array.
            query: Natural-language object description, e.g. ``"blue bottle"``.

        Returns:
            List of :class:`Detection` with pixel bboxes.  Empty list if
            the model found no matching objects or JSON parse fails.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
        """
        prompt = _DETECT_PROMPT_TEMPLATE.format(query=query)
        raw = self._call_vlm(image, prompt)

        # Response may be a JSON array directly or wrapped in markdown fences.
        # _parse_json_response handles dicts; for arrays we need special casing.
        parsed = _parse_array_response(raw)

        h, w = image.shape[:2]
        results: list[Detection] = []
        for item in parsed:
            det = _detection_from_raw(item, w, h)
            if det is not None:
                results.append(det)
        return results

    @property
    def cumulative_cost_usd(self) -> float:
        """Total USD spent on API calls since this instance was created."""
        with self._cost_lock:
            return self._cost_usd

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_frame(self, frame: Any) -> str:
        """Encode an RGB frame as a base64 JPEG, resizing for API limits."""
        pil_image = Image.fromarray(frame)
        max_dim = _VLM_IMAGE_MAX_DIM_LOCAL if self._local else _VLM_IMAGE_MAX_DIM
        w, h = pil_image.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            pil_image = pil_image.resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS
            )
        buf = BytesIO()
        pil_image.save(buf, format="JPEG", quality=_JPEG_QUALITY)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _call_vlm(self, frame: Any, prompt: str) -> str:
        """POST to the VLM endpoint with retry logic.

        Retries up to _MAX_RETRIES times on transport/5xx errors.
        4xx errors raise immediately (no retry).

        Args:
            frame: (H, W, 3) uint8 RGB array.
            prompt: Text prompt sent alongside the image.

        Returns:
            Raw text from model's first choice.

        Raises:
            RuntimeError: After all retries are exhausted.
        """
        image_b64 = self._encode_frame(frame)
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if not self._local:
            headers["Authorization"] = f"Bearer {self._api_key}"

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            t_start = time.monotonic()
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(
                        f"{self._base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    )

                elapsed = time.monotonic() - t_start

                # 4xx — do not retry
                if 400 <= response.status_code < 500:
                    raise RuntimeError(
                        f"Qwen VLM client error {response.status_code}: "
                        f"{response.text}"
                    )

                response.raise_for_status()
                data = response.json()

                usage: dict[str, int] = data.get("usage") or {}
                input_tokens = int(usage.get("prompt_tokens", 0))
                output_tokens = int(usage.get("completion_tokens", 0))
                call_cost = (
                    input_tokens * _COST_PER_INPUT_TOKEN
                    + output_tokens * _COST_PER_OUTPUT_TOKEN
                )
                with self._cost_lock:
                    self._cost_usd += call_cost

                logger.info(
                    "[vlm_qwen] ok attempt=%d elapsed=%.2fs "
                    "in=%d out=%d cost=$%.6f",
                    attempt,
                    elapsed,
                    input_tokens,
                    output_tokens,
                    call_cost,
                )

                choices: list[dict[str, Any]] = data.get("choices") or []
                if not choices:
                    last_exc = RuntimeError("Qwen VLM returned no choices.")
                    continue

                return str(choices[0].get("message", {}).get("content", ""))

            except (httpx.TransportError, httpx.TimeoutException) as exc:
                elapsed = time.monotonic() - t_start
                logger.warning(
                    "[vlm_qwen] attempt %d/%d failed (%.2fs): %s",
                    attempt,
                    _MAX_RETRIES,
                    elapsed,
                    exc,
                )
                last_exc = exc

            except RuntimeError:
                raise

            except Exception as exc:
                elapsed = time.monotonic() - t_start
                logger.warning(
                    "[vlm_qwen] unexpected error attempt %d/%d (%.2fs): %s",
                    attempt,
                    _MAX_RETRIES,
                    elapsed,
                    exc,
                )
                last_exc = exc

        raise RuntimeError(
            f"Qwen VLM API failed after {_MAX_RETRIES} attempts. "
            f"Last error: {last_exc}"
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _parse_array_response(text: str) -> list[dict[str, Any]]:
    """Parse a JSON array from the model response.

    Handles:
    - Raw JSON array ``[...]``
    - Markdown-fenced ``\\`\\`\\`json\\n[...]\\n\\`\\`\\`\\``
    - Object with ``objects`` key (graceful fallback)

    Returns:
        List of raw dicts, or ``[]`` on parse failure.
    """
    import json
    import re

    stripped = text.strip()

    # Attempt 1: direct parse
    try:
        result = json.loads(stripped)
        if isinstance(result, list):
            return result  # type: ignore[return-value]
        if isinstance(result, dict):
            # Might be {"objects": [...]}
            return result.get("objects") or []
    except json.JSONDecodeError:
        pass

    # Attempt 2: ```json ... ``` block
    match = re.search(r"```json\s*(.*?)\s*```", stripped, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1))
            if isinstance(result, list):
                return result  # type: ignore[return-value]
        except json.JSONDecodeError:
            pass

    # Attempt 3: any ``` ... ``` block
    match = re.search(r"```\s*(.*?)\s*```", stripped, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1))
            if isinstance(result, list):
                return result  # type: ignore[return-value]
        except json.JSONDecodeError:
            pass

    logger.error("[vlm_qwen] failed to parse array from response: %.200s", text)
    return []


def _detection_from_raw(
    raw: dict[str, Any], img_w: int, img_h: int
) -> Detection | None:
    """Convert a raw VLM dict to a Detection, handling bbox normalisation.

    If all bbox coords are <= 1.0, they are treated as normalised and scaled
    to pixel space using ``img_w`` x ``img_h``.

    Confidence is clamped to [0.0, 1.0].

    Returns:
        Detection, or None if the raw dict is malformed.
    """
    try:
        label = str(raw.get("label", ""))
        confidence = float(raw.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        bbox_raw = raw.get("bbox") or []
        if len(bbox_raw) < 4:
            return None

        x1, y1, x2, y2 = (float(v) for v in bbox_raw[:4])

        # Normalised-coord detection: scale to pixels
        if max(x1, y1, x2, y2) <= 1.0:
            x1 *= img_w
            y1 *= img_h
            x2 *= img_w
            y2 *= img_h

        return Detection(label=label, bbox=(x1, y1, x2, y2), confidence=confidence)
    except (TypeError, ValueError, KeyError):
        logger.warning("[vlm_qwen] malformed detection entry: %s", raw)
        return None
