# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""GPT-4o Vision-Language Model perception for the Go2 quadruped robot.

Calls the OpenRouter API (https://openrouter.ai/api/v1) with model
openai/gpt-4o to analyze camera frames.  No ROS2 dependency — pure Python.

Typical usage::

    from vector_os_nano.perception.vlm_go2 import Go2VLMPerception

    vlm = Go2VLMPerception()                       # key from env
    vlm = Go2VLMPerception(config={"api_key": ".."})  # key from config

    scene = vlm.describe_scene(rgb_frame)
    room  = vlm.identify_room(rgb_frame)
    objs  = vlm.find_objects(rgb_frame, query="chair")
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import httpx
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class VlmRateLimitError(RuntimeError):
    """Raised when the VLM endpoint returns HTTP 429 (rate limited).

    Subclass of RuntimeError so existing ``except RuntimeError`` paths (and the
    broad ``except Exception`` swallow in VlmTargetDetector) are unchanged; a
    caller that wants to DISTINGUISH a throttled endpoint from "object absent"
    can catch this type instead of blind-walking on an empty detection."""


# ---------------------------------------------------------------------------
# Cost model — GPT-4o via OpenRouter (as of 2026-04)
# ---------------------------------------------------------------------------
_COST_PER_INPUT_TOKEN: float = 2.50 / 1_000_000   # USD per input token
_COST_PER_OUTPUT_TOKEN: float = 10.00 / 1_000_000  # USD per output token

_OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
# DQ-6 (owner 2026-06-12): the VLM backbone is a 72B-class Qwen VL via
# OpenRouter — never a weak-vision text model. Owner sized it "72B 就够了";
# Qwen3-VL has no 72B dense on OpenRouter, so the 72B slot is
# qwen2.5-vl-72b-instruct (probed valid). Override: VECTOR_VLM_MODEL_OPENROUTER.
_MODEL: str = os.environ.get("VECTOR_VLM_MODEL_OPENROUTER", "") or "qwen/qwen2.5-vl-72b-instruct"
_TIMEOUT_S: float = 30.0
_MAX_RETRIES: int = 2
_JPEG_QUALITY: int = 50
_VLM_IMAGE_MAX_DIM: int = int(os.environ.get("VECTOR_VLM_MAX_DIM", "") or 512)
# ^ remote downsize, env-overridable. Was 160 (tiny payload), but campaign #12 R4
# proved that starved photoreal-VLN grounding on BOTH embodiments — a far target is
# only a few pixels at 160, so the VLM mis-grounded onto near objects/obstacles
# (Go2 AND G1 both ~2/5 arrival). At 512 the locate error dropped 2.1 m -> 0.29 m
# (the chair is grounded correctly every run); OpenRouter handles the larger inline
# image fine (~1-2 s). Matches the local-model dim.
_VLM_IMAGE_MAX_DIM_LOCAL: int = 512  # local models can handle larger images

# ---------------------------------------------------------------------------
# Local VLM backend (Ollama) — env var overrides
# Set VECTOR_VLM_URL=http://localhost:11434/v1 to use local Ollama
# Set VECTOR_VLM_MODEL=gemma4:e4b to select the local model
# ---------------------------------------------------------------------------
_LOCAL_VLM_URL: str | None = os.environ.get("VECTOR_VLM_URL")
_LOCAL_VLM_MODEL: str | None = os.environ.get("VECTOR_VLM_MODEL")
_USE_LOCAL_VLM: bool = _LOCAL_VLM_URL is not None


# ---------------------------------------------------------------------------
# Result dataclasses (immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectedObject:
    """A single object identified by the VLM."""

    name: str
    description: str
    confidence: float  # 0.0 – 1.0


@dataclass(frozen=True)
class SceneDescription:
    """Full scene analysis returned by describe_scene()."""

    summary: str                    # One-line summary
    objects: list[DetectedObject]
    room_type: str                  # e.g. "kitchen", "bedroom", "hallway"
    details: str                    # Full free-form description


@dataclass(frozen=True)
class RoomIdentification:
    """Room classification returned by identify_room()."""

    room: str           # Canonical room name (kitchen, living_room, …)
    confidence: float   # 0.0 – 1.0
    reasoning: str      # Model's rationale


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_DESCRIBE_SCENE_PROMPT: str = (
    "You are a robot dog's vision system (camera at 30cm height). "
    "Describe what you see in this first-person camera view from inside a house. "
    "Focus on: furniture (sofas, tables, beds, desks), appliances, floor type, "
    "wall colors, and any distinctive objects. These help identify which room this is. "
    'Respond ONLY in JSON: {"summary": "...", "objects": [{"name": "...", '
    '"description": "...", "confidence": 0.9}], "room_type": "...", "details": "..."}'
)

_IDENTIFY_ROOM_PROMPT: str = (
    "You are a robot dog (30cm tall) navigating a house. "
    "Based on this first-person camera view, identify which room you are in. "
    "Room features to look for:\n"
    "- living_room: blue sofa, coffee table, TV, rug, bookshelves\n"
    "- dining_room: dining table with chairs, pendant light\n"
    "- kitchen: counter, island, fridge (steel), range hood\n"
    "- study: desk with monitor, office chair, bookshelf\n"
    "- master_bedroom: large bed with frame, wardrobe, nightstands\n"
    "- guest_bedroom: bed, dresser\n"
    "- bathroom: bathtub, vanity, small tiles on floor\n"
    "- hallway: open corridor, no major furniture, connecting space\n\n"
    "Look at furniture, floor texture, colors, and objects to decide. "
    'Respond ONLY in JSON: {"room": "...", "confidence": 0.9, "reasoning": "..."}'
)

_FIND_OBJECTS_PROMPT_BASE: str = (
    "List all objects visible in this robot camera view"
)
_FIND_OBJECTS_PROMPT_SUFFIX: str = (
    '. Respond in JSON: {"objects": [{"name": "...", "description": "...", '
    '"confidence": 0.0}]}'
)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class Go2VLMPerception:
    """Vision-Language Model perception interface for the Go2 robot.

    Thread-safe.  Multiple threads may call describe_scene / identify_room /
    find_objects concurrently.  Cumulative cost is tracked under a lock.

    Args:
        config: Optional dict with keys ``api_key`` (str).
                Falls back to environment variable OPENROUTER_API_KEY.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}

        # Local VLM (Ollama) — no API key needed
        if _USE_LOCAL_VLM:
            self._api_key: str = "ollama"  # dummy, not sent
            self._base_url: str = _LOCAL_VLM_URL  # type: ignore[assignment]
            self._model: str = _LOCAL_VLM_MODEL or "gemma4:e4b"
            self._local: bool = True
            self._timeout: float = 45.0  # local: first call loads model (~30s), retry handles it
            logger.info("VLM: local Ollama at %s model=%s", self._base_url, self._model)
        else:
            api_key: str | None = cfg.get("api_key") or os.environ.get(
                "OPENROUTER_API_KEY"
            )
            if not api_key:
                raise ValueError(
                    "OpenRouter API key not found. "
                    "Pass config={'api_key': '...'} or set OPENROUTER_API_KEY."
                )
            self._api_key = api_key
            self._base_url = _OPENROUTER_BASE_URL
            self._model = _MODEL
            self._local = False
            self._timeout: float = _TIMEOUT_S

        # Per-instance image downsize (campaign #12 M4 DEBT-1): defaults to the
        # module global (512 remote / 512 local), but a caller can opt DOWN — e.g.
        # vlm_seek's 70-iter servo loop passes max_dim=256 (it approaches a target
        # and stalls on progress, it does NOT need far-locate precision), keeping
        # the recognize_navigate LOCATE path at the proven 512. Single-source: the
        # global stays the default; this is an additive opt-down (rule 6).
        self._max_dim: int = int(cfg.get("max_dim") or (
            _VLM_IMAGE_MAX_DIM_LOCAL if self._local else _VLM_IMAGE_MAX_DIM))
        self._cost_usd: float = 0.0
        self._cost_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def describe_scene(self, frame: np.ndarray) -> SceneDescription:
        """Analyze a camera frame and return a full scene description.

        Args:
            frame: (H, W, 3) uint8 RGB array.

        Returns:
            SceneDescription with summary, objects, room_type, and details.
        """
        raw = self._call_vlm(frame, _DESCRIBE_SCENE_PROMPT)
        data = _parse_json_response(raw)

        raw_objects: list[dict[str, Any]] = data.get("objects") or []
        objects = [_parse_detected_object(o) for o in raw_objects]

        return SceneDescription(
            summary=str(data.get("summary", "")),
            objects=objects,
            room_type=str(data.get("room_type", "unknown")),
            details=str(data.get("details", "")),
        )

    def identify_room(self, frame: np.ndarray) -> RoomIdentification:
        """Identify which room the robot is currently in.

        Args:
            frame: (H, W, 3) uint8 RGB array.

        Returns:
            RoomIdentification with room name, confidence, and reasoning.
        """
        raw = self._call_vlm(frame, _IDENTIFY_ROOM_PROMPT)
        data = _parse_json_response(raw)

        return RoomIdentification(
            room=str(data.get("room", "unknown")),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=str(data.get("reasoning", "")),
        )

    def find_objects(
        self, frame: np.ndarray, query: str | None = None
    ) -> list[DetectedObject]:
        """List objects visible in the camera frame.

        Args:
            frame: (H, W, 3) uint8 RGB array.
            query: Optional filter string, e.g. ``"chair"`` or ``"red objects"``.
                   If omitted, all visible objects are returned.

        Returns:
            List of DetectedObject sorted by descending confidence.
        """
        if query:
            prompt = (
                f"{_FIND_OBJECTS_PROMPT_BASE} that match: {query}"
                f"{_FIND_OBJECTS_PROMPT_SUFFIX}"
            )
        else:
            prompt = _FIND_OBJECTS_PROMPT_BASE + _FIND_OBJECTS_PROMPT_SUFFIX

        raw = self._call_vlm(frame, prompt)
        data = _parse_json_response(raw)

        raw_objects: list[dict[str, Any]] = data.get("objects") or []
        objects = [_parse_detected_object(o) for o in raw_objects]
        return sorted(objects, key=lambda o: o.confidence, reverse=True)

    # ------------------------------------------------------------------
    # Cost tracking
    # ------------------------------------------------------------------

    @property
    def cumulative_cost_usd(self) -> float:
        """Total USD spent on API calls since this instance was created."""
        with self._cost_lock:
            return self._cost_usd

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_frame(self, frame: np.ndarray) -> str:
        """Encode a numpy RGB frame as a base64 JPEG string.

        Resizes to at most _VLM_IMAGE_MAX_DIM on the longest side (default 512,
        ~a few-KB to ~20KB base64). Smaller is faster but starves grounding of a
        distant target — 512 is the campaign #12 balance (160 mis-grounded far
        objects); OpenRouter handles it reliably (~1-2s).

        Args:
            frame: (H, W, 3) uint8 RGB array.

        Returns:
            Base64-encoded JPEG bytes (no data URI prefix).
        """
        pil_image = Image.fromarray(frame)
        # Resize if larger than max dimension (per-instance, DEBT-1)
        max_dim = self._max_dim
        w, h = pil_image.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            pil_image = pil_image.resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS,
            )
        buf = BytesIO()
        pil_image.save(buf, format="JPEG", quality=_JPEG_QUALITY)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _call_vlm(self, frame: np.ndarray, prompt: str) -> str:
        """Send a vision request to GPT-4o via OpenRouter.

        Retries up to _MAX_RETRIES times on transient failures (network errors,
        5xx status codes).  4xx errors (bad auth, rate limit) are not retried.

        Args:
            frame: (H, W, 3) uint8 RGB array.
            prompt: System + user prompt to send alongside the image.

        Returns:
            Raw text content from the model's first choice.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
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

                # 4xx — auth/quota errors, do not retry. A 429 (rate limited) is
                # classified as a TYPED error so recognize_navigate can give up
                # honestly instead of mistaking the throttle for "object absent".
                # 529 (overloaded) is 5xx and intentionally keeps its retry path.
                if response.status_code == 429:
                    raise VlmRateLimitError(
                        f"OpenRouter rate limited (429): {response.text[:200]}"
                    )
                if 400 <= response.status_code < 500:
                    raise RuntimeError(
                        f"OpenRouter API client error {response.status_code}: "
                        f"{response.text}"
                    )

                response.raise_for_status()
                data = response.json()

                usage: dict[str, int] = data.get("usage") or {}
                input_tokens: int = int(usage.get("prompt_tokens", 0))
                output_tokens: int = int(usage.get("completion_tokens", 0))
                call_cost = (
                    input_tokens * _COST_PER_INPUT_TOKEN
                    + output_tokens * _COST_PER_OUTPUT_TOKEN
                )

                with self._cost_lock:
                    self._cost_usd += call_cost

                logger.info(
                    "VLM call ok | attempt=%d elapsed=%.2fs "
                    "input_tokens=%d output_tokens=%d cost=$%.6f",
                    attempt,
                    elapsed,
                    input_tokens,
                    output_tokens,
                    call_cost,
                )

                choices: list[dict[str, Any]] = data.get("choices") or []
                if not choices:
                    logger.warning(
                        "OpenRouter returned no choices (attempt %d/%d) — retrying",
                        attempt,
                        _MAX_RETRIES,
                    )
                    last_exc = RuntimeError("OpenRouter returned no choices.")
                    continue

                return str(
                    choices[0].get("message", {}).get("content", "")
                )

            except (httpx.TransportError, httpx.TimeoutException) as exc:
                elapsed = time.monotonic() - t_start
                logger.warning(
                    "VLM call failed (attempt %d/%d, %.2fs): %s",
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
                    "VLM call unexpected error (attempt %d/%d, %.2fs): %s",
                    attempt,
                    _MAX_RETRIES,
                    elapsed,
                    exc,
                )
                last_exc = exc

        raise RuntimeError(
            f"VLM API failed after {_MAX_RETRIES} attempts. "
            f"Last error: {last_exc}"
        )


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions)
# ---------------------------------------------------------------------------


def _parse_json_response(text: str) -> dict[str, Any]:
    """Parse a JSON object from the model response.

    Tries direct json.loads first.  Falls back to extracting a ```json
    fenced block if the model wrapped the JSON in Markdown.

    Args:
        text: Raw model output string.

    Returns:
        Parsed dict.  Returns empty dict on any parse failure.
    """
    stripped = text.strip()
    try:
        return json.loads(stripped)  # type: ignore[return-value]
    except json.JSONDecodeError:
        pass

    # Try ```json ... ``` block
    match = re.search(r"```json\s*(.*?)\s*```", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))  # type: ignore[return-value]
        except json.JSONDecodeError:
            pass

    # Try any ``` ... ``` block
    match = re.search(r"```\s*(.*?)\s*```", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))  # type: ignore[return-value]
        except json.JSONDecodeError:
            pass

    # Last resort: extract the FIRST balanced {...} object and ignore anything
    # around it. Handles an unclosed ```json fence (no trailing ```) and the
    # trailing-garbage Qwen sometimes appends (e.g. `{...}']` — campaign #9 R1),
    # both of which defeat json.loads on the whole string but leave a valid
    # object embedded. Brace counting respects string literals/escapes.
    obj = _first_balanced_object(stripped)
    if obj is not None:
        try:
            return json.loads(obj)  # type: ignore[return-value]
        except json.JSONDecodeError:
            pass

    logger.error("Failed to parse JSON from VLM response: %.200s", text)
    return {}


def _first_balanced_object(s: str) -> "str | None":
    """Return the first balanced ``{...}`` substring of *s*, or None.

    Tracks brace depth while skipping over string literals (so braces inside
    quotes don't miscount) and backslash escapes. Tolerates leading fence
    markers / trailing junk that make whole-string json.loads fail."""
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def _parse_detected_object(raw: dict[str, Any]) -> DetectedObject:
    """Construct a DetectedObject from a raw dict, clamping confidence.

    Args:
        raw: Dict with keys ``name``, ``description``, ``confidence``.

    Returns:
        DetectedObject with confidence clamped to [0.0, 1.0].
    """
    confidence = float(raw.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    return DetectedObject(
        name=str(raw.get("name", "")),
        description=str(raw.get("description", "")),
        confidence=confidence,
    )
