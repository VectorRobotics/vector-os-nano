# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Vision-language model scene perception for a camera-bearing base (optional).

An OpenAI-compatible chat endpoint selected by ``perception.scene_vlm`` in
config.yaml. A local server can declare ``api_key_env: null`` for unauthenticated calls.
It captions the scene, suggests the room and names objects; it is never required
(the detector over the world vocabulary discovers objects on its own). Frames go
out at up to 640 px, so small objects stay visible. No ROS2 dependency.

Typical usage::

    from vector_os_nano.perception.vlm_go2 import Go2VLMPerception

    vlm = Go2VLMPerception()                          # model config + environment credentials
    vlm = Go2VLMPerception(config={"api_key": "..", "rooms": ["kitchen", "hallway"]})

    scene = vlm.describe_scene(rgb_frame)
    room  = vlm.identify_room(rgb_frame)
    objs  = vlm.find_objects(rgb_frame, query="chair")
"""
from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any

import httpx
import numpy as np
from PIL import Image
from vector_os_nano.schemas import paths, user

logger = logging.getLogger(__name__)

_TIMEOUT_S: float = 30.0
_MAX_RETRIES: int = 2
_JPEG_QUALITY: int = 80
_VLM_IMAGE_MAX_DIM: int = 640  # >= 512 px: a 6 cm can at 2 m is ~20 px, not ~5


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
    "You are a mobile robot's vision system (a low, forward-facing head camera). "
    "Describe what you see in this first-person camera view from inside a house. "
    "Focus on: furniture (sofas, tables, beds, desks), appliances, floor type, "
    "wall colors, and any distinctive objects. These help identify which room this is. "
    'Respond ONLY in JSON: {"summary": "...", "objects": [{"name": "...", '
    '"description": "...", "confidence": 0.9}], "room_type": "...", "details": "..."}'
)

_IDENTIFY_ROOM_PROMPT: str = (
    "You are a mobile robot navigating a house. Based on this first-person camera "
    "view, identify which room you are in{rooms}. Look at furniture, floor texture, "
    "colors, and objects to decide. "
    'Respond ONLY in JSON: {{"room": "...", "confidence": 0.9, "reasoning": "..."}}'
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
        config: Optional dict: ``api_key`` (else OPENROUTER_API_KEY), ``model``,
            ``rooms`` (the room names the room question may answer), and
            ``price_in``/``price_out`` (USD per 1M tokens; cost tracking is 0
            unless they are given). ``base_url`` selects the server and
            ``api_key_env`` names its credential variable (null for no authentication).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        selected = user.load(paths.config_file()).perception.scene_vlm if config is None else user.SceneVlm()
        if selected is None:
            raise ValueError('scene VLM disabled by perception.scene_vlm: null')
        cfg = {**asdict(selected), **(config or {})}
        rooms = [str(r) for r in cfg.get("rooms") or ()]
        self._room_prompt = _IDENTIFY_ROOM_PROMPT.format(
            rooms=f" (one of: {', '.join(rooms)}; else \"unknown\")" if rooms else "")
        self._price = (float(cfg.get("price_in", 0.0)) / 1e6, float(cfg.get("price_out", 0.0)) / 1e6)

        key_env = cfg['api_key_env']
        self._api_key = cfg.get('api_key') or (os.environ.get(key_env, '') if key_env else '')
        if key_env and not self._api_key:
            raise ValueError(f'no {key_env} set for scene VLM; set the key in the environment')
        self._base_url = cfg['base_url'].rstrip('/')
        self._model = cfg['model']
        self._local = not bool(self._api_key)
        self._timeout = 45.0 if self._local else _TIMEOUT_S

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

        objects = _parse_objects(data.get("objects"))

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
        raw = self._call_vlm(frame, self._room_prompt)
        data = _parse_json_response(raw)

        return RoomIdentification(
            room=str(data.get("room") or "unknown"),
            confidence=_confidence(data.get("confidence")),
            reasoning=str(data.get("reasoning") or ""),
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

        objects = _parse_objects(data.get("objects"))
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

        Resizes to at most _VLM_IMAGE_MAX_DIM (640 px) on the longest side.

        Args:
            frame: (H, W, 3) uint8 RGB array.

        Returns:
            Base64-encoded JPEG bytes (no data URI prefix).
        """
        pil_image = Image.fromarray(frame)
        # Resize if larger than max dimension
        max_dim = _VLM_IMAGE_MAX_DIM
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
        """Send a vision request to the configured chat endpoint.

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

                # 4xx — auth/quota errors, do not retry
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
                call_cost = input_tokens * self._price[0] + output_tokens * self._price[1]

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
    """The first JSON object in a model reply ({} when there is none).

    Accepts a bare object, a fenced block, or an object inside prose ("Here is the
    JSON: {...}"): the first balanced ``{...}`` that parses. A non-object root (a
    list, a string) gives {}.
    """
    stripped = (text or "").strip()
    try:
        data = json.loads(stripped)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(stripped):
        if ch != "{":
            continue
        try:
            data, _end = decoder.raw_decode(stripped, i)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    logger.error("Failed to parse JSON from VLM response: %.200s", text)
    return {}


def _confidence(v: Any) -> float:
    """A confidence clamped to [0, 1]; null, words ("high") and junk are 0.0."""
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


def _parse_objects(raw: Any) -> list[DetectedObject]:
    """DetectedObjects from a reply's ``objects`` list; entries that are not dicts are skipped."""
    return [_parse_detected_object(o) for o in (raw if isinstance(raw, list) else []) if isinstance(o, dict)]


def _parse_detected_object(raw: dict[str, Any]) -> DetectedObject:
    """Construct a DetectedObject from a raw dict, clamping confidence."""
    return DetectedObject(
        name=str(raw.get("name") or ""),
        description=str(raw.get("description") or ""),
        confidence=_confidence(raw.get("confidence")),
    )
