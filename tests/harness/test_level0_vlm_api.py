# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Level 0 — VLM API connectivity verification.

Tests at this level verify that GPT-4o vision can be reached via OpenRouter
and returns valid structured responses. No MuJoCo, no ROS2, no simulation.

Uses a synthetic test image (solid color with text overlay) to avoid any
simulation dependency. The only requirement is a valid OPENROUTER_API_KEY.

Cost: ~$0.01 per full run (2-3 API calls with small images).
"""
from __future__ import annotations

import base64
import io
import json
import os
import time

import numpy as np
import pytest

# Billed VLM calls: opt in with VECTOR_LIVE_LLM=1 (a key in the env or
# config is NOT consent; tests:missed2).
pytestmark = pytest.mark.live_llm

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_API_KEY: str | None = None


def _get_api_key() -> str:
    """Load OpenRouter API key from config or environment."""
    global _API_KEY
    if _API_KEY is not None:
        return _API_KEY

    # Try environment first
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        # Try config/user.yaml
        try:
            import yaml
            cfg_path = (
                __import__("pathlib").Path(__file__).resolve().parents[2]
                / "config" / "user.yaml"
            )
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            key = cfg.get("llm", {}).get("api_key", "")
        except Exception:
            pass

    _API_KEY = key
    return key


def _make_test_image(width: int = 320, height: int = 240) -> np.ndarray:
    """Create a synthetic test image — a room-like scene with colored blocks.

    Returns (H, W, 3) uint8 RGB array.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Floor (brown)
    img[height // 2:, :] = [139, 90, 43]
    # Wall (beige)
    img[:height // 2, :] = [222, 207, 180]
    # "Window" (blue rectangle)
    img[20:80, 100:220] = [135, 206, 235]
    # "Table" (dark brown rectangle)
    img[140:180, 80:240] = [101, 67, 33]
    # "Chair" (small red block)
    img[150:190, 250:290] = [178, 34, 34]

    return img


def _encode_image_b64(frame: np.ndarray) -> str:
    """Encode RGB numpy array as base64 JPEG."""
    from PIL import Image

    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLevel0VLMApi:
    """L0: Verify GPT-4o vision API is reachable and returns valid responses."""

    @pytest.fixture(autouse=True)
    def _require_api_key(self):
        key = _get_api_key()
        if not key:
            pytest.skip("No OPENROUTER_API_KEY available")

    def test_api_reachable(self):
        """API returns a 200 response for a simple vision request."""
        import httpx

        key = _get_api_key()
        frame = _make_test_image()
        b64 = _encode_image_b64(frame)

        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What do you see? Reply in one sentence.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}",
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 150,
            },
            timeout=30.0,
        )

        assert response.status_code == 200, f"API returned {response.status_code}: {response.text}"
        data = response.json()
        assert "choices" in data
        text = data["choices"][0]["message"]["content"]
        assert len(text) > 5, f"Response too short: {text!r}"

    def test_json_structured_response(self):
        """API can return structured JSON when prompted."""
        import httpx

        key = _get_api_key()
        frame = _make_test_image()
        b64 = _encode_image_b64(frame)

        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Describe this image. Respond ONLY in JSON format: "
                                    '{"summary": "one sentence", "objects": ["item1", "item2"]}'
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}",
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 200,
            },
            timeout=30.0,
        )

        assert response.status_code == 200
        text = response.json()["choices"][0]["message"]["content"]

        # Parse JSON — strip markdown fences if present
        import re
        clean = text.strip()
        parsed = None
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            # Try extracting from markdown fence
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", clean, re.DOTALL)
            if match:
                parsed = json.loads(match.group(1))
            else:
                # Try finding any JSON object in text
                match = re.search(r"\{.*\}", clean, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
        assert "summary" in parsed, f"Missing 'summary' in response: {parsed}"
        assert "objects" in parsed, f"Missing 'objects' in response: {parsed}"
        assert isinstance(parsed["objects"], list)

    def test_response_latency(self):
        """API responds within 15 seconds for a simple vision query."""
        import httpx

        key = _get_api_key()
        frame = _make_test_image()
        b64 = _encode_image_b64(frame)

        start = time.monotonic()
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What color is the largest area? One word."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 20,
            },
            timeout=15.0,
        )
        elapsed = time.monotonic() - start

        assert response.status_code == 200
        assert elapsed < 15.0, f"API took {elapsed:.1f}s (limit: 15s)"

    def test_cost_tracking(self):
        """API response includes usage data for cost estimation."""
        import httpx

        key = _get_api_key()
        frame = _make_test_image()
        b64 = _encode_image_b64(frame)

        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Say 'ok'."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 10,
            },
            timeout=15.0,
        )

        assert response.status_code == 200
        data = response.json()
        usage = data.get("usage", {})
        # OpenRouter should return token counts
        assert "prompt_tokens" in usage or "total_tokens" in usage, (
            f"No usage data in response: {list(data.keys())}"
        )
