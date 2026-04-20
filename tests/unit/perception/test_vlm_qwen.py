"""Unit tests for QwenVLMDetector.

All tests mock httpx.Client — no real network calls are made.

Implementation note on imports:
    ``vector_os_nano/perception/__init__.py`` imports numpy-heavy modules
    (pipeline, tracker, realsense) that trigger coverage's C-tracer numpy
    reimport error on Ubuntu 24.04 / coverage 7.4.x.  We therefore load
    ``vlm_qwen`` and ``vlm_go2`` directly via ``importlib`` (bypassing the
    package ``__init__``) so that test collection and coverage can succeed
    without touching those heavy dependencies.

    ``vector_os_nano.core.types`` is safe (stdlib-only) and imported normally.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: load vlm_qwen directly without triggering perception/__init__
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parents[3]  # …/vector_os_nano/
_PERCEPTION = _REPO_ROOT / "vector_os_nano" / "perception"


def _load_direct(name: str, path: Path) -> types.ModuleType:
    """Load a .py file as a module without running parent package __init__."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"Cannot find {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _ensure_perception_modules() -> None:
    """Pre-register stub entries so vlm_qwen's package imports resolve.

    ``vector_os_nano/perception/__init__.py`` imports numpy-heavy modules
    (pipeline, tracker, realsense) that conflict with coverage's C tracer on
    Ubuntu 24.04 (numpy 2.x + coverage 7.4.x).  We bypass it by registering
    a minimal package stub and loading vlm_qwen.py directly via importlib.
    """
    # If the real module is already loaded (e.g., from a prior test run in
    # the same process), nothing to do.
    if "vector_os_nano.perception.vlm_qwen" in sys.modules:
        return

    # Ensure core.types is loaded (stdlib-only, safe).
    import vector_os_nano.core.types  # noqa: F401

    # Register a minimal stub for the perception *package* so that
    # ``import vector_os_nano.perception.vlm_qwen`` doesn't execute
    # the heavy __init__.py.
    if "vector_os_nano.perception" not in sys.modules:
        pkg_stub = types.ModuleType("vector_os_nano.perception")
        pkg_stub.__path__ = [str(_PERCEPTION)]  # type: ignore[assignment]
        pkg_stub.__package__ = "vector_os_nano.perception"
        sys.modules["vector_os_nano.perception"] = pkg_stub

    # Load vlm_qwen directly — it only imports httpx, PIL, and stdlib.
    # No numpy at module level (numpy is only used at runtime in _encode_frame
    # via PIL.Image.fromarray, which we mock in all tests).
    _load_direct(
        "vector_os_nano.perception.vlm_qwen",
        _PERCEPTION / "vlm_qwen.py",
    )


_ensure_perception_modules()

from vector_os_nano.perception.vlm_qwen import QwenVLMDetector  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeFrame:
    """Minimal stand-in for a numpy (H, W, 3) uint8 array.

    ``detect()`` reads ``frame.shape[:2]`` for bbox scaling, and
    ``_encode_frame`` calls ``Image.fromarray(frame)`` — but since we mock
    ``QwenVLMDetector._encode_frame`` in all tests, the PIL call never runs.
    """

    def __init__(self, h: int = 240, w: int = 320) -> None:
        self.shape = (h, w, 3)


def _make_frame(h: int = 240, w: int = 320) -> _FakeFrame:
    """Return a minimal frame stub suitable for mocked detect() calls."""
    return _FakeFrame(h, w)


_FAKE_B64 = "aGVsbG8="  # "hello" base64 — dummy encoded image


def _make_response(
    content: str,
    status_code: int = 200,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> MagicMock:
    """Build a fake httpx response object."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = content
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }
    resp.raise_for_status = MagicMock()
    return resp


def _make_client_cm(response: MagicMock) -> MagicMock:
    """Wrap a response in a context-manager-compatible httpx.Client mock."""
    client_instance = MagicMock()
    client_instance.post.return_value = response
    client_cm = MagicMock()
    client_cm.__enter__ = MagicMock(return_value=client_instance)
    client_cm.__exit__ = MagicMock(return_value=False)
    return client_cm


# ---------------------------------------------------------------------------
# Fixture: fresh QwenVLMDetector with a fake API key
# ---------------------------------------------------------------------------


@pytest.fixture()
def detector(monkeypatch: pytest.MonkeyPatch) -> QwenVLMDetector:
    """Return a QwenVLMDetector with a fake API key (no VECTOR_VLM_URL).

    Also patches _encode_frame so PIL/numpy are never invoked.
    """
    monkeypatch.delenv("VECTOR_VLM_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-abc")
    det = QwenVLMDetector()
    # Patch _encode_frame at the instance level — returns stable dummy b64
    monkeypatch.setattr(det, "_encode_frame", lambda frame: _FAKE_B64)
    return det


# ---------------------------------------------------------------------------
# Test 1 — parses plain JSON bbox list
# ---------------------------------------------------------------------------


def test_detect_parses_json_bbox_list(detector: QwenVLMDetector) -> None:
    payload = json.dumps(
        [{"label": "bottle", "bbox": [10, 20, 30, 40], "confidence": 0.9}]
    )
    resp = _make_response(payload)
    client_cm = _make_client_cm(resp)

    with patch("httpx.Client", return_value=client_cm):
        results = detector.detect(_make_frame(), "bottle")

    assert len(results) == 1
    det = results[0]
    assert det.label == "bottle"
    assert det.bbox == (10.0, 20.0, 30.0, 40.0)
    assert abs(det.confidence - 0.9) < 1e-6


# ---------------------------------------------------------------------------
# Test 2 — parses markdown-fenced JSON
# ---------------------------------------------------------------------------


def test_detect_parses_markdown_fenced_json(detector: QwenVLMDetector) -> None:
    inner = json.dumps(
        [{"label": "cup", "bbox": [5, 10, 50, 80], "confidence": 0.75}]
    )
    payload = f"```json\n{inner}\n```"
    resp = _make_response(payload)
    client_cm = _make_client_cm(resp)

    with patch("httpx.Client", return_value=client_cm):
        results = detector.detect(_make_frame(), "cup")

    assert len(results) == 1
    assert results[0].label == "cup"
    assert results[0].bbox == (5.0, 10.0, 50.0, 80.0)


# ---------------------------------------------------------------------------
# Test 3 — empty response returns empty list
# ---------------------------------------------------------------------------


def test_detect_empty_response_returns_empty_list(
    detector: QwenVLMDetector,
) -> None:
    resp = _make_response("[]")
    client_cm = _make_client_cm(resp)

    with patch("httpx.Client", return_value=client_cm):
        results = detector.detect(_make_frame(), "anything")

    assert results == []


# ---------------------------------------------------------------------------
# Test 4 — normalised bbox (coords <= 1.0) scaled to pixel space
# ---------------------------------------------------------------------------


def test_detect_scales_normalised_bbox_to_pixels(
    detector: QwenVLMDetector,
) -> None:
    # bbox coords all <= 1.0 → should be multiplied by (W, H, W, H)
    frame = _make_frame(h=240, w=320)
    payload = json.dumps(
        [{"label": "chair", "bbox": [0.1, 0.2, 0.5, 0.8], "confidence": 0.8}]
    )
    resp = _make_response(payload)
    client_cm = _make_client_cm(resp)

    with patch("httpx.Client", return_value=client_cm):
        results = detector.detect(frame, "chair")

    assert len(results) == 1
    x1, y1, x2, y2 = results[0].bbox
    assert abs(x1 - 0.1 * 320) < 1e-3
    assert abs(y1 - 0.2 * 240) < 1e-3
    assert abs(x2 - 0.5 * 320) < 1e-3
    assert abs(y2 - 0.8 * 240) < 1e-3


# ---------------------------------------------------------------------------
# Test 5 — confidence clamped to [0.0, 1.0]
# ---------------------------------------------------------------------------


def test_detect_clamps_confidence_to_unit_interval(
    detector: QwenVLMDetector,
) -> None:
    payload = json.dumps(
        [
            {"label": "box", "bbox": [0, 0, 10, 10], "confidence": 1.5},
            {"label": "table", "bbox": [10, 10, 20, 20], "confidence": -0.3},
        ]
    )
    resp = _make_response(payload)
    client_cm = _make_client_cm(resp)

    with patch("httpx.Client", return_value=client_cm):
        results = detector.detect(_make_frame(), "stuff")

    assert results[0].confidence == 1.0
    assert results[1].confidence == 0.0


# ---------------------------------------------------------------------------
# Test 6 — timeout retries then raises RuntimeError
# ---------------------------------------------------------------------------


def test_detect_timeout_retries_then_fails(detector: QwenVLMDetector) -> None:
    import httpx as _httpx

    client_instance = MagicMock()
    client_instance.post.side_effect = _httpx.TimeoutException("timed out")
    client_cm = MagicMock()
    client_cm.__enter__ = MagicMock(return_value=client_instance)
    client_cm.__exit__ = MagicMock(return_value=False)

    with patch("httpx.Client", return_value=client_cm):
        with pytest.raises(RuntimeError, match="failed after"):
            detector.detect(_make_frame(), "bottle")

    # Should have attempted _MAX_RETRIES times (2)
    assert client_instance.post.call_count == 2


# ---------------------------------------------------------------------------
# Test 7 — 4xx does NOT retry
# ---------------------------------------------------------------------------


def test_detect_4xx_does_not_retry(detector: QwenVLMDetector) -> None:
    resp = MagicMock()
    resp.status_code = 401
    resp.text = "Unauthorized"
    resp.raise_for_status = MagicMock()

    client_instance = MagicMock()
    client_instance.post.return_value = resp
    client_cm = MagicMock()
    client_cm.__enter__ = MagicMock(return_value=client_instance)
    client_cm.__exit__ = MagicMock(return_value=False)

    with patch("httpx.Client", return_value=client_cm):
        with pytest.raises(RuntimeError):
            detector.detect(_make_frame(), "bottle")

    # Must NOT retry on 4xx — exactly 1 post call
    assert client_instance.post.call_count == 1


# ---------------------------------------------------------------------------
# Test 8 — cumulative cost tracks across multiple calls
# ---------------------------------------------------------------------------


def test_detect_tracks_cumulative_cost(detector: QwenVLMDetector) -> None:
    # Cost rates from vlm_qwen.py:
    # input  = 0.40e-6 USD/tok
    # output = 1.20e-6 USD/tok
    prompt_tokens = 100
    completion_tokens = 50
    expected_per_call = (
        prompt_tokens * 0.40e-6 + completion_tokens * 1.20e-6
    )

    payload = "[]"
    resp = _make_response(
        payload,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    client_cm = _make_client_cm(resp)

    with patch("httpx.Client", return_value=client_cm):
        detector.detect(_make_frame(), "a")
        detector.detect(_make_frame(), "b")

    assert abs(detector.cumulative_cost_usd - expected_per_call * 2) < 1e-12


# ---------------------------------------------------------------------------
# Test 9 — VECTOR_VLM_URL env: no Auth header, uses VECTOR_VLM_MODEL override
# ---------------------------------------------------------------------------


def test_detect_honours_VECTOR_VLM_URL_env_no_auth_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VECTOR_VLM_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("VECTOR_VLM_MODEL", "qwen2.5vl:local")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    det = QwenVLMDetector()
    # Patch _encode_frame to avoid numpy/PIL
    monkeypatch.setattr(det, "_encode_frame", lambda frame: _FAKE_B64)

    payload = "[]"
    resp = _make_response(payload)
    client_instance = MagicMock()
    client_instance.post.return_value = resp
    client_cm = MagicMock()
    client_cm.__enter__ = MagicMock(return_value=client_instance)
    client_cm.__exit__ = MagicMock(return_value=False)

    with patch("httpx.Client", return_value=client_cm):
        det.detect(_make_frame(), "x")

    call_args = client_instance.post.call_args
    # Extract keyword args (headers, json)
    kwargs: dict[str, Any] = call_args[1] if call_args[1] else {}
    headers_sent: dict[str, str] = kwargs.get("headers", {})
    body_sent: dict[str, Any] = kwargs.get("json", {})

    assert "Authorization" not in headers_sent
    assert body_sent.get("model") == "qwen2.5vl:local"


# ---------------------------------------------------------------------------
# Test 10 — raises ValueError when neither API key nor local URL set
# ---------------------------------------------------------------------------


def test_detect_raises_if_no_api_key_and_no_local_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("VECTOR_VLM_URL", raising=False)

    with pytest.raises(ValueError, match="API key"):
        QwenVLMDetector()


# ---------------------------------------------------------------------------
# Coverage-boosting tests (uncovered branches from initial 10)
# ---------------------------------------------------------------------------


def test_detect_no_choices_retries_then_fails(
    detector: QwenVLMDetector,
) -> None:
    """API returns 200 but no 'choices' key → retries, then RuntimeError."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [], "usage": {}}

    client_instance = MagicMock()
    client_instance.post.return_value = resp
    client_cm = MagicMock()
    client_cm.__enter__ = MagicMock(return_value=client_instance)
    client_cm.__exit__ = MagicMock(return_value=False)

    with patch("httpx.Client", return_value=client_cm):
        with pytest.raises(RuntimeError):
            detector.detect(_make_frame(), "x")

    assert client_instance.post.call_count == 2  # retried once


def test_parse_array_response_dict_with_objects_key(
    detector: QwenVLMDetector,
) -> None:
    """Qwen sometimes returns {\"objects\": [...]} instead of a bare array."""
    inner_objs = [{"label": "lamp", "bbox": [5, 5, 50, 60], "confidence": 0.7}]
    payload = json.dumps({"objects": inner_objs})
    resp = _make_response(payload)
    client_cm = _make_client_cm(resp)

    with patch("httpx.Client", return_value=client_cm):
        results = detector.detect(_make_frame(), "lamp")

    assert len(results) == 1
    assert results[0].label == "lamp"


def test_parse_array_response_malformed_json_returns_empty(
    detector: QwenVLMDetector,
) -> None:
    """Completely unparseable response → empty list (no crash)."""
    resp = _make_response("not json at all")
    client_cm = _make_client_cm(resp)

    with patch("httpx.Client", return_value=client_cm):
        results = detector.detect(_make_frame(), "x")

    assert results == []


def test_detection_from_raw_short_bbox_returns_none(
    detector: QwenVLMDetector,
) -> None:
    """VLM returns bbox with < 4 elements → detection silently dropped."""
    payload = json.dumps(
        [{"label": "chair", "bbox": [10, 20], "confidence": 0.5}]
    )
    resp = _make_response(payload)
    client_cm = _make_client_cm(resp)

    with patch("httpx.Client", return_value=client_cm):
        results = detector.detect(_make_frame(), "chair")

    assert results == []


def test_detection_from_raw_type_error_returns_none(
    detector: QwenVLMDetector,
) -> None:
    """Bbox contains non-numeric values → detection silently dropped."""
    payload = json.dumps(
        [{"label": "box", "bbox": ["a", "b", "c", "d"], "confidence": 0.9}]
    )
    resp = _make_response(payload)
    client_cm = _make_client_cm(resp)

    with patch("httpx.Client", return_value=client_cm):
        results = detector.detect(_make_frame(), "box")

    assert results == []


def test_detect_unexpected_exception_retries_then_fails(
    detector: QwenVLMDetector,
) -> None:
    """Non-transport, non-RuntimeError exception → caught, retried, then RuntimeError."""
    client_instance = MagicMock()
    client_instance.post.side_effect = OSError("socket error")
    client_cm = MagicMock()
    client_cm.__enter__ = MagicMock(return_value=client_instance)
    client_cm.__exit__ = MagicMock(return_value=False)

    with patch("httpx.Client", return_value=client_cm):
        with pytest.raises(RuntimeError, match="failed after"):
            detector.detect(_make_frame(), "x")

    assert client_instance.post.call_count == 2


def test_parse_array_response_generic_fenced_block(
    detector: QwenVLMDetector,
) -> None:
    """Response wrapped in generic ``` block (no 'json' tag) — path 3."""
    inner = json.dumps([{"label": "pen", "bbox": [1, 2, 3, 4], "confidence": 0.6}])
    payload = f"```\n{inner}\n```"
    resp = _make_response(payload)
    client_cm = _make_client_cm(resp)

    with patch("httpx.Client", return_value=client_cm):
        results = detector.detect(_make_frame(), "pen")

    assert len(results) == 1
    assert results[0].label == "pen"


def test_parse_array_response_fenced_invalid_json_falls_through(
    detector: QwenVLMDetector,
) -> None:
    """```json block contains invalid JSON → fallback to generic block."""
    payload = "```json\nnot valid json\n```"
    resp = _make_response(payload)
    client_cm = _make_client_cm(resp)

    with patch("httpx.Client", return_value=client_cm):
        results = detector.detect(_make_frame(), "x")

    # No parseable array → empty list
    assert results == []


def test_encode_frame_small_image_no_resize(monkeypatch: pytest.MonkeyPatch) -> None:
    """_encode_frame: image within max_dim → no resize, returns b64 string."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("VECTOR_VLM_URL", raising=False)
    det = QwenVLMDetector()

    # Image size within _VLM_IMAGE_MAX_DIM (160) — no resize path
    fake_img = MagicMock()
    fake_img.size = (100, 80)  # max(100, 80) = 100 <= 160
    fake_buf = MagicMock()
    fake_buf.getvalue.return_value = b"fakeimagedata"

    with patch(
        "vector_os_nano.perception.vlm_qwen.Image.fromarray",
        return_value=fake_img,
    ), patch("vector_os_nano.perception.vlm_qwen.BytesIO", return_value=fake_buf):
        result = det._encode_frame(_make_frame())

    assert isinstance(result, str)
    assert len(result) > 0
    # resize() should NOT have been called
    fake_img.resize.assert_not_called()


def test_encode_frame_large_image_triggers_resize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_encode_frame: image exceeding max_dim → resized, returns b64 string."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("VECTOR_VLM_URL", raising=False)
    det = QwenVLMDetector()

    # Image size exceeds _VLM_IMAGE_MAX_DIM (160) — resize path (163-164)
    resized_img = MagicMock()
    resized_img.size = (160, 90)
    fake_img = MagicMock()
    fake_img.size = (640, 360)  # max(640,360)=640 > 160 → resize needed
    fake_img.resize.return_value = resized_img
    fake_buf = MagicMock()
    fake_buf.getvalue.return_value = b"resizedimagedata"

    with patch(
        "vector_os_nano.perception.vlm_qwen.Image.fromarray",
        return_value=fake_img,
    ), patch("vector_os_nano.perception.vlm_qwen.BytesIO", return_value=fake_buf):
        result = det._encode_frame(_make_frame())

    assert isinstance(result, str)
    assert len(result) > 0
    # resize() SHOULD have been called with (int(640*160/640), int(360*160/640))
    fake_img.resize.assert_called_once()
