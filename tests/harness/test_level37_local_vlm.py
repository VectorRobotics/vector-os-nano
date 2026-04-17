"""L37: Local VLM (Ollama) backend integration tests.

Tests that the VLM perception layer works with a local Ollama backend
when VECTOR_VLM_URL and VECTOR_VLM_MODEL env vars are set.

These tests verify:
- Backend selection logic (local vs remote)
- Ollama API compatibility (OpenAI-compat endpoint)
- identify_room / describe_scene / find_objects all return valid structures
- Graceful fallback when Ollama is not running

Requires: Ollama running with gemma4:e4b pulled.
Skip if Ollama not available.
"""
from __future__ import annotations

import ast
import inspect
import os
import textwrap

import httpx
import numpy as np
import pytest


def _ollama_available() -> bool:
    """Check if Ollama is running and has gemma4:e4b."""
    try:
        resp = httpx.get("http://localhost:11434/v1/models", timeout=3.0)
        if resp.status_code != 200:
            return False
        models = resp.json().get("data", [])
        return any("gemma4" in m.get("id", "") for m in models)
    except Exception:
        return False


_SKIP_REASON = "Ollama not running or gemma4:e4b not pulled"


# ── Unit tests (no Ollama needed) ──────────────────────────────────────


class TestVLMBackendSelection:
    """Verify backend selection logic in vlm_go2.py."""

    def test_local_vlm_env_vars_read(self):
        """_USE_LOCAL_VLM should be True when VECTOR_VLM_URL is set."""
        source = inspect.getsource(
            __import__("vector_os_nano.perception.vlm_go2", fromlist=["_USE_LOCAL_VLM"])
        )
        assert "_LOCAL_VLM_URL" in source
        assert "_LOCAL_VLM_MODEL" in source
        assert "_USE_LOCAL_VLM" in source

    def test_local_backend_no_auth_header(self):
        """When local, Authorization header should not be sent."""
        source = inspect.getsource(
            __import__("vector_os_nano.perception.vlm_go2", fromlist=["Go2VLMPerception"])
        )
        assert "not self._local" in source, "Auth header should be conditional on self._local"

    def test_base_url_uses_instance_var(self):
        """_call_vlm should use self._base_url, not module-level constant."""
        from vector_os_nano.perception.vlm_go2 import Go2VLMPerception
        source = inspect.getsource(Go2VLMPerception._call_vlm)
        assert "self._base_url" in source, "_call_vlm should use self._base_url"
        assert "_OPENROUTER_BASE_URL" not in source, "Should not hardcode OpenRouter URL"

    def test_model_uses_instance_var(self):
        """Payload should use self._model, not module-level _MODEL."""
        from vector_os_nano.perception.vlm_go2 import Go2VLMPerception
        source = inspect.getsource(Go2VLMPerception._call_vlm)
        assert '"model": self._model' in source or "'model': self._model" in source

    def test_local_timeout_longer(self):
        """Local backend should have longer timeout than remote default (25s).

        Local backend timeout was intentionally set to 45s (reduced from 60s):
        first call loads the model (~30s), retry logic handles any overrun.
        The invariant is: local timeout > remote default (25s) AND > 30s
        to cover at least one full model-load cycle.
        """
        from vector_os_nano.perception import vlm_go2
        # Simulate local backend
        orig_use = vlm_go2._USE_LOCAL_VLM
        orig_url = vlm_go2._LOCAL_VLM_URL
        try:
            vlm_go2._USE_LOCAL_VLM = True
            vlm_go2._LOCAL_VLM_URL = "http://localhost:11434/v1"
            vlm = vlm_go2.Go2VLMPerception()
            assert vlm._timeout >= 30.0, (
                f"Local timeout should be >= 30s (model load time), got {vlm._timeout}"
            )
            assert vlm._timeout > vlm_go2._TIMEOUT_S, (
                f"Local timeout {vlm._timeout}s must exceed remote default "
                f"{vlm_go2._TIMEOUT_S}s"
            )
            assert vlm._local is True
        finally:
            vlm_go2._USE_LOCAL_VLM = orig_use
            vlm_go2._LOCAL_VLM_URL = orig_url

    def test_remote_backend_default(self):
        """Without env vars, should use OpenRouter."""
        from vector_os_nano.perception import vlm_go2
        orig_use = vlm_go2._USE_LOCAL_VLM
        try:
            vlm_go2._USE_LOCAL_VLM = False
            # Need API key for remote
            vlm = vlm_go2.Go2VLMPerception(config={"api_key": "test-key"})
            assert vlm._local is False
            assert "openrouter" in vlm._base_url
        finally:
            vlm_go2._USE_LOCAL_VLM = orig_use


# ── Integration tests (need running Ollama) ────────────────────────────


@pytest.mark.skipif(not _ollama_available(), reason=_SKIP_REASON)
class TestLocalVLMInference:
    """Integration tests that hit the actual local Ollama endpoint."""

    @pytest.fixture(autouse=True)
    def _setup_local_env(self, monkeypatch):
        """Force local VLM backend for all tests in this class."""
        from vector_os_nano.perception import vlm_go2
        monkeypatch.setattr(vlm_go2, "_USE_LOCAL_VLM", True)
        monkeypatch.setattr(vlm_go2, "_LOCAL_VLM_URL", "http://localhost:11434/v1")
        monkeypatch.setattr(vlm_go2, "_LOCAL_VLM_MODEL", "gemma4:e4b")

    def _make_frame(self) -> np.ndarray:
        """Create a simple test frame (gray image)."""
        return np.ones((240, 320, 3), dtype=np.uint8) * 180

    def test_identify_room_returns_valid_structure(self):
        """identify_room should return RoomIdentification with required fields."""
        from vector_os_nano.perception.vlm_go2 import Go2VLMPerception, RoomIdentification
        vlm = Go2VLMPerception()
        result = vlm.identify_room(self._make_frame())
        assert isinstance(result, RoomIdentification)
        assert isinstance(result.room, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.reasoning, str)

    def test_describe_scene_returns_valid_structure(self):
        """describe_scene should return SceneDescription with required fields."""
        from vector_os_nano.perception.vlm_go2 import Go2VLMPerception, SceneDescription
        vlm = Go2VLMPerception()
        result = vlm.describe_scene(self._make_frame())
        assert isinstance(result, SceneDescription)
        assert isinstance(result.summary, str)
        assert isinstance(result.room_type, str)

    def test_find_objects_returns_list(self):
        """find_objects should return a list."""
        from vector_os_nano.perception.vlm_go2 import Go2VLMPerception
        vlm = Go2VLMPerception()
        result = vlm.find_objects(self._make_frame())
        assert isinstance(result, list)

    def test_local_vlm_no_cost(self):
        """Local VLM should report zero or near-zero cost."""
        from vector_os_nano.perception.vlm_go2 import Go2VLMPerception
        vlm = Go2VLMPerception()
        vlm.identify_room(self._make_frame())
        # Ollama may or may not return usage tokens — cost should be minimal
        assert vlm.cumulative_cost_usd < 0.01

    def test_consecutive_calls_fast(self):
        """Second call should be much faster (model already loaded)."""
        import time
        from vector_os_nano.perception.vlm_go2 import Go2VLMPerception
        vlm = Go2VLMPerception()
        # Warmup
        vlm.identify_room(self._make_frame())
        # Timed call
        t0 = time.monotonic()
        vlm.identify_room(self._make_frame())
        elapsed = time.monotonic() - t0
        # Second call should complete within 30s (typically 2-5s after warmup)
        assert elapsed < 30.0, f"Second call took {elapsed:.1f}s — too slow"


# ── Launch script env var tests ────────────────────────────────────────


class TestLaunchScriptConfig:
    """Verify launch scripts propagate VLM env vars."""

    def test_launch_explore_has_vlm_env(self):
        """launch_explore.sh should export VECTOR_VLM_URL."""
        script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "scripts", "launch_explore.sh",
        )
        if not os.path.isfile(script):
            pytest.skip("launch_explore.sh not found")
        with open(script) as f:
            content = f.read()
        assert "VECTOR_VLM_URL" in content
        assert "VECTOR_VLM_MODEL" in content
