"""Level 26 — VLM reliability tests.

Covers:
- JSON strip of markdown code blocks
- Timeout value >= 25s
- Retry logic on empty choices
- identify_room return format
"""
from __future__ import annotations

import ast
import re
import textwrap
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VLM_MODULE_PATH = (
    Path(__file__).parents[2]
    / "vector_os_nano"
    / "perception"
    / "vlm_go2.py"
)


def _load_vlm_source() -> str:
    return VLM_MODULE_PATH.read_text()


# ---------------------------------------------------------------------------
# Fix 1: JSON strip of markdown backticks
# ---------------------------------------------------------------------------


class TestJsonStripMarkdownBackticks:
    """_parse_json_response must handle markdown-wrapped JSON."""

    @pytest.fixture(autouse=True)
    def _import(self):
        import importlib
        import sys

        # Ensure fresh import each time
        mod_name = "vector_os_nano.perception.vlm_go2"
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
        else:
            mod = importlib.import_module(mod_name)
        self._parse = mod._parse_json_response

    def test_plain_json_unchanged(self):
        result = self._parse('{"room": "kitchen", "confidence": 0.9, "reasoning": "x"}')
        assert result["room"] == "kitchen"
        assert result["confidence"] == pytest.approx(0.9)

    def test_json_backtick_json_block(self):
        wrapped = textwrap.dedent("""\
            ```json
            {"room": "bedroom", "confidence": 0.8, "reasoning": "bed visible"}
            ```""")
        result = self._parse(wrapped)
        assert result["room"] == "bedroom"
        assert result["confidence"] == pytest.approx(0.8)

    def test_json_plain_backtick_block(self):
        wrapped = textwrap.dedent("""\
            ```
            {"room": "hallway", "confidence": 0.7, "reasoning": "corridor"}
            ```""")
        result = self._parse(wrapped)
        assert result["room"] == "hallway"

    def test_json_with_preamble_text(self):
        """Model sometimes adds text before the code block."""
        wrapped = textwrap.dedent("""\
            Sure, here is the JSON:
            ```json
            {"room": "kitchen", "confidence": 0.95, "reasoning": "fridge visible"}
            ```""")
        result = self._parse(wrapped)
        assert result["room"] == "kitchen"

    def test_invalid_json_returns_empty_dict(self):
        result = self._parse("not json at all")
        assert result == {}

    def test_nested_objects_preserved(self):
        wrapped = textwrap.dedent("""\
            ```json
            {"summary": "room", "objects": [{"name": "chair", "description": "wooden", "confidence": 0.9}], "room_type": "study", "details": "desk visible"}
            ```""")
        result = self._parse(wrapped)
        assert result["room_type"] == "study"
        assert len(result["objects"]) == 1
        assert result["objects"][0]["name"] == "chair"


# ---------------------------------------------------------------------------
# Fix 2: Timeout >= 25s
# ---------------------------------------------------------------------------


class TestVlmTimeoutAtLeast25s:
    """Verify the module-level timeout constant is at least 25 seconds."""

    def test_timeout_constant_at_least_25s(self):
        source = _load_vlm_source()
        # Find _TIMEOUT_S assignment
        match = re.search(r"_TIMEOUT_S\s*:\s*float\s*=\s*([\d.]+)", source)
        assert match is not None, "_TIMEOUT_S constant not found in vlm_go2.py"
        timeout_val = float(match.group(1))
        assert timeout_val >= 25.0, (
            f"_TIMEOUT_S={timeout_val} is below 25s minimum — "
            "VLM calls will timeout prematurely"
        )

    def test_httpx_client_uses_timeout_constant(self):
        source = _load_vlm_source()
        # After refactor: timeout is stored per-instance (self._timeout) to allow
        # local/remote backends to have different values. Accept both patterns:
        # - legacy: httpx.Client(timeout=_TIMEOUT_S)
        # - current: httpx.Client(timeout=self._timeout)
        uses_module_const = "httpx.Client(timeout=_TIMEOUT_S)" in source
        uses_instance_var = "httpx.Client(timeout=self._timeout)" in source
        assert uses_module_const or uses_instance_var, (
            "httpx.Client must use _TIMEOUT_S or self._timeout for the timeout parameter"
        )


# ---------------------------------------------------------------------------
# Fix 3: Retry on empty choices
# ---------------------------------------------------------------------------


class TestVlmRetryOnEmpty:
    """Verify that empty choices response triggers a retry."""

    def _make_vlm(self) -> object:
        import importlib
        mod = importlib.import_module("vector_os_nano.perception.vlm_go2")
        return mod.Go2VLMPerception(config={"api_key": "test-key-abc123"})

    def _dummy_frame(self) -> np.ndarray:
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def test_retry_logic_exists_in_source(self):
        """Source code must contain retry handling for no-choices case."""
        source = _load_vlm_source()
        # Should have a continue or loop mechanism after empty choices
        assert "no choices" in source.lower(), (
            "No retry comment/message found for empty choices case"
        )
        # The no-choices path should not immediately raise (should continue loop)
        # Verify RuntimeError is not immediately raised for no-choices in the loop
        # by checking for 'continue' after the empty-choices branch
        assert re.search(
            r'not choices.*?continue',
            source,
            re.DOTALL,
        ), "Empty choices branch must use 'continue' to retry, not raise immediately"

    def test_empty_choices_retries_before_failing(self):
        """A response with empty choices should trigger _MAX_RETRIES attempts."""
        vlm = self._make_vlm()
        frame = self._dummy_frame()

        # Build a fake HTTP response with no choices
        empty_response = MagicMock()
        empty_response.status_code = 200
        empty_response.json.return_value = {
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0},
        }
        empty_response.raise_for_status = MagicMock()

        call_count = 0

        def fake_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return empty_response

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = fake_post
            mock_client_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="VLM API failed"):
                vlm._call_vlm(frame, "test prompt")

        # Must have retried _MAX_RETRIES times (default 2)
        assert call_count == 2, (
            f"Expected 2 retry attempts for empty choices, got {call_count}"
        )

    def test_empty_choices_succeeds_on_second_attempt(self):
        """If second attempt returns valid choices, should succeed."""
        vlm = self._make_vlm()
        frame = self._dummy_frame()

        empty_response = MagicMock()
        empty_response.status_code = 200
        empty_response.json.return_value = {
            "choices": [],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0},
        }
        empty_response.raise_for_status = MagicMock()

        valid_response = MagicMock()
        valid_response.status_code = 200
        valid_response.json.return_value = {
            "choices": [{"message": {"content": '{"room": "kitchen"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        valid_response.raise_for_status = MagicMock()

        responses = [empty_response, valid_response]
        call_count = 0

        def fake_post(*args, **kwargs):
            nonlocal call_count
            resp = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            return resp

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = fake_post
            mock_client_cls.return_value = mock_client

            result = vlm._call_vlm(frame, "test prompt")

        assert result == '{"room": "kitchen"}'
        assert call_count == 2


# ---------------------------------------------------------------------------
# Fix 4: identify_room return format
# ---------------------------------------------------------------------------


class TestVlmRoomIdentificationFormat:
    """identify_room() must return a RoomIdentification with correct fields."""

    def _make_vlm(self) -> object:
        import importlib
        mod = importlib.import_module("vector_os_nano.perception.vlm_go2")
        return mod.Go2VLMPerception(config={"api_key": "test-key-xyz"})

    def _dummy_frame(self) -> np.ndarray:
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def test_identify_room_returns_correct_type(self):
        import importlib
        mod = importlib.import_module("vector_os_nano.perception.vlm_go2")
        vlm = self._make_vlm()
        frame = self._dummy_frame()

        valid_response = MagicMock()
        valid_response.status_code = 200
        valid_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"room": "living_room", "confidence": 0.85, "reasoning": "sofa visible"}'
                }
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }
        valid_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = valid_response
            mock_client_cls.return_value = mock_client

            result = vlm.identify_room(frame)

        assert isinstance(result, mod.RoomIdentification)
        assert result.room == "living_room"
        assert result.confidence == pytest.approx(0.85)
        assert result.reasoning == "sofa visible"

    def test_identify_room_handles_markdown_wrapped_response(self):
        import importlib
        mod = importlib.import_module("vector_os_nano.perception.vlm_go2")
        vlm = self._make_vlm()
        frame = self._dummy_frame()

        # VLM returns markdown-wrapped JSON
        markdown_json = (
            '```json\n'
            '{"room": "kitchen", "confidence": 0.9, "reasoning": "fridge seen"}\n'
            '```'
        )

        valid_response = MagicMock()
        valid_response.status_code = 200
        valid_response.json.return_value = {
            "choices": [{"message": {"content": markdown_json}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15},
        }
        valid_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = valid_response
            mock_client_cls.return_value = mock_client

            result = vlm.identify_room(frame)

        assert isinstance(result, mod.RoomIdentification)
        assert result.room == "kitchen"
        assert result.confidence == pytest.approx(0.9)

    def test_identify_room_unknown_on_parse_failure(self):
        import importlib
        mod = importlib.import_module("vector_os_nano.perception.vlm_go2")
        vlm = self._make_vlm()
        frame = self._dummy_frame()

        # VLM returns garbage (unparseable)
        bad_response = MagicMock()
        bad_response.status_code = 200
        bad_response.json.return_value = {
            "choices": [{"message": {"content": "Sorry, I cannot help with that."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8},
        }
        bad_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = bad_response
            mock_client_cls.return_value = mock_client

            result = vlm.identify_room(frame)

        assert isinstance(result, mod.RoomIdentification)
        assert result.room == "unknown"
        assert result.confidence == pytest.approx(0.0)
