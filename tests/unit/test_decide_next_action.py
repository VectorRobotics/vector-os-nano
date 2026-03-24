"""Unit tests for ClaudeProvider.decide_next_action() — TDD RED phase (T2)."""
from unittest.mock import patch

from vector_os_nano.llm.claude import ClaudeProvider


class TestDecideNextAction:
    def test_returns_action(self):
        provider = ClaudeProvider(api_key="test-key")
        with patch.object(
            provider,
            "_chat_completion",
            return_value='{"action": "pick", "params": {"object_label": "mug"}, "reasoning": "only object"}',
        ):
            result = provider.decide_next_action(
                goal="pick all objects",
                observation={"world_state": {}},
                skill_schemas=[],
                history=[],
            )
        assert result["action"] == "pick"
        assert result["params"]["object_label"] == "mug"

    def test_returns_done(self):
        provider = ClaudeProvider(api_key="test-key")
        with patch.object(
            provider,
            "_chat_completion",
            return_value='{"done": true, "summary": "Table is clear."}',
        ):
            result = provider.decide_next_action(
                goal="clean table", observation={}, skill_schemas=[], history=[]
            )
        assert result.get("done") is True

    def test_handles_llm_error_string(self):
        provider = ClaudeProvider(api_key="test-key")
        with patch.object(
            provider,
            "_chat_completion",
            return_value="LLM error: request timed out",
        ):
            result = provider.decide_next_action(
                goal="test", observation={}, skill_schemas=[], history=[]
            )
        # Should fallback to scan, not crash
        assert result["action"] == "scan"
