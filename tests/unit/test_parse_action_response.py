"""Unit tests for parse_action_response() — TDD RED phase (T2)."""
from vector_os_nano.llm.claude import parse_action_response


class TestParseActionResponse:
    def test_parses_action(self):
        raw = '{"action": "pick", "params": {"object_label": "banana"}, "reasoning": "closest"}'
        result = parse_action_response(raw)
        assert result["action"] == "pick"
        assert result["params"]["object_label"] == "banana"

    def test_parses_done(self):
        raw = '{"done": true, "summary": "All 3 objects removed."}'
        result = parse_action_response(raw)
        assert result.get("done") is True
        assert "All 3 objects" in result["summary"]

    def test_handles_markdown_fence(self):
        raw = '```json\n{"action": "scan", "params": {}}\n```'
        result = parse_action_response(raw)
        assert result["action"] == "scan"

    def test_handles_broken_json_regex_fallback(self):
        raw = 'Here is my answer: {"action": "detect", broken...'
        result = parse_action_response(raw)
        assert result["action"] == "detect"

    def test_handles_empty_response(self):
        result = parse_action_response("")
        assert result["action"] == "scan"

    def test_handles_unparseable(self):
        result = parse_action_response("I cannot decide what to do.")
        assert result["action"] == "scan"

    def test_fills_missing_params(self):
        raw = '{"action": "home"}'
        result = parse_action_response(raw)
        assert result["action"] == "home"
        assert result["params"] == {}

    def test_fills_missing_reasoning(self):
        raw = '{"action": "scan", "params": {}}'
        result = parse_action_response(raw)
        assert "reasoning" in result
