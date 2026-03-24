"""Unit tests for build_agent_loop_prompt() — TDD RED phase (T2)."""
from vector_os_nano.llm.prompts import build_agent_loop_prompt


class TestAgentLoopPrompt:
    def test_contains_goal(self):
        prompt = build_agent_loop_prompt(
            goal="clean the table",
            observation={"world_state": {}},
            skill_schemas=[],
            history=[],
            max_iterations=10,
        )
        assert "clean the table" in prompt

    def test_contains_max_iterations(self):
        prompt = build_agent_loop_prompt(
            goal="test",
            observation={},
            skill_schemas=[],
            history=[],
            max_iterations=7,
        )
        assert "7" in prompt

    def test_contains_skills(self):
        prompt = build_agent_loop_prompt(
            goal="test",
            observation={},
            skill_schemas=[{"name": "pick", "description": "picks objects"}],
            history=[],
            max_iterations=5,
        )
        assert "pick" in prompt

    def test_contains_observation(self):
        prompt = build_agent_loop_prompt(
            goal="test",
            observation={"world_state": {"objects": [{"label": "banana"}]}},
            skill_schemas=[],
            history=[],
            max_iterations=5,
        )
        assert "banana" in prompt

    def test_contains_history(self):
        prompt = build_agent_loop_prompt(
            goal="test",
            observation={},
            skill_schemas=[],
            history=[{"action": "pick", "success": True}],
            max_iterations=5,
        )
        assert "pick" in prompt
