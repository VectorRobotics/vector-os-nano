"""Tests for AgentLoop skeleton — instantiation and basic run paths."""
from unittest.mock import MagicMock
from vector_os_nano.core.agent_loop import AgentLoop
from vector_os_nano.core.types import GoalResult


class TestAgentLoopBasic:
    def test_instantiates(self):
        agent = MagicMock()
        agent._arm = None
        loop = AgentLoop(agent_ref=agent, config={})
        assert loop is not None

    def test_run_returns_goal_result_on_done(self):
        agent = MagicMock()
        agent._arm = None
        agent._world_model.to_dict.return_value = {"objects": []}
        agent._skill_registry.to_schemas.return_value = []
        agent._llm.decide_next_action.return_value = {"done": True, "summary": "Already done."}

        loop = AgentLoop(agent_ref=agent, config={})
        result = loop.run(goal="test", max_iterations=5, verify=False, on_step=None, on_message=None)
        assert isinstance(result, GoalResult)
        assert result.success is True
        assert result.summary == "Already done."
        assert result.iterations == 1

    def test_max_iterations_returns_failure(self):
        agent = MagicMock()
        agent._arm = None
        agent._world_model.to_dict.return_value = {"objects": []}
        agent._skill_registry.to_schemas.return_value = []
        agent._llm.decide_next_action.return_value = {"action": "scan", "params": {}, "reasoning": "loop"}
        mock_skill = MagicMock()
        mock_skill.execute.return_value = MagicMock(success=True, result_data={})
        agent._skill_registry.get.return_value = mock_skill
        agent._build_context.return_value = MagicMock()

        loop = AgentLoop(agent_ref=agent, config={})
        result = loop.run(goal="test", max_iterations=3, verify=False, on_step=None, on_message=None)
        assert result.success is False
        assert result.iterations == 3
        assert len(result.actions) == 3

    def test_no_llm_returns_done(self):
        agent = MagicMock()
        agent._arm = None
        agent._llm = None
        agent._world_model.to_dict.return_value = {}

        loop = AgentLoop(agent_ref=agent, config={})
        result = loop.run(goal="test", max_iterations=5, verify=False, on_step=None, on_message=None)
        assert result.success is True
        assert "No LLM" in result.summary
