"""Comprehensive tests for AgentLoop — full observe-decide-act-verify cycle."""
from unittest.mock import MagicMock, call
from vector_os_nano.core.agent_loop import AgentLoop
from vector_os_nano.core.types import GoalResult, SkillResult, ActionRecord


def _make_agent(decide_responses, skill_result=None):
    """Factory for mock agent with pre-set LLM decisions."""
    agent = MagicMock()
    agent._arm = None  # not sim by default
    agent._world_model.to_dict.return_value = {"objects": [{"label": "banana"}]}
    agent._world_model.apply_skill_effects.return_value = None
    agent._skill_registry.to_schemas.return_value = [{"name": "pick"}, {"name": "scan"}]
    agent._llm.decide_next_action.side_effect = decide_responses
    if skill_result is None:
        skill_result = SkillResult(success=True, result_data={"diagnosis": "ok"})
    mock_skill = MagicMock()
    mock_skill.execute.return_value = skill_result
    agent._skill_registry.get.return_value = mock_skill
    agent._build_context.return_value = MagicMock()
    return agent


class TestFullLoop:
    def test_two_picks_then_done(self):
        responses = [
            {"action": "pick", "params": {"object_label": "banana"}, "reasoning": "first"},
            {"action": "pick", "params": {"object_label": "mug"}, "reasoning": "second"},
            {"done": True, "summary": "All picked."},
        ]
        agent = _make_agent(responses)
        loop = AgentLoop(agent_ref=agent, config={})
        result = loop.run(goal="pick all", max_iterations=10, verify=False, on_step=None, on_message=None)

        assert result.success is True
        assert result.iterations == 3
        assert len(result.actions) == 2
        assert result.actions[0].action == "pick"
        assert result.actions[1].action == "pick"
        assert result.summary == "All picked."

    def test_single_action_then_done(self):
        responses = [
            {"action": "scan", "params": {}, "reasoning": "look around"},
            {"done": True, "summary": "Nothing to do."},
        ]
        agent = _make_agent(responses)
        loop = AgentLoop(agent_ref=agent, config={})
        result = loop.run(goal="check", max_iterations=10, verify=False, on_step=None, on_message=None)

        assert result.success is True
        assert result.iterations == 2
        assert len(result.actions) == 1
        assert result.actions[0].action == "scan"

    def test_immediate_done(self):
        responses = [{"done": True, "summary": "Already clean."}]
        agent = _make_agent(responses)
        loop = AgentLoop(agent_ref=agent, config={})
        result = loop.run(goal="clean", max_iterations=10, verify=False, on_step=None, on_message=None)

        assert result.success is True
        assert result.iterations == 1
        assert len(result.actions) == 0


class TestMaxIterations:
    def test_cap_at_max(self):
        responses = [{"action": "scan", "params": {}, "reasoning": "loop"}] * 20
        agent = _make_agent(responses)
        loop = AgentLoop(agent_ref=agent, config={})
        result = loop.run(goal="infinite", max_iterations=3, verify=False, on_step=None, on_message=None)

        assert result.success is False
        assert result.iterations == 3
        assert len(result.actions) == 3
        assert "Max iterations" in result.summary


class TestVerify:
    def test_verify_called_for_pick(self):
        responses = [
            {"action": "pick", "params": {"object_label": "mug"}, "reasoning": "grab"},
            {"done": True, "summary": "Done."},
        ]
        agent = _make_agent(responses)
        # Make it sim mode so _verify calls _refresh_objects
        agent._arm = MagicMock()
        agent._arm.get_object_positions = MagicMock(return_value={})

        loop = AgentLoop(agent_ref=agent, config={})
        result = loop.run(goal="pick mug", max_iterations=10, verify=True, on_step=None, on_message=None)

        assert result.success is True
        assert result.actions[0].verified is True

    def test_verify_not_called_for_scan(self):
        responses = [
            {"action": "scan", "params": {}, "reasoning": "look"},
            {"done": True, "summary": "Done."},
        ]
        agent = _make_agent(responses)
        loop = AgentLoop(agent_ref=agent, config={})
        result = loop.run(goal="look", max_iterations=10, verify=True, on_step=None, on_message=None)

        assert result.actions[0].verified is False

    def test_verify_false_skips_all(self):
        responses = [
            {"action": "pick", "params": {"object_label": "mug"}, "reasoning": "grab"},
            {"done": True, "summary": "Done."},
        ]
        agent = _make_agent(responses)
        loop = AgentLoop(agent_ref=agent, config={})
        result = loop.run(goal="pick", max_iterations=10, verify=False, on_step=None, on_message=None)

        assert result.actions[0].verified is False


class TestErrorHandling:
    def test_unknown_skill(self):
        responses = [
            {"action": "nonexistent", "params": {}, "reasoning": "bad"},
            {"done": True, "summary": "Gave up."},
        ]
        agent = _make_agent(responses)
        agent._skill_registry.get.side_effect = lambda name: None if name == "nonexistent" else MagicMock()
        loop = AgentLoop(agent_ref=agent, config={})
        result = loop.run(goal="test", max_iterations=5, verify=False, on_step=None, on_message=None)

        assert len(result.actions) == 1
        assert result.actions[0].skill_success is False
        assert result.actions[0].action == "nonexistent"

    def test_skill_failure_recorded(self):
        responses = [
            {"action": "pick", "params": {"object_label": "heavy"}, "reasoning": "try"},
            {"done": True, "summary": "Failed."},
        ]
        agent = _make_agent(responses, skill_result=SkillResult(success=False, error_message="too heavy"))
        loop = AgentLoop(agent_ref=agent, config={})
        result = loop.run(goal="test", max_iterations=5, verify=False, on_step=None, on_message=None)

        assert result.actions[0].skill_success is False


class TestHistory:
    def test_history_trimmed(self):
        n = 10
        responses = [
            {"action": "scan", "params": {}, "reasoning": f"step {i}"} for i in range(n)
        ] + [{"done": True, "summary": "Done."}]
        agent = _make_agent(responses)
        loop = AgentLoop(agent_ref=agent, config={"agent": {"agent_loop": {"history_max_actions": 3}}})
        result = loop.run(goal="test", max_iterations=n + 2, verify=False, on_step=None, on_message=None)

        # Verify decide was called with trimmed history
        for c in agent._llm.decide_next_action.call_args_list:
            history_arg = c.kwargs.get("history", c.args[3] if len(c.args) > 3 else [])
            assert len(history_arg) <= 3


class TestCallbacks:
    def test_on_step_called(self):
        responses = [
            {"action": "scan", "params": {}, "reasoning": "look"},
            {"action": "pick", "params": {"object_label": "mug"}, "reasoning": "grab"},
            {"done": True, "summary": "Done."},
        ]
        agent = _make_agent(responses)
        calls = []
        loop = AgentLoop(agent_ref=agent, config={})
        result = loop.run(goal="test", max_iterations=10, verify=False,
                          on_step=lambda action, i, total: calls.append((action, i, total)),
                          on_message=None)
        assert len(calls) == 2
        assert calls[0] == ("scan", 0, 10)
        assert calls[1] == ("pick", 1, 10)

    def test_on_message_called_on_done(self):
        responses = [{"done": True, "summary": "Complete!"}]
        agent = _make_agent(responses)
        messages = []
        loop = AgentLoop(agent_ref=agent, config={})
        result = loop.run(goal="test", max_iterations=5, verify=False,
                          on_step=None, on_message=lambda msg: messages.append(msg))
        assert messages == ["Complete!"]


class TestSimMode:
    def test_sim_calls_refresh_objects(self):
        responses = [{"done": True, "summary": "Done."}]
        agent = _make_agent(responses)
        agent._arm = MagicMock()
        agent._arm.get_object_positions = MagicMock(return_value={"banana": [0.1, 0.0, 0.05]})

        loop = AgentLoop(agent_ref=agent, config={})
        loop.run(goal="test", max_iterations=5, verify=False, on_step=None, on_message=None)

        agent._refresh_objects.assert_called()
