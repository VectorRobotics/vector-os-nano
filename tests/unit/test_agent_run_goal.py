"""Tests for Agent.run_goal() — delegates to AgentLoop.

TDD: RED phase — written before implementation.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from vector_os_nano.core.agent import Agent
from vector_os_nano.core.types import GoalResult


class TestAgentRunGoal:
    def test_delegates_to_agent_loop(self):
        agent = Agent()  # no hardware
        mock_result = GoalResult(
            success=True,
            goal="test",
            iterations=1,
            total_duration_sec=1.0,
            actions=[],
            summary="Done.",
            final_world_state={},
        )
        with patch("vector_os_nano.core.agent_loop.AgentLoop") as MockLoop:
            MockLoop.return_value.run.return_value = mock_result
            result = agent.run_goal("test goal", max_iterations=5, verify=False)

        assert isinstance(result, GoalResult)
        assert result.success is True
        MockLoop.return_value.run.assert_called_once()

    def test_uses_config_defaults(self):
        agent = Agent(
            config={"agent": {"agent_loop": {"max_iterations": 7, "verify": False}}}
        )
        mock_result = GoalResult(
            success=True,
            goal="test",
            iterations=1,
            total_duration_sec=0.5,
            actions=[],
            summary="Done.",
            final_world_state={},
        )
        with patch("vector_os_nano.core.agent_loop.AgentLoop") as MockLoop:
            MockLoop.return_value.run.return_value = mock_result
            result = agent.run_goal("test goal")  # no explicit max_iterations or verify

        call_kwargs = MockLoop.return_value.run.call_args
        # Support both positional (args) and keyword (kwargs) invocation
        assert (
            call_kwargs.kwargs.get("max_iterations") == 7
            or call_kwargs[1].get("max_iterations") == 7
        )

    def test_returns_goal_result_type(self):
        agent = Agent()
        mock_result = GoalResult(
            success=False,
            goal="fail",
            iterations=10,
            total_duration_sec=30.0,
            actions=[],
            summary="Max.",
            final_world_state={},
        )
        with patch("vector_os_nano.core.agent_loop.AgentLoop") as MockLoop:
            MockLoop.return_value.run.return_value = mock_result
            result = agent.run_goal("fail")
        assert isinstance(result, GoalResult)
        assert result.success is False

    def test_passes_on_step_callback(self):
        agent = Agent()
        mock_result = GoalResult(
            success=True,
            goal="step test",
            iterations=2,
            total_duration_sec=5.0,
            actions=[],
            summary="Done.",
            final_world_state={},
        )
        on_step_cb = MagicMock()
        with patch("vector_os_nano.core.agent_loop.AgentLoop") as MockLoop:
            MockLoop.return_value.run.return_value = mock_result
            agent.run_goal("step test", on_step=on_step_cb)

        call_kwargs = MockLoop.return_value.run.call_args
        assert (
            call_kwargs.kwargs.get("on_step") is on_step_cb
            or call_kwargs[1].get("on_step") is on_step_cb
        )

    def test_passes_on_message_callback(self):
        agent = Agent()
        mock_result = GoalResult(
            success=True,
            goal="msg test",
            iterations=1,
            total_duration_sec=1.0,
            actions=[],
            summary="Done.",
            final_world_state={},
        )
        on_message_cb = MagicMock()
        with patch("vector_os_nano.core.agent_loop.AgentLoop") as MockLoop:
            MockLoop.return_value.run.return_value = mock_result
            agent.run_goal("msg test", on_message=on_message_cb)

        call_kwargs = MockLoop.return_value.run.call_args
        assert (
            call_kwargs.kwargs.get("on_message") is on_message_cb
            or call_kwargs[1].get("on_message") is on_message_cb
        )

    def test_default_max_iterations_fallback(self):
        """When no config and no explicit max_iterations, defaults to 10."""
        agent = Agent(config={})  # empty config, no agent_loop section
        mock_result = GoalResult(
            success=True,
            goal="fallback",
            iterations=1,
            total_duration_sec=1.0,
            actions=[],
            summary="Done.",
            final_world_state={},
        )
        with patch("vector_os_nano.core.agent_loop.AgentLoop") as MockLoop:
            MockLoop.return_value.run.return_value = mock_result
            agent.run_goal("fallback goal")

        call_kwargs = MockLoop.return_value.run.call_args
        assert (
            call_kwargs.kwargs.get("max_iterations") == 10
            or call_kwargs[1].get("max_iterations") == 10
        )
