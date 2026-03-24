"""Integration tests for Agent.run_goal() — full path from Agent through AgentLoop."""
import json
from unittest.mock import MagicMock, patch
from vector_os_nano.core.agent import Agent
from vector_os_nano.core.types import GoalResult, SkillResult, ExecutionResult


def _make_mock_arm(objects=None):
    """Mock arm with sim-like get_object_positions."""
    arm = MagicMock()
    arm.get_joint_positions = MagicMock(return_value=[0.0, 0.0, 0.0, 0.0, 0.0])
    arm.move_joints = MagicMock(return_value=True)
    arm.ik = MagicMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
    if objects is not None:
        arm.get_object_positions = MagicMock(return_value=objects)
    return arm


def _make_mock_llm(decide_responses):
    """Mock LLM with pre-set decide_next_action responses."""
    llm = MagicMock()
    llm.decide_next_action = MagicMock(side_effect=decide_responses)
    llm.classify = MagicMock(return_value="task")
    return llm


class TestRunGoalIntegration:
    def test_clean_table_three_objects(self):
        """Simulate picking 3 objects then done."""
        objects = {"banana": [0.1, 0.0, 0.05], "mug": [0.2, 0.0, 0.05], "bottle": [0.15, 0.05, 0.05]}
        arm = _make_mock_arm(objects)
        llm = _make_mock_llm([
            {"action": "pick", "params": {"object_label": "banana"}, "reasoning": "first"},
            {"action": "pick", "params": {"object_label": "mug"}, "reasoning": "second"},
            {"action": "pick", "params": {"object_label": "bottle"}, "reasoning": "third"},
            {"done": True, "summary": "All 3 objects removed."},
        ])

        agent = Agent(arm=arm, llm=llm, config={
            "agent": {"max_planning_retries": 3, "agent_loop": {"max_iterations": 10, "verify": False}},
            "skills": {"pick": {"hardware_offsets": False, "z_offset": 0.0, "pre_grasp_height": 0.04},
                       "home": {"joint_values": [0]*5}},
        })

        result = agent.run_goal("clean the table", verify=False)

        assert isinstance(result, GoalResult)
        assert result.success is True
        assert len(result.actions) == 3
        assert all(a.action == "pick" for a in result.actions)
        assert result.summary == "All 3 objects removed."

    def test_max_iterations_terminates(self):
        """Loop terminates when LLM never returns done."""
        arm = _make_mock_arm({"mug": [0.1, 0.0, 0.05]})
        llm = _make_mock_llm([
            {"action": "scan", "params": {}, "reasoning": "stuck"}
        ] * 20)  # way more than max_iterations

        agent = Agent(arm=arm, llm=llm, config={
            "agent": {"max_planning_retries": 3, "agent_loop": {"max_iterations": 3}},
            "skills": {"home": {"joint_values": [0]*5}},
        })

        result = agent.run_goal("test", max_iterations=3, verify=False)

        assert result.success is False
        assert result.iterations == 3
        assert "Max iterations" in result.summary

    def test_existing_execute_unaffected(self):
        """Agent.execute() still works — no regression."""
        agent = Agent()  # no hardware
        result = agent.execute("home")
        assert isinstance(result, ExecutionResult)
        assert isinstance(result.success, bool)

    def test_run_goal_via_mcp_handler(self):
        """MCP handle_tool_call routes run_goal correctly."""
        import asyncio
        from vector_os_nano.mcp.tools import handle_tool_call

        arm = _make_mock_arm({"banana": [0.1, 0.0, 0.05]})
        llm = _make_mock_llm([
            {"action": "pick", "params": {"object_label": "banana"}, "reasoning": "only object"},
            {"done": True, "summary": "Table clear."},
        ])

        agent = Agent(arm=arm, llm=llm, config={
            "agent": {"max_planning_retries": 3, "agent_loop": {"max_iterations": 10, "verify": False}},
            "skills": {"pick": {"hardware_offsets": False, "z_offset": 0.0, "pre_grasp_height": 0.04},
                       "home": {"joint_values": [0]*5}},
        })

        raw = asyncio.run(handle_tool_call(
            agent, "run_goal", {"goal": "pick all objects", "max_iterations": 5, "verify": False}
        ))
        data = json.loads(raw)
        assert data["success"] is True
        assert data["goal"] == "pick all objects"

    def test_run_goal_no_llm(self):
        """Without LLM, run_goal returns done immediately."""
        agent = Agent()  # no LLM, no arm
        result = agent.run_goal("test")
        assert isinstance(result, GoalResult)
        assert result.success is True
        assert "No LLM" in result.summary
