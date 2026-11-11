# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Integration tests for vector_os_nano.core.agent.Agent.

Tests cover the public Agent methods with mock hardware.
No real hardware or network calls are made.
"""
from __future__ import annotations

from vector_os_nano.core.types import SkillResult
from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.world_model import WorldModel


# ---------------------------------------------------------------------------
# Mock hardware helpers
# ---------------------------------------------------------------------------


class MockArm:
    """Minimal ArmProtocol implementation for tests."""

    name = "mock"
    joint_names = ["j1", "j2", "j3", "j4", "j5"]
    dof = 5
    _connected = False
    _bus = None  # No bus — prevents auto-gripper creation

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_joint_positions(self) -> list[float]:
        return [0.0] * 5

    def move_joints(self, pos: list[float], duration: float = 3.0) -> bool:
        return True

    def move_cartesian(self, xyz: list[float], duration: float = 3.0) -> bool:
        return True

    def fk(self, joints: list[float]):
        return ([0.2, 0.0, 0.1], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def ik(self, xyz: list[float], current=None) -> list[float]:
        return [0.0] * 5

    def stop(self) -> None:
        pass


class MockGripper:
    """Minimal GripperProtocol implementation for tests."""

    def open(self) -> bool:
        return True

    def close(self) -> bool:
        return True

    def is_holding(self) -> bool:
        return True

    def get_position(self) -> float:
        return 0.5

    def get_force(self):
        return None


class AlwaysSucceedSkill:
    """Custom skill that always succeeds, for testing registration."""

    name = "always_succeed"
    description = "A test skill that always succeeds"
    parameters: dict = {}
    preconditions: list = []
    postconditions: list = []
    effects: dict = {}

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        return SkillResult(success=True)


# ---------------------------------------------------------------------------
# Test: Agent creation
# ---------------------------------------------------------------------------


class TestAgentCreation:
    """Tests for Agent constructor with various combinations of arguments."""

    def test_agent_creation_minimal(self):
        """Agent(arm=MockArm()) should succeed with no other args."""
        from vector_os_nano.core.agent import Agent

        arm = MockArm()
        agent = Agent(arm=arm)
        assert agent is not None

    def test_agent_creation_no_args(self):
        """Agent() with zero arguments should succeed."""
        from vector_os_nano.core.agent import Agent

        agent = Agent()
        assert agent is not None

    def test_agent_creation_with_gripper(self):
        """Explicit gripper should be stored on the agent."""
        from vector_os_nano.core.agent import Agent

        arm = MockArm()
        gripper = MockGripper()
        agent = Agent(arm=arm, gripper=gripper)
        assert agent._gripper is gripper

    def test_agent_creation_with_custom_config(self):
        """Agent accepts dict config without crashing."""
        from vector_os_nano.core.agent import Agent

        cfg = {"agent": {"max_planning_retries": 2}, "llm": {"provider": "claude"}}
        agent = Agent(arm=MockArm(), config=cfg)
        assert agent is not None

    def test_agent_creation_no_arm(self):
        """Agent with no arm should set _arm to None."""
        from vector_os_nano.core.agent import Agent

        agent = Agent()
        assert agent._arm is None


# ---------------------------------------------------------------------------
# Test: the skills an Agent is built with
# ---------------------------------------------------------------------------


class TestAgentSkills:
    """The skills= argument and the skills property."""

    def test_agent_custom_skill(self):
        """A skill passed to the constructor appears in agent.skills and runs."""
        from vector_os_nano.core.agent import Agent

        agent = Agent(arm=MockArm(), skills=[AlwaysSucceedSkill()])
        assert "always_succeed" in agent.skills
        assert agent.execute_skill("always_succeed").success

    def test_agent_skills_property_lists_the_given_set(self):
        """agent.skills lists the set the body was built with (here the SO-101 one)."""
        from vector_os_nano.core.agent import Agent
        from vector_os_nano.skills import get_so101_skills

        agent = Agent(arm=MockArm(), skills=get_so101_skills())
        skill_names = agent.skills
        assert "home" in skill_names
        assert "scan" in skill_names
        assert "pick" in skill_names
        assert "place" in skill_names
        assert "detect" in skill_names

    def test_agent_same_name_registered_once(self):
        """Two skills with one name: the later one wins, listed once."""
        from vector_os_nano.core.agent import Agent

        agent = Agent(arm=MockArm(), skills=[AlwaysSucceedSkill(), AlwaysSucceedSkill()])
        assert agent.skills.count("always_succeed") == 1

# ---------------------------------------------------------------------------
# Test: world property
# ---------------------------------------------------------------------------


class TestAgentWorldProperty:
    """Tests for the agent.world property."""

    def test_agent_world_property(self):
        """agent.world should return the WorldModel instance."""
        from vector_os_nano.core.agent import Agent

        agent = Agent(arm=MockArm())
        assert isinstance(agent.world, WorldModel)

    def test_agent_world_is_same_instance(self):
        """Repeated access to agent.world returns the same object."""
        from vector_os_nano.core.agent import Agent

        agent = Agent(arm=MockArm())
        w1 = agent.world
        w2 = agent.world
        assert w1 is w2


# ---------------------------------------------------------------------------
# Test: skills property
# ---------------------------------------------------------------------------


class TestAgentSkillsProperty:
    """Tests for the agent.skills property."""

    def test_agent_skills_property(self):
        """agent.skills should return a list of strings."""
        from vector_os_nano.core.agent import Agent

        agent = Agent(arm=MockArm())
        skill_names = agent.skills
        assert isinstance(skill_names, list)
        assert all(isinstance(n, str) for n in skill_names)

    def test_agent_has_no_default_skills(self):
        """The Agent registers nothing by itself: the body's builder chooses."""
        from vector_os_nano.core.agent import Agent

        assert Agent(arm=MockArm()).skills == []


# ---------------------------------------------------------------------------
# Test: connect() / disconnect()
# ---------------------------------------------------------------------------


class TestAgentConnectDisconnect:
    """Tests for agent.connect() and agent.disconnect()."""

    def test_agent_connect_calls_arm_connect(self):
        """agent.connect() should call arm.connect()."""
        from vector_os_nano.core.agent import Agent

        arm = MockArm()
        agent = Agent(arm=arm)
        agent.connect()
        assert arm._connected is True

    def test_agent_disconnect_calls_arm_disconnect(self):
        """agent.disconnect() should call arm.disconnect()."""
        from vector_os_nano.core.agent import Agent

        arm = MockArm()
        agent = Agent(arm=arm)
        agent.connect()
        agent.disconnect()
        assert arm._connected is False

    def test_agent_connect_no_arm(self):
        """agent.connect() should not crash when arm is None."""
        from vector_os_nano.core.agent import Agent

        agent = Agent()
        agent.connect()  # Should not raise

    def test_agent_disconnect_no_arm(self):
        """agent.disconnect() should not crash when arm is None."""
        from vector_os_nano.core.agent import Agent

        agent = Agent()
        agent.disconnect()  # Should not raise


# ---------------------------------------------------------------------------
# Test: context manager
# ---------------------------------------------------------------------------


class TestAgentContextManager:
    """Tests for Agent used as a context manager (with statement)."""

    def test_agent_context_manager_connects_and_disconnects(self):
        """with Agent() as a: should connect on enter and disconnect on exit."""
        from vector_os_nano.core.agent import Agent

        arm = MockArm()
        with Agent(arm=arm) as a:
            assert arm._connected is True
            assert a is not None
        assert arm._connected is False

    def test_agent_context_manager_returns_agent(self):
        """The 'as' target should be the Agent instance."""
        from vector_os_nano.core.agent import Agent

        with Agent(arm=MockArm()) as a:
            assert isinstance(a, Agent)

    def test_agent_context_manager_disconnects_on_exception(self):
        """disconnect() should be called even if body raises an exception."""
        from vector_os_nano.core.agent import Agent

        arm = MockArm()
        try:
            with Agent(arm=arm):
                assert arm._connected is True
                raise RuntimeError("simulated error")
        except RuntimeError:
            pass
        assert arm._connected is False
