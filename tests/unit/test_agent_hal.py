"""Tests for Agent HAL integration — dict-based hardware registries.

Verifies that Agent.__init__ correctly wires legacy flat kwargs (arm=, base=)
into SkillContext dict registries introduced in T4 (SkillContext redesign).
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock


class TestAgentDictRegistries:
    def test_agent_with_base_dict(self):
        from vector_os_nano.core.agent import Agent

        base = MagicMock()
        base.get_position.return_value = [1.0, 2.0, 0.27]
        base.get_heading.return_value = 0.0
        agent = Agent(base=base)
        ctx = agent._build_context()
        assert ctx.base is base
        assert ctx.has_base()

    def test_agent_with_arm_and_base(self):
        from vector_os_nano.core.agent import Agent

        arm = MagicMock()
        base = MagicMock()
        base.get_position.return_value = [0, 0, 0.27]
        base.get_heading.return_value = 0.0
        agent = Agent(arm=arm, base=base)
        ctx = agent._build_context()
        assert ctx.arm is arm
        assert ctx.base is base
        assert ctx.has_arm()
        assert ctx.has_base()

    def test_agent_skills_property(self):
        from vector_os_nano.core.agent import Agent

        agent = Agent()
        # Should have default skills
        assert isinstance(agent.skills, list)

    def test_agent_sync_robot_state_with_base(self):
        from vector_os_nano.core.agent import Agent

        base = MagicMock()
        base.get_position.return_value = [5.0, 3.0, 0.27]
        base.get_heading.return_value = 1.57
        agent = Agent(base=base)
        agent._sync_robot_state()
        robot = agent._world_model.get_robot()
        assert robot.position_xy == (5.0, 3.0)
        assert abs(robot.heading - 1.57) < 0.01

    def test_agent_context_capabilities(self):
        from vector_os_nano.core.agent import Agent

        base = MagicMock()
        base.get_position.return_value = [0, 0, 0]
        base.get_heading.return_value = 0.0
        agent = Agent(base=base)
        ctx = agent._build_context()
        caps = ctx.capabilities()
        assert caps["has_base"] is True
        assert caps["has_arm"] is False

    # ------------------------------------------------------------------
    # Additional coverage: arm-only agent
    # ------------------------------------------------------------------

    def test_agent_with_arm_only(self):
        from vector_os_nano.core.agent import Agent

        arm = MagicMock()
        arm.get_joint_positions.return_value = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        agent = Agent(arm=arm)
        ctx = agent._build_context()
        assert ctx.arm is arm
        assert ctx.has_arm()
        assert not ctx.has_base()

    def test_agent_no_hardware_context(self):
        from vector_os_nano.core.agent import Agent

        agent = Agent()
        ctx = agent._build_context()
        assert not ctx.has_arm()
        assert not ctx.has_base()
        assert not ctx.has_gripper()

    def test_agent_build_context_uses_dict_registry_for_base(self):
        """_build_context must store base in bases dict, not legacy flat field."""
        from vector_os_nano.core.agent import Agent

        base = MagicMock()
        base.get_position.return_value = [0.0, 0.0, 0.27]
        base.get_heading.return_value = 0.0
        agent = Agent(base=base)
        ctx = agent._build_context()
        # bases dict must be populated
        assert "default" in ctx.bases
        assert ctx.bases["default"] is base

    def test_agent_build_context_uses_dict_registry_for_arm(self):
        """_build_context must store arm in arms dict, not legacy flat field."""
        from vector_os_nano.core.agent import Agent

        arm = MagicMock()
        arm.get_joint_positions.return_value = []
        agent = Agent(arm=arm)
        ctx = agent._build_context()
        assert "default" in ctx.arms
        assert ctx.arms["default"] is arm

    def test_agent_sync_robot_state_position_uses_first_two_elements(self):
        """get_position() returns [x, y, z] — only x,y stored in position_xy."""
        from vector_os_nano.core.agent import Agent

        base = MagicMock()
        base.get_position.return_value = [7.5, -2.3, 0.27]
        base.get_heading.return_value = 3.14
        agent = Agent(base=base)
        robot = agent._world_model.get_robot()
        assert robot.position_xy == pytest.approx((7.5, -2.3))
        assert robot.heading == pytest.approx(3.14, abs=0.01)

    def test_agent_sync_robot_state_base_none_does_not_update_position(self):
        """Without a base, position_xy stays at default (0, 0)."""
        from vector_os_nano.core.agent import Agent

        agent = Agent()
        robot = agent._world_model.get_robot()
        assert robot.position_xy == (0.0, 0.0)
        assert robot.heading == 0.0

    def test_agent_context_capabilities_arm_and_base(self):
        from vector_os_nano.core.agent import Agent

        arm = MagicMock()
        arm.get_joint_positions.return_value = []
        # Prevent auto-gripper: remove _bus so the auto-create branch is skipped
        del arm._bus
        base = MagicMock()
        base.get_position.return_value = [0.0, 0.0, 0.0]
        base.get_heading.return_value = 0.0
        agent = Agent(arm=arm, base=base)
        caps = agent._build_context().capabilities()
        assert caps["has_arm"] is True
        assert caps["has_base"] is True
        assert caps["has_gripper"] is False

    def test_agent_with_gripper(self):
        from vector_os_nano.core.agent import Agent

        arm = MagicMock()
        arm.get_joint_positions.return_value = []
        # Prevent auto-gripper creation: no _bus attribute
        del arm._bus
        gripper = MagicMock()
        agent = Agent(arm=arm, gripper=gripper)
        ctx = agent._build_context()
        assert ctx.has_gripper()
        assert ctx.gripper is gripper

    def test_agent_with_perception(self):
        from vector_os_nano.core.agent import Agent

        perception = MagicMock()
        agent = Agent(perception=perception)
        ctx = agent._build_context()
        assert ctx.has_perception()
        assert ctx.perception is perception
