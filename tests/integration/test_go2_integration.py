# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Integration tests for Go2 MuJoCo — full Agent + Skills pipeline."""
import pytest

pytest.importorskip("vector_os_nano.third_party.convex_mpc")


class TestGo2AgentIntegration:
    def test_agent_registers_go2_skills(self):
        """Agent with Go2 base should be able to register Go2 skills."""
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
        from vector_os_nano.core.agent import Agent
        from vector_os_nano.skills.go2 import get_go2_skills

        go2 = MuJoCoGo2(gui=False)
        go2.connect()
        go2.stand()

        agent = Agent(base=go2)
        for skill in get_go2_skills():
            agent._skill_registry.register(skill)

        skill_names = agent.skills
        assert "walk" in skill_names
        assert "turn" in skill_names
        assert "stand" in skill_names
        assert "sit" in skill_names
        assert "lie_down" in skill_names
        go2.disconnect()

    def test_walk_turn_sit_sequence(self):
        """Full sequence: stand -> walk -> turn -> sit without crash."""
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2

        go2 = MuJoCoGo2(gui=False)
        go2.connect()
        go2.stand()

        # Walk forward
        go2.walk(vx=0.3, vy=0.0, vyaw=0.0, duration=1.0)
        pos_after_walk = go2.get_position()
        assert pos_after_walk[2] > 0.15  # still upright

        # Turn left
        go2.walk(vx=0.0, vy=0.0, vyaw=0.5, duration=1.0)
        pos_after_turn = go2.get_position()
        assert pos_after_turn[2] > 0.15

        # Sit
        go2.sit()
        pos_after_sit = go2.get_position()
        assert pos_after_sit[2] < pos_after_walk[2]  # lower when sitting

        go2.disconnect()

    def test_world_model_syncs_base_state(self):
        """Agent._sync_robot_state should populate base position/heading."""
        from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
        from vector_os_nano.core.agent import Agent
        from vector_os_nano.skills.go2 import get_go2_skills

        go2 = MuJoCoGo2(gui=False)
        go2.connect()
        go2.stand()

        agent = Agent(base=go2)
        for skill in get_go2_skills():
            agent._skill_registry.register(skill)

        agent._sync_robot_state()
        robot = agent._world_model.get_robot()
        # At minimum, heading should be a float
        assert isinstance(robot.heading, float)

        go2.disconnect()
