"""Agent + Go2 integration tests.

Verifies that the Vector OS Nano Agent can control MuJoCoGo2
through the skill system — the full SDK pipeline.
"""
from __future__ import annotations

import math
import time

import pytest


@pytest.fixture
def agent_go2(go2_standing):
    """Create an Agent with Go2 base and Go2 skills registered."""
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.skills.go2 import get_go2_skills

    agent = Agent(base=go2_standing, config={"agent": {"max_planning_retries": 1}})
    for skill in get_go2_skills():
        agent._skill_registry.register(skill)
    return agent


@pytest.mark.agent
def test_agent_constructs_with_go2(go2_standing):
    """Agent(base=MuJoCoGo2) constructs without error."""
    from vector_os_nano.core.agent import Agent
    agent = Agent(base=go2_standing)
    assert agent._base is go2_standing
    assert agent._arm is None


@pytest.mark.agent
def test_go2_skills_registered(agent_go2):
    """Agent has all 6 Go2 skills registered."""
    names = sorted(agent_go2.skills)
    assert "walk" in names
    assert "turn" in names
    assert "stand" in names
    assert "sit" in names
    assert "lie_down" in names
    assert "navigate" in names
    assert len(names) >= 6


@pytest.mark.agent
def test_walk_skill_executes(agent_go2):
    """WalkSkill.execute moves the robot forward."""
    from vector_os_nano.core.skill import SkillResult

    base = agent_go2._base
    start = base.get_position()

    context = agent_go2._build_context()
    skill = agent_go2._skill_registry.get("walk")
    result = skill.execute({"direction": "forward", "distance": 1.0, "speed": 0.3}, context)

    end = base.get_position()
    dx = end[0] - start[0]

    assert isinstance(result, SkillResult)
    assert result.success, f"WalkSkill failed: {result}"
    assert dx > 0.1, f"Robot barely moved: dx={dx:.3f}"


@pytest.mark.agent
def test_turn_skill_executes(agent_go2):
    """TurnSkill.execute changes heading."""
    base = agent_go2._base
    h0 = base.get_heading()

    context = agent_go2._build_context()
    skill = agent_go2._skill_registry.get("turn")
    result = skill.execute({"direction": "left", "angle": 90.0}, context)

    h1 = base.get_heading()
    dh = abs(h1 - h0)
    if dh > math.pi:
        dh = 2 * math.pi - dh

    assert result.success, f"TurnSkill failed: {result}"
    assert dh > 0.2, f"Heading barely changed: dh={math.degrees(dh):.1f} deg"


@pytest.mark.agent
def test_sit_skill_executes(agent_go2):
    """SitSkill.execute lowers the robot."""
    base = agent_go2._base
    stand_z = base.get_position()[2]

    context = agent_go2._build_context()
    skill = agent_go2._skill_registry.get("sit")
    result = skill.execute({}, context)

    sit_z = base.get_position()[2]
    assert result.success
    assert sit_z < stand_z, f"Robot did not lower: stand_z={stand_z:.3f}, sit_z={sit_z:.3f}"

    # Stand back up for subsequent tests
    base.stand(duration=1.0)
