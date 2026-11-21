# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Capability binding has no robot-name dispatch; imported extensions join it."""
from dataclasses import replace

import pytest

from vector_os_nano.core.agent import Agent
from vector_os_nano.core.skill import SKILL_TYPES, skill
from vector_os_nano.embodiments import registry
from vector_os_nano.skills import get_skills
from vector_os_nano.vcli.tools.skill_wrapper import wrap_skills


def test_an_agent_registers_nothing_by_itself():
    assert Agent(config={}).skills == []


@pytest.mark.parametrize('embodiment', registry.embodiment_ids())
def test_tools_match_the_body_capabilities_and_always_offer_stop(embodiment):
    cfg = registry.load(embodiment)
    offered = get_skills(cfg.capabilities)
    names = {t.name for t in wrap_skills(Agent(config={}, skills=offered))}
    assert len(names) == len(offered)
    assert 'stop' in names
    assert ('home' in names) == cfg.capabilities.has_arm
    assert ('grasp' in names) == (cfg.capabilities.has_arm and cfg.capabilities.has_gripper
                                and cfg.capabilities.camera and cfg.capabilities.depth)
    assert ('navigate' in names) == cfg.capabilities.has_base
    assert ('look' in names) == (cfg.capabilities.has_base and cfg.capabilities.camera)
    assert ('explore' in names) == (cfg.capabilities.has_base and cfg.capabilities.lidar
                                  and cfg.capabilities.camera)
    from vector_os_nano.vcli.engine import VectorEngine
    eng = VectorEngine(backend=None)
    eng._agent = Agent(config={}, skills=offered)
    assert eng.is_stop('停') and eng.is_stop('stop')


def test_new_registered_skill_needs_no_pack_edit():
    try:
        @skill(requires=('arm', 'depth'), actuates=False)
        class DepthProbe:
            name = 'test_depth_probe'
        caps = registry.load('piper').capabilities
        assert 'test_depth_probe' in {s.name for s in get_skills(caps)}
        assert 'test_depth_probe' not in {s.name for s in get_skills(replace(caps, depth=False))}
        assert 'test_depth_probe' not in {s.name for s in get_skills(registry.load('go2').capabilities)}
    finally:
        SKILL_TYPES.pop('test_depth_probe', None)


def test_requirement_typo_and_duplicate_name_fail_before_binding():
    with pytest.raises(ValueError, match='unknown capabilities'):
        @skill(requires=('amr',))
        class Typo:
            name = 'test_typo'
    get_skills(registry.load('go2').capabilities)
    with pytest.raises(ValueError, match='duplicate skill name'):
        @skill(requires=())
        class Duplicate:
            name = 'stop'


@pytest.mark.parametrize('embodiment', registry.embodiment_ids())
def test_persona_skill_list_matches_the_offered_tools(embodiment):
    from vector_os_nano.vcli.worlds.robot import task_policy
    agent = Agent(config={}, skills=get_skills(registry.load(embodiment).capabilities))
    line = next(line for line in task_policy(agent).splitlines() if line.startswith('Available skills: '))
    assert set(line.removeprefix('Available skills: ').split(', ')) == {t.name for t in wrap_skills(agent)}
