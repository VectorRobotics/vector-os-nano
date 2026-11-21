# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Registration by body: the tool list a body offers holds only what it can do.

core:default-so101-skills-on-every-body / skills-manip:so101-skills-registered-on-all-
embodiments: the Agent registered the SO-101 set on every body (go2, g1, go2_piper).
Now the Agent registers nothing and the session's table chooses per body.
"""
from __future__ import annotations

import pytest

from vector_os_nano.core.agent import Agent
from vector_os_nano.embodiments import registry
from vector_os_nano.sim.session import _skill_pack
from vector_os_nano.skills import get_piper_skills, get_so101_skills
from vector_os_nano.skills.go2 import get_go2_skills
from vector_os_nano.vcli.tools.skill_wrapper import wrap_skills

_GRIPPER = {"gripper_open", "gripper_close"}  # body-agnostic: any body with a gripper
_SO101 = {s.name for s in get_so101_skills()} - {"stop"} - _GRIPPER  # the 5-DoF pose skills
_PIPER = {s.name for s in get_piper_skills()} - _GRIPPER
_BASE = {s.name for s in get_go2_skills()} - {"stop"}


def _tools(embodiment: str) -> set[str]:
    """The skill tools the model is offered on this body."""
    return {t.name for t in wrap_skills(Agent(config={}, skills=_skill_pack(embodiment)))}


def test_an_agent_registers_nothing_by_itself() -> None:
    assert Agent(config={}).skills == []


@pytest.mark.parametrize("embodiment", registry.embodiment_ids())
def test_every_body_has_a_pack(embodiment: str) -> None:
    _skill_pack(embodiment)


def test_go2_offers_no_arm_skill() -> None:
    tools = _tools("go2")
    assert _BASE <= tools and not tools & (_SO101 | _PIPER | _GRIPPER), tools


def test_go2_piper_offers_the_piper_set_and_no_so101_skill() -> None:
    tools = _tools("go2_piper")
    assert (_BASE | _PIPER) <= tools and not tools & (_SO101 - _PIPER - {"home"}), tools


def test_so101_offers_no_base_skill() -> None:
    tools = _tools("so101")
    assert _SO101 <= tools and not tools & (_BASE | (_PIPER - _SO101)), tools


def test_the_fixed_piper_offers_the_piper_set_a_home_and_no_base_skill() -> None:
    tools = _tools("piper")
    assert (_PIPER | _GRIPPER | {"home", "stop"}) <= tools, tools
    assert not tools & (_BASE | (_SO101 - _PIPER - {"home"})), tools


def test_g1_offers_no_arm_skill() -> None:
    assert not _tools("g1") & (_SO101 | _PIPER | _GRIPPER)


@pytest.mark.parametrize("embodiment", ["so101", "piper", "go2_piper"])
def test_every_body_with_a_gripper_offers_open_and_close(embodiment: str) -> None:
    """R2V-01: go2_piper lost gripper_open/close (e10's scripted mid-carry drop)."""
    assert _GRIPPER <= _tools(embodiment)


@pytest.mark.parametrize("embodiment", registry.embodiment_ids())
def test_every_body_offers_the_stop_its_p0_phrase_needs(embodiment: str) -> None:
    """producers:stop-skill-go2-only: 'stop' on so101 went to the model."""
    from vector_os_nano.vcli.engine import VectorEngine

    eng = VectorEngine(backend=None)
    eng._agent = Agent(config={}, skills=_skill_pack(embodiment))
    assert "stop" in _tools(embodiment) and eng.is_stop("停") and eng.is_stop("stop")
