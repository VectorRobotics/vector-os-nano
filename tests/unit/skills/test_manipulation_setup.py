# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Unit tests for the single-source go2+arm manipulation wiring helper.

``register_manipulation_skills`` is the ONE place the go2+arm SimSession registers
the Piper manipulation skill set + perception-grasp (Rule 3 / Rule 11). It REUSES the
launcher's one perception (built at the manifest camera size) and never replaces it.
These tests pin the contract WITHOUT a real sim.
"""
from __future__ import annotations

import pytest

from vector_os_nano.core.agent import Agent
from vector_os_nano.skills.manipulation_setup import register_manipulation_skills


class _StubBase:
    """A no-op base — Go2GraspPerception only stores it (construction is lazy)."""


def _agent() -> Agent:
    # arm/gripper truthy stubs so the helper treats the arm as present; the
    # launcher has already built the ONE perception.
    from vector_os_nano.perception.go2_grasp_perception import Go2GraspPerception

    agent = Agent(base=_StubBase(), arm=object(), gripper=object(), config={})
    agent._perception = Go2GraspPerception(agent._base, width=320, height=240)
    return agent


def test_registers_perception_grasp_and_manip_skills() -> None:
    agent = _agent()
    ok = register_manipulation_skills(agent)
    assert ok is True
    names = set(agent._skill_registry.list_skills())
    for expected in (
        "perception_grasp", "pick_top_down", "place_top_down",
        "mobile_pick", "mobile_place",
    ):
        assert expected in names, f"{expected} not registered: {names}"


def test_reuses_the_launcher_perception() -> None:
    """launcher-perception-drift: the helper used to REPLACE the perception with a
    hardcoded 320x240 one, so two launchers ended at different sizes."""
    agent = _agent()
    from vector_os_nano.perception.go2_grasp_perception import Go2GraspPerception

    agent._perception = mine = Go2GraspPerception(agent._base, width=640, height=480)
    register_manipulation_skills(agent)
    assert agent._perception is mine


def test_no_perception_fails_loud() -> None:
    agent = Agent(base=_StubBase(), arm=object(), gripper=object(), config={})
    agent._perception = None
    with pytest.raises(RuntimeError, match="perception"):
        register_manipulation_skills(agent)


def test_perception_grasp_wins_grab_alias() -> None:
    """perception_grasp must register LAST so it wins the shared 抓/grab aliases."""
    agent = _agent()
    register_manipulation_skills(agent)
    m = agent._skill_registry.match("抓绿色的瓶子")
    assert m is not None
    assert m.skill_name == "perception_grasp", m.skill_name


def test_env_disable_skips_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_ENABLE_MANIPULATION", "0")
    agent = _agent()
    ok = register_manipulation_skills(agent)
    assert ok is False
    assert "perception_grasp" not in set(agent._skill_registry.list_skills())
