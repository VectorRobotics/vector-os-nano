# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tests for Wave 2/3 skill registration wiring in sim_tool.py.

Test strategy: The full behavioural test (calling _start_go2 end-to-end with
with_arm=True) requires mocking subprocess.Popen, os.setsid, time.sleep(20),
Go2ROS2Proxy, PiperROS2Proxy, PiperGripperROS2Proxy, Agent, SceneGraph, and
multiple lazy imports inside the method body. That mock surface is fragile and
couples the test to implementation details of unrelated code paths.

Instead we use two focused guards:

1. test_manipulation_skills_importable_and_instantiable — verifies all 4 skill
   classes can be imported and constructed with no arguments. A failure here
   means any register(XyzSkill()) call in sim_tool will raise ImportError or
   TypeError at runtime.

2. test_register_manipulation_skills_* — run the single-source helper both
   launchers call (skills/manipulation_setup.py) on a real SkillRegistry, and
   check both launchers call it.
"""
from __future__ import annotations

import inspect


def test_manipulation_skills_importable_and_instantiable() -> None:
    """Regression guard: all 4 Wave 2/3 skill classes are importable and
    instantiable with no arguments. A failure means the register(...) calls
    added to _start_go2 will crash at runtime.
    """
    from vector_os_nano.skills.pick_top_down import PickTopDownSkill
    from vector_os_nano.skills.place_top_down import PlaceTopDownSkill
    from vector_os_nano.skills.mobile_pick import MobilePickSkill
    from vector_os_nano.skills.mobile_place import MobilePlaceSkill

    PickTopDownSkill()
    PlaceTopDownSkill()
    MobilePickSkill()
    MobilePlaceSkill()


def test_register_manipulation_skills_wires_the_full_set() -> None:
    """Behaviour, not source text: the single-source helper registers every
    manipulation skill + the grasp perception on a real registry, and honours the
    VECTOR_ENABLE_MANIPULATION=0 opt-out.

    Regression (tests:undocumented-gate-red): the old guard grepped sim_tool.py for
    ``PickTopDownSkill()``, which went red when registration moved into
    skills/manipulation_setup.py although the wiring was correct.
    """
    from types import SimpleNamespace

    from vector_os_nano.core.skill import SkillRegistry
    from vector_os_nano.perception.go2_grasp_perception import Go2GraspPerception
    from vector_os_nano.skills.manipulation_setup import register_manipulation_skills

    agent = SimpleNamespace(_skill_registry=SkillRegistry(), _perception=None)
    assert register_manipulation_skills(agent, base=object()) is True
    assert {"pick_top_down", "place_top_down", "mobile_pick", "mobile_place",
            "perception_grasp"} <= set(agent._skill_registry.list_skills())
    assert isinstance(agent._perception, Go2GraspPerception)


def test_register_manipulation_skills_opt_out(monkeypatch) -> None:
    from types import SimpleNamespace

    from vector_os_nano.core.skill import SkillRegistry
    from vector_os_nano.skills.manipulation_setup import register_manipulation_skills

    monkeypatch.setenv("VECTOR_ENABLE_MANIPULATION", "0")
    agent = SimpleNamespace(_skill_registry=SkillRegistry(), _perception=None)
    assert register_manipulation_skills(agent, base=object()) is False
    assert agent._skill_registry.list_skills() == []


def test_both_go2_arm_launchers_use_the_single_source_helper() -> None:
    """The NL launcher and the --sim-go2 launcher both route through the helper."""
    from vector_os_nano.vcli import cli
    from vector_os_nano.vcli.tools import sim_tool

    for mod in (cli, sim_tool):
        assert "register_manipulation_skills(agent, base)" in inspect.getsource(mod), mod.__name__
