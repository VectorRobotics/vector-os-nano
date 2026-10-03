# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""R2-5: motor skills in simulation are auto-allowed; everything else keeps the confirm.

Sim-ness comes from the SimSession (tools-worlds:sim-by-module-path): an agent is
"simulated" only when it is the body of a live in-process ``SimSession``. A module
path that merely LOOKS simulated (``vector_os_nano.hardware.sim.*`` — also where the
ROS proxies that can drive a real robot live) no longer auto-allows anything.
Motor-ness is the skill's declared ``@skill(actuates=...)`` (fail closed: undeclared
actuates). All stubs — no MuJoCo.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vector_os_nano.sim import session as sim_session
from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool


def _make_skill(name: str = "scan", motor: bool = True) -> Any:
    skill = SimpleNamespace(name=name, description="x", parameters={}, preconditions=[], effects={})
    if not motor:
        skill.__skill_actuates__ = False
    return skill


def _sim_module_hw() -> Any:
    class _Arm:
        pass

    _Arm.__module__ = "vector_os_nano.hardware.sim.mujoco_arm"
    return _Arm()


@pytest.fixture
def live_sim(monkeypatch):
    """Register *agent* as the body of a (fake) live SimSession."""
    def _register(agent: Any) -> Any:
        monkeypatch.setattr(sim_session, "_LIVE", [SimpleNamespace(agent=agent)])
        return agent
    return _register


def test_motor_skill_on_a_live_sim_session_body_is_allowed(live_sim) -> None:
    agent = live_sim(SimpleNamespace(_arm=object(), _base=None, _gripper=None))
    tool = SkillWrapperTool(_make_skill(motor=True), agent)
    assert tool._is_motor
    assert tool.check_permissions({}, None).behavior == "allow"


def test_sim_looking_module_path_without_a_session_asks() -> None:
    """The old module-path heuristic auto-allowed this; a ROS proxy lives there too."""
    agent = SimpleNamespace(_arm=_sim_module_hw(), _base=None, _gripper=None)
    tool = SkillWrapperTool(_make_skill(motor=True), agent)
    assert tool._robot_is_simulated() is False
    assert tool.check_permissions({}, None).behavior == "ask"


def test_a_different_agent_than_the_sessions_asks(live_sim) -> None:
    live_sim(SimpleNamespace(_arm=object()))
    other = SimpleNamespace(_arm=_sim_module_hw())
    assert SkillWrapperTool(_make_skill(), other).check_permissions({}, None).behavior == "ask"


@pytest.mark.parametrize("agent", [None, SimpleNamespace(), SimpleNamespace(_arm=None, _base=None)])
def test_no_session_fails_safe_to_ask(agent) -> None:
    assert SkillWrapperTool(_make_skill(motor=True), agent).check_permissions({}, None).behavior == "ask"


@pytest.mark.parametrize("sim", [True, False])
def test_read_only_skill_is_always_allowed(live_sim, sim) -> None:
    agent = SimpleNamespace(_arm=object())
    if sim:
        live_sim(agent)
    tool = SkillWrapperTool(_make_skill(motor=False), agent)
    assert not tool._is_motor
    assert tool.check_permissions({}, None).behavior == "allow"
