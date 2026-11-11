# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Level 71 — vector-cli robot-control paths (arm sim).

Locks down the fixes that make `vector-cli --sim` and the in-REPL
`start_simulation` flow actually control the SO-101 arm on macOS:

- RobotContextProvider is arm-aware (no false "No hardware connected").
- start_simulation lives in the always-enabled 'sim' category (reachable from
  the bare dev world).
- SimStartTool binds the engine to the new body and re-enables the robot/diag
  categories; SimStopTool reverts cleanly.
- MuJoCoPerception.detect uses word-boundary matching ('all' != 'ball') and
  exposes caption()/visual_query() so 'describe' works.
- SkillWrapperTool runs pick (which moves to its scan pose itself) and returns
  pick's own data; it classifies wave/scan as motor skills.

Pure-logic tests run with zero robot deps; sim tests skip if mujoco is absent.
"""
from __future__ import annotations

import argparse
import threading
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Pure-logic (no mujoco)
# ---------------------------------------------------------------------------


def test_robot_context_arm_only_reports_connected():
    from vector_os_nano.vcli.robot_context import RobotContextProvider

    class _FakeArm:
        name = "TestArm"
        dof = 5

        def get_joint_positions(self):
            return [0.1, 0.2, 0.3, 0.4, 0.5]

    text = RobotContextProvider(base=None, scene_graph=None, arm=_FakeArm()).get_context_block()["text"]
    assert "No hardware connected" not in text
    assert "Arm: TestArm" in text
    assert "Joints:" in text
    # arm-only sessions must not get quadruped nav/explore noise
    assert "Exploring" not in text and "Nav stack" not in text


def test_robot_context_no_hardware_still_disconnected():
    from vector_os_nano.vcli.robot_context import RobotContextProvider

    block = RobotContextProvider().get_context_block()
    assert block["text"] == "[Robot State]\nNo hardware connected."


def test_start_simulation_in_sim_category():
    from vector_os_nano.vcli.tools import discover_categorized_tools

    _, cat_map = discover_categorized_tools()
    assert "start_simulation" in cat_map.get("sim", [])
    assert "stop_simulation" in cat_map.get("sim", [])
    assert "start_simulation" not in cat_map.get("system", [])


def test_model_flag_no_sentinel():
    from vector_os_nano.vcli.cli import parse_args

    assert parse_args([]).model is None
    # explicit override is preserved (not collapsed to None by a sentinel)
    assert parse_args(["--model", "claude-sonnet-4-6"]).model == "claude-sonnet-4-6"
    assert parse_args(["--model", "google/gemini-2.5-flash"]).model == "google/gemini-2.5-flash"


def test_wave_scan_classified_motor():
    from vector_os_nano.skills.wave import WaveSkill
    from vector_os_nano.skills.scan import ScanSkill
    from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

    for skill_cls in (WaveSkill, ScanSkill):
        wrapper = SkillWrapperTool(skill_cls(), None)
        assert wrapper._is_motor is True
        assert wrapper.is_concurrency_safe({}) is False


# ---------------------------------------------------------------------------
# Sim-backed (skip without mujoco) — all headless, gui=False
# ---------------------------------------------------------------------------


@pytest.fixture
def arm():
    pytest.importorskip("mujoco")
    from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm

    a = MuJoCoArm(gui=False)
    a.connect()
    yield a
    try:
        a.disconnect()
    except Exception:
        pass


def test_sim_agent_has_gripper_and_perception():
    pytest.importorskip("mujoco")
    from vector_os_nano.vcli import cli
    from vector_os_nano.hardware.sim.mujoco_gripper import MuJoCoGripper
    from vector_os_nano.hardware.sim.mujoco_perception import MuJoCoPerception

    # cli._init_agent reads `headless` (not `gui`): without it a viewer opens.
    ns = argparse.Namespace(sim=True, sim_go2=False, headless=True, api_key=None)
    session = cli._init_agent(ns)  # --sim -> the so101 SimSession
    try:
        agent = session.agent
        assert getattr(agent._arm, "_viewer", None) is None
        assert isinstance(agent._gripper, MuJoCoGripper)
        assert isinstance(agent._perception, MuJoCoPerception)
        assert agent.execute_skill("gripper_open").success
    finally:
        session.stop()


def test_perception_detect_word_boundary(arm):
    from vector_os_nano.hardware.sim.mujoco_perception import MuJoCoPerception

    p = MuJoCoPerception(arm)
    assert p.detect("red ball") == []   # 'all' must not fire on 'ball'
    assert p.detect("ball") == []
    assert len(p.detect("all objects")) > 0
    assert len(p.detect("all")) > 0


def test_perception_caption_and_describe(arm):
    from vector_os_nano.hardware.sim.mujoco_perception import MuJoCoPerception
    from vector_os_nano.core.agent import Agent

    p = MuJoCoPerception(arm)
    cap = p.caption()
    assert isinstance(cap, str) and cap
    assert p.visual_query("what is there?") == cap
    from vector_os_nano.skills import get_default_skills

    agent = Agent(arm=arm, perception=p, skills=get_default_skills())
    assert agent.execute_skill("describe").success


def test_skill_wrapper_pick_returns_picks_own_data(arm):
    from vector_os_nano.hardware.sim.mujoco_gripper import MuJoCoGripper
    from vector_os_nano.hardware.sim.mujoco_perception import MuJoCoPerception
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.skills import get_default_skills
    from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool
    from vector_os_nano.vcli.tools.base import ToolContext

    agent = Agent(arm=arm, gripper=MuJoCoGripper(arm), perception=MuJoCoPerception(arm),
                  skills=get_default_skills(),
                  config={"skills": {"pick": {"hardware_offsets": False}}})
    pick = agent._skill_registry.get("pick")
    ctx = ToolContext(agent=agent, cwd=Path("."), session=None, permissions=None,
                      abort=threading.Event())
    result = SkillWrapperTool(pick, agent).execute({"object_label": "mug", "mode": "hold"}, ctx)
    assert not result.is_error, result.content  # mug is in the scene -> grasp succeeds (physics)
    assert result.metadata["picked_object"] == "mug"  # pick's own data, nothing appended


def test_sim_tool_lifecycle_dev_to_arm_to_dev():
    pytest.importorskip("mujoco")
    from vector_os_nano.vcli.tools.base import CategorizedToolRegistry, ToolContext
    from vector_os_nano.vcli.tools import discover_categorized_tools
    from vector_os_nano.vcli.tools.sim_tool import SimStartTool, SimStopTool

    class _FakeEngine:
        def __init__(self):
            self.bound = []

        def bind(self, agent, world):
            self.bound.append((agent, world.name))

    reg = CategorizedToolRegistry()
    tools, cat_map = discover_categorized_tools()
    for t in tools:
        reg.register(t, category=next((c for c, n in cat_map.items() if t.name in n), "default"))
    eng = _FakeEngine()
    app = {"agent": None, "registry": reg, "engine": eng}
    ctx = ToolContext(agent=None, cwd=Path.cwd(), session=None, permissions=None,
                      abort=threading.Event(), app_state=app)

    tools_before = set(reg.list_tools())
    res = SimStartTool().execute({"embodiment": "so101", "gui": False}, ctx)
    assert not res.is_error and "Started so101" in res.content, res.content
    assert set(reg.list_tools()) - tools_before, "skills should be registered on start"
    assert eng.bound[-1] == (app["agent"], "robot")
    assert app["agent"] is not None
    # gui=False: no viewer in-process (the product default is a window)
    assert getattr(app["agent"]._arm, "_viewer", None) is None

    SimStopTool().execute({}, ctx)
    assert eng.bound[-1] == (None, "dev")
    assert app["agent"] is None and app["sim_session"] is None
    # skill tools are actually unregistered (not merely hidden), so restart is clean
    assert set(reg.list_tools()) == tools_before


