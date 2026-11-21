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
- Grasp/home are motor skills; the wrapper preserves the grasp's own result.
- The generic camera provides SO-101 RGB-D perception.

Pure-logic tests run with zero robot deps; sim tests skip if mujoco is absent.
"""
from __future__ import annotations

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
    from vector_os_nano.vcli.runtime import build_registry

    cat_map = build_registry().list_categories()
    assert "start_simulation" in cat_map.get("sim", [])
    assert "stop_simulation" in cat_map.get("sim", [])
    assert "start_simulation" not in cat_map.get("system", [])


def test_model_flag_no_sentinel():
    from vector_os_nano.vcli.cli import parse_args

    assert parse_args([]).model is None
    # explicit override is preserved (not collapsed to None by a sentinel)
    assert parse_args(["--model", "claude-sonnet-4-6"]).model == "claude-sonnet-4-6"
    assert parse_args(["--model", "google/gemini-2.5-flash"]).model == "google/gemini-2.5-flash"


def test_grasp_home_classified_motor():
    from vector_os_nano.skills.grasp import GraspSkill
    from vector_os_nano.skills.home import HomeSkill
    from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool

    for skill_cls in (GraspSkill, HomeSkill):
        wrapper = SkillWrapperTool(skill_cls(), None)
        assert wrapper._is_motor is True
        assert wrapper.is_concurrency_safe({}) is False


# ---------------------------------------------------------------------------
# Sim-backed (skip without mujoco) — all headless, gui=False
# ---------------------------------------------------------------------------


_TABLETOP = {n: [0.2, 0.0, 0.03] for n in ("banana", "mug", "bottle", "screwdriver", "duck", "lego")}


@pytest.mark.sim
def test_sim_agent_has_gripper_and_perception():
    from vector_os_nano.perception.camera_perception import CameraPerception
    from vector_os_nano.sim.drivers.gripper import Gripper
    from vector_os_nano.vcli import runtime

    session = runtime.start_sim("so101", gui=False)  # what --sim starts
    try:
        agent = session.agent
        assert session.runtime.world._viewer is None
        assert isinstance(agent._gripper, Gripper)
        assert isinstance(agent._perception, CameraPerception)
        assert agent.execute_skill("gripper_open").success
    finally:
        session.stop()


@pytest.mark.sim
def test_sim_tool_lifecycle_dev_to_arm_to_dev():
    pytest.importorskip("mujoco")
    from vector_os_nano.vcli.runtime import build_registry
    from vector_os_nano.vcli.tools.base import ToolContext
    from vector_os_nano.vcli.tools.sim_tool import SimStartTool, SimStopTool

    class _FakeEngine:
        def __init__(self):
            self.bound = []

        def bind(self, agent, world):
            self.bound.append((agent, world.name))

    reg = build_registry()
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
    assert app["sim_session"].runtime.world._viewer is None

    SimStopTool().execute({}, ctx)
    assert eng.bound[-1] == (None, "dev")
    assert app["agent"] is None and app["sim_session"] is None
    # skill tools are actually unregistered (not merely hidden), so restart is clean
    assert set(reg.list_tools()) == tools_before


