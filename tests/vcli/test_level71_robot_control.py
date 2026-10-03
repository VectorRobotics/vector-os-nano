# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Level 71 — vector-cli robot-control paths (arm sim).

Locks down the fixes that make `vector-cli --sim` and the in-REPL
`start_simulation` flow actually control the SO-101 arm on macOS:

- RobotContextProvider is arm-aware (no false "No hardware connected").
- start_simulation lives in the always-enabled 'sim' category (reachable from
  the bare dev world).
- SimStartTool rebuilds a live, arm-aware DynamicSystemPrompt and re-enables the
  robot/diag categories; SimStopTool reverts cleanly.
- The VGG gate admits arm-only agents.
- MuJoCoPerception.detect uses word-boundary matching ('all' != 'ball') and
  exposes caption()/visual_query() so 'describe' works.
- SkillWrapperTool honours __skill_auto_steps__ (pick = scan->detect->pick) and
  classifies wave/scan as motor skills.

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


def test_start_simulation_visible_in_dev_world():
    from vector_os_nano.vcli.tools import discover_categorized_tools
    from vector_os_nano.vcli.tools.base import CategorizedToolRegistry

    reg = CategorizedToolRegistry()
    tools, cat_map = discover_categorized_tools()
    for t in tools:
        cat = next((c for c, names in cat_map.items() if t.name in names), "default")
        reg.register(t, category=cat)
    for c in ("robot", "diag", "system"):  # dev-world disable set (matches cli.py)
        reg.disable_category(c)
    names = [s["name"] for s in reg.to_anthropic_schemas()]
    assert "start_simulation" in names  # 'sim' category stays enabled
    # robot-world 'system' helpers must stay hidden in the bare dev world
    for hidden in ("robot_status", "skill_reload", "open_foxglove"):
        assert hidden not in names


def test_sim_start_reachable_via_router_in_dev_world():
    """'start arm sim' must route to a still-enabled category in the dev world,
    or the routed tool_use path starves on zero tools and the LLM cannot start a
    sim conversationally."""
    from vector_os_nano.vcli.tools import discover_categorized_tools
    from vector_os_nano.vcli.tools.base import CategorizedToolRegistry
    from vector_os_nano.vcli.intent_router import IntentRouter

    reg = CategorizedToolRegistry()
    tools, cat_map = discover_categorized_tools()
    for t in tools:
        reg.register(t, category=next((c for c, n in cat_map.items() if t.name in n), "default"))
    for c in ("robot", "diag", "system"):  # exact dev-world disable set
        reg.disable_category(c)

    router = IntentRouter()
    for phrase in ("start arm sim", "start the arm simulation", "启动仿真"):
        cats = router.route(phrase)
        names = [s["name"] for s in reg.to_anthropic_schemas(categories=cats)]
        assert "start_simulation" in names, f"{phrase!r} routed to {cats} with no start_simulation"


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
    agent = cli._init_agent(ns)
    try:
        assert getattr(agent._arm, "_viewer", None) is None
        assert isinstance(agent._gripper, MuJoCoGripper)
        assert isinstance(agent._perception, MuJoCoPerception)
        assert agent.execute_skill("gripper_open").success
    finally:
        agent._arm.disconnect()


def test_vgg_gate_admits_arm_only(arm):
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.tools.base import CategorizedToolRegistry
    from vector_os_nano.vcli.intent_router import IntentRouter
    from vector_os_nano.vcli.worlds import resolve_world

    agent = Agent(arm=arm)
    eng = VectorEngine(backend=None, registry=CategorizedToolRegistry(), system_prompt=[],
                       permissions=None, intent_router=IntentRouter())
    eng._world = resolve_world(agent)
    eng.init_vgg(agent=agent, skill_registry=agent._skill_registry)
    assert eng.vgg_decompose("go home") is not None


def test_vgg_gate_blocks_disconnected():
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.tools.base import CategorizedToolRegistry
    from vector_os_nano.vcli.intent_router import IntentRouter

    # a fully-disconnected agent (no base, no arm) stays gated
    agent = Agent.__new__(Agent)
    agent._base = None
    agent._arm = None
    agent._skill_registry = None
    eng = VectorEngine(backend=None, registry=CategorizedToolRegistry(), system_prompt=[],
                       permissions=None, intent_router=IntentRouter())

    class _RobotWorld:
        def is_robot(self):
            return True

    eng._world = _RobotWorld()
    eng.init_vgg(agent=agent, skill_registry=None)
    assert eng.vgg_decompose("go home") is None


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
    assert Agent(arm=arm, perception=p).execute_skill("describe").success


def test_skill_wrapper_pick_honors_auto_steps(arm, monkeypatch):
    from vector_os_nano.hardware.sim.mujoco_gripper import MuJoCoGripper
    from vector_os_nano.hardware.sim.mujoco_perception import MuJoCoPerception
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool
    from vector_os_nano.vcli.tools.base import ToolContext

    agent = Agent(arm=arm, gripper=MuJoCoGripper(arm), perception=MuJoCoPerception(arm),
                  config={"skills": {"pick": {"hardware_offsets": False}}})
    pick = agent._skill_registry.get("pick")
    assert getattr(pick, "__skill_auto_steps__", None), "pick should declare auto_steps"

    seen = {}
    real_execute_skill = agent.execute_skill

    def _spy(name, params=None, **kw):
        seen["name"] = name
        return real_execute_skill(name, params, **kw)

    monkeypatch.setattr(agent, "execute_skill", _spy)
    ctx = ToolContext(agent=agent, cwd=Path("."), session=None, permissions=None,
                      abort=threading.Event())
    result = SkillWrapperTool(pick, agent).execute({"object_label": "mug"}, ctx)
    # auto_steps path must route through agent.execute_skill (runs scan->detect->pick)
    assert seen.get("name") == "pick"
    assert not result.is_error  # mug is in the scene -> grasp succeeds (physics)
    # grasp data is surfaced from ExecutionResult.trace[-1] (not the top-level result)
    assert result.metadata


def test_sim_tool_lifecycle_dev_to_arm_to_dev():
    pytest.importorskip("mujoco")
    from vector_os_nano.vcli.tools.base import CategorizedToolRegistry, ToolContext
    from vector_os_nano.vcli.tools import discover_categorized_tools
    from vector_os_nano.vcli.tools.sim_tool import SimStartTool, SimStopTool
    from vector_os_nano.vcli.dynamic_prompt import DynamicSystemPrompt

    class _FakeEngine:
        def __init__(self):
            self._system_prompt = []

        def init_vgg(self, **kw):
            pass

    reg = CategorizedToolRegistry()
    tools, cat_map = discover_categorized_tools()
    for t in tools:
        reg.register(t, category=next((c for c, n in cat_map.items() if t.name in n), "default"))
    for c in ("robot", "diag"):
        reg.disable_category(c)

    for c in ("system",):  # mirror cli.py dev-world entry state fully
        reg.disable_category(c)

    eng = _FakeEngine()
    app = {"agent": None, "registry": reg, "engine": eng, "vgg_step_callback": None}
    ctx = ToolContext(agent=None, cwd=Path.cwd(), session=None, permissions=None,
                      abort=threading.Event(), app_state=app)

    tools_before = set(reg.list_tools())
    SimStartTool().execute({"sim_type": "arm", "gui": False}, ctx)
    assert set(reg.list_tools()) - tools_before, "skills should be registered on start"
    sp = eng._system_prompt
    assert isinstance(sp, DynamicSystemPrompt)
    assert sp._provider is not None and sp._provider._arm is not None
    assert "No hardware connected" not in "".join(b.get("text", "") for b in sp)
    assert reg.is_category_enabled("robot") and reg.is_category_enabled("diag")
    assert app["agent"] is not None
    # gui=False: no viewer in-process (the product default is a window)
    assert getattr(app["agent"]._arm, "_viewer", None) is None

    SimStopTool().execute({}, ctx)
    sp2 = eng._system_prompt
    assert isinstance(sp2, DynamicSystemPrompt)
    assert sp2._provider._arm is None
    assert not reg.is_category_enabled("robot")
    assert app["agent"] is None
    # skill tools are actually unregistered (not merely hidden), so restart is clean
    assert set(reg.list_tools()) == tools_before


def test_dynamic_prompt_no_corruption_across_turns(arm):
    """The per-turn [Robot State] refresh must not clobber the static
    tool-instructions block (which itself contains the '[Robot State]'
    substring). Regression guard for the DynamicSystemPrompt.__init__ scan.
    """
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.vcli.prompt import build_system_prompt
    from vector_os_nano.vcli.dynamic_prompt import DynamicSystemPrompt
    from vector_os_nano.vcli.robot_context import RobotContextProvider
    from vector_os_nano.vcli.worlds import resolve_world

    agent = Agent(arm=arm)
    prov = RobotContextProvider(arm=arm)
    blocks = build_system_prompt(agent=agent, cwd=None, robot_context=prov,
                                 world=resolve_world(agent))
    dsp = DynamicSystemPrompt(blocks, prov)

    def texts():
        return [b.get("text", "") for b in list(dsp)]

    static_ref = [t for t in texts() if "[Robot State]" in t and not t.startswith("[Robot State]")]
    assert static_ref, "expected a static block that references [Robot State]"

    for _ in range(3):
        _ = list(dsp)

    after = texts()
    robot_state = [t for t in after if t.startswith("[Robot State]")]
    survived = [t for t in after if "[Robot State]" in t and not t.startswith("[Robot State]")]
    assert len(robot_state) == 1, f"expected exactly one [Robot State] block, got {len(robot_state)}"
    assert survived, "static tool-instructions block was clobbered by the refresh"
