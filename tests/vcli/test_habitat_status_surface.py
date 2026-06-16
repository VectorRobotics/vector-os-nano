# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""STATUS OPEN #0 — the status/persona surface must know the habitat world.

Owner live-test finding (2026-06-11): with `--scenario apartment` the banner
showed `Base: habitat_kinematic` yet the persona answered "还没跑仿真" and
"怎么启动habitat" sent the LLM into bash exploration. Root causes locked here:

1. PlaygroundWorld served the MuJoCo robot persona (launch_explore.sh guidance)
   for habitat scenarios.
2. RobotContextProvider injected go2 nav-stack lines ("Nav stack: stopped")
   for ANY base and never mentioned the running world.
3. robot_status reported hardware only — no world/scenario/live-objects line.
4. No NL path existed to start habitat ("启动habitat模拟") or SysNav
   ("启动sysnav") — start_simulation knew only arm|go2.

Everything here is headless: fake bases, real WorldModel, real scenarios.
"""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from vector_os_nano.core.world_model import ObjectState, WorldModel
from vector_os_nano.playground.catalog import SCENARIOS
from vector_os_nano.playground.world import PlaygroundWorld
from vector_os_nano.vcli.robot_context import RobotContextProvider
from vector_os_nano.vcli.tools.base import ToolContext


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeHabitatBase:
    name = "habitat_kinematic"

    def get_position(self):
        return [1.0, 2.0, 0.1]

    def get_heading(self):
        return 0.5

    def get_pano(self):  # the capability the SysNav feed needs
        return {}

    def disconnect(self):
        self.disconnected = True


def _apartment_world() -> PlaygroundWorld:
    return PlaygroundWorld(SCENARIOS["apartment"])


def _wm(*objs: ObjectState) -> WorldModel:
    wm = WorldModel()
    for o in objs:
        wm.add_object(o)
    return wm


def _ctx(agent, app_state=None) -> ToolContext:
    return ToolContext(
        agent=agent,
        cwd=Path.cwd(),
        session=None,
        permissions=None,
        abort=threading.Event(),
        app_state=app_state,
    )


# ---------------------------------------------------------------------------
# 1. RobotContextProvider — the [Robot State] block
# ---------------------------------------------------------------------------


class TestRobotContextHabitat:
    def test_habitat_world_block_says_running(self) -> None:
        wm = _wm(
            ObjectState("sysnav_1", "sofa", x=1.0, y=1.0, confidence=0.7),
            ObjectState("sysnav_2", "light", x=2.0, y=0.5, confidence=0.7),
        )
        provider = RobotContextProvider(
            base=_FakeHabitatBase(), world=_apartment_world(), world_model=wm
        )
        text = provider.get_context_block()["text"]
        assert "apartment" in text
        assert "habitat" in text
        assert "RUNNING" in text          # the world is already live
        assert "Live objects: 2" in text
        assert "Position: (1.0, 2.0" in text
        # go2 nav-stack noise must NOT leak into the habitat block
        assert "Nav stack" not in text
        assert "Exploring" not in text

    def test_habitat_empty_world_model_points_at_sysnav(self) -> None:
        provider = RobotContextProvider(
            base=_FakeHabitatBase(), world=_apartment_world(), world_model=_wm()
        )
        text = provider.get_context_block()["text"]
        assert "Live objects: 0" in text
        assert "sysnav" in text.lower()   # tells the LLM how to populate

    def test_legacy_base_path_unchanged(self) -> None:
        # No world passed (the MuJoCo go2 path) — nav-stack lines stay.
        provider = RobotContextProvider(base=_FakeHabitatBase())
        text = provider.get_context_block()["text"]
        assert "Nav stack" in text


# ---------------------------------------------------------------------------
# 2. PlaygroundWorld persona — backend-selected (ADR-006 thing #4)
# ---------------------------------------------------------------------------


class TestHabitatPersona:
    def test_habitat_scenario_gets_habitat_persona(self) -> None:
        role, tools = _apartment_world().persona_blocks()
        assert role and tools
        assert "habitat" in (role + tools).lower()
        # The world is already running — the persona must say so and must NOT
        # teach the MuJoCo launch path (bash + launch_explore.sh) that sent
        # the LLM exploring the filesystem.
        assert "already running" in (role + tools).lower()
        assert "launch_explore.sh" not in tools
        # NL lifecycle guidance: habitat start + sysnav perception
        assert "start_simulation" in tools
        assert "sysnav_perception" in tools

    def test_mujoco_scenarios_keep_robot_persona(self) -> None:
        from vector_os_nano.vcli.prompt import (
            ROBOT_ROLE_PROMPT,
            ROBOT_TOOL_INSTRUCTIONS,
        )

        role, tools = PlaygroundWorld(SCENARIOS["tabletop"]).persona_blocks()
        assert role == ROBOT_ROLE_PROMPT
        assert tools == ROBOT_TOOL_INSTRUCTIONS

    def test_robot_persona_mentions_habitat_start(self) -> None:
        # The generic robot persona's sim-start guidance must include the
        # habitat option so "怎么启动habitat" never falls back to bash.
        from vector_os_nano.vcli.prompt import ROBOT_TOOL_INSTRUCTIONS

        assert 'sim_type="habitat"' in ROBOT_TOOL_INSTRUCTIONS

    def test_robot_persona_names_switch_embodiment(self) -> None:
        # Campaign #11 R11: the persona must teach switch_embodiment as the way
        # to CHANGE the running embodiment, and must NOT teach the old
        # "use bash + launch_explore.sh instead" directive that biased the LLM
        # toward raw shell for "switch to go2".
        from vector_os_nano.vcli.prompt import ROBOT_TOOL_INSTRUCTIONS

        assert "switch_embodiment" in ROBOT_TOOL_INSTRUCTIONS
        assert "use bash + launch_explore.sh instead" not in ROBOT_TOOL_INSTRUCTIONS

    def test_dev_persona_teaches_sim_tools(self) -> None:
        # Dev world keeps the sim category enabled; the persona must say how
        # NL sim startup works ("启动habitat模拟" from a bare REPL).
        from vector_os_nano.vcli.prompt import DEV_TOOL_INSTRUCTIONS

        assert "start_simulation" in DEV_TOOL_INSTRUCTIONS
        assert "habitat" in DEV_TOOL_INSTRUCTIONS


# ---------------------------------------------------------------------------
# 3. robot_status tool — world/scenario/live-objects aware
# ---------------------------------------------------------------------------


class TestRobotStatusWorldAware:
    def _agent(self, wm: WorldModel | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            _arm=None,
            _gripper=None,
            _base=_FakeHabitatBase(),
            _perception=None,
            _world_model=wm if wm is not None else _wm(),
        )

    def test_reports_habitat_world_running(self) -> None:
        from vector_os_nano.vcli.tools.robot import RobotStatusTool

        agent = self._agent(_wm(ObjectState("sysnav_1", "sofa", x=1.0, y=2.0)))
        app = {"world": _apartment_world(), "scenario": "apartment"}
        res = RobotStatusTool().execute({}, _ctx(agent, app))
        assert not res.is_error
        assert "World: apartment (habitat" in res.content
        assert "running" in res.content
        assert "Base: habitat_kinematic" in res.content
        assert "Live objects: 1" in res.content

    def test_reports_sysnav_state(self) -> None:
        from vector_os_nano.vcli.tools.robot import RobotStatusTool

        agent = self._agent()
        agent._sysnav_feed = object()
        agent._sysnav_proc = SimpleNamespace(poll=lambda: None, pid=4242)
        app = {"world": _apartment_world(), "scenario": "apartment"}
        res = RobotStatusTool().execute({}, _ctx(agent, app))
        assert "SysNav feed: up" in res.content
        assert "SysNav nodes: running" in res.content

    def test_no_world_keeps_legacy_output(self) -> None:
        from vector_os_nano.vcli.tools.robot import RobotStatusTool

        res = RobotStatusTool().execute({}, _ctx(self._agent(), None))
        assert "World:" not in res.content
        assert "Base: habitat_kinematic" in res.content

    def test_falls_back_to_app_state_agent_same_turn(self) -> None:
        # start_simulation swaps app_state["agent"] mid-turn while
        # ToolContext.agent was captured at turn start (None) — the status
        # tool must see the freshly started agent, not error out.
        from vector_os_nano.vcli.tools.robot import RobotStatusTool

        app = {"agent": self._agent()}
        res = RobotStatusTool().execute({}, _ctx(None, app))
        assert not res.is_error
        assert "Base: habitat_kinematic" in res.content


# ---------------------------------------------------------------------------
# 4. start_simulation — sim_type="habitat" (the NL startup path)
# ---------------------------------------------------------------------------


class _FakeEngine:
    def __init__(self) -> None:
        self._system_prompt: list = []
        self.vgg_world = None

    def init_vgg(self, **kwargs) -> None:
        self.vgg_world = kwargs.get("world")


def _mobile_agent() -> SimpleNamespace:
    from vector_os_nano.core.skill import SkillRegistry
    from vector_os_nano.skills.go2.stop import StopSkill
    from vector_os_nano.skills.go2.turn import TurnSkill
    from vector_os_nano.skills.go2.walk import WalkSkill
    from vector_os_nano.skills.navigate_to_point import NavigateToPointSkill

    registry = SkillRegistry()
    for s in (WalkSkill(), TurnSkill(), StopSkill(), NavigateToPointSkill()):
        registry.register(s)
    return SimpleNamespace(
        _arm=None,
        _gripper=None,
        _base=_FakeHabitatBase(),
        _perception=None,
        _world_model=_wm(),
        _skill_registry=registry,
        _spatial_memory=None,
    )


class TestSimStartHabitat:
    def test_schema_offers_habitat(self) -> None:
        from vector_os_nano.vcli.tools.sim_tool import SimStartTool

        schema = SimStartTool.input_schema
        assert "habitat" in schema["properties"]["sim_type"]["enum"]
        assert "scenario" in schema["properties"]

    def _app(self) -> dict:
        from vector_os_nano.vcli.tools import discover_categorized_tools
        from vector_os_nano.vcli.tools.base import CategorizedToolRegistry

        reg = CategorizedToolRegistry()
        tools, cat_map = discover_categorized_tools()
        for t in tools:
            cat = next((c for c, ns in cat_map.items() if t.name in ns), "default")
            reg.register(t, category=cat)
        for c in ("robot", "diag", "system"):
            reg.disable_category(c)
        return {"agent": None, "registry": reg, "engine": _FakeEngine()}

    def test_default_scenario_is_house(self, monkeypatch) -> None:
        # N5: "启动habitat模拟" with no scenario boots the FLAGSHIP house
        # world (multi-room ReplicaCAD), not the bare-scan apartment.
        from vector_os_nano.vcli import habitat_runtime
        from vector_os_nano.vcli.tools.sim_tool import SimStartTool

        booted: dict = {}

        def _boot(world, on_status=None, gui=None):
            booted["world"] = world
            return _mobile_agent()

        monkeypatch.setattr(habitat_runtime, "boot_habitat_agent", _boot)
        app = self._app()
        res = SimStartTool().execute({"sim_type": "habitat"}, _ctx(None, app))
        assert not res.is_error, res.content
        assert booted["world"].name == "house"
        assert app["scenario"] == "house"

    def test_starts_habitat_via_runtime(self, monkeypatch) -> None:
        from vector_os_nano.vcli import habitat_runtime
        from vector_os_nano.vcli.dynamic_prompt import DynamicSystemPrompt
        from vector_os_nano.vcli.tools.sim_tool import SimStartTool

        fake_agent = _mobile_agent()
        booted: dict = {}

        def _boot(world, on_status=None, gui=None):
            booted["world"] = world
            booted["gui"] = gui
            return fake_agent

        monkeypatch.setattr(habitat_runtime, "boot_habitat_agent", _boot)
        app = self._app()
        res = SimStartTool().execute(
            {"sim_type": "habitat", "scenario": "apartment"}, _ctx(None, app)
        )
        assert not res.is_error, res.content
        assert booted["world"].name == "apartment"
        assert booted["gui"] is True  # tool default: a visible viewer window
        assert app["agent"] is fake_agent
        assert app["world"].name == "apartment"
        assert app["scenario"] == "apartment"
        # status surface re-enabled for the started world (incl. robot_status)
        reg = app["registry"]
        assert reg.is_category_enabled("robot")
        assert reg.is_category_enabled("system")
        # prompt rebuilt LIVE with the HABITAT persona, not resolve_world(agent)
        eng = app["engine"]
        assert isinstance(eng._system_prompt, DynamicSystemPrompt)
        prompt_text = " ".join(
            b.get("text", "") for b in eng._system_prompt if isinstance(b, dict)
        )
        assert "already running" in prompt_text.lower()
        assert eng.vgg_world is app["world"]
        assert "apartment" in res.content

    def test_unknown_scenario_fails_loud_with_valid_set(self) -> None:
        from vector_os_nano.vcli.tools.sim_tool import SimStartTool

        res = SimStartTool().execute(
            {"sim_type": "habitat", "scenario": "游泳池"}, _ctx(None, self._app())
        )
        assert res.is_error
        assert "apartment" in res.content  # the valid set names the real ones

    def test_non_habitat_scenario_rejected(self) -> None:
        from vector_os_nano.vcli.tools.sim_tool import SimStartTool

        res = SimStartTool().execute(
            {"sim_type": "habitat", "scenario": "tabletop"}, _ctx(None, self._app())
        )
        assert res.is_error
        assert "habitat" in res.content

    def test_already_running_guard_names_the_base(self) -> None:
        from vector_os_nano.vcli.tools.sim_tool import SimStartTool

        app = self._app()
        app["agent"] = _mobile_agent()
        res = SimStartTool().execute(
            {"sim_type": "habitat", "scenario": "apartment"}, _ctx(None, app)
        )
        assert not res.is_error
        assert "habitat_kinematic" in res.content
        assert "already" in res.content.lower()


# ---------------------------------------------------------------------------
# 5. sysnav_perception tool — NL start/stop/status for semantic perception
# ---------------------------------------------------------------------------


class TestSysnavPerceptionTool:
    def test_registered_in_sim_category(self) -> None:
        from vector_os_nano.vcli.tools import discover_categorized_tools

        tools, cat_map = discover_categorized_tools()
        assert "sysnav_perception" in {t.name for t in tools}
        assert "sysnav_perception" in cat_map["sim"]

    def test_start_without_agent_points_at_habitat(self) -> None:
        from vector_os_nano.vcli.tools.sysnav_tool import SysnavPerceptionTool

        res = SysnavPerceptionTool().execute({"action": "start"}, _ctx(None, {}))
        assert res.is_error
        assert "start_simulation" in res.content
        assert "habitat" in res.content

    def test_start_requires_pano_capable_base(self) -> None:
        from vector_os_nano.vcli.tools.sysnav_tool import SysnavPerceptionTool

        agent = SimpleNamespace(_base=SimpleNamespace(name="go2"), _world_model=_wm())
        res = SysnavPerceptionTool().execute(
            {"action": "start"}, _ctx(agent, {"agent": agent})
        )
        assert res.is_error

    def test_start_wires_feed_and_launches_nodes(self, monkeypatch) -> None:
        from vector_os_nano.vcli import habitat_runtime
        from vector_os_nano.vcli.tools.sysnav_tool import SysnavPerceptionTool

        agent = _mobile_agent()
        calls: list[str] = []

        def _wire(a, on_status=None):
            calls.append("wire")
            a._sysnav_feed = object()
            a._sysnav_consumer = object()

        fake_proc = SimpleNamespace(poll=lambda: None, pid=4242)

        def _launch(on_status=None):
            calls.append("launch")
            return fake_proc, None

        monkeypatch.setattr(habitat_runtime, "wire_sysnav_feed", _wire)
        monkeypatch.setattr(habitat_runtime, "launch_sysnav_nodes", _launch)
        res = SysnavPerceptionTool().execute(
            {"action": "start"}, _ctx(agent, {"agent": agent})
        )
        assert not res.is_error, res.content
        assert calls == ["wire", "launch"]
        assert agent._sysnav_proc is fake_proc
        assert "model" in res.content.lower()  # warns about ~1 min model load

    def test_start_is_idempotent(self, monkeypatch) -> None:
        from vector_os_nano.vcli.tools.sysnav_tool import SysnavPerceptionTool

        agent = _mobile_agent()
        agent._sysnav_feed = object()
        agent._sysnav_consumer = object()
        agent._sysnav_proc = SimpleNamespace(poll=lambda: None, pid=1)
        res = SysnavPerceptionTool().execute(
            {"action": "start"}, _ctx(agent, {"agent": agent})
        )
        assert not res.is_error
        assert "already" in res.content.lower()

    def test_status_reports_components(self) -> None:
        from vector_os_nano.vcli.tools.sysnav_tool import SysnavPerceptionTool

        agent = _mobile_agent()
        agent._world_model = _wm(ObjectState("sysnav_1", "sofa", x=1, y=2))
        res = SysnavPerceptionTool().execute(
            {"action": "status"}, _ctx(agent, {"agent": agent})
        )
        assert not res.is_error
        assert "not wired" in res.content
        assert "Live objects: 1" in res.content

    def test_stop_tears_down(self, monkeypatch) -> None:
        from vector_os_nano.vcli import habitat_runtime
        from vector_os_nano.vcli.tools.sysnav_tool import SysnavPerceptionTool

        agent = _mobile_agent()
        agent._sysnav_proc = SimpleNamespace(poll=lambda: None, pid=1)
        stopped: list[str] = []
        monkeypatch.setattr(
            habitat_runtime,
            "shutdown_sysnav",
            lambda a: stopped.append("yes") or ["nodes stopped"],
        )
        res = SysnavPerceptionTool().execute(
            {"action": "stop"}, _ctx(agent, {"agent": agent})
        )
        assert not res.is_error
        assert stopped == ["yes"]


# ---------------------------------------------------------------------------
# 6. habitat_runtime — shared boot/wire/teardown helpers
# ---------------------------------------------------------------------------


class TestHabitatRuntime:
    def test_shutdown_sysnav_kills_proc_and_stops_consumer(self) -> None:
        from vector_os_nano.vcli import habitat_runtime

        events: list[str] = []
        agent = SimpleNamespace(
            _sysnav_proc=SimpleNamespace(
                poll=lambda: None,
                pid=999999999,  # nonexistent pid — killpg path must not raise
            ),
            _sysnav_consumer=SimpleNamespace(stop=lambda: events.append("consumer")),
            _sysnav_feed=SimpleNamespace(destroy=lambda: events.append("feed")),
        )
        lines = habitat_runtime.shutdown_sysnav(agent)
        assert "consumer" in events and "feed" in events
        assert agent._sysnav_proc is None
        assert isinstance(lines, list)

    def test_shutdown_sysnav_noop_without_refs(self) -> None:
        from vector_os_nano.vcli import habitat_runtime

        assert habitat_runtime.shutdown_sysnav(SimpleNamespace()) == []

    def test_launch_sysnav_nodes_preflights_workspace(self, monkeypatch, tmp_path) -> None:
        from vector_os_nano.vcli import habitat_runtime

        monkeypatch.setenv("VECTOR_SYSNAV_WS", str(tmp_path / "nope"))
        with pytest.raises(RuntimeError, match="SysNav"):
            habitat_runtime.launch_sysnav_nodes()

    def test_sim_stop_shutdown_agent_covers_sysnav(self) -> None:
        # SimStopTool._shutdown_agent must tear down the sysnav refs too.
        from vector_os_nano.vcli.tools.sim_tool import SimStartTool

        events: list[str] = []
        base = _FakeHabitatBase()
        agent = SimpleNamespace(
            _base=base,
            _arm=None,
            _gripper=None,
            _sysnav_proc=None,
            _sysnav_consumer=SimpleNamespace(stop=lambda: events.append("consumer")),
            _sysnav_feed=SimpleNamespace(destroy=lambda: events.append("feed")),
        )
        summary = SimStartTool._shutdown_agent(agent)
        assert "consumer" in events and "feed" in events
        assert getattr(base, "disconnected", False) is True
        assert "habitat" in summary.lower() or "disconnected" in summary


# ---------------------------------------------------------------------------
# 7. GUI viewer plumbing — "我需要能看到的sim" (owner live-test finding #2)
#    The conda habitat build is HEADLESS (no native window possible); the
#    server instead opens a live first-person OpenCV viewer when --gui is on.
# ---------------------------------------------------------------------------


class TestHabitatGuiPlumbing:
    def test_resolve_gui_env_override_wins(self, monkeypatch) -> None:
        from vector_os_nano.vcli.habitat_runtime import resolve_habitat_gui

        monkeypatch.setenv("VECTOR_HABITAT_GUI", "0")
        assert resolve_habitat_gui(requested=True) is False
        monkeypatch.setenv("VECTOR_HABITAT_GUI", "1")
        assert resolve_habitat_gui(requested=False) is True

    def test_resolve_gui_requested_beats_display_default(self, monkeypatch) -> None:
        from vector_os_nano.vcli.habitat_runtime import resolve_habitat_gui

        monkeypatch.delenv("VECTOR_HABITAT_GUI", raising=False)
        monkeypatch.setenv("DISPLAY", ":0")
        assert resolve_habitat_gui(requested=False) is False
        assert resolve_habitat_gui(requested=True) is True
        assert resolve_habitat_gui() is True  # desktop default: visible

    def test_resolve_gui_headless_box_defaults_off(self, monkeypatch) -> None:
        from vector_os_nano.vcli.habitat_runtime import resolve_habitat_gui

        monkeypatch.delenv("VECTOR_HABITAT_GUI", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        assert resolve_habitat_gui() is False

    def test_bridge_argv_carries_gui_flag(self) -> None:
        from vector_os_nano.playground.habitat.bridge import HabitatBridge

        on = HabitatBridge("scene.glb", gui=True)._server_argv()
        off = HabitatBridge("scene.glb", gui=False)._server_argv()
        assert "--gui" in on
        assert "--gui" not in off

    def test_habitat_base_passes_gui_to_bridge(self, monkeypatch) -> None:
        import vector_os_nano.playground.habitat.base as base_mod

        seen: dict = {}

        class _StubBridge:
            def __init__(self, scene, gui=False, **kw):
                seen["scene"] = scene
                seen["gui"] = gui

        monkeypatch.setattr(base_mod, "HabitatBridge", _StubBridge)
        base_mod.HabitatBase(scene="x.glb", gui=True)
        assert seen == {"scene": "x.glb", "gui": True}

    def test_sim_tool_gui_false_propagates(self, monkeypatch) -> None:
        from vector_os_nano.vcli import habitat_runtime
        from vector_os_nano.vcli.tools.sim_tool import SimStartTool

        booted: dict = {}

        def _boot(world, on_status=None, gui=None):
            booted["gui"] = gui
            return _mobile_agent()

        monkeypatch.setattr(habitat_runtime, "boot_habitat_agent", _boot)
        app = TestSimStartHabitat()._app()
        res = SimStartTool().execute(
            {"sim_type": "habitat", "scenario": "apartment", "gui": False},
            _ctx(None, app),
        )
        assert not res.is_error, res.content
        assert booted["gui"] is False


# ---------------------------------------------------------------------------
# 8. IntentRouter — "启动habitat模拟" / "启动sysnav" reach the sim tools
# ---------------------------------------------------------------------------


class TestIntentRoutingHabitat:
    def test_habitat_start_routes_to_sim_category(self) -> None:
        from vector_os_nano.vcli.intent_router import IntentRouter

        cats = IntentRouter().route("启动habitat模拟")
        assert cats is not None and "sim" in cats
        cats = IntentRouter().route("start habitat")
        assert cats is not None and "sim" in cats

    def test_sysnav_routes_to_sim_category(self) -> None:
        from vector_os_nano.vcli.intent_router import IntentRouter

        cats = IntentRouter().route("启动sysnav")
        assert cats is not None and "sim" in cats

    def test_lifecycle_phrases_bypass_vgg(self) -> None:
        from vector_os_nano.vcli.intent_router import IntentRouter

        router = IntentRouter()
        assert router.should_use_vgg("启动habitat模拟") is False
        assert router.should_use_vgg("启动sysnav") is False
        assert router.should_use_vgg("start sysnav") is False
