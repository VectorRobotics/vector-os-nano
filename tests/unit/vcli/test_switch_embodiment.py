# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #11 M1 — SwitchEmbodimentTool pure-logic tests (ADR-011).

The runtime g1<->go2 switch: no-op / fail-loud / boot-then-swap rollback /
embodiment guard / prefer_daemon hard-wire / single-source stale-tool drop.
Real-sim leak + walk verification lives in the sandbox harness (not pytest)."""
from __future__ import annotations

import types

import pytest

from vector_os_nano.vcli.tools.sim_tool import SimStartTool
from vector_os_nano.vcli.tools.switch_tool import SwitchEmbodimentTool


def _ctx(app):
    return types.SimpleNamespace(app_state=app, cwd="/tmp")


class _FakeBase:
    pass


class _MuJoCoGo2(_FakeBase):
    pass


class _MuJoCoG1(_FakeBase):
    pass


def _agent_with(base_cls):
    a = types.SimpleNamespace()
    a._base = base_cls()
    a._skill_registry = types.SimpleNamespace(list_skills=lambda: ["walk", "turn"])
    return a


# -- fail loud --------------------------------------------------------------

def test_unknown_target_fails_loud():
    res = SwitchEmbodimentTool().execute({"target": "spot"}, _ctx({"agent": object()}))
    assert res.is_error and "g1" in res.content and "go2" in res.content


def test_no_agent_fails_loud():
    res = SwitchEmbodimentTool().execute({"target": "g1"}, _ctx({"agent": None}))
    assert res.is_error and "start_simulation" in res.content


def test_no_app_state_fails_loud():
    res = SwitchEmbodimentTool().execute({"target": "g1"}, _ctx(None))
    assert res.is_error and "app state" in res.content.lower()


# -- no-op ------------------------------------------------------------------

def test_noop_same_embodiment(monkeypatch):
    called = {"boot": False}
    monkeypatch.setattr(SwitchEmbodimentTool, "_boot_target",
                        staticmethod(lambda *a, **k: called.__setitem__("boot", True)))
    app = {"agent": _agent_with(_MuJoCoGo2)}
    res = SwitchEmbodimentTool().execute({"target": "go2"}, _ctx(app))
    assert not res.is_error and "Already running go2" in res.content
    assert called["boot"] is False          # never booted


# -- boot-then-swap rollback ------------------------------------------------

def test_rollback_on_boot_failure(monkeypatch):
    old = _agent_with(_MuJoCoGo2)
    app = {"agent": old, "sim_gui": False}
    monkeypatch.setattr(SwitchEmbodimentTool, "_boot_target",
                        staticmethod(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))))
    shut = {"called": False}
    monkeypatch.setattr(SimStartTool, "_shutdown_agent",
                        staticmethod(lambda a: shut.__setitem__("called", True) or "x"))
    res = SwitchEmbodimentTool().execute({"target": "g1"}, _ctx(app))
    assert res.is_error and "Kept the current" in res.content
    assert app["agent"] is old              # unmutated
    assert shut["called"] is False          # old never torn down


# -- happy path wiring (mocked sim) -----------------------------------------

def test_happy_path_switch_wiring(monkeypatch):
    old = _agent_with(_MuJoCoGo2)
    new = _agent_with(_MuJoCoG1)
    app = {"agent": old, "sim_gui": False}
    monkeypatch.setattr(SwitchEmbodimentTool, "_boot_target",
                        staticmethod(lambda t, s, g: (new, object())))
    monkeypatch.setattr(SimStartTool, "_shutdown_agent",
                        staticmethod(lambda a: "Go2 disconnected"))
    rebound = {}
    monkeypatch.setattr(SimStartTool, "_rebind_agent",
                        staticmethod(lambda app, ctx, agent, world=None: app.__setitem__("agent", agent)))
    res = SwitchEmbodimentTool().execute({"target": "g1"}, _ctx(app))
    assert not res.is_error and "Switched to g1" in res.content
    assert app["agent"] is new


# -- _boot_target embodiment guard + prefer_daemon --------------------------

def test_boot_target_embodiment_guard(monkeypatch):
    import vector_os_nano.vcli.worlds as worlds
    fake_world = types.SimpleNamespace(scenario=types.SimpleNamespace(embodiment="go2"))
    monkeypatch.setattr(worlds, "resolve_world_named", lambda name: fake_world)
    with pytest.raises(ValueError, match="not a g1 scenario"):
        SwitchEmbodimentTool._boot_target("g1", "go2_room_vlm", gui=False)


def test_boot_target_prefer_daemon_for_g1(monkeypatch):
    import vector_os_nano.vcli.worlds as worlds
    from vector_os_nano.vcli import g1_runtime, go2_runtime
    fake_g1 = types.SimpleNamespace(scenario=types.SimpleNamespace(embodiment="g1"))
    monkeypatch.setattr(worlds, "resolve_world_named", lambda name: fake_g1)
    seen = {}
    monkeypatch.setattr(g1_runtime, "boot_g1_agent",
                        lambda world, gui=False, prefer_daemon=False: seen.update(prefer_daemon=prefer_daemon) or "g1agent")
    agent, world = SwitchEmbodimentTool._boot_target("g1", "g1_room_vlm", gui=False)
    assert agent == "g1agent" and seen["prefer_daemon"] is True

    fake_go2 = types.SimpleNamespace(scenario=types.SimpleNamespace(embodiment="go2"))
    monkeypatch.setattr(worlds, "resolve_world_named", lambda name: fake_go2)
    seen2 = {}
    monkeypatch.setattr(go2_runtime, "boot_go2_agent",
                        lambda world, gui=False: seen2.update(called=True) or "go2agent")
    agent2, _ = SwitchEmbodimentTool._boot_target("go2", "go2_room_vlm", gui=False)
    assert agent2 == "go2agent" and seen2["called"] is True


# -- _rebind_agent single-source stale-tool drop ----------------------------

def test_rebind_drops_stale_skill_tools(monkeypatch):
    """The shared rebind drops the old embodiment's skill-wrapper tools and
    registers the new agent's — so a switch never leaves a stale skill."""
    dropped, added = [], []

    class _Reg:
        def __init__(self):
            self._tools = {"old_walk": types.SimpleNamespace(_is_skill_wrapper=True),
                           "file_read": types.SimpleNamespace(_is_skill_wrapper=False)}
        def list_tools(self):
            return list(self._tools)
        def get(self, n):
            return self._tools.get(n)
        def unregister(self, n):
            dropped.append(n); self._tools.pop(n, None)
        def register(self, t, category=None):
            added.append(getattr(t, "name", "new"))
        def enable_category(self, c):
            pass

    reg = _Reg()
    new_agent = _agent_with(_MuJoCoG1)
    app = {"registry": reg, "engine": None}
    monkeypatch.setattr("vector_os_nano.vcli.tools.skill_wrapper.wrap_skills",
                        lambda agent: [types.SimpleNamespace(name="g1_walk")])
    SimStartTool._rebind_agent(app, _ctx(app), new_agent, None)
    assert "old_walk" in dropped and "file_read" not in dropped   # only skill wrappers
    assert "g1_walk" in added
    assert app["agent"] is new_agent
