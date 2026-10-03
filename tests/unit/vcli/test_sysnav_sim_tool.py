# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""SysnavSimTool unit tests (v2.4 T7).

Stubs SimStartTool + LiveSysnavBridge so the orchestrator's behaviour
is tested in isolation. No real MuJoCo or rclpy is involved here.
"""
from __future__ import annotations

import builtins
import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from vector_os_nano.core.world_model import WorldModel
from vector_os_nano.vcli.tools import ToolContext, ToolResult, discover_all_tools
from vector_os_nano.vcli.tools.sysnav_sim_tool import SysnavSimTool


def _ctx(agent: object) -> ToolContext:
    """Build a minimal ToolContext for tests."""
    return ToolContext(
        agent=agent,
        cwd=Path("/tmp"),
        session=SimpleNamespace(),
        permissions=SimpleNamespace(),
        abort=threading.Event(),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubSim:
    """Mimics SimStartTool.execute returning a benign ToolResult."""

    def __init__(self) -> None:
        self.executions: list[dict] = []

    def execute(self, params, context):
        self.executions.append(dict(params))
        return ToolResult(content="sim ok", metadata={"booted": True})


class _StubBridge:
    def __init__(self, world_model, *, start_returns: bool = True) -> None:
        self.world_model = world_model
        self.started = False
        self.stopped = False
        self._start_returns = start_returns

    def start(self) -> bool:
        self.started = True
        return self._start_returns

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def agent_with_world_model():
    return SimpleNamespace(_world_model=WorldModel())


@pytest.fixture
def context(agent_with_world_model):
    return _ctx(agent_with_world_model)


def _build_tool(*, bridge_start: bool = True) -> tuple[SysnavSimTool, _StubSim, list[_StubBridge]]:
    """Return tool with stubbed sim + bridge factories. Bridges captured
    for assertion (a list because some tests call execute twice)."""
    stub_sim = _StubSim()
    bridges: list[_StubBridge] = []

    def bridge_factory(wm):
        b = _StubBridge(wm, start_returns=bridge_start)
        bridges.append(b)
        return b

    tool = SysnavSimTool(sim_tool_factory=lambda: stub_sim)
    tool.set_bridge_factory(bridge_factory)
    return tool, stub_sim, bridges


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


def test_preflight_when_tare_planner_msg_present(monkeypatch) -> None:
    """A real-or-stubbed `tare_planner.msg` import succeeds → True."""
    fake_module = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "tare_planner", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "tare_planner.msg", fake_module)
    assert SysnavSimTool.preflight_sysnav_workspace() is True


def test_preflight_when_tare_planner_msg_absent(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("tare_planner"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "tare_planner", raising=False)
    monkeypatch.delitem(sys.modules, "tare_planner.msg", raising=False)
    assert SysnavSimTool.preflight_sysnav_workspace() is False


# ---------------------------------------------------------------------------
# execute() — happy path
# ---------------------------------------------------------------------------


def test_run_starts_sim_then_bridge(context) -> None:
    tool, stub_sim, bridges = _build_tool(bridge_start=True)
    result = tool.execute({"gui": False}, context)
    # Sim called exactly once with our params merged in
    assert len(stub_sim.executions) == 1
    assert stub_sim.executions[0]["embodiment"] == "go2_piper"
    # Bridge constructed and started
    assert len(bridges) == 1
    assert bridges[0].started is True
    assert "bridge live" in result.content or "MuJoCo + bridge live" in result.content
    assert result.metadata["bridge_active"] is True


def test_run_continues_when_bridge_start_fails(context, caplog) -> None:
    tool, _stub_sim, bridges = _build_tool(bridge_start=False)
    with caplog.at_level(logging.WARNING):
        result = tool.execute({}, context)
    # Bridge construction succeeded, but start returned False
    assert len(bridges) == 1
    assert bridges[0].started is True
    # Tool result still reports completion (no exception)
    assert result.metadata["bridge_active"] is False


def test_run_continues_when_agent_has_no_world_model() -> None:
    """If the agent doesn't expose a WorldModel, bridge is skipped.

    Tool still reports sim started.
    """
    tool = SysnavSimTool(sim_tool_factory=lambda: _StubSim())
    ctx = _ctx(SimpleNamespace())                # no _world_model attr
    result = tool.execute({}, ctx)
    assert "bridge_active" in result.metadata
    assert result.metadata["bridge_active"] is False


def test_double_run_idempotent_returns_already_running_message(context) -> None:
    tool, stub_sim, bridges = _build_tool()
    tool.execute({}, context)
    second = tool.execute({}, context)
    # Second call must not boot another sim
    assert len(stub_sim.executions) == 1
    assert len(bridges) == 1
    assert "already running" in second.content


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------


def test_stop_cleans_bridge_then_sim(context, monkeypatch) -> None:
    tool, _stub_sim, bridges = _build_tool()
    tool.execute({}, context)

    # Patch SimStopTool that stop() will lazy-import
    class _StubStop:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, params, ctx):
            self.calls += 1
            return ToolResult(content="stopped")

    stub_stop = _StubStop()
    monkeypatch.setattr(tool, "_build_sim_stop_tool", lambda: stub_stop)

    tool.stop()
    assert bridges[0].stopped is True
    assert stub_stop.calls == 1
    assert tool._started is False


def test_stop_idempotent_when_never_started() -> None:
    tool = SysnavSimTool()
    tool.stop()
    tool.stop()              # must not raise


# ---------------------------------------------------------------------------
# Tool descriptor / registration
# ---------------------------------------------------------------------------


def test_help_text_lists_required_topics() -> None:
    desc = SysnavSimTool.description
    assert "/object_nodes_list" in desc
    assert "/registered_scan" in desc or "/state_estimation" in desc
    assert "/camera/image" in desc


def test_tool_paused_not_registered_in_init() -> None:
    # SysNav sim integration PAUSED for the master merge — v2.4 T6
    # (vnav_bridge wiring) + G5 (launch) were not delivered. The tool
    # class is retained in-tree but intentionally NOT registered.
    # Flip this back to `in names` when v2.4 resumes.
    tools = discover_all_tools()
    names = {getattr(t, "name") for t in tools}
    assert "start_sysnav_sim" not in names
    # Code retained, not deleted — class still importable.
    from vector_os_nano.vcli.tools.sysnav_sim_tool import SysnavSimTool

    assert SysnavSimTool.__name__ == "SysnavSimTool"


def test_input_schema_documents_optional_gui_and_backend() -> None:
    schema = SysnavSimTool.input_schema
    assert "gui" in schema["properties"]
    assert "backend" in schema["properties"]
    assert schema["required"] == []


def test_pre_flight_log_warning_when_missing(context, monkeypatch, caplog) -> None:
    """Pre-flight failure surfaces a log line so users know to source SysNav."""
    monkeypatch.setattr(
        SysnavSimTool, "preflight_sysnav_workspace",
        staticmethod(lambda: False),
    )
    tool, _, _ = _build_tool()
    with caplog.at_level(logging.WARNING):
        tool.execute({}, context)
    assert any("tare_planner.msg" in r.message for r in caplog.records)
