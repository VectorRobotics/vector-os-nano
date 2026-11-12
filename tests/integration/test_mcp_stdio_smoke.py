# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""A2 MCP stdio smoke: the real ``vector-os-mcp`` subprocess, driven by the SDK client.

One server, one session: tools/list, a dev-world turn, an NL simulator launch, a
grasp graded GROUNDED with goal_check proven (the same verdict as ``-p``'s
VECTOR_VERDICT line), status, the camera frame and the P0 stop.
"""
from __future__ import annotations

import pytest

from tests.harness.mcp_client import drive_mcp, run

_SCRIPT = {"steps": [
    {"when": "你好", "text": "你好！", "tool_calls": [], "stop_reason": "end_turn"},
    {"when": "启动机械臂仿真", "tool_calls": [
        {"name": "start_simulation", "input": {"embodiment": "so101", "gui": False}}]},
    {"when": "", "is_error": False, "text": "机械臂仿真已启动", "tool_calls": []},
    {"when": "抓起香蕉", "tool_calls": [
        {"name": "pick", "input": {"object_label": "banana", "mode": "hold"}}]},
    {"when": "", "tool_calls": [{"name": "verify", "input": {"expr": "holding_object('banana')"}}]},
    {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
]}


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.timeout(600)
def test_mcp_stdio_dev_turn_then_sim_grasp_goal_proven(tmp_path) -> None:
    async def calls(c):
        out = {"tools": {t.name for t in (await c.list_tools()).tools}}
        out["chat"] = await run(c, "你好")
        out["launch"] = await run(c, "启动机械臂仿真 无窗口")
        out["grasp"] = await run(c, "抓起香蕉")
        out["status"] = (await c.call_tool("status", {})).structured_content
        out["camera"] = (await c.read_resource("camera://frame")).contents[0]
        out["stop"] = (await c.call_tool("stop", {})).structured_content["result"]
        return out

    out = drive_mcp(tmp_path, _SCRIPT, calls)
    assert out["tools"] == {"run", "stop", "status"}
    assert out["chat"]["answer"] == "你好！" and out["chat"]["verdict"]["evidence"] == "NO_TRACE"
    assert out["launch"]["refused"] == [], out["launch"]  # start_simulation is the CLI's allow
    v = out["grasp"]["verdict"]
    assert v["verified"] is True and v["evidence"] == "GROUNDED", v
    assert v["goal_check"] == "proven" and v["contract"] == ["holding_object('banana')"], v
    assert [(s["verify"], s["actor_caused"]) for s in v["per_step"]] == [
        ("holding_object('banana')", "CAUSED")], v
    assert out["grasp"]["refused"] == []  # a sim body's motor skills: the CLI's auto-allow
    assert out["status"]["body"] == "so101" and "ready in" in out["status"]["sim"]
    cam = out["camera"]
    assert cam.mime_type == "image/png" and cam.blob  # base64 PNG of the body's camera
    assert out["stop"].startswith("stopped (")
