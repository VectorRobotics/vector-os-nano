# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""A2 MCP stdio smoke: the real ``vector-os-mcp`` subprocess, driven by the SDK client.

One server, one session: tools/list, a dev-world turn, an NL simulator launch, a
grasp graded GROUNDED with goal_check proven (the same verdict as ``-p``'s
VECTOR_VERDICT line), status, the camera frame and the P0 stop. A second server,
go2: stop() mid-walk never waits for the running turn, and run('停') is the same P0 stop.
"""
from __future__ import annotations

import json
import re
import time

import anyio
import pytest

from tests.harness.mcp_client import drive_mcp, run

_SCRIPT = {"steps": [
    {"when": "你好", "text": "你好！", "tool_calls": [], "stop_reason": "end_turn"},
    {"when": "启动机械臂仿真", "tool_calls": [
        {"name": "start_simulation", "input": {"embodiment": "so101", "gui": False}}]},
    {"when": "", "is_error": False, "text": "机械臂仿真已启动", "tool_calls": []},
    {"when": "抓起香蕉", "tool_calls": [
        {"name": "grasp", "input": {"query": "banana"}}]},
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


_WALK = {"steps": [
    # 5 m at 0.3 m/s: ~17 s of walking, far longer than the stop takes.
    {"when": "往前走五米", "tool_calls": [
        {"name": "walk", "input": {"direction": "forward", "distance": 5.0, "speed": 0.3}}]},
    {"when": "", "text": "walked", "tool_calls": [], "stop_reason": "end_turn"},
]}
_SETTLED = re.compile(r"no LLM call\); base [\d.]+ m/s at 0\.2 sim-s, peak [\d.]+ m/s to 1\.0 sim-s, "
                      r"([\d.]+) m/s at 1\.0 sim-s")


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.timeout(600)
def test_mcp_stop_mid_turn_never_waits_for_the_turn_and_a_stop_phrase_is_p0(tmp_path, monkeypatch) -> None:
    log = tmp_path / "llm_calls.jsonl"
    monkeypatch.setenv("VECTOR_FAKE_LLM_LOG", str(log))
    monkeypatch.setenv("VECTOR_NAV_STACK", "")

    async def calls(c):
        out = {}

        async def walk():
            out["walk"] = await run(c, "往前走五米")

        async def stopper():
            with anyio.fail_after(60):
                while not (await c.call_tool("status", {})).structured_content["robot_busy"]:
                    await anyio.sleep(0.2)
            await anyio.sleep(2.0)  # the body is trotting
            out["mid"] = (await c.call_tool("status", {})).structured_content
            t0 = time.monotonic()
            out["stop"] = (await c.call_tool("stop", {})).structured_content["result"]
            out["stop_s"] = time.monotonic() - t0

        async with anyio.create_task_group() as tg:
            tg.start_soon(walk)
            tg.start_soon(stopper)
        out["p0"] = await run(c, "停")
        return out

    out = drive_mcp(tmp_path, _WALK, calls, args=("--embodiment", "go2", "--headless"))
    assert out["mid"]["turn_running"] is True, out["mid"]  # stop() landed while run() held the turn
    # (1) stop() never waits for the turn: measured 1.0 s, which is the speed report's
    # 1.0 sim-s of sampling; the walk alone had ~15 s left.
    assert out["stop_s"] < 3.0, out
    m = _SETTLED.search(out["stop"])
    assert out["stop"].startswith("stopped (") and m, out["stop"]
    assert float(m.group(1)) < 0.05, out["stop"]  # still at 1.0 sim-s (E5's settled bound)
    # (2) the running turn's walk was aborted, and nothing is claimed.
    v = out["walk"]["verdict"]
    assert v["verified"] is False and v["evidence"] == "FAILED", v
    assert [s["strategy"] for s in v["per_step"]] == ["walk"], v
    assert "aborted" in v["per_step"][0]["reject_reason"], v
    # (3) a stop phrase through run() is the same P0 stop: no LLM call, no trace.
    p0 = out["p0"]
    assert p0["answer"].startswith("stopped (") and "no LLM call" in p0["answer"], p0
    assert p0["verdict"]["evidence"] == "NO_TRACE" and p0["verdict"]["verified"] is False, p0
    assert p0["verdict"]["error"] == "stopped (P0, no LLM call)" and p0["refused"] == [], p0
    obs = [json.loads(line)["obs"] for line in log.read_text(encoding="utf-8").splitlines()]
    assert obs == ["往前走五米"], obs  # the walk's one call; '停' never reached the model
