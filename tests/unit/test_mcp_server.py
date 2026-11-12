# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""vector-os-mcp (hermetic): the loopback-only bind, a dev-world stdio smoke, and the
permission rule (nobody can be asked over MCP: an ``ask`` is refused unless --allow
names the tool; an intrinsic deny is never lifted).

The server is the real ``python -m vector_os_nano.mcp`` driven by the SDK's client;
the simulator half of the smoke is tests/e2e_repl/test_e13_mcp_stdio.py.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from tests.harness.mcp_client import drive_mcp, run
from vector_os_nano.mcp.server import LOOPBACK, bind_refusal


@pytest.mark.parametrize("transport", ["sse", "streamable-http"])
@pytest.mark.parametrize("host", ["0.0.0.0", "::", "", "192.168.1.5", "127.0.0.2",
                                  "localhost.attacker.example", "127.0.0.1 "])
def test_a_network_transport_refuses_every_non_loopback_host(transport, host) -> None:
    why = bind_refusal(transport, host)
    assert why and repr(host) in why and "authentication" in why


@pytest.mark.parametrize("host", LOOPBACK)
def test_loopback_binds_and_stdio_never_binds(host) -> None:
    assert bind_refusal("streamable-http", host) == "" and bind_refusal("sse", host) == ""
    assert bind_refusal("stdio", "0.0.0.0") == ""  # stdio opens no socket


def test_the_server_exits_before_building_anything_on_a_public_bind() -> None:
    r = subprocess.run([sys.executable, "-m", "vector_os_nano.mcp", "--transport", "streamable-http",
                        "--host", "0.0.0.0"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2, r
    assert "refusing to listen on '0.0.0.0'" in r.stderr and "authentication" in r.stderr
    assert "Starting" not in r.stderr and r.stdout == ""


def test_stdio_smoke_dev_world(tmp_path) -> None:
    script = {"steps": [{"when": "你好", "text": "你好！", "tool_calls": [], "stop_reason": "end_turn"}]}

    async def calls(c):
        tools = {t.name for t in (await c.list_tools()).tools}
        resources = [str(r.uri) for r in (await c.list_resources()).resources]
        status = (await c.call_tool("status", {})).structured_content
        turn = await run(c, "你好")
        stopped = (await c.call_tool("stop", {})).structured_content["result"]
        try:
            await c.read_resource("camera://frame")
            camera = "read"
        except Exception as exc:  # noqa: BLE001 — the SDK raises the server's error
            camera = str(exc)
        return tools, resources, status, turn, stopped, camera

    tools, resources, status, turn, stopped, camera = drive_mcp(tmp_path, script, calls)
    assert tools == {"run", "stop", "status"}  # skills are the model's choice inside run
    assert resources == ["camera://frame"]
    assert status["body"] == "none (dev world)" and status["world"] == "dev"
    assert status["turn_running"] is False and status["allowed"] == []
    assert turn["answer"] == "你好！"
    v = turn["verdict"]  # a chat turn: the -p VECTOR_VERDICT shape, nothing claimed
    assert v["evidence"] == "NO_TRACE" and v["verified"] is False and v["goal_check"] == "unchecked"
    assert v["goal"] == "你好" and v["error"] == "" and turn["refused"] == []
    assert stopped.startswith("stopped (") and "no LLM call" in stopped
    assert "no camera" in camera


def _write_turn(tmp_path, args: tuple[str, ...]):
    target = tmp_path / "hello.txt"
    script = {"steps": [
        {"when": "hello.txt", "tool_calls": [
            {"name": "file_write", "input": {"file_path": str(target), "content": "hello\n"}}]},
        {"when": "Wrote", "tool_calls": [
            {"name": "verify", "input": {"expr": f"path_contains('{target}', 'hello')"}}]},
        {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
        {"when": "Permission denied", "text": "I was not allowed to write it.", "tool_calls": [],
         "stop_reason": "end_turn"},
    ]}
    turn = drive_mcp(tmp_path, script, lambda c: run(c, f"在 {tmp_path} 写一个 hello.txt"),
                     args=args, cwd=tmp_path)
    return turn, target


@pytest.mark.parametrize("args", [(), ("--allow", "bash,file_edit")], ids=["no-allow", "other-tools"])
def test_an_ask_is_refused_with_the_reason_unless_allowed_by_name(tmp_path, args) -> None:
    turn, target = _write_turn(tmp_path, args)
    assert not target.exists()  # never silently allowed
    assert any(r.startswith("file_write:") for r in turn["refused"]), turn
    assert "--allow" in turn["note"]
    assert turn["verdict"]["verified"] is False


def test_the_allow_flag_allows_exactly_the_named_tool(tmp_path) -> None:
    turn, target = _write_turn(tmp_path, ("--allow", "file_write"))
    assert target.read_text(encoding="utf-8") == "hello\n"
    assert turn["refused"] == [] and "note" not in turn
    v = turn["verdict"]
    assert v["verified"] is True and v["evidence"] == "GROUNDED", v
    assert [s["verify"] for s in v["per_step"]] == [f"path_contains('{target}', 'hello')"]


def test_allow_never_lifts_an_intrinsic_deny(tmp_path) -> None:
    keep = tmp_path / "keep" / "keep.txt"
    keep.parent.mkdir()
    keep.write_text("keep\n", encoding="utf-8")
    script = {"steps": [
        {"when": "清空", "tool_calls": [{"name": "bash", "input": {"command": f"rm -rf {keep.parent}"}}]},
        {"when": "", "text": "refused", "tool_calls": [], "stop_reason": "end_turn"},
    ]}
    turn = drive_mcp(tmp_path, script, lambda c: run(c, f"清空 {keep.parent}"),
                     args=("--allow", "bash"), cwd=tmp_path)
    assert keep.read_text(encoding="utf-8") == "keep\n"
    assert any(r.startswith("bash:") for r in turn["refused"]), turn
