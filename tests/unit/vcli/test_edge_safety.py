# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Safety at the edges: the permission prompt's default, unreadable tool-call
arguments, the sim lifecycle tools' declared permission, a bash child that outlives
its caller."""
from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.unit.vcli.test_native_loop import _session
from tests.unit.vcli.test_native_safety import _dev_engine, _script, _SpyTool
from vector_os_nano.vcli import repl
from vector_os_nano.vcli.backends.fake import tool_turn
from vector_os_nano.interfaces.llm import LLMResponse, LLMToolCall


def test_the_permission_prompt_defaults_to_no(monkeypatch) -> None:
    """cli-entry:permission-prompt-default-allow: Enter alone (or type-ahead) never
    approves bash / file_write / robot motion."""
    seen: dict[str, Any] = {}
    monkeypatch.setattr(repl.Prompt, "ask", lambda *a, **k: seen.update(k) or k["default"])
    monkeypatch.setattr(repl.console, "file", open(os.devnull, "w"))
    assert repl.ask_permission("bash", {"command": "rm -rf build"}) == "n"
    assert seen["default"] == "n"


def test_an_unreadable_tool_call_is_an_error_the_model_sees_never_a_dispatch() -> None:
    """producers:malformed-toolcall-runs-with-defaults: a truncated argument string used
    to run the tool as {'_raw': ...}, i.e. with its defaults."""
    spy = _SpyTool("walk", "allow")
    bad = LLMResponse(text="", tool_calls=[LLMToolCall("c1", "walk", {}, parse_error="arguments are not "
                                                       "valid JSON (Unterminated string)")], stop_reason="tool_use")
    session = _session()
    eng = _dev_engine(_script(bad, tool_turn(end=True)), spy)
    trace = eng.run_turn_native("walk 2 m", session=session)
    assert spy.calls == 0 and trace.actions_dispatched == 0
    result = next(r for e in session._entries if e["type"] == "tool_result" for r in e["results"])
    assert result["is_error"] and "walk was not run: its arguments are not valid JSON" in result["content"]


@pytest.mark.parametrize("arguments, error", [('{"distance": 2', "not valid JSON"), ("[1, 2]", "not a JSON object")])
def test_the_openai_compatible_backend_marks_unreadable_arguments(arguments, error) -> None:
    from vector_os_nano.vcli.backends.openai_compat import OpenAICompatBackend

    with patch("openai.OpenAI"):
        backend = OpenAICompatBackend(api_key="k", model="m")
    fn = SimpleNamespace(name="walk", arguments=arguments)
    delta = SimpleNamespace(content=None, reasoning_content=None, reasoning=None,
                            tool_calls=[SimpleNamespace(index=0, id="c1", function=fn)])
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason="tool_calls")], usage=None)
    backend._client = MagicMock()
    backend._client.chat.completions.create.return_value = iter([chunk])
    call = backend.call(messages=[{"role": "user", "content": "go"}], tools=[], system=[], max_tokens=64).tool_calls[0]
    assert call.input == {} and error in call.parse_error


@pytest.mark.parametrize("cls", ["SimStartTool", "SimStopTool"])
def test_the_sim_lifecycle_tools_do_what_they_declare(cls) -> None:
    """tools-worlds:start-sim-permission-mismatch: the decorator said 'ask' while
    check_permissions allowed; declared and enforced permission are one value now."""
    from vector_os_nano.vcli.tools import sim_tool

    tool = getattr(sim_tool, cls)()
    assert tool.__tool_permission__ == tool.check_permissions({}, None).behavior == "allow"


@pytest.mark.parametrize("body", ["so101", "piper", "go2"])
def test_a_window_without_mjpython_runs_headless_and_never_replaces_the_repl(monkeypatch, body) -> None:
    """The NL start used to os.execve the whole CLI under mjpython (the conversation lost);
    every body falls back the same way (it was the so101 alone)."""
    from vector_os_nano.sim import session as sim_session
    from vector_os_nano.vcli import runtime
    from vector_os_nano.vcli.tools.base import ToolContext
    from vector_os_nano.vcli.tools.sim_tool import SimStartTool

    started: list[Any] = []
    fake = SimpleNamespace(summary=lambda: "ready", stop=lambda: "stopped", agent=None)
    monkeypatch.setattr(sim_session.SimSession, "validate", staticmethod(lambda spec: None))
    monkeypatch.setattr(sim_session.SimSession, "start", staticmethod(lambda spec, **_: started.append(spec) or fake))
    monkeypatch.setattr(runtime, "bind_body", lambda app, session: None)
    monkeypatch.setattr(SimStartTool, "_viewer_available", staticmethod(lambda: False))
    monkeypatch.setattr(os, "execve", lambda *a: pytest.fail("the REPL was replaced"))
    ctx = ToolContext(agent=None, cwd=Path.cwd(), session=None, permissions=None,
                      abort=threading.Event(), app_state={})
    res = SimStartTool().execute({"embodiment": body, "gui": True}, ctx)
    assert not res.is_error and started[0].gui is False and "needs mjpython" in res.content


def _alive(marker: str) -> list[int]:
    out = []
    for pid in os.listdir("/proc"):
        try:
            if pid.isdigit() and marker in Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode():
                out.append(int(pid))
        except OSError:
            continue
    return out


def test_a_bash_child_dies_with_its_caller() -> None:
    """A SIGTERM (SystemExit) or Ctrl-C landing while bash waits kills its process
    group, so -p and the REPL leave no child behind."""
    import time

    from vector_os_nano.vcli.tools.bash_tool import BashTool
    from vector_os_nano.vcli.tools.base import ToolContext

    marker = "987.6543"
    calls = {"n": 0}

    class _Term:
        def is_set(self) -> bool:
            calls["n"] += 1
            if calls["n"] > 3:
                raise SystemExit(143)
            return False

    ctx = ToolContext(agent=None, cwd=Path.cwd(), session=None, permissions=None, abort=_Term(), app_state={})
    with pytest.raises(SystemExit):
        BashTool().execute({"command": f"sleep {marker}"}, ctx)
    deadline = time.monotonic() + 5
    while _alive(marker) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert _alive(marker) == []


def test_a_raising_verify_tells_the_model_why(tmp_path, monkeypatch) -> None:
    """R1:verify-error-text-lost: a predicate that raised verified FAIL with no reason;
    the tool result now carries the exception's type and message (never a value)."""
    monkeypatch.chdir(tmp_path)
    session = _session()
    eng = _dev_engine(_script(tool_turn(("deploy", {})),
                              tool_turn(("verify", {"expr": "path_contains('a.py', '')"})),
                              tool_turn(("deploy", {})),
                              tool_turn(("verify", {"expr": "file_exists('a.py') + "})),
                              tool_turn(end=True)), _SpyTool("deploy", "allow"))
    eng.run_turn_native("deploy", session=session)
    results = [r["content"] for e in session._entries if e["type"] == "tool_result" for r in e["results"]
               if r["content"].startswith("verify(")]
    assert "FAIL" in results[0] and \
        "could not be evaluated: ValueError: path_contains needs a non-empty substring" in results[0]
    assert "could not be evaluated: SyntaxError" in results[1]


def test_navigates_hint_comes_from_the_arrival_oracle() -> None:
    """MR1V:kernel-at-position-text: the hint binds the oracle's own parameters and the
    description quotes the oracle's own signature and docstring; no oracle, no hint."""
    from vector_os_nano.vcli.native_tools import _NativeBaseNavigateTool

    def at_position(x: float, y: float, tol: float = 0.3) -> bool:
        """True when the base is within tol metres of (x, y)."""
        return True

    tool = _NativeBaseNavigateTool(at_position)
    assert tool.verify_hint == "at_position({x}, {y})"
    assert "at_position(x, y, tol=0.3): True when the base is within tol metres" in tool.description

    def renamed(px: float, py: float) -> bool:
        return True

    assert _NativeBaseNavigateTool(renamed).verify_hint == ""  # nothing binds: no guess
    assert _NativeBaseNavigateTool().verify_hint == "" and "verified" not in _NativeBaseNavigateTool().description


def test_navigate_auto_allows_only_a_live_sim_body_of_its_world() -> None:
    """NL1: the kernel asks the bound world whether the body is a live sim (it imports
    no simulator). Only a body a live session owns, asked of the robot world, is
    auto-allowed; a body with no session, the dev world or no world at all asks."""
    from tests.harness.fake_gt import bind_fake_sim, unbind_fake_sim
    from vector_os_nano.vcli.native_tools import _NativeBaseNavigateTool
    from vector_os_nano.vcli.worlds.dev import DevWorld
    from vector_os_nano.vcli.worlds.robot import RobotWorld

    body, stray = SimpleNamespace(), SimpleNamespace()
    sim = bind_fake_sim(body, "go2")
    try:
        def asked(world: Any, agent: Any) -> str:
            return _NativeBaseNavigateTool(None, world).check_permissions({}, SimpleNamespace(agent=agent)).behavior

        assert asked(RobotWorld(), body) == "allow"
        assert asked(RobotWorld(), stray) == "ask"
        assert asked(RobotWorld(), None) == "ask"
        assert asked(DevWorld(), body) == "ask"
        assert asked(None, body) == "ask"
    finally:
        unbind_fake_sim(sim)
    assert _NativeBaseNavigateTool(None, RobotWorld()).check_permissions(
        {}, SimpleNamespace(agent=body)).behavior == "ask"  # the session ended
