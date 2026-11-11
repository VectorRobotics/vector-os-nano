# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The reactive fake LLM, and the tool_result feed the native recover loop rests on.

Regression (tests:fake-backend-open-loop): the open-loop FakeToolScriptBackend
ignored ``messages``, so nothing proved the native loop feeds a verify FAIL back to
the model. Here a recording backend branches on that FAIL: turn N+1 must see a
tool_result whose ``tool_use_id`` is turn N's verify call and whose content says FAIL.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.harness.fake_backend import FakeToolScriptBackend, ReactiveFakeBackend


class _WriteFileSkill:
    name = "write_file"
    description = "TEST-ONLY: write text content to a file."
    parameters = {
        "file_path": {"type": "string", "required": True},
        "content": {"type": "string", "required": True},
    }
    preconditions: list[str] = []
    effects: dict = {}

    def execute(self, params, context):
        from vector_os_nano.core.types import SkillResult

        Path(params["file_path"]).write_text(str(params["content"]), encoding="utf-8")
        return SkillResult(success=True, result_data={"written": params["file_path"]})


_VERIFY = {"name": "verify", "input": {"expr": "path_contains('out.txt', 'ready')"}}
_STEPS = [
    {"when": "create out.txt", "tool_calls": [_VERIFY]},          # 0: verify first -> FAIL
    {"when": "FAIL", "tool_calls": [                              # 1: recover on the FAIL
        {"name": "write_file", "input": {"file_path": "out.txt", "content": "ready\n"}}]},
    {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    {"when": "", "is_error": False, "tool_calls": [_VERIFY]},     # 3: re-verify after the write
]


def test_native_loop_feeds_verify_fail_back_with_matching_tool_use_id(tmp_path, monkeypatch) -> None:
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.permissions import PermissionContext
    from vector_os_nano.vcli.session import Session
    from vector_os_nano.vcli.tools.base import CategorizedToolRegistry
    from vector_os_nano.vcli.worlds.dev import DevWorld

    monkeypatch.chdir(tmp_path)
    agent = Agent(config={})
    agent._skill_registry.register(_WriteFileSkill())
    backend = ReactiveFakeBackend(_STEPS)
    eng = VectorEngine(backend=backend, registry=CategorizedToolRegistry(), permissions=PermissionContext(no_permission=True))
    eng.bind(agent, DevWorld())
    session = Session(session_id="reactive", created_at="t", updated_at="t", path=tmp_path / "s.jsonl")

    trace = eng.run_turn_native("create out.txt containing ready", session=session)

    # Call 2 is the model's view right after its first verify: the FAIL result, keyed
    # to that verify call's id.
    messages, _offered = backend.calls[1]
    last = messages[-1]
    assert last["role"] == "user"
    (result,) = [b for b in last["content"] if b.get("type") == "tool_result"]
    assert result["tool_use_id"] == "call_0_0"
    assert "FAIL" in result["content"]
    # The branch on that feed recovered: write, re-verify PASS, finish.
    assert [s.verify_result for s in trace.steps] == [False, True]
    assert Path("out.txt").read_text(encoding="utf-8") == "ready\n"
    assert len(backend.calls) == 4


def test_step_fires_only_when_its_tools_are_offered() -> None:
    backend = ReactiveFakeBackend([
        {"when": "go", "tool_calls": [{"name": "start_simulation", "input": {}}]},
        {"when": "go", "text": "fallback", "tool_calls": []},
    ])
    user = [{"role": "user", "content": "go now"}]
    native = backend.call(user, [{"name": "verify"}], [], 100)
    legacy = backend.call(user, [{"name": "start_simulation"}], [], 100)
    assert native.text == "fallback" and not native.tool_calls
    assert [tc.name for tc in legacy.tool_calls] == ["start_simulation"]
    # Exhausted / unmatched -> a visible end_turn, never a stale replay.
    done = backend.call(user, [{"name": "start_simulation"}], [], 100)
    assert done.tool_calls == [] and "no step matched" in done.text


def test_is_error_condition_and_json_seam(tmp_path) -> None:
    path = tmp_path / "script.json"
    path.write_text(json.dumps({"steps": [
        {"when": "boom", "is_error": True, "text": "saw the error", "tool_calls": []},
    ]}), encoding="utf-8")
    backend = FakeToolScriptBackend.from_json_file(path)  # the cli.py seam
    assert isinstance(backend, ReactiveFakeBackend)
    ok = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "boom"}]}]
    err = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "boom",
                                         "is_error": True}]}]
    assert "no step matched" in backend.call(ok, [], [], 1).text
    assert backend.call(err, [], [], 1).text == "saw the error"


def test_bad_payload_fails_loud(tmp_path) -> None:
    path = tmp_path / "script.json"
    path.write_text(json.dumps({"steps": [{"tool_calls": [{"input": {}}]}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="must name a tool"):
        FakeToolScriptBackend.from_json_file(path)
