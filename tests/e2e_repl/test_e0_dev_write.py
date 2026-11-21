# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E0 dev world: '在 <dir> 写一个 hello.txt', permission prompt answered 'y'.

The native producer runs the write behind the permission gate and grades it: the
session starts IN that (per-run temp) dir, because the dev file predicates only read
inside the working directory (an observational sandbox that never widens).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.harness.pty_cli import run_repl_session


@pytest.fixture(scope="module")
def e0_session(tmp_path_factory):
    target = tmp_path_factory.mktemp("vos_e0") / "hello.txt"
    # Reactive: each step fires on what the model just saw.
    script = {"steps": [
        {"when": "hello.txt", "tool_calls": [
            {"name": "file_write", "input": {"file_path": str(target), "content": "hello\n"}}]},
        {"when": "Wrote", "tool_calls": [
            {"name": "verify", "input": {"expr": f"path_contains('{target}', 'hello')"}}]},
        {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]}
    result = run_repl_session(
        [f"在 {target.parent} 写一个 hello.txt"], tool_script=script, answers=["y"],
        timeout_sec=120, cwd=target.parent,
    )
    return SimpleNamespace(text=result.text, timed_out=result.timed_out, target=target)


@pytest.mark.cli_main
def test_e0_permission_prompt_answered_then_file_written(e0_session) -> None:
    text = e0_session.text
    assert not e0_session.timed_out, text[-3000:]
    assert "Permission required" in text and "file_write" in text, text[-3000:]
    assert e0_session.target.read_text(encoding="utf-8") == "hello\n"


@pytest.mark.cli_main
def test_e0_grounded_path_contains(e0_session) -> None:
    text = e0_session.text
    assert "verify path_contains" in " ".join(text.split()), text[-3000:]
    assert "verdict GROUNDED verified=True" in text, text[-3000:]


# E0-deny: destructive dev commands are refused and never run, on the bare REPL.
@pytest.mark.cli_main
def test_e0_deny_destructive_commands_never_execute(tmp_path) -> None:
    keep = tmp_path / "vos_e0deny" / "keep.txt"
    keep.parent.mkdir()
    keep.write_text("keep\n", encoding="utf-8")
    script = {"steps": [
        # The bash deny-list refuses this outright: no prompt, no exec.
        {"when": "清空", "tool_calls": [{"name": "bash", "input": {"command": f"rm -rf {keep.parent}"}}]},
        # An 'ask' tool the user answers 'n' to: not run either.
        {"when": "删掉", "tool_calls": [{"name": "bash", "input": {"command": f"rm {keep}"}}]},
        {"when": "denied", "text": "没有执行", "tool_calls": []},
    ]}
    r = run_repl_session(
        [f"清空 {keep.parent}", f"删掉 {keep}"], tool_script=script,
        answers=["n"], timeout_sec=120, cwd=keep.parent,
    )
    text = r.text
    assert not r.timed_out, text[-3000:]
    assert "Dangerous command blocked" in text or "Permission denied" in text, text[-3000:]
    assert "Permission required" in text, text[-3000:]  # the second one was asked, and refused
    assert keep.read_text(encoding="utf-8") == "keep\n"  # the spy: nothing ran


# E0-read: a read request shows the model's answer (one producer, no verify nudges).
@pytest.mark.cli_main
def test_e0_a_read_request_prints_the_models_answer(tmp_path) -> None:
    doc = tmp_path / "vos_e0read" / "README.md"
    doc.parent.mkdir()
    doc.write_text("# E0READ-MARKER\nA demo project.\n", encoding="utf-8")
    script = {"steps": [
        {"when": "总结一下", "tool_calls": [{"name": "file_read", "input": {"file_path": str(doc)}}]},
        {"when": "E0READ-MARKER", "text": "README 摘要：E0READ 是一个演示项目。", "tool_calls": []},
    ]}
    log = tmp_path / "llm.jsonl"
    r = run_repl_session(
        ["总结一下 README.md 的内容"], tool_script=script, timeout_sec=120,
        cwd=doc.parent, extra_env={"VECTOR_FAKE_LLM_LOG": str(log)},
    )
    text = r.text
    assert not r.timed_out, text[-3000:]
    assert "README 摘要：E0READ 是一个演示项目。" in text, text[-3000:]
    calls = log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2, calls  # the read, the answer: no nudge, no second producer


# E0-chat: a question is the model's own answer: one LLM round-trip, no verdict claim,
# and nothing else re-plans the turn.
_CHAT_SCRIPT = {"steps": [
    {"when": "这个仓库", "text": "这是 Vector OS Nano：一个机器人 agent 运行时。", "tool_calls": []},
]}


@pytest.mark.cli_main
def test_e0_a_chat_question_is_the_models_text_in_one_round_trip(tmp_path) -> None:
    log = tmp_path / "llm.jsonl"
    r = run_repl_session(
        ["这个仓库是做什么的？"], tool_script=_CHAT_SCRIPT, timeout_sec=120, cwd=tmp_path,
        extra_env={"VECTOR_FAKE_LLM_LOG": str(log)},
    )
    text = r.text
    assert not r.timed_out, text[-3000:]
    assert "这是 Vector OS Nano：一个机器人 agent 运行时。" in text, text[-3000:]
    assert "verdict" not in text and "verified=" not in text, text[-3000:]
    calls = log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 1, calls  # the answer: no second producer, no extra LLM call
