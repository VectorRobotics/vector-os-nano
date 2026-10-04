# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E0 dev world: '在 /tmp/vos_e0 写一个 hello.txt', permission prompt answered 'y'.

The native producer runs the write behind the permission gate and grades it: the
session starts IN /tmp/vos_e0, because the dev file predicates only read inside the
working directory (an observational sandbox that never widens).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.harness.pty_cli import run_repl_session

_TARGET = Path("/tmp/vos_e0/hello.txt")
_PROMPT = f"在 {_TARGET.parent} 写一个 hello.txt"
# Reactive: the step fires on what the model just saw. The native producer gets the
# verify/finish steps; the tool_use path (no verify tool offered) gets the text step.
_SCRIPT = {"steps": [
    {"when": "hello.txt", "tool_calls": [
        {"name": "file_write", "input": {"file_path": str(_TARGET), "content": "hello\n"}}]},
    {"when": "Wrote", "tool_calls": [
        {"name": "verify", "input": {"expr": f"path_contains('{_TARGET}', 'hello')"}}]},
    {"when": "PASS", "tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    {"when": "Wrote", "text": "已写入 hello.txt", "tool_calls": []},
]}


@pytest.fixture(scope="module")
def e0_session():
    shutil.rmtree(_TARGET.parent, ignore_errors=True)
    _TARGET.parent.mkdir(parents=True)
    result = run_repl_session(
        [_PROMPT], tool_script=_SCRIPT, answers=["y"], timeout_sec=120, cwd=_TARGET.parent,
    )
    yield result
    shutil.rmtree(_TARGET.parent, ignore_errors=True)


@pytest.mark.cli_main
def test_e0_permission_prompt_answered_then_file_written(e0_session) -> None:
    text = e0_session.text
    assert not e0_session.timed_out, text[-3000:]
    assert "Permission required" in text and "file_write" in text, text[-3000:]
    assert _TARGET.read_text(encoding="utf-8") == "hello\n"


@pytest.mark.cli_main
def test_e0_grounded_path_contains(e0_session) -> None:
    text = e0_session.text
    assert "verify path_contains" in text, text[-3000:]
    assert "verdict GROUNDED verified=True" in text, text[-3000:]


# E0-deny: destructive dev commands are refused and never run, on the bare REPL.
_KEEP = Path("/tmp/vos_e0deny/keep.txt")
_DENY_SCRIPT = {"steps": [
    # The bash deny-list refuses this outright: no prompt, no exec.
    {"when": "清空", "tool_calls": [{"name": "bash", "input": {"command": f"rm -rf {_KEEP.parent}"}}]},
    # An 'ask' tool the user answers 'n' to: not run either.
    {"when": "删掉", "tool_calls": [{"name": "bash", "input": {"command": f"rm {_KEEP}"}}]},
    {"when": "denied", "text": "没有执行", "tool_calls": []},
]}


@pytest.mark.cli_main
def test_e0_deny_destructive_commands_never_execute() -> None:
    shutil.rmtree(_KEEP.parent, ignore_errors=True)
    _KEEP.parent.mkdir(parents=True)
    _KEEP.write_text("keep\n", encoding="utf-8")
    try:
        r = run_repl_session(
            [f"清空 {_KEEP.parent}", f"删掉 {_KEEP}"], tool_script=_DENY_SCRIPT,
            answers=["n"], timeout_sec=120, cwd=_KEEP.parent,
        )
        text = r.text
        assert not r.timed_out, text[-3000:]
        assert "Dangerous command blocked" in text or "Permission denied" in text, text[-3000:]
        assert "Permission required" in text, text[-3000:]  # the second one was asked, and refused
        assert _KEEP.read_text(encoding="utf-8") == "keep\n"  # the spy: nothing ran
    finally:
        shutil.rmtree(_KEEP.parent, ignore_errors=True)


# E0-read: a read request shows the model's answer (one producer, no verify nudges).
_DOC = Path("/tmp/vos_e0read/README.md")
_READ_SCRIPT = {"steps": [
    {"when": "总结一下", "tool_calls": [{"name": "file_read", "input": {"file_path": str(_DOC)}}]},
    {"when": "E0READ-MARKER", "text": "README 摘要：E0READ 是一个演示项目。", "tool_calls": []},
]}


@pytest.mark.cli_main
def test_e0_a_read_request_prints_the_models_answer(tmp_path) -> None:
    shutil.rmtree(_DOC.parent, ignore_errors=True)
    _DOC.parent.mkdir(parents=True)
    _DOC.write_text("# E0READ-MARKER\nA demo project.\n", encoding="utf-8")
    log = tmp_path / "llm.jsonl"
    try:
        r = run_repl_session(
            ["总结一下 README.md 的内容"], tool_script=_READ_SCRIPT, timeout_sec=120,
            cwd=_DOC.parent, extra_env={"VECTOR_FAKE_LLM_LOG": str(log)},
        )
        text = r.text
        assert not r.timed_out, text[-3000:]
        assert "README 摘要：E0READ 是一个演示项目。" in text, text[-3000:]
        calls = log.read_text(encoding="utf-8").splitlines()
        assert len(calls) == 2, calls  # the read, the answer: no nudge, no second producer
    finally:
        shutil.rmtree(_DOC.parent, ignore_errors=True)
