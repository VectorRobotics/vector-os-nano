# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E0 dev world: '在 /tmp/vos_e0 写一个 hello.txt', permission prompt answered 'y'."""
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
    result = run_repl_session([_PROMPT], tool_script=_SCRIPT, answers=["y"], turn_timeout_sec=120)
    yield result
    shutil.rmtree(_TARGET.parent, ignore_errors=True)


@pytest.mark.cli_main
def test_e0_permission_prompt_answered_then_file_written(e0_session) -> None:
    text = e0_session.text
    assert not e0_session.timed_out, text[-3000:]
    assert "Permission required" in text and "file_write" in text, text[-3000:]
    assert _TARGET.read_text(encoding="utf-8") == "hello\n"


@pytest.mark.cli_main
@pytest.mark.xfail(strict=True, reason=(
    "WP05: a dev-world action routes to the tool_use path, which renders no verdict "
    "(the native producer that would verify it has no permission gate yet)"))
def test_e0_grounded_path_contains(e0_session) -> None:
    text = e0_session.text
    assert "verify path_contains" in text, text[-3000:]
    assert "verdict GROUNDED verified=True" in text, text[-3000:]
