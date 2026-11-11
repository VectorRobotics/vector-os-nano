# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Bare ``vector-cli`` on a PTY with no key: the first run prints ONE instruction
naming the variable to set, every slash command runs, an NL line is answered with the
instruction (no traceback), and the REPL exits cleanly.
"""
from __future__ import annotations

import pytest

from tests.harness.pty_cli import run_repl_session
from vector_os_nano.vcli.slash import COMMANDS

INSTRUCTION = "No DEEPSEEK_API_KEY set"


@pytest.mark.cli_main
@pytest.mark.timeout(180)
def test_first_run_without_a_key_and_every_slash_command() -> None:
    lines = [f"/{c.name}" for c in COMMANDS if c.name != "quit"]
    lines += ["/model openrouter", "/resume-me", "hello there", "/quit"]
    res = run_repl_session(lines, extra_env={"VECTOR_FAKE_LLM_TOOLS": ""}, timeout_sec=150)
    text = res.text
    assert not res.timed_out and res.exit_code == 0, text[-2000:]
    startup = text[: text.index("vector>")]
    assert startup.count(INSTRUCTION) == 1, startup  # ONE instruction, at the first run
    assert "Traceback" not in text and " failed:" not in text, text[-3000:]
    assert "No OPENROUTER_API_KEY set" in text  # /model openrouter names that provider's variable
    assert "Unknown: /resume-me" in text
    assert text.count(INSTRUCTION) >= 3  # /login and the NL line repeat it
    assert "Session saved" in text
