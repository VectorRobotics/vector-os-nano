# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Unified personality layer + project-context (AGENTS.md/CLAUDE.md) extension.

Personality is a kernel-owned, WORLD-AGNOSTIC prompt layer: the same Vector
identity/voice across every world. It sits below the system/developer prompt
(the world persona) and above project context, loaded from
``~/.vector/personality.md`` with a built-in default.
"""
from __future__ import annotations

import vector_os_nano.vcli.prompt as prompt


def test_personality_block_present_after_persona(tmp_path):
    blocks = prompt.build_system_prompt(cwd=tmp_path)
    texts = [b["text"] for b in blocks]
    idxs = [i for i, t in enumerate(texts) if "[personality]" in t.lower()]
    assert idxs, "personality block missing from the system prompt"
    i = idxs[0]
    # After the world persona (role[0] + tool_instructions[1]).
    assert i >= 2
    # Static + cacheable, like the persona blocks.
    assert blocks[i].get("cache_control", {}).get("type") == "ephemeral"


def test_personality_before_project_context(tmp_path):
    (tmp_path / "VECTOR.md").write_text("proj-ctx-marker", encoding="utf-8")
    texts = [b["text"] for b in prompt.build_system_prompt(cwd=tmp_path)]
    pers = next(i for i, t in enumerate(texts) if "[personality]" in t.lower())
    proj = next(i for i, t in enumerate(texts) if "proj-ctx-marker" in t)
    assert pers < proj  # personality above project context (the layered order)


def test_load_personality_default_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.vector/personality.md here
    assert prompt._load_personality() == prompt.DEFAULT_PERSONALITY.strip()


def test_load_personality_reads_user_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    vdir = tmp_path / ".vector"
    vdir.mkdir()
    (vdir / "personality.md").write_text("[Personality]\nCustom Vector Vibe", encoding="utf-8")
    assert "Custom Vector Vibe" in prompt._load_personality()


def test_personality_is_world_agnostic_default():
    # The default carries no embodiment-specific role content (that belongs to the
    # world persona). It must read as one identity across bodies.
    d = prompt.DEFAULT_PERSONALITY.lower()
    assert "one identity" in d


def test_default_personality_curbs_over_eager_tool_use():
    # Regression: deepseek over-explored on "hello" (read files, ran pwd). The
    # personality must tell the agent to answer greetings/questions directly.
    d = prompt.DEFAULT_PERSONALITY.lower()
    assert "answer directly" in d
    assert "do not read files" in d or "do not run commands" in d


def test_project_context_recognizes_agents_md(tmp_path, monkeypatch):
    # Another agent's file is opt-in (cli-entry:robot-persona-wrong-launch-instructions).
    monkeypatch.setenv("VECTOR_PROJECT_CONTEXT", "1")
    (tmp_path / "AGENTS.md").write_text("agents-rules-here", encoding="utf-8")
    assert "agents-rules-here" in prompt._load_vector_md(tmp_path)


def test_project_context_recognizes_claude_md(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTOR_PROJECT_CONTEXT", "1")
    (tmp_path / "CLAUDE.md").write_text("claude-rules-here", encoding="utf-8")
    assert "claude-rules-here" in prompt._load_vector_md(tmp_path)


def test_project_context_precedence_vector_first(tmp_path):
    (tmp_path / "VECTOR.md").write_text("vector-wins", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("claude-loses", encoding="utf-8")
    out = prompt._load_vector_md(tmp_path)
    assert "vector-wins" in out
    assert "claude-loses" not in out  # first file in precedence wins, only one used
