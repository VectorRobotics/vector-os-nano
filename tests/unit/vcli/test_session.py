# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tests for vcli session persistence (JSONL format).

TDD: Tests written before implementation.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from vector_os_nano.vcli.session import (
    Session,
    SessionSummary,
    TokenUsage,
    create_session,
    get_latest_session,
    list_sessions,
    load_session,
)


# ---------------------------------------------------------------------------
# TokenUsage tests
# ---------------------------------------------------------------------------


def test_token_usage_accumulates() -> None:
    """add() returns a new TokenUsage with summed fields."""
    base = TokenUsage(input_tokens=100, output_tokens=50, cache_read_tokens=10, cache_creation_tokens=5)
    delta = TokenUsage(input_tokens=20, output_tokens=10, cache_read_tokens=0, cache_creation_tokens=2)
    result = base.add(delta)
    assert result.input_tokens == 120
    assert result.output_tokens == 60
    assert result.cache_read_tokens == 10
    assert result.cache_creation_tokens == 7
    # Original is unchanged (frozen)
    assert base.input_tokens == 100


def test_token_usage_immutable() -> None:
    """TokenUsage is a frozen dataclass — mutations raise."""
    usage = TokenUsage(input_tokens=5)
    with pytest.raises((AttributeError, TypeError)):
        usage.input_tokens = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Session creation tests
# ---------------------------------------------------------------------------


def test_create_session_generates_uuid(tmp_path: Path) -> None:
    """create_session() assigns a valid UUID4 as session_id."""
    session = create_session(directory=tmp_path)
    parsed = uuid.UUID(session.session_id, version=4)
    assert str(parsed) == session.session_id


def test_a_session_is_written_on_its_first_entry(tmp_path: Path) -> None:
    """cli-entry:session-litter-and-resume: an empty session leaves no file (it never
    litters /sessions or --resume latest); the first entry writes it."""
    session = create_session(directory=tmp_path)
    expected_path = tmp_path / f"{session.session_id}.jsonl"
    session.save()
    assert not expected_path.exists()
    session.append_user("hi")
    session.save()
    assert expected_path.exists()


def test_latest_is_never_an_empty_or_a_print_session(tmp_path: Path) -> None:
    """--resume latest resumes the newest conversation, never a one-shot -p turn."""
    repl = create_session(metadata={"face": "repl"}, directory=tmp_path)
    repl.append_user("a conversation")
    repl.save()
    once = create_session(metadata={"face": "print"}, directory=tmp_path)
    once.append_user("a -p turn")
    once.save()
    create_session(directory=tmp_path).save()  # empty: no file at all
    assert get_latest_session(directory=tmp_path).session_id == repl.session_id


def test_session_default_directory_follows_home_at_call_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default dir is ~/.vector/sessions resolved per call, not at import.

    Regression (tests:dotenv-credential-leak): an import-time constant ignored a
    test's HOME override, so in-process REPL tests wrote into the real ~/.vector.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    session = create_session()
    session.append_user("hi")
    session.save()
    assert (tmp_path / ".vector" / "sessions" / f"{session.session_id}.jsonl").exists()
    assert get_latest_session().session_id == session.session_id


def test_session_metadata(tmp_path: Path) -> None:
    """Metadata dict provided at creation is preserved across save/load."""
    meta = {"robot": "go2", "user": "yusen"}
    session = create_session(metadata=meta, directory=tmp_path)
    session.append_user("x")  # a session is written on its first entry
    session.save()

    loaded = load_session(session.session_id, directory=tmp_path)
    assert loaded.metadata["robot"] == "go2"
    assert loaded.metadata["user"] == "yusen"


# ---------------------------------------------------------------------------
# Append tests
# ---------------------------------------------------------------------------


def test_append_user_message(tmp_path: Path) -> None:
    """append_user() adds exactly one JSONL line with type=user."""
    session = create_session(directory=tmp_path)
    session.append_user("hello robot")
    session.save()

    path = tmp_path / f"{session.session_id}.jsonl"
    lines = [json.loads(ln) for ln in path.read_text().strip().splitlines() if ln.strip()]
    user_lines = [ln for ln in lines if ln.get("type") == "user"]
    assert len(user_lines) == 1
    assert user_lines[0]["content"] == "hello robot"
    assert "ts" in user_lines[0]


def test_append_assistant_message(tmp_path: Path) -> None:
    """append_assistant() adds a line with type=assistant, text, and tool_use."""
    session = create_session(directory=tmp_path)
    tool_blocks = [{"type": "tool_use", "id": "tu_1", "name": "walk", "input": {}}]
    session.append_assistant(text="Walking now", tool_use_blocks=tool_blocks)
    session.save()

    path = tmp_path / f"{session.session_id}.jsonl"
    lines = [json.loads(ln) for ln in path.read_text().strip().splitlines() if ln.strip()]
    asst_lines = [ln for ln in lines if ln.get("type") == "assistant"]
    assert len(asst_lines) == 1
    entry = asst_lines[0]
    assert entry["text"] == "Walking now"
    assert entry["tool_use"] == tool_blocks
    assert "ts" in entry


def test_append_assistant_message_no_tools(tmp_path: Path) -> None:
    """append_assistant() with no tools sets tool_use to null/None."""
    session = create_session(directory=tmp_path)
    session.append_assistant(text="Just text, no tools")
    session.save()

    path = tmp_path / f"{session.session_id}.jsonl"
    lines = [json.loads(ln) for ln in path.read_text().strip().splitlines() if ln.strip()]
    asst_lines = [ln for ln in lines if ln.get("type") == "assistant"]
    assert asst_lines[0]["tool_use"] is None


def test_append_tool_results(tmp_path: Path) -> None:
    """append_tool_results() adds a line with type=tool_result."""
    session = create_session(directory=tmp_path)
    results = [{"tool_use_id": "tu_1", "content": "ok", "is_error": False}]
    session.append_tool_results(results)
    session.save()

    path = tmp_path / f"{session.session_id}.jsonl"
    lines = [json.loads(ln) for ln in path.read_text().strip().splitlines() if ln.strip()]
    tr_lines = [ln for ln in lines if ln.get("type") == "tool_result"]
    assert len(tr_lines) == 1
    assert tr_lines[0]["results"] == results


# ---------------------------------------------------------------------------
# to_messages() reconstruction
# ---------------------------------------------------------------------------


def test_to_messages_reconstructs_api_format(tmp_path: Path) -> None:
    """to_messages() returns list[dict] in Anthropic API role/content format."""
    session = create_session(directory=tmp_path)
    session.append_user("pick up the cube")
    session.append_assistant(
        text="I will pick up the cube.",
        tool_use_blocks=[{"type": "tool_use", "id": "tu_1", "name": "pick", "input": {}}],
    )
    session.append_tool_results([{"tool_use_id": "tu_1", "content": "picked", "is_error": False}])
    session.append_assistant(text="Done!")

    messages = session.to_messages()

    assert len(messages) == 4

    # First: user message
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "pick up the cube"

    # Second: assistant with text + tool_use
    assert messages[1]["role"] == "assistant"
    content_1 = messages[1]["content"]
    assert isinstance(content_1, list)
    text_blocks = [b for b in content_1 if b.get("type") == "text"]
    tool_blocks = [b for b in content_1 if b.get("type") == "tool_use"]
    assert len(text_blocks) == 1
    assert text_blocks[0]["text"] == "I will pick up the cube."
    assert len(tool_blocks) == 1

    # Third: tool_result as user role
    assert messages[2]["role"] == "user"
    content_2 = messages[2]["content"]
    assert isinstance(content_2, list)
    tr_blocks = [b for b in content_2 if b.get("type") == "tool_result"]
    assert len(tr_blocks) == 1
    assert tr_blocks[0]["tool_use_id"] == "tu_1"

    # Fourth: assistant text only
    assert messages[3]["role"] == "assistant"
    content_3 = messages[3]["content"]
    assert isinstance(content_3, list)
    text_blocks_2 = [b for b in content_3 if b.get("type") == "text"]
    assert text_blocks_2[0]["text"] == "Done!"


def test_to_messages_skips_meta_entries(tmp_path: Path) -> None:
    """to_messages() skips type=meta entries (internal tracking only)."""
    session = create_session(directory=tmp_path)
    session.append_user("hello")
    session.add_usage(TokenUsage(input_tokens=10, output_tokens=5))
    session.save()

    messages = session.to_messages()
    # Only the user message, no meta entry
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


# ---------------------------------------------------------------------------
# Load / persistence tests
# ---------------------------------------------------------------------------


def test_load_session_parses_jsonl(tmp_path: Path) -> None:
    """load_session() recovers all entries written by create_session + appends."""
    session = create_session(directory=tmp_path)
    session.append_user("msg 1")
    session.append_assistant(text="reply 1")
    session.save()

    loaded = load_session(session.session_id, directory=tmp_path)
    messages = loaded.to_messages()
    assert len(messages) == 2
    assert messages[0]["content"] == "msg 1"


def test_load_session_skips_corrupt_tail(tmp_path: Path) -> None:
    """load_session() skips a corrupt last line, recovers remaining entries."""
    session = create_session(directory=tmp_path)
    session.append_user("good message")
    session.save()

    # Corrupt the file by appending a malformed line
    path = tmp_path / f"{session.session_id}.jsonl"
    with path.open("a") as f:
        f.write("{corrupt json\n")

    loaded = load_session(session.session_id, directory=tmp_path)
    messages = loaded.to_messages()
    # The corrupt line is skipped; good message survives
    assert len(messages) == 1
    assert messages[0]["content"] == "good message"


# ---------------------------------------------------------------------------
# list_sessions / get_latest_session tests
# ---------------------------------------------------------------------------


def test_list_sessions_sorted_by_mtime(tmp_path: Path) -> None:
    """list_sessions() returns SessionSummary list, newest first by mtime."""
    s1 = create_session(directory=tmp_path)
    s1.append_user("first")
    s1.save()

    s2 = create_session(directory=tmp_path)
    s2.append_user("second")
    s2.save()

    summaries = list_sessions(directory=tmp_path)
    assert len(summaries) == 2
    assert isinstance(summaries[0], SessionSummary)
    # Newest (s2) should be first
    assert summaries[0].session_id == s2.session_id


def test_list_sessions_returns_correct_message_count(tmp_path: Path) -> None:
    """SessionSummary.message_count reflects number of non-meta entries."""
    session = create_session(directory=tmp_path)
    session.append_user("a")
    session.append_assistant(text="b")
    session.save()

    summaries = list_sessions(directory=tmp_path)
    assert summaries[0].message_count == 2


def test_get_latest_session(tmp_path: Path) -> None:
    """get_latest_session() returns most recently modified Session."""
    s1 = create_session(directory=tmp_path)
    s1.save()

    s2 = create_session(directory=tmp_path)
    s2.append_user("latest")
    s2.save()

    latest = get_latest_session(directory=tmp_path)
    assert latest is not None
    assert latest.session_id == s2.session_id


def test_get_latest_session_empty_directory(tmp_path: Path) -> None:
    """get_latest_session() returns None when no sessions exist."""
    result = get_latest_session(directory=tmp_path)
    assert result is None


def test_list_sessions_empty_directory(tmp_path: Path) -> None:
    """list_sessions() returns empty list when no sessions exist."""
    summaries = list_sessions(directory=tmp_path)
    assert summaries == []


# ---------------------------------------------------------------------------
# Session.read_files tracking
# ---------------------------------------------------------------------------


def test_new_session_has_empty_read_files(tmp_path: Path) -> None:
    """A freshly created session starts with an empty read_files set."""
    session = create_session(directory=tmp_path)
    assert isinstance(session.read_files, set)
    assert len(session.read_files) == 0


def test_can_add_paths_to_read_files(tmp_path: Path) -> None:
    """Paths can be added to read_files and are retrievable."""
    session = create_session(directory=tmp_path)
    session.read_files.add("/tmp/foo.txt")
    session.read_files.add("/tmp/bar.txt")
    assert "/tmp/foo.txt" in session.read_files
    assert "/tmp/bar.txt" in session.read_files
    assert len(session.read_files) == 2


def test_read_files_not_persisted_on_save_and_load(tmp_path: Path) -> None:
    """read_files is runtime-only — it is NOT written to or loaded from disk."""
    session = create_session(directory=tmp_path)
    session.read_files.add("/tmp/important.txt")
    session.append_user("x")  # a session is written on its first entry
    session.save()

    loaded = load_session(session.session_id, directory=tmp_path)
    # After loading, read_files must be empty (runtime tracking resets)
    assert len(loaded.read_files) == 0
    assert "/tmp/important.txt" not in loaded.read_files
