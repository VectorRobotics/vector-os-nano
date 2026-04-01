"""End-to-end integration tests for the Vector CLI tool_use agent loop.

These tests exercise the full path from user message → VectorEngine →
tool execution → session persistence, using mocked Anthropic API calls.
No real API calls are made.

Covered scenarios:
1. test_full_tool_use_loop          — FileReadTool round-trip, 4-entry session
2. test_multi_tool_loop             — GlobTool + GrepTool concurrent execution
3. test_permission_deny_flow        — BashTool dangerous command blocked unconditionally
4. test_permission_ask_flow         — BashTool safe command approved by callback
5. test_session_persistence_roundtrip — save/load session, messages identical
6. test_skill_wrapper_end_to_end    — SkillWrapperTool routes to skill.execute()
7. test_system_prompt_included      — system prompt forwarded to API
8. test_cli_help_exits_zero         — main(["--help"]) exits with SystemExit(0)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vector_os_nano.vcli.engine import TurnResult, VectorEngine
from vector_os_nano.vcli.permissions import PermissionContext
from vector_os_nano.vcli.session import Session, create_session, load_session
from vector_os_nano.vcli.tools.base import ToolRegistry, ToolResult
from vector_os_nano.vcli.tools.bash_tool import BashTool
from vector_os_nano.vcli.tools.file_tools import FileReadTool
from vector_os_nano.vcli.tools.search_tools import GlobTool, GrepTool
from vector_os_nano.vcli.tools.skill_wrapper import SkillWrapperTool, wrap_skills
from vector_os_nano.vcli.prompt import build_system_prompt


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------


def make_mock_client(responses: list[Any]) -> MagicMock:
    """Return a mock Anthropic client whose messages.create returns *responses* in sequence."""
    client = MagicMock()
    messages_mock = MagicMock()
    messages_mock.create = MagicMock(side_effect=responses)
    client.messages = messages_mock
    return client


def make_response(
    content_blocks: list[Any],
    stop_reason: str = "end_turn",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> MagicMock:
    """Build a mock Anthropic response object."""
    resp = MagicMock()
    resp.content = content_blocks
    resp.stop_reason = stop_reason
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_read_input_tokens = 0
    usage.cache_creation_input_tokens = 0
    resp.usage = usage
    return resp


def make_text_block(text: str) -> MagicMock:
    """Build a mock text content block."""
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def make_tool_use_block(tool_id: str, name: str, input_dict: dict[str, Any]) -> MagicMock:
    """Build a mock tool_use content block."""
    b = MagicMock()
    b.type = "tool_use"
    b.id = tool_id
    b.name = name
    b.input = input_dict
    return b


def _make_engine_with_client(
    mock_client: MagicMock,
    registry: ToolRegistry,
    permissions: PermissionContext | None = None,
    system_prompt: list[dict[str, Any]] | None = None,
) -> VectorEngine:
    """Create a VectorEngine whose internal _client is the supplied mock."""
    with patch("vector_os_nano.vcli.engine.anthropic.Anthropic", return_value=mock_client):
        engine = VectorEngine(
            api_key="test-key",
            registry=registry,
            permissions=permissions or PermissionContext(no_permission=True),
            system_prompt=system_prompt or [],
        )
    engine._client = mock_client
    return engine


def _tmp_session(tmp_path: Path) -> Session:
    """Create a new session persisted inside *tmp_path*."""
    return create_session(directory=tmp_path)


# ---------------------------------------------------------------------------
# 1. Full tool_use loop with FileReadTool
# ---------------------------------------------------------------------------


class TestFullToolUseLoop:
    def test_full_tool_use_loop(self, tmp_path: Path) -> None:
        """User message → API returns tool_use for file_read → execute → final text.

        Verifies:
        - TurnResult has text + 1 tool_call
        - Session accumulates exactly 4 API-level messages:
          [user, assistant(tool_use), user(tool_result), assistant(final)]
        """
        # Create a real file for FileReadTool to read
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello\n")

        registry = ToolRegistry()
        registry.register(FileReadTool())

        first = make_response(
            [make_tool_use_block("id1", "file_read", {"file_path": str(test_file)})],
            stop_reason="tool_use",
        )
        second = make_response([make_text_block("The file contains hello")])

        mock_client = make_mock_client([first, second])
        session = _tmp_session(tmp_path)
        engine = _make_engine_with_client(mock_client, registry)

        result: TurnResult = engine.run_turn("read the file", session)

        # TurnResult assertions
        assert result.text == "The file contains hello"
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.tool_name == "file_read"
        assert tc.params["file_path"] == str(test_file)
        assert not tc.result.is_error
        assert "hello" in tc.result.content

        # Session should have 4 messages
        messages = session.to_messages()
        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"   # tool_result injected as user turn
        assert messages[3]["role"] == "assistant"


# ---------------------------------------------------------------------------
# 2. Multi-tool loop (GlobTool + GrepTool)
# ---------------------------------------------------------------------------


class TestMultiToolLoop:
    def test_multi_tool_loop(self, tmp_path: Path) -> None:
        """API returns 2 concurrent tool_use blocks (glob + grep); both executed.

        Both GlobTool and GrepTool are read-only/concurrency-safe, so they are
        dispatched in a single concurrent batch.
        """
        registry = ToolRegistry()
        registry.register(GlobTool())
        registry.register(GrepTool())

        first = make_response(
            [
                make_tool_use_block("id_g", "glob", {"pattern": "*.txt", "path": str(tmp_path)}),
                make_tool_use_block("id_r", "grep", {"pattern": "hello", "path": str(tmp_path)}),
            ],
            stop_reason="tool_use",
        )
        second = make_response([make_text_block("Search complete")])

        mock_client = make_mock_client([first, second])
        session = _tmp_session(tmp_path)
        engine = _make_engine_with_client(mock_client, registry)

        result: TurnResult = engine.run_turn("search files", session)

        assert result.text == "Search complete"
        assert len(result.tool_calls) == 2
        tool_names = {tc.tool_name for tc in result.tool_calls}
        assert tool_names == {"glob", "grep"}


# ---------------------------------------------------------------------------
# 3. Permission deny flow (BashTool with dangerous command)
# ---------------------------------------------------------------------------


class TestPermissionDenyFlow:
    def test_permission_deny_flow(self, tmp_path: Path) -> None:
        """BashTool denies 'rm -rf /' unconditionally; tool is never executed.

        Verifies:
        - TurnResult shows 1 tool_call with is_error=True
        - permission_action == "denied"
        """
        registry = ToolRegistry()
        bash = BashTool()
        registry.register(bash)

        first = make_response(
            [make_tool_use_block("id1", "bash", {"command": "rm -rf /"})],
            stop_reason="tool_use",
        )
        second = make_response([make_text_block("Command was denied")])

        mock_client = make_mock_client([first, second])
        session = _tmp_session(tmp_path)

        # Default PermissionContext (no_permission=False) + BashTool.check_permissions denies
        permissions = PermissionContext(no_permission=False)
        engine = _make_engine_with_client(mock_client, registry, permissions=permissions)

        result: TurnResult = engine.run_turn("delete everything", session)

        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.result.is_error is True
        assert tc.permission_action == "denied"
        assert "Permission denied" in tc.result.content or "denied" in tc.result.content.lower()


# ---------------------------------------------------------------------------
# 4. Permission ask flow (BashTool with safe command)
# ---------------------------------------------------------------------------


class TestPermissionAskFlow:
    def test_permission_ask_flow(self, tmp_path: Path) -> None:
        """BashTool with 'ls' returns ask; callback returning True allows execution."""
        registry = ToolRegistry()
        registry.register(BashTool())

        first = make_response(
            [make_tool_use_block("id1", "bash", {"command": "echo hello"})],
            stop_reason="tool_use",
        )
        second = make_response([make_text_block("Command ran")])

        mock_client = make_mock_client([first, second])
        session = _tmp_session(tmp_path)

        permissions = PermissionContext(no_permission=False)
        engine = _make_engine_with_client(mock_client, registry, permissions=permissions)

        result: TurnResult = engine.run_turn(
            "echo something",
            session,
            ask_permission=lambda name, params: True,
        )

        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert not tc.result.is_error
        assert tc.permission_action == "asked_allowed"

    def test_permission_ask_denied_by_callback(self, tmp_path: Path) -> None:
        """ask_permission returning False prevents execution; error result returned."""
        registry = ToolRegistry()
        registry.register(BashTool())

        first = make_response(
            [make_tool_use_block("id1", "bash", {"command": "echo hello"})],
            stop_reason="tool_use",
        )
        second = make_response([make_text_block("Denied")])

        mock_client = make_mock_client([first, second])
        session = _tmp_session(tmp_path)

        permissions = PermissionContext(no_permission=False)
        engine = _make_engine_with_client(mock_client, registry, permissions=permissions)

        result: TurnResult = engine.run_turn(
            "echo something",
            session,
            ask_permission=lambda name, params: False,
        )

        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.result.is_error is True
        assert tc.permission_action == "asked_denied"


# ---------------------------------------------------------------------------
# 5. Session persistence round-trip
# ---------------------------------------------------------------------------


class TestSessionPersistenceRoundtrip:
    def test_session_persistence_roundtrip(self, tmp_path: Path) -> None:
        """Run a turn, save session, reload from disk — messages must match."""
        registry = ToolRegistry()
        registry.register(FileReadTool())

        test_file = tmp_path / "data.txt"
        test_file.write_text("content\n")

        first = make_response(
            [make_tool_use_block("id1", "file_read", {"file_path": str(test_file)})],
            stop_reason="tool_use",
        )
        second = make_response([make_text_block("File read done")])

        mock_client = make_mock_client([first, second])
        session = _tmp_session(tmp_path)
        engine = _make_engine_with_client(mock_client, registry)

        engine.run_turn("read the data file", session)
        original_messages = session.to_messages()

        # Persist to disk
        session.save()

        # Reload from disk
        loaded_session = load_session(session.session_id, directory=tmp_path)
        loaded_messages = loaded_session.to_messages()

        assert loaded_messages == original_messages


# ---------------------------------------------------------------------------
# 6. Skill wrapper end-to-end
# ---------------------------------------------------------------------------


class TestSkillWrapperEndToEnd:
    def test_skill_wrapper_end_to_end(self, tmp_path: Path) -> None:
        """wrap_skills() wraps a mock skill; API tool_use routes to skill.execute()."""
        # Build a minimal mock skill
        mock_skill = MagicMock()
        mock_skill.name = "mock_skill"
        mock_skill.description = "A test skill"
        mock_skill.parameters = {}
        mock_skill.preconditions = []
        mock_skill.effects = {}

        skill_result = MagicMock()
        skill_result.success = True
        skill_result.result_data = {"value": 42}
        skill_result.error_message = None
        mock_skill.execute.return_value = skill_result

        # Build a minimal mock skill registry
        mock_registry = MagicMock()
        mock_registry.list_skills.return_value = ["mock_skill"]
        mock_registry.get.return_value = mock_skill

        # Build a minimal mock agent
        mock_agent = MagicMock()
        mock_agent._skill_registry = mock_registry
        mock_ctx = MagicMock()
        mock_agent._build_context.return_value = mock_ctx
        mock_agent._sync_robot_state.return_value = None

        # Wrap skills and register in a real registry
        registry = ToolRegistry()
        for tool in wrap_skills(mock_agent):
            registry.register(tool)

        assert "mock_skill" in registry.list_tools()

        first = make_response(
            [make_tool_use_block("id1", "mock_skill", {})],
            stop_reason="tool_use",
        )
        second = make_response([make_text_block("Skill ran")])

        mock_client = make_mock_client([first, second])
        session = _tmp_session(tmp_path)
        engine = _make_engine_with_client(mock_client, registry)

        result: TurnResult = engine.run_turn("run mock_skill", session, agent=mock_agent)

        mock_skill.execute.assert_called_once()
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.tool_name == "mock_skill"
        assert not tc.result.is_error


# ---------------------------------------------------------------------------
# 7. System prompt included in API call
# ---------------------------------------------------------------------------


class TestSystemPromptIncluded:
    def test_system_prompt_included(self, tmp_path: Path) -> None:
        """build_system_prompt() result is forwarded as system= to messages.create()."""
        system_prompt = build_system_prompt(agent=None)
        assert len(system_prompt) >= 2  # at least ROLE + TOOL_INSTRUCTIONS

        registry = ToolRegistry()
        single_response = make_response([make_text_block("hello")])
        mock_client = make_mock_client([single_response])

        engine = _make_engine_with_client(
            mock_client,
            registry,
            system_prompt=system_prompt,
        )
        session = _tmp_session(tmp_path)

        engine.run_turn("hi", session)

        # Verify the system= kwarg was passed to messages.create
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs is not None
        passed_system = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system")
        assert passed_system == system_prompt

    def test_system_prompt_contains_role_text(self) -> None:
        """build_system_prompt returns blocks containing the ROLE_PROMPT text."""
        from vector_os_nano.vcli.prompt import ROLE_PROMPT

        blocks = build_system_prompt(agent=None)
        all_text = " ".join(b.get("text", "") for b in blocks)
        assert "Vector" in all_text
        assert ROLE_PROMPT.strip()[:40] in all_text


# ---------------------------------------------------------------------------
# 8. CLI --help exits zero
# ---------------------------------------------------------------------------


class TestCliHelpExitsZero:
    def test_cli_help_exits_zero(self) -> None:
        """main(['--help']) prints usage and exits with code 0."""
        from vector_os_nano.vcli.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])

        assert exc_info.value.code == 0
