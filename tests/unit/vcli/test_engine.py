"""Unit tests for VectorEngine — TDD RED phase.

Covers:
- test_single_text_turn: text-only response returns TurnResult with no tool_calls
- test_tool_use_turn: tool_use block → execute tool → send tool_result → final text
- test_multi_tool_turn: 2 tool_use blocks both executed, results sent
- test_tool_concurrency_partition: _partition_tools groups read-only vs write
- test_streaming_text_callback: on_text called with text content
- test_tool_start_end_callbacks: on_tool_start and on_tool_end called correctly
- test_max_turns_respected: engine stops after max_turns
- test_permission_deny_returns_error: deny → tool not executed, error result sent
- test_permission_ask_flow: ask_permission callback returns True → tool executed
- test_permission_ask_denied: ask_permission returns False → denial sent
- test_turn_result_usage: TurnResult includes token usage
- test_session_messages_updated: session.to_messages() has all exchanges
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from vector_os_nano.vcli.engine import TurnResult, VectorEngine
from vector_os_nano.vcli.permissions import PermissionContext
from vector_os_nano.vcli.session import Session, TokenUsage
from vector_os_nano.vcli.tools.base import (
    PermissionResult,
    ToolContext,
    ToolRegistry,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def make_mock_response(
    content_blocks: list,
    stop_reason: str = "end_turn",
    usage: Any = None,
) -> MagicMock:
    """Create a mock Anthropic API response."""
    response = MagicMock()
    response.content = content_blocks
    response.stop_reason = stop_reason
    if usage is None:
        usage = MagicMock(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
    response.usage = usage
    return response


def make_text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def make_tool_use_block(tool_id: str, name: str, input_dict: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_dict
    return block


def make_session() -> Session:
    """Return a transient in-memory session (no file I/O)."""
    session = Session(
        session_id="test-session-1",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        path=Path("/tmp/test_session.jsonl"),
    )
    return session


def make_read_only_tool(name: str, return_content: str = "ok") -> Any:
    """Return a mock tool that is read-only and concurrency-safe."""
    tool = MagicMock()
    tool.name = name
    tool.description = f"Read-only tool {name}"
    tool.input_schema = {"type": "object", "properties": {}}
    tool.is_read_only.return_value = True
    tool.is_concurrency_safe.return_value = True
    tool.check_permissions.return_value = PermissionResult(behavior="allow")
    tool.execute.return_value = ToolResult(content=return_content, is_error=False)
    return tool


def make_write_tool(name: str, return_content: str = "written") -> Any:
    """Return a mock tool that is write (sequential, requires permission ask)."""
    tool = MagicMock()
    tool.name = name
    tool.description = f"Write tool {name}"
    tool.input_schema = {"type": "object", "properties": {}}
    tool.is_read_only.return_value = False
    tool.is_concurrency_safe.return_value = False
    tool.check_permissions.return_value = PermissionResult(behavior="ask")
    tool.execute.return_value = ToolResult(content=return_content, is_error=False)
    return tool


def make_engine(
    registry: ToolRegistry | None = None,
    permissions: PermissionContext | None = None,
    max_turns: int = 50,
) -> tuple[VectorEngine, MagicMock]:
    """Return (engine, mock_client) with patched anthropic.Anthropic."""
    mock_client = MagicMock()
    with patch("vector_os_nano.vcli.engine.anthropic.Anthropic", return_value=mock_client):
        engine = VectorEngine(
            api_key="test-key",
            registry=registry or ToolRegistry(),
            permissions=permissions or PermissionContext(no_permission=True),
            max_turns=max_turns,
        )
    # Directly replace the private _client so mock is in place
    engine._client = mock_client
    return engine, mock_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleTextTurn:
    def test_single_text_turn(self) -> None:
        """User sends 'hello', API returns text-only → TurnResult with text, no tool_calls."""
        engine, mock_client = make_engine()
        session = make_session()

        response = make_mock_response([make_text_block("Hello, world!")])
        mock_client.messages.create.return_value = response

        result = engine.run_turn("hello", session)

        assert isinstance(result, TurnResult)
        assert result.text == "Hello, world!"
        assert result.tool_calls == []
        assert result.stop_reason == "end_turn"

    def test_turn_result_usage(self) -> None:
        """TurnResult.usage reflects the token counts from the API response."""
        engine, mock_client = make_engine()
        session = make_session()

        usage_mock = MagicMock(
            input_tokens=200,
            output_tokens=75,
            cache_read_input_tokens=10,
            cache_creation_input_tokens=5,
        )
        response = make_mock_response([make_text_block("hi")], usage=usage_mock)
        mock_client.messages.create.return_value = response

        result = engine.run_turn("ping", session)

        assert result.usage.input_tokens == 200
        assert result.usage.output_tokens == 75
        assert result.usage.cache_read_tokens == 10
        assert result.usage.cache_creation_tokens == 5

    def test_session_messages_updated(self) -> None:
        """After run_turn, session.to_messages() contains user + assistant entries."""
        engine, mock_client = make_engine()
        session = make_session()

        response = make_mock_response([make_text_block("I'm fine.")])
        mock_client.messages.create.return_value = response

        engine.run_turn("How are you?", session)

        messages = session.to_messages()
        # First message: user
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "How are you?"
        # Second message: assistant
        assert messages[1]["role"] == "assistant"
        assert any(
            block.get("text") == "I'm fine."
            for block in messages[1]["content"]
            if isinstance(block, dict)
        )


class TestStreamingCallback:
    def test_streaming_text_callback(self) -> None:
        """on_text callback receives text chunks during the turn."""
        engine, mock_client = make_engine()
        session = make_session()

        response = make_mock_response([make_text_block("chunk one"), make_text_block(" chunk two")])
        mock_client.messages.create.return_value = response

        received: list[str] = []
        engine.run_turn("hello", session, on_text=received.append)

        assert received == ["chunk one", " chunk two"]


class TestToolUseTurn:
    def test_tool_use_turn(self) -> None:
        """API returns tool_use block → engine executes tool → sends tool_result → returns final text."""
        registry = ToolRegistry()
        tool = make_read_only_tool("read_file", return_content="file contents")
        registry.register(tool)

        engine, mock_client = make_engine(registry=registry)
        session = make_session()

        # First response: tool call; second: final text after tool result
        first_response = make_mock_response(
            [make_tool_use_block("tool_1", "read_file", {"path": "/tmp/x.txt"})],
            stop_reason="tool_use",
        )
        second_response = make_mock_response([make_text_block("Done.")])
        mock_client.messages.create.side_effect = [first_response, second_response]

        result = engine.run_turn("read that file", session)

        assert result.text == "Done."
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.tool_name == "read_file"
        assert tc.params == {"path": "/tmp/x.txt"}
        assert tc.result.content == "file contents"
        assert not tc.result.is_error

    def test_multi_tool_turn(self) -> None:
        """API returns 2 tool_use blocks → both executed → results sent back."""
        registry = ToolRegistry()
        tool_a = make_read_only_tool("tool_a", return_content="result_a")
        tool_b = make_read_only_tool("tool_b", return_content="result_b")
        registry.register(tool_a)
        registry.register(tool_b)

        engine, mock_client = make_engine(registry=registry)
        session = make_session()

        first_response = make_mock_response(
            [
                make_tool_use_block("id_a", "tool_a", {}),
                make_tool_use_block("id_b", "tool_b", {}),
            ],
            stop_reason="tool_use",
        )
        second_response = make_mock_response([make_text_block("Both done.")])
        mock_client.messages.create.side_effect = [first_response, second_response]

        result = engine.run_turn("run both", session)

        assert result.text == "Both done."
        assert len(result.tool_calls) == 2
        tool_names = {tc.tool_name for tc in result.tool_calls}
        assert tool_names == {"tool_a", "tool_b"}


class TestToolConcurrencyPartition:
    def test_tool_concurrency_partition_read_only_batch(self) -> None:
        """Consecutive read-only tools are grouped into a concurrent batch."""
        registry = ToolRegistry()
        tool_a = make_read_only_tool("tool_a")
        tool_b = make_read_only_tool("tool_b")
        registry.register(tool_a)
        registry.register(tool_b)

        engine, _ = make_engine(registry=registry)

        block_a = make_tool_use_block("id_a", "tool_a", {})
        block_b = make_tool_use_block("id_b", "tool_b", {})
        batches = engine._partition_tools([block_a, block_b])

        assert len(batches) == 1
        assert batches[0].concurrent is True
        assert len(batches[0].blocks) == 2

    def test_tool_concurrency_partition_write_sequential(self) -> None:
        """Write tools each get their own sequential batch."""
        registry = ToolRegistry()
        tool_a = make_write_tool("write_a")
        tool_b = make_write_tool("write_b")
        registry.register(tool_a)
        registry.register(tool_b)

        engine, _ = make_engine(registry=registry)

        block_a = make_tool_use_block("id_a", "write_a", {})
        block_b = make_tool_use_block("id_b", "write_b", {})
        batches = engine._partition_tools([block_a, block_b])

        assert len(batches) == 2
        assert all(not b.concurrent for b in batches)

    def test_tool_concurrency_partition_mixed(self) -> None:
        """Read-only batch, then write tool breaks into new sequential batch."""
        registry = ToolRegistry()
        ro_tool = make_read_only_tool("ro")
        wr_tool = make_write_tool("wr")
        registry.register(ro_tool)
        registry.register(wr_tool)

        engine, _ = make_engine(registry=registry)

        block_ro = make_tool_use_block("id_ro", "ro", {})
        block_wr = make_tool_use_block("id_wr", "wr", {})
        batches = engine._partition_tools([block_ro, block_wr])

        assert len(batches) == 2
        assert batches[0].concurrent is True
        assert batches[1].concurrent is False


class TestCallbacks:
    def test_tool_start_end_callbacks(self) -> None:
        """on_tool_start and on_tool_end are called with correct tool name and result."""
        registry = ToolRegistry()
        tool = make_read_only_tool("my_tool", return_content="output")
        registry.register(tool)

        engine, mock_client = make_engine(registry=registry)
        session = make_session()

        first_response = make_mock_response(
            [make_tool_use_block("tool_1", "my_tool", {"key": "val"})],
            stop_reason="tool_use",
        )
        second_response = make_mock_response([make_text_block("done")])
        mock_client.messages.create.side_effect = [first_response, second_response]

        started: list[tuple[str, dict]] = []
        ended: list[tuple[str, ToolResult]] = []

        engine.run_turn(
            "go",
            session,
            on_tool_start=lambda name, params: started.append((name, params)),
            on_tool_end=lambda name, result: ended.append((name, result)),
        )

        assert len(started) == 1
        assert started[0] == ("my_tool", {"key": "val"})
        assert len(ended) == 1
        assert ended[0][0] == "my_tool"
        assert ended[0][1].content == "output"


class TestMaxTurns:
    def test_max_turns_respected(self) -> None:
        """Engine stops after max_turns even if model keeps calling tools."""
        registry = ToolRegistry()
        tool = make_read_only_tool("loop_tool", return_content="looping")
        registry.register(tool)

        engine, mock_client = make_engine(registry=registry, max_turns=2)
        session = make_session()

        # Every response calls loop_tool forever
        looping_response = make_mock_response(
            [make_tool_use_block("t1", "loop_tool", {})],
            stop_reason="tool_use",
        )
        mock_client.messages.create.return_value = looping_response

        result = engine.run_turn("loop", session)

        # Engine must stop — API was called at most max_turns + 1 times
        assert mock_client.messages.create.call_count <= 3  # initial + 2 tool loops


class TestPermissions:
    def test_permission_deny_returns_error(self) -> None:
        """Permission deny → tool not executed, error result sent to API."""
        registry = ToolRegistry()
        tool = make_write_tool("dangerous_tool")
        tool.check_permissions.return_value = PermissionResult(behavior="deny", reason="nope")
        registry.register(tool)

        permissions = PermissionContext(deny_tools={"dangerous_tool"})
        engine, mock_client = make_engine(registry=registry, permissions=permissions)
        session = make_session()

        first_response = make_mock_response(
            [make_tool_use_block("t1", "dangerous_tool", {})],
            stop_reason="tool_use",
        )
        second_response = make_mock_response([make_text_block("Denied.")])
        mock_client.messages.create.side_effect = [first_response, second_response]

        result = engine.run_turn("do it", session)

        # Tool.execute should NOT have been called
        tool.execute.assert_not_called()

        # ToolCall recorded with permission_action denied
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.result.is_error is True
        assert tc.permission_action == "denied"

    def test_permission_ask_flow(self) -> None:
        """ask_permission callback returning True → tool is executed."""
        registry = ToolRegistry()
        tool = make_write_tool("write_tool")
        registry.register(tool)

        permissions = PermissionContext()  # default: will ask
        engine, mock_client = make_engine(registry=registry, permissions=permissions)
        session = make_session()

        first_response = make_mock_response(
            [make_tool_use_block("t1", "write_tool", {"data": "hello"})],
            stop_reason="tool_use",
        )
        second_response = make_mock_response([make_text_block("Written.")])
        mock_client.messages.create.side_effect = [first_response, second_response]

        # User approves
        result = engine.run_turn("write it", session, ask_permission=lambda n, p: True)

        tool.execute.assert_called_once()
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.permission_action == "asked_allowed"
        assert not tc.result.is_error

    def test_permission_ask_denied(self) -> None:
        """ask_permission returning False → tool not executed, denial sent."""
        registry = ToolRegistry()
        tool = make_write_tool("write_tool")
        registry.register(tool)

        permissions = PermissionContext()
        engine, mock_client = make_engine(registry=registry, permissions=permissions)
        session = make_session()

        first_response = make_mock_response(
            [make_tool_use_block("t1", "write_tool", {})],
            stop_reason="tool_use",
        )
        second_response = make_mock_response([make_text_block("User declined.")])
        mock_client.messages.create.side_effect = [first_response, second_response]

        result = engine.run_turn("write it", session, ask_permission=lambda n, p: False)

        tool.execute.assert_not_called()
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.result.is_error is True
        assert tc.permission_action == "asked_denied"

    def test_unknown_tool_returns_error(self) -> None:
        """API calls a tool name not in registry → error ToolResult, not a crash."""
        engine, mock_client = make_engine()
        session = make_session()

        first_response = make_mock_response(
            [make_tool_use_block("t1", "nonexistent_tool", {})],
            stop_reason="tool_use",
        )
        second_response = make_mock_response([make_text_block("ok")])
        mock_client.messages.create.side_effect = [first_response, second_response]

        result = engine.run_turn("call unknown", session)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].result.is_error is True
        assert "Unknown tool" in result.tool_calls[0].result.content

    def test_tool_exception_returns_error(self) -> None:
        """Tool.execute() raising an exception produces an error ToolResult."""
        registry = ToolRegistry()
        tool = make_read_only_tool("crashy_tool")
        tool.execute.side_effect = RuntimeError("boom")
        registry.register(tool)

        engine, mock_client = make_engine(registry=registry)
        session = make_session()

        first_response = make_mock_response(
            [make_tool_use_block("t1", "crashy_tool", {})],
            stop_reason="tool_use",
        )
        second_response = make_mock_response([make_text_block("handled")])
        mock_client.messages.create.side_effect = [first_response, second_response]

        result = engine.run_turn("crash it", session)

        assert result.tool_calls[0].result.is_error is True
        assert "boom" in result.tool_calls[0].result.content


class TestTurnResultFrozen:
    def test_turn_result_is_frozen(self) -> None:
        """TurnResult is a frozen dataclass — mutations raise."""
        from vector_os_nano.vcli.engine import ToolCall

        tc = ToolCall(
            tool_name="t",
            params={},
            result=ToolResult(content="x"),
            duration_sec=0.1,
            permission_action="allowed",
        )
        tr = TurnResult(
            text="hello",
            tool_calls=[tc],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        with pytest.raises((AttributeError, TypeError)):
            tr.text = "changed"  # type: ignore[misc]

    def test_tool_call_is_frozen(self) -> None:
        """ToolCall is frozen."""
        from vector_os_nano.vcli.engine import ToolCall

        tc = ToolCall(
            tool_name="t",
            params={},
            result=ToolResult(content="x"),
            duration_sec=0.0,
            permission_action="allowed",
        )
        with pytest.raises((AttributeError, TypeError)):
            tc.tool_name = "changed"  # type: ignore[misc]


class TestSessionIntegration:
    def test_session_tool_results_recorded(self) -> None:
        """After a tool turn, session contains tool_result entry."""
        registry = ToolRegistry()
        tool = make_read_only_tool("my_tool", return_content="data")
        registry.register(tool)

        engine, mock_client = make_engine(registry=registry)
        session = make_session()

        first_response = make_mock_response(
            [make_tool_use_block("t1", "my_tool", {})],
            stop_reason="tool_use",
        )
        second_response = make_mock_response([make_text_block("done")])
        mock_client.messages.create.side_effect = [first_response, second_response]

        engine.run_turn("go", session)

        messages = session.to_messages()
        # Should have: user, assistant (tool_use), user (tool_result), assistant (final text)
        roles = [m["role"] for m in messages]
        assert roles.count("user") == 2  # initial + tool_result
        assert roles.count("assistant") == 2  # tool call response + final

    def test_session_usage_accumulated(self) -> None:
        """Token usage is accumulated across all API calls in a turn."""
        registry = ToolRegistry()
        tool = make_read_only_tool("my_tool")
        registry.register(tool)

        engine, mock_client = make_engine(registry=registry)
        session = make_session()

        usage1 = MagicMock(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        usage2 = MagicMock(
            input_tokens=200,
            output_tokens=75,
            cache_read_input_tokens=5,
            cache_creation_input_tokens=0,
        )
        first_response = make_mock_response(
            [make_tool_use_block("t1", "my_tool", {})],
            stop_reason="tool_use",
            usage=usage1,
        )
        second_response = make_mock_response(
            [make_text_block("done")],
            usage=usage2,
        )
        mock_client.messages.create.side_effect = [first_response, second_response]

        result = engine.run_turn("go", session)

        # TurnResult carries cumulative usage
        assert result.usage.input_tokens == 300
        assert result.usage.output_tokens == 125
        assert result.usage.cache_read_tokens == 5
