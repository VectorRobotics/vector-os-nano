# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Unit tests for VectorEngine — updated for backend-agnostic API.

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
from unittest.mock import MagicMock, call

import pytest

from vector_os_nano.vcli.backends.types import LLMResponse, LLMToolCall
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


def make_llm_response(
    text: str = "",
    tool_calls: list[LLMToolCall] | None = None,
    stop_reason: str = "end_turn",
    usage: TokenUsage | None = None,
) -> LLMResponse:
    """Build a canonical LLMResponse for test use."""
    return LLMResponse(
        text=text,
        tool_calls=tool_calls or [],
        stop_reason=stop_reason,
        usage=usage or TokenUsage(input_tokens=100, output_tokens=50),
    )


def make_tool_call(tool_id: str, name: str, input_dict: dict) -> LLMToolCall:
    """Build a frozen LLMToolCall."""
    return LLMToolCall(id=tool_id, name=name, input=input_dict)


def make_session() -> Session:
    """Return a transient in-memory session (no file I/O)."""
    return Session(
        session_id="test-session-1",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        path=Path("/tmp/test_session.jsonl"),
    )


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
    """Return (engine, mock_backend) with a MagicMock backend."""
    mock_backend = MagicMock()
    engine = VectorEngine(
        backend=mock_backend,
        registry=registry or ToolRegistry(),
        permissions=permissions or PermissionContext(no_permission=True),
        max_turns=max_turns,
    )
    return engine, mock_backend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleTextTurn:
    def test_single_text_turn(self) -> None:
        """User sends 'hello', backend returns text-only → TurnResult with text, no tool_calls."""
        engine, mock_backend = make_engine()
        session = make_session()

        mock_backend.call.return_value = make_llm_response(text="Hello, world!")

        result = engine.run_turn("hello", session)

        assert isinstance(result, TurnResult)
        assert result.text == "Hello, world!"
        assert result.tool_calls == []
        assert result.stop_reason == "end_turn"

    def test_turn_result_usage(self) -> None:
        """TurnResult.usage reflects the token counts from the backend response."""
        engine, mock_backend = make_engine()
        session = make_session()

        mock_backend.call.return_value = make_llm_response(
            text="hi",
            usage=TokenUsage(
                input_tokens=200,
                output_tokens=75,
                cache_read_tokens=10,
                cache_creation_tokens=5,
            ),
        )

        result = engine.run_turn("ping", session)

        assert result.usage.input_tokens == 200
        assert result.usage.output_tokens == 75
        assert result.usage.cache_read_tokens == 10
        assert result.usage.cache_creation_tokens == 5

    def test_session_messages_updated(self) -> None:
        """After run_turn, session.to_messages() contains user + assistant entries."""
        engine, mock_backend = make_engine()
        session = make_session()

        mock_backend.call.return_value = make_llm_response(text="I'm fine.")

        engine.run_turn("How are you?", session)

        messages = session.to_messages()
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "How are you?"
        assert messages[1]["role"] == "assistant"
        assert any(
            block.get("text") == "I'm fine."
            for block in messages[1]["content"]
            if isinstance(block, dict)
        )


class TestStreamingCallback:
    def test_streaming_text_callback(self) -> None:
        """on_text callback receives text chunks during the turn.

        The backend's call() invokes the on_text callback that was passed to it.
        We simulate this with a side_effect that calls on_text with two chunks,
        then returns a final LLMResponse.
        """
        engine, mock_backend = make_engine()
        session = make_session()

        def streaming_side_effect(messages, tools, system, max_tokens, on_text=None):
            if on_text is not None:
                on_text("chunk one")
                on_text(" chunk two")
            return make_llm_response(text="chunk one chunk two")

        mock_backend.call.side_effect = streaming_side_effect

        received: list[str] = []
        engine.run_turn("hello", session, on_text=received.append)

        assert received == ["chunk one", " chunk two"]


class TestToolUseTurn:
    def test_tool_use_turn(self) -> None:
        """Backend returns tool_use → engine executes tool → sends result → returns final text."""
        registry = ToolRegistry()
        tool = make_read_only_tool("read_file", return_content="file contents")
        registry.register(tool)

        engine, mock_backend = make_engine(registry=registry)
        session = make_session()

        first_response = make_llm_response(
            tool_calls=[make_tool_call("tool_1", "read_file", {"path": "/tmp/x.txt"})],
            stop_reason="tool_use",
        )
        second_response = make_llm_response(text="Done.")
        mock_backend.call.side_effect = [first_response, second_response]

        result = engine.run_turn("read that file", session)

        assert result.text == "Done."
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.tool_name == "read_file"
        assert tc.params == {"path": "/tmp/x.txt"}
        assert tc.result.content == "file contents"
        assert not tc.result.is_error

    def test_multi_tool_turn(self) -> None:
        """Backend returns 2 tool_use blocks → both executed → results sent back."""
        registry = ToolRegistry()
        tool_a = make_read_only_tool("tool_a", return_content="result_a")
        tool_b = make_read_only_tool("tool_b", return_content="result_b")
        registry.register(tool_a)
        registry.register(tool_b)

        engine, mock_backend = make_engine(registry=registry)
        session = make_session()

        first_response = make_llm_response(
            tool_calls=[
                make_tool_call("id_a", "tool_a", {}),
                make_tool_call("id_b", "tool_b", {}),
            ],
            stop_reason="tool_use",
        )
        second_response = make_llm_response(text="Both done.")
        mock_backend.call.side_effect = [first_response, second_response]

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

        tc_a = make_tool_call("id_a", "tool_a", {})
        tc_b = make_tool_call("id_b", "tool_b", {})
        batches = engine._partition_tools([tc_a, tc_b])

        assert len(batches) == 1
        assert batches[0].concurrent is True
        assert len(batches[0].tool_calls) == 2

    def test_tool_concurrency_partition_write_sequential(self) -> None:
        """Write tools each get their own sequential batch."""
        registry = ToolRegistry()
        tool_a = make_write_tool("write_a")
        tool_b = make_write_tool("write_b")
        registry.register(tool_a)
        registry.register(tool_b)

        engine, _ = make_engine(registry=registry)

        tc_a = make_tool_call("id_a", "write_a", {})
        tc_b = make_tool_call("id_b", "write_b", {})
        batches = engine._partition_tools([tc_a, tc_b])

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

        tc_ro = make_tool_call("id_ro", "ro", {})
        tc_wr = make_tool_call("id_wr", "wr", {})
        batches = engine._partition_tools([tc_ro, tc_wr])

        assert len(batches) == 2
        assert batches[0].concurrent is True
        assert batches[1].concurrent is False


class TestCallbacks:
    def test_tool_start_end_callbacks(self) -> None:
        """on_tool_start and on_tool_end are called with correct tool name and result."""
        registry = ToolRegistry()
        tool = make_read_only_tool("my_tool", return_content="output")
        registry.register(tool)

        engine, mock_backend = make_engine(registry=registry)
        session = make_session()

        first_response = make_llm_response(
            tool_calls=[make_tool_call("tool_1", "my_tool", {"key": "val"})],
            stop_reason="tool_use",
        )
        second_response = make_llm_response(text="done")
        mock_backend.call.side_effect = [first_response, second_response]

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

        engine, mock_backend = make_engine(registry=registry, max_turns=2)
        session = make_session()

        # Every response calls loop_tool forever
        looping_response = make_llm_response(
            tool_calls=[make_tool_call("t1", "loop_tool", {})],
            stop_reason="tool_use",
        )
        mock_backend.call.return_value = looping_response

        engine.run_turn("loop", session)

        # Engine must stop — backend was called at most max_turns + 1 times
        assert mock_backend.call.call_count <= 3  # initial + 2 tool loops


class TestPermissions:
    def test_permission_deny_returns_error(self) -> None:
        """Permission deny → tool not executed, error result sent to backend."""
        registry = ToolRegistry()
        tool = make_write_tool("dangerous_tool")
        tool.check_permissions.return_value = PermissionResult(behavior="deny", reason="nope")
        registry.register(tool)

        permissions = PermissionContext(deny_tools={"dangerous_tool"})
        engine, mock_backend = make_engine(registry=registry, permissions=permissions)
        session = make_session()

        first_response = make_llm_response(
            tool_calls=[make_tool_call("t1", "dangerous_tool", {})],
            stop_reason="tool_use",
        )
        second_response = make_llm_response(text="Denied.")
        mock_backend.call.side_effect = [first_response, second_response]

        result = engine.run_turn("do it", session)

        tool.execute.assert_not_called()

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
        engine, mock_backend = make_engine(registry=registry, permissions=permissions)
        session = make_session()

        first_response = make_llm_response(
            tool_calls=[make_tool_call("t1", "write_tool", {"data": "hello"})],
            stop_reason="tool_use",
        )
        second_response = make_llm_response(text="Written.")
        mock_backend.call.side_effect = [first_response, second_response]

        result = engine.run_turn("write it", session, ask_permission=lambda n, p: "y")

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
        engine, mock_backend = make_engine(registry=registry, permissions=permissions)
        session = make_session()

        first_response = make_llm_response(
            tool_calls=[make_tool_call("t1", "write_tool", {})],
            stop_reason="tool_use",
        )
        second_response = make_llm_response(text="User declined.")
        mock_backend.call.side_effect = [first_response, second_response]

        result = engine.run_turn("write it", session, ask_permission=lambda n, p: "n")

        tool.execute.assert_not_called()
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.result.is_error is True
        assert tc.permission_action == "asked_denied"

    def test_unknown_tool_returns_error(self) -> None:
        """Backend calls a tool name not in registry → error ToolResult, not a crash."""
        engine, mock_backend = make_engine()
        session = make_session()

        first_response = make_llm_response(
            tool_calls=[make_tool_call("t1", "nonexistent_tool", {})],
            stop_reason="tool_use",
        )
        second_response = make_llm_response(text="ok")
        mock_backend.call.side_effect = [first_response, second_response]

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

        engine, mock_backend = make_engine(registry=registry)
        session = make_session()

        first_response = make_llm_response(
            tool_calls=[make_tool_call("t1", "crashy_tool", {})],
            stop_reason="tool_use",
        )
        second_response = make_llm_response(text="handled")
        mock_backend.call.side_effect = [first_response, second_response]

        result = engine.run_turn("crash it", session)

        assert result.tool_calls[0].result.is_error is True
        assert "boom" in result.tool_calls[0].result.content


class TestAlwaysAllowPermission:
    def test_ask_permission_returns_a_adds_to_session_allow(self) -> None:
        """ask_permission returning 'a' calls permissions.add_always_allow and executes tool."""
        registry = ToolRegistry()
        tool = make_write_tool("write_tool")
        registry.register(tool)

        permissions = PermissionContext()  # default: will ask
        engine, mock_backend = make_engine(registry=registry, permissions=permissions)
        session = make_session()

        first_response = make_llm_response(
            tool_calls=[make_tool_call("t1", "write_tool", {"data": "hello"})],
            stop_reason="tool_use",
        )
        second_response = make_llm_response(text="Written always.")
        mock_backend.call.side_effect = [first_response, second_response]

        result = engine.run_turn("write it", session, ask_permission=lambda n, p: "a")

        tool.execute.assert_called_once()
        assert "write_tool" in permissions.session_allow
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].permission_action == "asked_allowed"
        assert not result.tool_calls[0].result.is_error


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

        engine, mock_backend = make_engine(registry=registry)
        session = make_session()

        first_response = make_llm_response(
            tool_calls=[make_tool_call("t1", "my_tool", {})],
            stop_reason="tool_use",
        )
        second_response = make_llm_response(text="done")
        mock_backend.call.side_effect = [first_response, second_response]

        engine.run_turn("go", session)

        messages = session.to_messages()
        roles = [m["role"] for m in messages]
        assert roles.count("user") == 2  # initial + tool_result
        assert roles.count("assistant") == 2  # tool call response + final

    def test_session_usage_accumulated(self) -> None:
        """Token usage is accumulated across all backend calls in a turn."""
        registry = ToolRegistry()
        tool = make_read_only_tool("my_tool")
        registry.register(tool)

        engine, mock_backend = make_engine(registry=registry)
        session = make_session()

        first_response = make_llm_response(
            tool_calls=[make_tool_call("t1", "my_tool", {})],
            stop_reason="tool_use",
            usage=TokenUsage(input_tokens=100, output_tokens=50),
        )
        second_response = make_llm_response(
            text="done",
            usage=TokenUsage(
                input_tokens=200,
                output_tokens=75,
                cache_read_tokens=5,
            ),
        )
        mock_backend.call.side_effect = [first_response, second_response]

        result = engine.run_turn("go", session)

        assert result.usage.input_tokens == 300
        assert result.usage.output_tokens == 125
        assert result.usage.cache_read_tokens == 5


def test_legacy_turn_never_sends_a_tool_use_absent_from_its_tool_list() -> None:
    """I23: the native turn (with its synthetic verify/finish tool_use) lives in the
    real session; a later legacy turn must not send those blocks (a tool_use naming
    a tool absent from ``tools``). The exchange survives as text."""
    from vector_os_nano.vcli.backends.openai_compat import convert_messages

    registry = ToolRegistry()
    registry.register(make_read_only_tool("file_read"))
    engine, mock_backend = make_engine(registry=registry)
    session = make_session()
    session.append_user("找到绿色瓶子")
    session.append_assistant("", [
        {"type": "tool_use", "id": "l1", "name": "file_read", "input": {"path": "a"}},
        {"type": "tool_use", "id": "v1", "name": "verify", "input": {"expr": "localized('green bottle')"}},
    ])
    session.append_tool_results([
        {"tool_use_id": "l1", "content": "a-text"},
        {"tool_use_id": "v1", "content": "verify localized('green bottle') -> True"},
    ])
    session.append_assistant("", [{"type": "tool_use", "id": "f1", "name": "finish", "input": {}}])
    session.append_tool_results([{"tool_use_id": "f1", "content": "finished"}])
    mock_backend.call.return_value = make_llm_response(text="hi")

    engine.run_turn("你好", session)

    sent = mock_backend.call.call_args.kwargs["messages"]
    names = {t["name"] for t in mock_backend.call.call_args.kwargs["tools"]}
    blocks = [b for m in sent if isinstance(m["content"], list) for b in m["content"]]
    used = {b["id"]: b["name"] for b in blocks if b.get("type") == "tool_use"}
    assert used == {"l1": "file_read"} and set(used.values()) <= names
    assert [b["tool_use_id"] for b in blocks if b.get("type") == "tool_result"] == ["l1"]
    text = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    assert "localized('green bottle') -> True" in text and "finish" in text
    roles = [m["role"] for m in convert_messages(sent, "")]
    assert all(a != b or a == "tool" for a, b in zip(roles, roles[1:])), roles
