"""Unit tests for vcli BashTool — TDD RED phase.

Covers:
- execute: echo, stderr capture, timeout, output truncation
- check_permissions: deny patterns, ask by default
- is_read_only returns False
- is_concurrency_safe returns False
- __tool_name__ == "bash"
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(**kwargs: Any):
    """Return a minimal ToolContext for testing."""
    from vector_os_nano.vcli.tools.base import ToolContext

    defaults: dict[str, Any] = {
        "agent": None,
        "cwd": Path("/tmp"),
        "session": None,
        "permissions": None,
        "abort": threading.Event(),
    }
    defaults.update(kwargs)
    return ToolContext(**defaults)


def _get_tool():
    """Return a fresh BashTool instance."""
    from vector_os_nano.vcli.tools.bash_tool import BashTool

    return BashTool()


# ---------------------------------------------------------------------------
# execute()
# ---------------------------------------------------------------------------


class TestBashToolExecute:
    def test_bash_runs_echo(self) -> None:
        tool = _get_tool()
        ctx = _make_context()
        result = tool.execute({"command": "echo hello"}, ctx)
        assert "hello" in result.content
        assert result.is_error is False

    def test_bash_captures_stderr(self) -> None:
        tool = _get_tool()
        ctx = _make_context()
        # Write to stderr explicitly; exit 0 so returncode is clean
        result = tool.execute({"command": "echo error-msg >&2"}, ctx)
        assert "error-msg" in result.content

    def test_bash_timeout(self) -> None:
        tool = _get_tool()
        ctx = _make_context()
        result = tool.execute({"command": "sleep 10", "timeout_ms": 500}, ctx)
        assert result.is_error is True
        assert "timed out" in result.content.lower() or "timeout" in result.content.lower()

    def test_bash_output_truncation(self) -> None:
        tool = _get_tool()
        ctx = _make_context()
        # Generate ~200 KB of output via yes | head to ensure truncation
        result = tool.execute(
            {"command": "yes | head -c 200000"},
            ctx,
        )
        # Content bytes should not exceed MAX_OUTPUT_BYTES + small overhead for message
        assert len(result.content.encode()) <= 110_000
        assert "truncated" in result.content.lower()

    def test_bash_nonzero_exit_is_error(self) -> None:
        tool = _get_tool()
        ctx = _make_context()
        result = tool.execute({"command": "exit 1"}, ctx)
        assert result.is_error is True


# ---------------------------------------------------------------------------
# check_permissions()
# ---------------------------------------------------------------------------


class TestBashToolPermissions:
    def test_bash_denies_rm_rf_root(self) -> None:
        tool = _get_tool()
        ctx = _make_context()
        result = tool.check_permissions({"command": "rm -rf /"}, ctx)
        assert result.behavior == "deny"

    def test_bash_denies_rm_rf_root_wildcard(self) -> None:
        tool = _get_tool()
        ctx = _make_context()
        result = tool.check_permissions({"command": "rm -rf /*"}, ctx)
        assert result.behavior == "deny"

    def test_bash_denies_fork_bomb(self) -> None:
        tool = _get_tool()
        ctx = _make_context()
        result = tool.check_permissions({"command": ":(){ :|:& };:"}, ctx)
        assert result.behavior == "deny"

    def test_bash_denies_mkfs(self) -> None:
        tool = _get_tool()
        ctx = _make_context()
        result = tool.check_permissions({"command": "mkfs.ext4 /dev/sda"}, ctx)
        assert result.behavior == "deny"

    def test_bash_denies_dd(self) -> None:
        tool = _get_tool()
        ctx = _make_context()
        result = tool.check_permissions(
            {"command": "dd if=/dev/zero of=/dev/sda"}, ctx
        )
        assert result.behavior == "deny"

    def test_bash_ask_by_default(self) -> None:
        """Safe commands should return 'ask', not 'allow' or 'deny'."""
        tool = _get_tool()
        ctx = _make_context()
        result = tool.check_permissions({"command": "ls"}, ctx)
        assert result.behavior == "ask"

    def test_bash_deny_reason_non_empty(self) -> None:
        tool = _get_tool()
        ctx = _make_context()
        result = tool.check_permissions({"command": "mkfs.ext4 /dev/sda"}, ctx)
        assert result.behavior == "deny"
        assert len(result.reason) > 0


# ---------------------------------------------------------------------------
# is_read_only / is_concurrency_safe
# ---------------------------------------------------------------------------


class TestBashToolFlags:
    def test_bash_not_read_only(self) -> None:
        tool = _get_tool()
        assert tool.is_read_only({}) is False

    def test_bash_not_concurrency_safe(self) -> None:
        tool = _get_tool()
        assert tool.is_concurrency_safe({}) is False


# ---------------------------------------------------------------------------
# @tool decorator metadata
# ---------------------------------------------------------------------------


class TestBashToolDecorator:
    def test_bash_registered_with_decorator(self) -> None:
        from vector_os_nano.vcli.tools.bash_tool import BashTool

        assert BashTool.__tool_name__ == "bash"

    def test_bash_tool_name_attribute(self) -> None:
        tool = _get_tool()
        assert tool.name == "bash"

    def test_bash_tool_has_description(self) -> None:
        tool = _get_tool()
        assert isinstance(tool.description, str)
        assert len(tool.description) > 0

    def test_bash_tool_has_input_schema(self) -> None:
        tool = _get_tool()
        assert isinstance(tool.input_schema, dict)
        assert "command" in tool.input_schema.get("properties", {})
