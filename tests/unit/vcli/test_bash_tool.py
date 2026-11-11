# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

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


def _gone(pid: int) -> bool:
    """True once *pid* has exited (missing, or a zombie awaiting its reaper)."""
    import time

    for _ in range(60):
        try:
            with open(f"/proc/{pid}/stat") as fh:
                if fh.read().rsplit(")", 1)[1].split()[0] == "Z":
                    return True
        except FileNotFoundError:
            return True
        time.sleep(0.05)
    return False


class TestBashToolProcessGroup:
    """R3: abort and timeout stop the WHOLE command, and a background job cannot
    hold the call open (tools-worlds:bash-gate-weak-and-blocking)."""

    def test_abort_stops_the_command_and_its_children(self, tmp_path: Path) -> None:
        import os
        import time

        abort = threading.Event()
        pidfile = tmp_path / "child.pid"
        threading.Timer(0.5, abort.set).start()
        t0 = time.monotonic()
        result = _get_tool().execute(
            {"command": f"sleep 30 & echo $! > {pidfile}; wait"},
            _make_context(abort=abort),
        )
        assert time.monotonic() - t0 < 5.0
        assert result.is_error and "aborted" in result.content.lower()
        child = int(pidfile.read_text())
        try:
            assert _gone(child), "the backgrounded grandchild outlived the abort"
        finally:
            try:
                os.kill(child, 9)
            except ProcessLookupError:
                pass

    def test_timeout_kills_grandchildren(self, tmp_path: Path) -> None:
        import os

        pidfile = tmp_path / "child.pid"
        result = _get_tool().execute(
            {"command": f"sleep 30 & echo $! > {pidfile}; wait", "timeout_ms": 500},
            _make_context(),
        )
        assert result.is_error and "timed out" in result.content.lower()
        child = int(pidfile.read_text())
        try:
            assert _gone(child), "the grandchild outlived the timeout"
        finally:
            try:
                os.kill(child, 9)
            except ProcessLookupError:
                pass

    def test_a_background_job_does_not_hang_the_call(self, tmp_path: Path) -> None:
        import os
        import time

        pidfile = tmp_path / "bg.pid"
        t0 = time.monotonic()
        result = _get_tool().execute(
            {"command": f"sleep 30 & echo $! > {pidfile}; echo started", "timeout_ms": 20_000},
            _make_context(),
        )
        try:
            assert time.monotonic() - t0 < 5.0, "the call waited for the background job"
            assert not result.is_error and "started" in result.content
        finally:
            try:
                os.kill(int(pidfile.read_text()), 9)
            except ProcessLookupError:
                pass

    def test_stdin_is_closed(self) -> None:
        result = _get_tool().execute({"command": "cat; echo done", "timeout_ms": 5000}, _make_context())
        assert not result.is_error and "done" in result.content


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
