"""Unit tests for vcli file tools (read/write/edit) — TDD RED phase.

Covers:
- FileReadTool: content, offset/limit, nonexistent, binary rejection, dangerous path
- FileWriteTool: create new file, refuse overwrite of unread file
- FileEditTool: unique replacement, not-unique error, not-found error
- Decorator metadata: __tool_name__ on all three
- is_read_only semantics per tool
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(tmp_path: Path, session: Any = None) -> Any:
    from vector_os_nano.vcli.tools.base import ToolContext

    return ToolContext(
        agent=None,
        cwd=tmp_path,
        session=session,
        permissions=None,
        abort=threading.Event(),
    )


# ---------------------------------------------------------------------------
# FileReadTool
# ---------------------------------------------------------------------------


class TestFileReadTool:
    def test_file_read_returns_content(self, tmp_path: Path) -> None:
        """Reads a temp file and returns line-numbered content (cat -n style)."""
        from vector_os_nano.vcli.tools.file_tools import FileReadTool

        target = tmp_path / "hello.txt"
        target.write_text("line one\nline two\nline three\n")

        tool = FileReadTool()
        result = tool.execute({"file_path": str(target)}, _make_context(tmp_path))

        assert not result.is_error
        assert "line one" in result.content
        assert "line two" in result.content
        assert "line three" in result.content
        # Must be line-numbered (cat -n style): "     1\t..." or "  1\t..."
        assert "\t" in result.content

    def test_file_read_offset_limit(self, tmp_path: Path) -> None:
        """offset=5, limit=3 returns only lines 6-8 (1-based)."""
        from vector_os_nano.vcli.tools.file_tools import FileReadTool

        lines = [f"line {i}\n" for i in range(1, 12)]  # lines 1-11
        target = tmp_path / "multi.txt"
        target.write_text("".join(lines))

        tool = FileReadTool()
        result = tool.execute(
            {"file_path": str(target), "offset": 5, "limit": 3},
            _make_context(tmp_path),
        )

        assert not result.is_error
        assert "line 6" in result.content
        assert "line 7" in result.content
        assert "line 8" in result.content
        # Lines outside the window must not appear
        assert "line 5" not in result.content
        assert "line 9" not in result.content

    def test_file_read_nonexistent(self, tmp_path: Path) -> None:
        """Returns is_error=True when the file does not exist."""
        from vector_os_nano.vcli.tools.file_tools import FileReadTool

        tool = FileReadTool()
        result = tool.execute(
            {"file_path": str(tmp_path / "no_such_file.txt")},
            _make_context(tmp_path),
        )

        assert result.is_error

    def test_file_read_binary_rejected(self, tmp_path: Path) -> None:
        """Binary file returns is_error with 'binary' in message."""
        from vector_os_nano.vcli.tools.file_tools import FileReadTool

        binary_file = tmp_path / "data.bin"
        binary_file.write_bytes(bytes(range(256)))

        tool = FileReadTool()
        result = tool.execute({"file_path": str(binary_file)}, _make_context(tmp_path))

        assert result.is_error
        assert "binary" in result.content.lower()

    def test_file_read_dangerous_path(self, tmp_path: Path) -> None:
        """Reading /etc/shadow returns is_error=True."""
        from vector_os_nano.vcli.tools.file_tools import FileReadTool

        tool = FileReadTool()
        result = tool.execute({"file_path": "/etc/shadow"}, _make_context(tmp_path))

        assert result.is_error

    def test_file_read_is_read_only(self) -> None:
        """FileReadTool.is_read_only() returns True."""
        from vector_os_nano.vcli.tools.file_tools import FileReadTool

        tool = FileReadTool()
        assert tool.is_read_only({}) is True


# ---------------------------------------------------------------------------
# FileWriteTool
# ---------------------------------------------------------------------------


class TestFileWriteTool:
    def test_file_write_creates_file(self, tmp_path: Path) -> None:
        """Creates a new file with the given content."""
        from vector_os_nano.vcli.tools.file_tools import FileWriteTool

        target = tmp_path / "new_file.txt"
        content = "hello world\nsecond line\n"

        tool = FileWriteTool()
        result = tool.execute(
            {"file_path": str(target), "content": content},
            _make_context(tmp_path),
        )

        assert not result.is_error
        assert target.exists()
        assert target.read_text() == content

    def test_file_write_refuses_overwrite_unread(self, tmp_path: Path) -> None:
        """Overwriting an existing file not previously read in session -> is_error."""
        from vector_os_nano.vcli.tools.file_tools import FileWriteTool

        existing = tmp_path / "existing.txt"
        existing.write_text("original content\n")

        # No session tracking of a read — session=None means nothing was read
        tool = FileWriteTool()
        result = tool.execute(
            {"file_path": str(existing), "content": "new content\n"},
            _make_context(tmp_path, session=None),
        )

        assert result.is_error

    def test_file_write_not_read_only(self) -> None:
        """FileWriteTool.is_read_only() returns False."""
        from vector_os_nano.vcli.tools.file_tools import FileWriteTool

        tool = FileWriteTool()
        assert tool.is_read_only({}) is False


# ---------------------------------------------------------------------------
# FileEditTool
# ---------------------------------------------------------------------------


class TestFileEditTool:
    def test_file_edit_replaces_unique(self, tmp_path: Path) -> None:
        """old_string found exactly once -> replaced with new_string."""
        from vector_os_nano.vcli.tools.file_tools import FileEditTool

        target = tmp_path / "edit_me.txt"
        target.write_text("foo bar baz\nhello world\n")

        tool = FileEditTool()
        result = tool.execute(
            {
                "file_path": str(target),
                "old_string": "hello world",
                "new_string": "goodbye world",
            },
            _make_context(tmp_path),
        )

        assert not result.is_error
        assert target.read_text() == "foo bar baz\ngoodbye world\n"

    def test_file_edit_fails_not_unique(self, tmp_path: Path) -> None:
        """old_string found 2+ times -> is_error."""
        from vector_os_nano.vcli.tools.file_tools import FileEditTool

        target = tmp_path / "dupe.txt"
        target.write_text("dup\ndup\n")

        tool = FileEditTool()
        result = tool.execute(
            {
                "file_path": str(target),
                "old_string": "dup",
                "new_string": "unique",
            },
            _make_context(tmp_path),
        )

        assert result.is_error
        # File must be unchanged
        assert target.read_text() == "dup\ndup\n"

    def test_file_edit_fails_not_found(self, tmp_path: Path) -> None:
        """old_string not present -> is_error."""
        from vector_os_nano.vcli.tools.file_tools import FileEditTool

        target = tmp_path / "nostr.txt"
        target.write_text("alpha beta gamma\n")

        tool = FileEditTool()
        result = tool.execute(
            {
                "file_path": str(target),
                "old_string": "delta",
                "new_string": "epsilon",
            },
            _make_context(tmp_path),
        )

        assert result.is_error


# ---------------------------------------------------------------------------
# Decorator metadata
# ---------------------------------------------------------------------------


class TestToolDecorators:
    def test_all_registered_with_tool_decorator(self) -> None:
        """All three file tools carry __tool_name__ stamped by @tool."""
        from vector_os_nano.vcli.tools.file_tools import (
            FileEditTool,
            FileReadTool,
            FileWriteTool,
        )

        assert hasattr(FileReadTool, "__tool_name__")
        assert hasattr(FileWriteTool, "__tool_name__")
        assert hasattr(FileEditTool, "__tool_name__")

    def test_file_read_tool_name(self) -> None:
        from vector_os_nano.vcli.tools.file_tools import FileReadTool

        assert FileReadTool.__tool_name__ == "file_read"

    def test_file_write_tool_name(self) -> None:
        from vector_os_nano.vcli.tools.file_tools import FileWriteTool

        assert FileWriteTool.__tool_name__ == "file_write"

    def test_file_edit_tool_name(self) -> None:
        from vector_os_nano.vcli.tools.file_tools import FileEditTool

        assert FileEditTool.__tool_name__ == "file_edit"
