"""File tools for Vector CLI agentic harness: read, write, edit.

Public exports:
    FileReadTool  — read file with optional offset/limit, line-numbered output
    FileWriteTool — write/create file (refuses to overwrite unread files)
    FileEditTool  — search-and-replace (requires unique match)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from vector_os_nano.vcli.tools.base import ToolContext, ToolResult, tool

# ---------------------------------------------------------------------------
# Security constants
# ---------------------------------------------------------------------------

DANGEROUS_PATHS: frozenset[str] = frozenset(
    {"/etc/shadow", "/etc/passwd", "/etc/sudoers"}
)

DANGEROUS_PREFIXES: tuple[str, ...] = (
    "/etc/ssh/",
    "/etc/ssl/private/",
    "~/.ssh/",
    "~/.claude/",
    ".git/",
)

_BINARY_DETECT_BYTES = 8192


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_binary(path: Path) -> bool:
    """Return True when the file contains null bytes in the first 8 KB."""
    try:
        chunk = path.read_bytes()[:_BINARY_DETECT_BYTES]
    except OSError:
        return False
    return b"\x00" in chunk


def _resolve(file_path: str) -> Path:
    """Expand ~ and return a resolved Path (without requiring it to exist)."""
    return Path(file_path).expanduser()


def _is_dangerous(path: Path) -> bool:
    """Return True when the path hits a known sensitive location."""
    path_str = str(path)
    if path_str in DANGEROUS_PATHS:
        return True
    for prefix in DANGEROUS_PREFIXES:
        expanded = str(Path(prefix).expanduser())
        if path_str.startswith(expanded) or path_str.startswith(prefix):
            return True
    return False


def _line_numbered(lines: list[str], start_lineno: int) -> str:
    """Format lines with cat-n style numbering (1-based from *start_lineno*)."""
    parts: list[str] = []
    for i, line in enumerate(lines, start=start_lineno):
        parts.append(f"{i:6}\t{line}")
    return "".join(parts)


def _get_read_files(context: ToolContext) -> set[str]:
    """Return the set of absolute file paths read in this session."""
    session = context.session
    if session is None:
        return set()
    # Session object may expose a read_files set or similar attribute.
    if hasattr(session, "read_files"):
        return session.read_files
    return set()


def _record_read(context: ToolContext, path: Path) -> None:
    """Mark a file as read in the current session."""
    session = context.session
    if session is None:
        return
    if hasattr(session, "read_files"):
        session.read_files.add(str(path))


# ---------------------------------------------------------------------------
# FileReadTool
# ---------------------------------------------------------------------------


@tool(
    name="file_read",
    description="Read a file from disk with optional offset/limit",
    read_only=True,
    permission="allow",
)
class FileReadTool:
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to read",
            },
            "offset": {
                "type": "integer",
                "description": "Line offset (0-based, skips this many lines)",
                "default": 0,
            },
            "limit": {
                "type": "integer",
                "description": "Max lines to return",
                "default": 2000,
            },
        },
        "required": ["file_path"],
    }

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path_str: str = params["file_path"]
        offset: int = int(params.get("offset", 0))
        limit: int = int(params.get("limit", 2000))

        path = _resolve(file_path_str)

        # 1. Security check
        if _is_dangerous(path):
            return ToolResult(
                content=f"Access denied: {file_path_str} is a protected path.",
                is_error=True,
            )

        # 2. Existence check
        if not path.exists():
            return ToolResult(
                content=f"File not found: {file_path_str}",
                is_error=True,
            )

        if not path.is_file():
            return ToolResult(
                content=f"Not a regular file: {file_path_str}",
                is_error=True,
            )

        # 3. Binary check
        if _is_binary(path):
            return ToolResult(
                content=f"Cannot read binary file: {file_path_str}",
                is_error=True,
            )

        # 4. Read with offset/limit
        try:
            all_lines = path.read_text(errors="replace").splitlines(keepends=True)
        except OSError as exc:
            return ToolResult(content=f"Read error: {exc}", is_error=True)

        window = all_lines[offset : offset + limit]
        start_lineno = offset + 1
        output = _line_numbered(window, start_lineno)

        # 5. Record in session for write-guard
        _record_read(context, path)

        return ToolResult(content=output)


# ---------------------------------------------------------------------------
# FileWriteTool
# ---------------------------------------------------------------------------


@tool(
    name="file_write",
    description="Write content to a file (create or overwrite)",
    read_only=False,
    permission="ask",
)
class FileWriteTool:
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to write",
            },
            "content": {
                "type": "string",
                "description": "File content to write",
            },
        },
        "required": ["file_path", "content"],
    }

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path_str: str = params["file_path"]
        content: str = params["content"]

        path = _resolve(file_path_str)

        # Refuse to overwrite an existing file that was not read in this session
        if path.exists() and str(path) not in _get_read_files(context):
            return ToolResult(
                content=(
                    f"Refusing to overwrite '{file_path_str}': file exists but has not "
                    "been read in this session. Read it first to confirm intent."
                ),
                is_error=True,
            )

        # Create parent directories if needed
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        except OSError as exc:
            return ToolResult(content=f"Write error: {exc}", is_error=True)

        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return ToolResult(
            content=f"Wrote {line_count} line(s) to {file_path_str}",
        )


# ---------------------------------------------------------------------------
# FileEditTool
# ---------------------------------------------------------------------------


@tool(
    name="file_edit",
    description="Search and replace text in a file (old_string must match exactly once)",
    read_only=False,
    permission="ask",
)
class FileEditTool:
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to edit",
            },
            "old_string": {
                "type": "string",
                "description": "Exact string to find (must appear exactly once)",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement string",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path_str: str = params["file_path"]
        old_string: str = params["old_string"]
        new_string: str = params["new_string"]

        path = _resolve(file_path_str)

        # 1. Read current content
        if not path.exists():
            return ToolResult(
                content=f"File not found: {file_path_str}",
                is_error=True,
            )

        try:
            current = path.read_text()
        except OSError as exc:
            return ToolResult(content=f"Read error: {exc}", is_error=True)

        # 2. Count occurrences
        count = current.count(old_string)

        if count == 0:
            return ToolResult(
                content=f"old_string not found in {file_path_str}",
                is_error=True,
            )

        if count > 1:
            return ToolResult(
                content=(
                    f"old_string is not unique in {file_path_str}: "
                    f"found {count} occurrences. Provide more context to make it unique."
                ),
                is_error=True,
            )

        # 3. Replace (exactly once)
        updated = current.replace(old_string, new_string, 1)

        try:
            path.write_text(updated)
        except OSError as exc:
            return ToolResult(content=f"Write error: {exc}", is_error=True)

        return ToolResult(
            content=f"Replaced 1 occurrence in {file_path_str}",
        )
