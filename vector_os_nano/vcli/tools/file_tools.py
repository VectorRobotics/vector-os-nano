# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""File tools for Vector CLI agentic harness: read, write, edit.

Public exports:
    FileReadTool  — read file with optional offset/limit, line-numbered output
    FileWriteTool — write/create file (refuses to overwrite unread files)
    FileEditTool  — search-and-replace (requires unique match)
"""
from __future__ import annotations

import difflib
import os
import re
from pathlib import Path
from typing import Any

from vector_os_nano.vcli.tools.base import PermissionResult, ToolContext, ToolResult, tool

# ---------------------------------------------------------------------------
# Path guard (shared by every file / search tool)
# ---------------------------------------------------------------------------
# Every check runs on the REAL path (os.path.realpath): "..", symlinks and
# relative spellings resolve before any comparison, so none of them bypass it.

# Credential stores under $HOME: refused outright, whatever the permission mode.
_HOME_SECRET_DIRS: tuple[str, ...] = (
    ".ssh", ".aws", ".gnupg", ".claude", ".config/gh", ".config/gcloud", ".docker", ".kube",
)
_HOME_SECRET_FILES: tuple[str, ...] = (
    ".vector/config.yaml", ".vector/oauth_credentials.json",
    ".netrc", ".git-credentials", ".pgpass",
)
_SYSTEM_SECRETS: tuple[str, ...] = (
    "/etc/shadow", "/etc/gshadow", "/etc/passwd", "/etc/sudoers", "/etc/sudoers.d",
    "/etc/ssh", "/etc/ssl/private",
)
# Secret-looking file names anywhere (.env, private keys); templates stay readable.
_SECRET_NAME_RE: re.Pattern[str] = re.compile(
    r"^(\.env(\..+)?|id_(rsa|dsa|ecdsa|ed25519)(_sk)?|.+\.(pem|key|p12|pfx))$"
)
_ENV_TEMPLATE_SUFFIXES: tuple[str, ...] = (".example", ".sample", ".template")

_BINARY_DETECT_BYTES = 8192


def real_path(path_str: str, cwd: Any = None) -> Path:
    """Expand ~, anchor a relative path at *cwd*, and resolve it (realpath)."""
    p = Path(os.path.expanduser(path_str))
    if not p.is_absolute():
        p = Path(cwd or os.getcwd()) / p
    return Path(os.path.realpath(p))


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def is_protected(path: Path) -> bool:
    """True when the REAL *path* is a credential store, a secret file or inside .git.
    Compared case-insensitively (a harmless false positive on Linux): on macOS or
    Windows .AWS/credentials and .GIT/config ARE the real secrets."""
    low = Path(str(path).lower())
    if ".git" in low.parts:
        return True
    if _SECRET_NAME_RE.match(low.name) and not low.name.endswith(_ENV_TEMPLATE_SUFFIXES):
        return True
    home = os.path.realpath(Path.home())
    roots = [os.path.realpath(p) for p in _SYSTEM_SECRETS]
    roots += [os.path.join(home, p) for p in _HOME_SECRET_DIRS + _HOME_SECRET_FILES]
    return any(_within(low, Path(r.lower())) for r in roots)


def project_root(context: Any) -> Path:
    """The real project root: the cwd the CLI was started in."""
    return Path(os.path.realpath(getattr(context, "cwd", None) or os.getcwd()))


def in_project(path: Path, root: Path) -> bool:
    """The REAL *path* lies inside *root*, and root is neither $HOME nor /."""
    return root not in (Path("/"), Path(os.path.realpath(Path.home()))) and _within(path, root)


def read_permission(path_str: str, context: Any) -> PermissionResult:
    """The ONE permission rule for every auto-allowed read tool.

    deny  — the real path is protected (credential deny-list, .git);
    allow — the real path lies inside the project root (in_project);
    ask   — anything else (a home dir or / as the root never auto-allows).
    """
    root = project_root(context)
    path = real_path(path_str, root)
    if is_protected(path):
        return PermissionResult("deny", f"{path_str} is a protected path")
    if in_project(path, root):
        return PermissionResult("allow")
    return PermissionResult("ask", f"Read {path} (outside the project {root})?")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_binary(path: Path) -> bool:
    """Return True when the file contains null bytes in the first 8 KB."""
    try:
        with path.open("rb") as fh:
            chunk = fh.read(_BINARY_DETECT_BYTES)
    except OSError:
        return False
    return b"\x00" in chunk


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

    def check_permissions(self, params: dict[str, Any], context: ToolContext) -> PermissionResult:
        return read_permission(str(params.get("file_path", "")), context)

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path_str: str = params["file_path"]
        offset: int = int(params.get("offset", 0))
        limit: int = int(params.get("limit", 2000))

        path = real_path(file_path_str, getattr(context, "cwd", None))

        # 1. Security check (defence in depth: also holds when no permission gate ran)
        if is_protected(path):
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

    def check_permissions(self, params: dict[str, Any], context: ToolContext) -> PermissionResult:
        # Hard-deny writes to protected paths, mirroring the read path's guard.
        # A "deny" here is unconditional (survives --no-permission / always-allow).
        if is_protected(real_path(str(params.get("file_path", "")), getattr(context, "cwd", None))):
            return PermissionResult("deny", f"{params.get('file_path')} is a protected path")
        return PermissionResult("ask")

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path_str: str = params["file_path"]
        content: str = params["content"]

        path = real_path(file_path_str, getattr(context, "cwd", None))

        # Defence in depth: also block protected paths at execute() so the guard
        # holds even if a caller bypasses PermissionContext.
        if is_protected(path):
            return ToolResult(
                content=f"Access denied: {file_path_str} is a protected path.",
                is_error=True,
            )

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
# Diff helper
# ---------------------------------------------------------------------------

_MAX_DIFF_LINES: int = 20


def _format_edit_diff(before: str, after: str, file_path_str: str) -> str:
    """Return a compact unified-diff summary of a single edit.

    Uses ``difflib.unified_diff`` with n=2 context lines and caps the
    output at *_MAX_DIFF_LINES* lines to keep it readable.

    Returns a string like::

        Edited /path/to/file.py:
        @@ line 7 @@
        -old line content
        +new line content
         context line
    """
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)

    diff_iter = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=file_path_str,
        tofile=file_path_str,
        n=2,
    )

    # Collect and cap the diff output
    diff_lines: list[str] = []
    for line in diff_iter:
        diff_lines.append(line)
        if len(diff_lines) >= _MAX_DIFF_LINES:
            break

    if not diff_lines:
        # Identical content after replacement (old_string == new_string edge-case)
        return f"Edited {file_path_str}: no visible changes"

    # Strip the two "--- / +++" header lines produced by unified_diff;
    # they are redundant since we already show the filename.
    body_lines = [ln for ln in diff_lines if not ln.startswith("--- ") and not ln.startswith("+++ ")]

    # Rewrite the @@ hunk header to a friendlier "line N" format
    formatted: list[str] = [f"Edited {file_path_str}:"]
    for ln in body_lines:
        if ln.startswith("@@"):
            # Parse first line number from "@@ -L,C +L,C @@"
            try:
                after_marker = ln.split("+")[1].split(",")[0]
                line_num = int(after_marker)
                formatted.append(f"@@ line {line_num} @@")
            except (IndexError, ValueError):
                formatted.append(ln.rstrip())
        else:
            formatted.append(ln.rstrip())

    return "\n".join(formatted)


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

    def check_permissions(self, params: dict[str, Any], context: ToolContext) -> PermissionResult:
        # Hard-deny edits to protected paths (mirrors the read/write guards).
        if is_protected(real_path(str(params.get("file_path", "")), getattr(context, "cwd", None))):
            return PermissionResult("deny", f"{params.get('file_path')} is a protected path")
        return PermissionResult("ask")

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path_str: str = params["file_path"]
        old_string: str = params["old_string"]
        new_string: str = params["new_string"]

        path = real_path(file_path_str, getattr(context, "cwd", None))

        # Defence in depth: block protected paths even outside PermissionContext.
        if is_protected(path):
            return ToolResult(
                content=f"Access denied: {file_path_str} is a protected path.",
                is_error=True,
            )

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

        # 4. Build a compact unified diff for the caller
        diff_text = _format_edit_diff(
            before=current,
            after=updated,
            file_path_str=file_path_str,
        )
        return ToolResult(content=diff_text)
