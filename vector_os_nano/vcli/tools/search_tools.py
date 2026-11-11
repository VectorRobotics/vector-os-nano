# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""vcli search tools — GlobTool and GrepTool for the agentic harness.

Both tools are read-only; a call inside the project needs no prompt and is concurrency-safe,
anything else asks (see file_tools.read_permission) and runs alone.

GlobTool: find files matching a glob pattern using pathlib, sorted by mtime (newest first).
GrepTool: search file contents via ripgrep (rg) with Python re fallback.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

from vector_os_nano.vcli.tools.base import PermissionResult, ToolContext, ToolResult, tool
from vector_os_nano.vcli.tools.file_tools import is_protected, read_permission, real_path


def _search_root(params: dict[str, Any], context: ToolContext) -> Path:
    return real_path(str(params.get("path") or context.cwd), context.cwd)


def _needs_no_prompt(params: dict[str, Any]) -> bool:
    return read_permission(str(params.get("path") or "."), None).behavior == "allow"


def _readable(path: str | Path, root: Path) -> bool:
    """A file found under *root* may be read: not protected, and its real path
    stays inside the root the permission was granted for (no symlink escape)."""
    real = Path(os.path.realpath(path))
    return not is_protected(real) and real.is_relative_to(root)


# ---------------------------------------------------------------------------
# GlobTool
# ---------------------------------------------------------------------------


@tool(name="glob", description="Find files matching a glob pattern", read_only=True, permission="allow")
class GlobTool:
    """Find files using pathlib.Path.glob, sorted by mtime (newest first)."""

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern (e.g. '**/*.py')"},
            "path": {"type": "string", "description": "Directory to search in", "default": "."},
            "limit": {"type": "integer", "description": "Max results", "default": 200},
        },
        "required": ["pattern"],
    }

    def is_concurrency_safe(self, params: dict[str, Any]) -> bool:  # noqa: D401
        # Only a call that needs no prompt may run in a parallel batch (prompts from
        # several threads at once would interleave on the terminal).
        return _needs_no_prompt(params)

    def check_permissions(self, params: dict[str, Any], context: ToolContext) -> PermissionResult:
        return read_permission(str(params.get("path") or context.cwd), context)

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        pattern: str = params["pattern"]
        search_path = _search_root(params, context)
        limit: int = int(params.get("limit", 200))

        if is_protected(search_path):
            return ToolResult(content=f"Access denied: {search_path} is a protected path.", is_error=True)
        # The pattern must not climb out of the approved root ("../../.ssh/*").
        if ".." in Path(pattern).parts:
            return ToolResult(content="Error: '..' is not allowed in a glob pattern; set path instead.", is_error=True)

        try:
            # A dir symlink in the tree must not list a secret dir or an outside dir.
            matches = [m for m in search_path.glob(pattern) if _readable(m, search_path)]
        except (ValueError, OSError, NotImplementedError) as exc:
            return ToolResult(content=f"Error: {exc}", is_error=True)

        if not matches:
            return ToolResult(content="No files found")

        # Sort by mtime descending (newest first)
        matches.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        matches = matches[:limit]

        content = "\n".join(str(p) for p in matches)
        return ToolResult(content=content)


# ---------------------------------------------------------------------------
# GrepTool
# ---------------------------------------------------------------------------


@tool(name="grep", description="Search file contents using regex", read_only=True, permission="allow")
class GrepTool:
    """Search file contents.

    Tries ripgrep (rg) for speed; falls back to a pure-Python re + os.walk
    implementation when rg is unavailable.
    """

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search"},
            "path": {"type": "string", "description": "File or directory to search", "default": "."},
            "limit": {"type": "integer", "description": "Max results", "default": 100},
        },
        "required": ["pattern"],
    }

    def is_concurrency_safe(self, params: dict[str, Any]) -> bool:  # noqa: D401
        # Only a call that needs no prompt may run in a parallel batch (prompts from
        # several threads at once would interleave on the terminal).
        return _needs_no_prompt(params)

    def check_permissions(self, params: dict[str, Any], context: ToolContext) -> PermissionResult:
        return read_permission(str(params.get("path") or context.cwd), context)

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        pattern: str = params["pattern"]
        search_path = _search_root(params, context)
        limit: int = int(params.get("limit", 100))

        if is_protected(search_path):
            return ToolResult(content=f"Access denied: {search_path} is a protected path.", is_error=True)

        matches = self._run_rg(pattern, search_path, limit)
        if matches is None:
            try:
                matches = self._run_python(pattern, search_path, limit)
            except re.error as exc:
                return ToolResult(content=f"Error: invalid regex {pattern!r}: {exc}", is_error=True)

        if not matches:
            return ToolResult(content="No matches found")

        content = "\n".join(matches[:limit])
        return ToolResult(content=content)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_rg(self, pattern: str, path: Path, limit: int) -> list[str] | None:
        """Run ripgrep and return lines, or None if rg is not available.

        The model's pattern goes in after ``-e`` and the path after ``--``, so
        neither can ever be parsed as an rg option (``--pre=CMD`` runs CMD);
        ``--no-config`` keeps RIPGREP_CONFIG_PATH out too. ``--null`` puts a NUL
        after each file name so every hit is filtered through the path guard.
        """
        try:
            proc = subprocess.run(
                ["rg", "--no-config", "--no-heading", "--with-filename", "--null", "-n",
                 "-e", pattern, "--", str(path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode not in (0, 1):
                # rg not found or hard error — fall back
                return None
            lines: list[str] = []
            for ln in proc.stdout.splitlines():
                fname, sep, rest = ln.partition("\0")
                if sep and _readable(fname, path):
                    lines.append(f"{fname}:{rest}")
                    if len(lines) >= limit:
                        break
            return lines
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None

    def _run_python(self, pattern: str, path: Path, limit: int) -> list[str]:
        """Pure-Python fallback: re.search over every regular file (an invalid regex raises
        re.error; a FIFO or socket is skipped, since opening one blocks)."""
        compiled = re.compile(pattern)

        results: list[str] = []
        targets: list[Path] = [path] if path.is_file() else []

        if not targets:
            for root, _dirs, files in os.walk(path):
                for fname in files:
                    targets.append(Path(root) / fname)
                    if len(targets) > 10_000:
                        break
                if len(targets) > 10_000:
                    break

        for fpath in targets:
            if len(results) >= limit:
                break
            if not _readable(fpath, path) or not fpath.is_file():
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if compiled.search(line):
                    results.append(f"{fpath}:{lineno}:{line}")
                    if len(results) >= limit:
                        break

        return results
