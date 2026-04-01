"""vcli search tools — GlobTool and GrepTool for the agentic harness.

Both tools are read-only and concurrency-safe.

GlobTool: find files matching a glob pattern using pathlib, sorted by mtime (newest first).
GrepTool: search file contents via ripgrep (rg) with Python re fallback.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

from vector_os_nano.vcli.tools.base import ToolContext, ToolResult, tool


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
        return True

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        pattern: str = params["pattern"]
        search_path = Path(params.get("path", str(context.cwd)))
        limit: int = int(params.get("limit", 200))

        try:
            matches = list(search_path.glob(pattern))
        except (ValueError, OSError) as exc:
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
        return True

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        pattern: str = params["pattern"]
        search_path = Path(params.get("path", str(context.cwd)))
        limit: int = int(params.get("limit", 100))

        matches = self._run_rg(pattern, search_path, limit)
        if matches is None:
            matches = self._run_python(pattern, search_path, limit)

        if not matches:
            return ToolResult(content="No matches found")

        content = "\n".join(matches[:limit])
        return ToolResult(content=content)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_rg(self, pattern: str, path: Path, limit: int) -> list[str] | None:
        """Run ripgrep and return lines, or None if rg is not available."""
        try:
            proc = subprocess.run(
                ["rg", "--no-heading", "-n", pattern, str(path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode not in (0, 1):
                # rg not found or hard error — fall back
                return None
            lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
            return lines[:limit]
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None

    def _run_python(self, pattern: str, path: Path, limit: int) -> list[str]:
        """Pure-Python fallback: re.search over every text file."""
        try:
            compiled = re.compile(pattern)
        except re.error:
            return []

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
