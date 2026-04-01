"""Unit tests for vcli search tools (GlobTool, GrepTool) — TDD RED phase.

Covers:
- GlobTool: finds files, respects limit, handles empty results
- GlobTool: is_read_only, is_concurrency_safe, __tool_name__
- GrepTool: finds pattern, supports regex, handles no matches, respects limit
- GrepTool: is_read_only, is_concurrency_safe, __tool_name__
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(tmp_path: Path) -> Any:
    from vector_os_nano.vcli.tools.base import ToolContext

    return ToolContext(
        agent=None,
        cwd=tmp_path,
        session=None,
        permissions=None,
        abort=threading.Event(),
    )


# ---------------------------------------------------------------------------
# GlobTool tests
# ---------------------------------------------------------------------------


class TestGlobTool:
    def test_glob_finds_python_files(self, tmp_path: Path) -> None:
        """glob('*.py') in a dir with .py files returns matches."""
        from vector_os_nano.vcli.tools.search_tools import GlobTool

        (tmp_path / "alpha.py").write_text("# alpha")
        (tmp_path / "beta.py").write_text("# beta")
        (tmp_path / "data.txt").write_text("not python")

        tool = GlobTool()
        ctx = _make_context(tmp_path)
        result = tool.execute({"pattern": "*.py", "path": str(tmp_path)}, ctx)

        assert not result.is_error
        assert "alpha.py" in result.content
        assert "beta.py" in result.content

    def test_glob_respects_limit(self, tmp_path: Path) -> None:
        """limit=2 with 5 matching files returns only 2 results."""
        from vector_os_nano.vcli.tools.search_tools import GlobTool

        for i in range(5):
            (tmp_path / f"file{i}.py").write_text(f"# {i}")

        tool = GlobTool()
        ctx = _make_context(tmp_path)
        result = tool.execute(
            {"pattern": "*.py", "path": str(tmp_path), "limit": 2}, ctx
        )

        assert not result.is_error
        lines = [ln for ln in result.content.splitlines() if ln.strip()]
        assert len(lines) == 2

    def test_glob_empty_result(self, tmp_path: Path) -> None:
        """No matching files returns a 'No files found' message."""
        from vector_os_nano.vcli.tools.search_tools import GlobTool

        tool = GlobTool()
        ctx = _make_context(tmp_path)
        result = tool.execute({"pattern": "*.xyz", "path": str(tmp_path)}, ctx)

        assert not result.is_error
        assert "No files found" in result.content

    def test_glob_is_read_only(self) -> None:
        """GlobTool.is_read_only({}) is True."""
        from vector_os_nano.vcli.tools.search_tools import GlobTool

        assert GlobTool().is_read_only({}) is True

    def test_glob_is_concurrency_safe(self) -> None:
        """GlobTool.is_concurrency_safe({}) is True."""
        from vector_os_nano.vcli.tools.search_tools import GlobTool

        assert GlobTool().is_concurrency_safe({}) is True

    def test_glob_registered(self) -> None:
        """GlobTool class carries __tool_name__ set by @tool decorator."""
        from vector_os_nano.vcli.tools.search_tools import GlobTool

        assert hasattr(GlobTool, "__tool_name__")
        assert GlobTool.__tool_name__ == "glob"


# ---------------------------------------------------------------------------
# GrepTool tests
# ---------------------------------------------------------------------------


class TestGrepTool:
    def test_grep_finds_pattern(self, tmp_path: Path) -> None:
        """Search 'def main' in a dir with a Python file returns the match."""
        from vector_os_nano.vcli.tools.search_tools import GrepTool

        (tmp_path / "main.py").write_text("def main():\n    pass\n")

        tool = GrepTool()
        ctx = _make_context(tmp_path)
        result = tool.execute({"pattern": "def main", "path": str(tmp_path)}, ctx)

        assert not result.is_error
        assert "def main" in result.content

    def test_grep_regex_support(self, tmp_path: Path) -> None:
        r"""Pattern 'def\s+\w+' (regex) matches function definitions."""
        from vector_os_nano.vcli.tools.search_tools import GrepTool

        (tmp_path / "funcs.py").write_text(
            "def  foo():\n    pass\ndef bar():\n    pass\n"
        )

        tool = GrepTool()
        ctx = _make_context(tmp_path)
        result = tool.execute(
            {"pattern": r"def\s+\w+", "path": str(tmp_path)}, ctx
        )

        assert not result.is_error
        assert "foo" in result.content or "bar" in result.content

    def test_grep_no_matches(self, tmp_path: Path) -> None:
        """Pattern not found returns a 'No matches found' message."""
        from vector_os_nano.vcli.tools.search_tools import GrepTool

        (tmp_path / "empty.py").write_text("x = 1\n")

        tool = GrepTool()
        ctx = _make_context(tmp_path)
        result = tool.execute(
            {"pattern": "DEFINITELY_NOT_IN_FILE_XYZ123", "path": str(tmp_path)}, ctx
        )

        assert not result.is_error
        assert "No matches found" in result.content

    def test_grep_respects_limit(self, tmp_path: Path) -> None:
        """limit=2 with many matches returns at most 2 lines."""
        from vector_os_nano.vcli.tools.search_tools import GrepTool

        content = "\n".join(f"match_line_{i} = {i}" for i in range(10))
        (tmp_path / "many.py").write_text(content)

        tool = GrepTool()
        ctx = _make_context(tmp_path)
        result = tool.execute(
            {"pattern": "match_line", "path": str(tmp_path), "limit": 2}, ctx
        )

        assert not result.is_error
        lines = [ln for ln in result.content.splitlines() if ln.strip()]
        assert len(lines) == 2

    def test_grep_is_read_only(self) -> None:
        """GrepTool.is_read_only({}) is True."""
        from vector_os_nano.vcli.tools.search_tools import GrepTool

        assert GrepTool().is_read_only({}) is True

    def test_grep_is_concurrency_safe(self) -> None:
        """GrepTool.is_concurrency_safe({}) is True."""
        from vector_os_nano.vcli.tools.search_tools import GrepTool

        assert GrepTool().is_concurrency_safe({}) is True

    def test_grep_registered(self) -> None:
        """GrepTool class carries __tool_name__ set by @tool decorator."""
        from vector_os_nano.vcli.tools.search_tools import GrepTool

        assert hasattr(GrepTool, "__tool_name__")
        assert GrepTool.__tool_name__ == "grep"
