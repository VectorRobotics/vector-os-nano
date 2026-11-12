# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The dev-world verify oracles fail closed (R3).

A model-authored predicate runs with no prompt and its truth grades the step. An
oracle that cannot read what it was asked about must RAISE (GoalVerifier turns that
into False), never answer 0/False: a negated form ('== 0', 'not ...') would then
verify True although the task never happened.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.worlds import dev
from vector_os_nano.vcli.worlds.dev import dev_verify_namespace, grep_count


def _verify(expr: str) -> bool:
    return GoalVerifier(dev_verify_namespace()).verify(expr)


def _layout_walk_attacks(proj: Path) -> None:
    """The trees that bypassed the old directory walk in earlier rounds; the only TODO
    sits where a walk skipped, stopped or followed nothing."""
    for rel in ("build/gen.py", "src/build/steps.py", "node_modules/m.js", ".venv/lib.py",
                "__pycache__/c.py", "sub/a.py", "many/f0.py"):
        (proj / rel).parent.mkdir(parents=True, exist_ok=True)
        (proj / rel).write_text("TODO\n")
    (proj / "lib").symlink_to(proj / "build")               # a symlinked directory
    (proj / ".git").write_text("gitdir: /elsewhere/.git/worktrees/x\n")   # a .git pointer file
    (proj / "sub" / ".git").write_text("gitdir: ../.git/modules/sub\n")
    for i in range(1, 2001):                                # more than 2000 files
        (proj / "many" / f"f{i}.py").write_text("x\n")


@pytest.mark.parametrize("d", [".", "build", "src", "src/build", "lib", "sub", "many", "node_modules"])
def test_a_directory_verifies_false_in_both_polarities(tmp_path: Path, monkeypatch, d: str) -> None:
    """The oracles read ONE regular file: a directory raises, so neither a count nor its
    negation can ground (a walk that skipped build/, a source package named build, a
    symlinked dir or a .git pointer, or stopped at a file cap, grounded '== 0' falsely)."""
    proj = tmp_path / "build" / "proj"            # a project that itself lives under build/
    proj.mkdir(parents=True)
    _layout_walk_attacks(proj)
    monkeypatch.chdir(proj)
    for expr in (f"grep_count('TODO', {d!r}) == 0", f"grep_count('TODO', {d!r}) > 0",
                 f"path_contains({d!r}, 'TODO')", f"not path_contains({d!r}, 'TODO')"):
        assert _verify(expr) is False, expr
    with pytest.raises(OSError, match="one regular file per call"):
        grep_count("TODO", d)


def test_naming_the_files_still_grounds(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "proj").mkdir()
    _layout_walk_attacks(tmp_path / "proj")
    monkeypatch.chdir(tmp_path / "proj")
    assert _verify("grep_count('TODO', 'src/build/steps.py') == 1 and grep_count('TODO', 'build/gen.py') == 1")
    assert _verify("grep_count('TODO', 'src/build/steps.py') == 0") is False
    assert _verify("path_contains('.git', 'gitdir')") is False      # a .git pointer: still protected


def test_grep_count_has_no_default_path(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.py").write_text("x\n")
    monkeypatch.chdir(tmp_path)
    assert _verify("grep_count('TODO') == 0") is False
    with pytest.raises(TypeError):
        grep_count("TODO")  # type: ignore[call-arg]


@pytest.mark.parametrize("expr", [
    "grep_count('[', 'a.py') == 0",
    "grep_count('(', 'a.py') == 0",
    "not grep_count('[unclosed', 'a.py')",
])
def test_an_invalid_regex_verifies_false_in_the_negated_form(tmp_path: Path, monkeypatch, expr: str) -> None:
    (tmp_path / "a.py").write_text("x = [1]\n")
    monkeypatch.chdir(tmp_path)
    assert _verify(expr) is False


@pytest.mark.parametrize("expr", [
    "grep_count('needle', 'big.txt') == 0",
    "not path_contains('big.txt', 'needle')",
])
def test_an_oversized_file_verifies_false_in_the_negated_form(tmp_path: Path, monkeypatch, expr: str) -> None:
    monkeypatch.setattr(dev, "_MAX_BYTES_PER_FILE", 100)
    (tmp_path / "big.txt").write_text("x" * 200 + "needle\n")   # the match lies past the cap
    monkeypatch.chdir(tmp_path)
    assert _verify(expr) is False


@pytest.mark.parametrize("expr", [
    "grep_count('x', 'locked.txt') == 0",
    "not path_contains('locked.txt', 'x')",
])
def test_an_unreadable_file_verifies_false_in_the_negated_form(tmp_path: Path, monkeypatch, expr: str) -> None:
    (tmp_path / "locked.txt").write_text("x\n")
    monkeypatch.chdir(tmp_path)
    real_open = Path.open

    def refuse(self: Path, *a, **kw):  # root ignores chmod, so fail the open itself
        if self.name == "locked.txt":
            raise PermissionError(13, "Permission denied", str(self))
        return real_open(self, *a, **kw)

    monkeypatch.setattr(Path, "open", refuse)   # Path.read_text opens through it
    assert _verify(expr) is False


def test_ordinary_greps_are_unchanged(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def f():\n    pass\ndef g():\n    pass\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text("def f():\n")
    monkeypatch.chdir(tmp_path)
    assert _verify("grep_count('def ', 'src/a.py') == 2")
    assert _verify("path_contains('src/a.py', 'def g')")
    assert _verify("not file_exists('missing.py')")   # absence is stated with file_exists


@pytest.mark.parametrize("expr", [
    "not path_contains('srcc/main.py', 'TODO')",
    "grep_count('TODO', 'srcc/main.py') == 0",
    "grep_count('TODO', 'srcc') == 0",
    "not path_contains('src', 'TODO')",       # a directory
    "not path_contains('pipe', 'x')",         # a FIFO
    "grep_count('x', 'pipe') == 0",
])
def test_a_negation_over_a_missing_or_non_regular_path_verifies_false(
        tmp_path: Path, monkeypatch, expr: str) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("ok\n")
    os.mkfifo(tmp_path / "pipe")
    monkeypatch.chdir(tmp_path)
    assert _verify(expr) is False


_HOME_FILES = {".npmrc": "npm_SECRET", ".pypirc": "password=pypi_S",
               ".bash_history": "export K=sk_S", ".cache/huggingface/token": "hf_S"}


def test_home_as_cwd_is_no_content_oracle(tmp_path: Path, monkeypatch) -> None:
    """read_permission never auto-allows with $HOME as the root, so neither do the oracles."""
    for rel, text in _HOME_FILES.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for rel, text in _HOME_FILES.items():
        prefix, wrong = text[:-1], text[:-1] + "#"
        for expr in (f"path_contains({rel!r}, {prefix!r})", f"not path_contains({rel!r}, {wrong!r})",
                     f"grep_count({prefix!r}, {rel!r}) == 1", f"grep_count({wrong!r}, {rel!r}) == 0",
                     f"file_exists({rel!r})", f"not file_exists({rel!r} + 'x')"):
            assert _verify(expr) is False, expr
    assert _verify("grep_count('S', '.npmrc') == 0") is False


def test_an_ancestor_of_home_as_cwd_is_no_content_oracle(tmp_path: Path, monkeypatch) -> None:
    """Started in /home with HOME=/home/alice: 'alice/.npmrc' lies under the root."""
    home = tmp_path / "home" / "alice"
    for rel, text in _HOME_FILES.items():
        (home / rel).parent.mkdir(parents=True, exist_ok=True)
        (home / rel).write_text(text)
    monkeypatch.setenv("HOME", str(home))
    for cwd in (home.parent, tmp_path):
        monkeypatch.chdir(cwd)
        base = os.path.relpath(home, cwd)
        for rel, text in _HOME_FILES.items():
            rel = f"{base}/{rel}"
            prefix, wrong = text[:-1], text[:-1] + "#"
            for expr in (f"path_contains({rel!r}, {prefix!r})", f"not path_contains({rel!r}, {wrong!r})",
                         f"grep_count({prefix!r}, {rel!r}) == 1", f"grep_count({wrong!r}, {rel!r}) == 0"):
                assert _verify(expr) is False, (cwd, expr)
        assert _verify("grep_count('S', '.npmrc') == 0") is False


@pytest.mark.parametrize("cwd,rel", [("/", "proc/self/environ"), ("/proc", "self/environ"),
                                     ("/proc/self", "environ")])
def test_root_or_proc_as_cwd_is_no_probe_of_the_process_environment(monkeypatch, cwd, rel) -> None:
    """Started in / or /proc, a per-character probe of /proc/self/environ (where an LLM key lives)."""
    monkeypatch.chdir(cwd)
    for expr in (f"path_contains({rel!r}, 'PATH=')",
                 f"not path_contains({rel!r}, 'PATH=#no-such-value#')",
                 f"grep_count('PATH=', {rel!r}) > 0",
                 f"grep_count('PATH=#no-such-value#', {rel!r}) == 0"):
        assert _verify(expr) is False, expr


def test_the_model_is_told_one_file_per_call() -> None:
    """The verify tool shows each oracle's signature + first docstring paragraph."""
    from vector_os_nano.vcli.native_loop import _oracle_line

    lines = {n: _oracle_line(n, f) for n, f in dev_verify_namespace().items()}
    for name in ("grep_count", "path_contains"):
        assert "One file per call" in lines[name] and "'and' the calls together" in lines[name]
        assert "directory" in lines[name]
    assert lines["grep_count"].startswith("grep_count(pattern, path): ")  # no default path
    assert "not file_exists" in lines["file_exists"]


@pytest.mark.parametrize("expr", [
    "file_exists('.')", "not file_exists('.')",                    # the project root always exists
    "file_exists('pkg')", "not file_exists('pkg')",                # any directory
    "path_contains('a.py', '')", "not path_contains('a.py', '')",  # every file contains ''
    "grep_count('', 'a.py') > 0", "grep_count('', 'a.py') == 0",   # '' matches everywhere
])
def test_a_vacuous_argument_verifies_false_in_both_polarities(tmp_path: Path, monkeypatch, expr: str) -> None:
    """spine:dev-world-trivial-evidence: file_exists('.') and path_contains(f, '') were
    True for any project and any file, so either grounded any dev write."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "a.py").write_text("print(1)\n")
    monkeypatch.chdir(tmp_path)
    assert _verify(expr) is False, expr


def test_the_guards_leave_real_predicates_grounding(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "a.py").write_text("print(1)\n")
    monkeypatch.chdir(tmp_path)
    assert _verify("file_exists('a.py') and not file_exists('b.py')")
    assert _verify("path_contains('a.py', 'print') and grep_count('print', 'a.py') == 1")
