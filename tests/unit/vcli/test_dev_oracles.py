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


@pytest.mark.parametrize("parent", ["build", "dist", "venv", "node_modules", ".venv"])
def test_skip_dirs_are_judged_relative_to_the_walk_root(tmp_path: Path, monkeypatch, parent: str) -> None:
    """A project that itself lives under build/ (or dist/, venv/, ...) is still walked."""
    proj = tmp_path / parent / "proj"
    (proj / "src").mkdir(parents=True)
    (proj / "src" / "a.py").write_text("TODO\n")
    (proj / "build").mkdir()
    (proj / "build" / "gen.py").write_text("TODO\n")   # a skip dir INSIDE the project
    monkeypatch.chdir(proj)
    assert grep_count("TODO", ".") == 1
    assert _verify("grep_count('TODO') == 0") is False
    (proj / ".env").write_text("OLD_KEY=abc\n")
    assert _verify("grep_count('OLD_KEY', '.') == 0") is False   # the walk meets .env


def test_the_walk_raises_at_its_file_cap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dev, "_MAX_FILES", 3)
    for i in range(3):
        (tmp_path / f"f{i}.py").write_text("x\n")
    monkeypatch.chdir(tmp_path)
    assert grep_count("x", ".") == 3                 # at the cap: still a full count
    (tmp_path / "f3.py").write_text("x\n")
    with pytest.raises(OSError, match="more than 3 files"):
        grep_count("x", ".")
    assert _verify("grep_count('x') < 4") is False   # no partial count


@pytest.mark.parametrize("expr", [
    "grep_count('[', 'a.py') == 0",
    "grep_count('(', '.') == 0",
    "not grep_count('[unclosed', 'a.py')",
])
def test_an_invalid_regex_verifies_false_in_the_negated_form(tmp_path: Path, monkeypatch, expr: str) -> None:
    (tmp_path / "a.py").write_text("x = [1]\n")
    monkeypatch.chdir(tmp_path)
    assert _verify(expr) is False


@pytest.mark.parametrize("expr", [
    "grep_count('needle', 'big.txt') == 0",
    "grep_count('needle', '.') == 0",
    "not path_contains('big.txt', 'needle')",
])
def test_an_oversized_file_verifies_false_in_the_negated_form(tmp_path: Path, monkeypatch, expr: str) -> None:
    monkeypatch.setattr(dev, "_MAX_BYTES_PER_FILE", 100)
    (tmp_path / "big.txt").write_text("x" * 200 + "needle\n")   # the match lies past the cap
    monkeypatch.chdir(tmp_path)
    assert _verify(expr) is False


@pytest.mark.parametrize("expr", [
    "grep_count('x', 'locked.txt') == 0",
    "grep_count('x', '.') == 0",
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
    assert _verify("grep_count('def ', '.') == 2")
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


@pytest.mark.parametrize("rel", ["proj", "proj/src"])
def test_a_symlinked_directory_fails_the_walk(tmp_path: Path, monkeypatch, rel: str) -> None:
    """os.walk lists a dir symlink but never descends: a silently partial count."""
    proj = tmp_path / "proj"
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "x.py").write_text("TODO\n")
    (proj / "src").mkdir(parents=True)
    (proj / "build" / "gen").mkdir(parents=True)
    (proj / "build" / "gen" / "y.py").write_text("TODO\n")
    (proj / "lib").symlink_to(tmp_path / "shared")                 # outside the project
    (proj / "src" / "gen").symlink_to(proj / "build" / "gen")     # into a skipped dir
    monkeypatch.chdir(proj)
    where = "." if rel == "proj" else "src"
    assert _verify(f"grep_count('TODO', '{where}') == 0") is False
    with pytest.raises(OSError, match="symlinked directory"):
        grep_count("TODO", where)


def test_an_unlistable_directory_fails_the_walk(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.py").write_text("TODO\n")
    monkeypatch.chdir(tmp_path)
    real_scandir = os.scandir

    def refuse(path):  # root ignores chmod, so fail the listing itself
        if os.fspath(path).endswith("sub"):
            raise PermissionError(13, "Permission denied", os.fspath(path))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", refuse)
    assert _verify("grep_count('TODO') == 0") is False


@pytest.mark.parametrize("name", [".git", ".GIT"])
def test_a_git_pointer_file_does_not_refuse_the_dot_walk(tmp_path: Path, monkeypatch, name: str) -> None:
    """A worktree's or submodule's .git is a FILE ('gitdir: ...'), not project content."""
    (tmp_path / name).write_text("gitdir: /elsewhere/.git/worktrees/x\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / ".git").write_text("gitdir: ../.git/modules/sub\n")
    (tmp_path / "sub" / "a.py").write_text("TODO\n")
    monkeypatch.chdir(tmp_path)
    assert _verify("grep_count('TODO') == 1")
    assert _verify("grep_count('gitdir', '.git') == 1") is False   # named directly: still refused


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
    assert _verify("grep_count('S') == 0") is False


def test_root_as_cwd_is_no_probe_of_the_process_environment(monkeypatch) -> None:
    """Started in /, a per-character probe of /proc/self/environ (where an LLM key lives)."""
    monkeypatch.chdir("/")
    for expr in ("path_contains('proc/self/environ', 'PATH=')",
                 "not path_contains('proc/self/environ', 'PATH=#no-such-value#')",
                 "grep_count('PATH=', 'proc/self/environ') > 0",
                 "grep_count('PATH=#no-such-value#', 'proc/self/environ') == 0"):
        assert _verify(expr) is False, expr


def test_the_model_is_told_the_dot_walk_is_refused_over_a_secret() -> None:
    sig = dev.DEV_VOCAB.verify_fn_signatures["grep_count"]
    assert "secret" in sig and "subdirectory" in sig and "symlinked" in sig
    assert "not file_exists" in dev.DEV_VOCAB.verify_fn_signatures["file_exists"]


def test_the_oracle_error_reaches_the_step_failure_reason(tmp_path: Path, monkeypatch) -> None:
    from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
    from vector_os_nano.vcli.cognitive.strategy_selector import StrategyResult
    from vector_os_nano.vcli.cognitive.types import SubGoal

    (tmp_path / ".env").write_text("K=v\n")
    monkeypatch.chdir(tmp_path)

    class _Noop:
        def select(self, sub_goal: SubGoal) -> StrategyResult:
            return StrategyResult("primitive", "noop", {})

    ex = GoalExecutor(_Noop(), GoalVerifier(dev_verify_namespace()),
                      primitives={"noop": lambda: True})

    def run(verify: str) -> str:
        sg = SubGoal(name="s", description="d", verify=verify)
        step = ex._execute_sub_goal(sg)
        assert step.success is False and step.failure_class == "verify_fail"
        return step.error

    reason = run("grep_count('K', '.') == 0")
    assert reason.startswith("verification failed: PermissionError") and ".env" in reason
    assert run("1 == 2") == "verification failed"      # a plain False carries no stale error
