# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The dev-world verify oracles fail closed (R3).

A model-authored predicate runs with no prompt and its truth grades the step. An
oracle that cannot read what it was asked about must RAISE (GoalVerifier turns that
into False), never answer 0/False: a negated form ('== 0', 'not ...') would then
verify True although the task never happened.
"""
from __future__ import annotations

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
    assert _verify("grep_count('x', 'missing.py') == 0")


def test_the_model_is_told_the_dot_walk_is_refused_over_a_secret() -> None:
    sig = dev.DEV_VOCAB.verify_fn_signatures["grep_count"]
    assert "secret" in sig and "subdirectory" in sig


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
