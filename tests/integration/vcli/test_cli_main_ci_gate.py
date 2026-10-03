# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""R2a CI gate + REPL-unregressed smoke tests.

Two responsibilities:

1. GATE GUARD — assert ``pytest -m cli_main`` selects >= 1 test, so the acceptance
   instrument can never silently disappear (a future refactor that drops every
   ``cli_main`` test would fail this guard). The fail-closed
   ``pytest_collection_modifyitems`` gate (a ``capability`` test MUST carry
   ``cli_main``) lives in ``tests/conftest.py``; this guard proves the marker is
   actually in use.

2. REPL UNREGRESSED — the ``run_one_turn`` / ``-p`` refactor extracted shared
   setup but the interactive REPL path must stay byte-identical. A smoke test runs
   ``main`` with a stubbed prompt session feeding one input then EOF, and asserts
   the REPL starts, decomposes+executes the turn, and exits cleanly — i.e. the
   refactor did not break the loop.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# 1. GATE GUARD — `pytest -m cli_main` selects >= 1
# ---------------------------------------------------------------------------


def test_cli_main_marker_selects_real_pty_runs() -> None:
    """`-m cli_main` selects >= 1 test that really SPAWNS cli.main on a PTY.

    Regression (tests:self-satisfying-gate-guard): the guard used to count its own
    file's cli_main tests, so it stayed green with every real PTY test deleted.
    Now a selected test only counts if its file drives run_cli_turn /
    run_repl_session, and this guard file never counts.
    """
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", "tests/vcli", "tests/integration/vcli",
            "tests/e2e_repl", "-m", "cli_main", "--co", "-q", "-p", "no:cacheprovider",
        ],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"collection failed:\n{proc.stdout}\n{proc.stderr}"
    files = {ln.split("::", 1)[0] for ln in proc.stdout.splitlines() if "::test_" in ln}
    spawning = sorted(
        f for f in files
        if Path(f).name != Path(__file__).name
        and re.search(r"run_(cli_turn|repl_session)\(", (_REPO_ROOT / f).read_text(encoding="utf-8"))
    )
    assert spawning, f"no cli_main test spawns cli.main on a PTY; selected files: {sorted(files)}"


# ---------------------------------------------------------------------------
# 2. REPL UNREGRESSED — main()'s interactive loop still runs a turn end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.cli_main
def test_repl_runs_one_turn_then_exits(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The REPL path (no -p) still decomposes+executes a turn and exits cleanly.

    Drives the REAL ``main()`` with a fake PromptSession that yields one input
    then EOF (clean exit). Uses the VECTOR_FAKE_LLM seam for a deterministic plan.
    Asserts the engine's VGG layer executed the turn (a trace was produced) — i.e.
    the run_one_turn refactor did NOT regress the interactive loop.
    """
    from vector_os_nano.vcli import cli

    # Run the REPL IN tmp_path so the dev-world file oracle (cwd-scoped) reads the
    # marker the actor writes. Isolate HOME so the persistent dev template tier
    # (~/.vector) is per-test — a compiled template from another run must not
    # short-circuit this decompose with a stale plan.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    marker = "repl_marker.txt"
    plan = {
        "goal": "repl turn writes a marker",
        "sub_goals": [{
            "name": "w", "description": "write marker",
            "verify": f"path_contains({marker!r}, 'ok')",
            "strategy": "tool_call",
            "strategy_params": {"tool": "file_write", "args": {"file_path": marker, "content": "ok\n"}},
        }],
    }
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setenv("VECTOR_FAKE_LLM", str(plan_file))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")

    # One input then EOF — the REPL reads it, runs the turn, then breaks on EOF.
    inputs = iter(["write the marker then verify it"])

    class _FakePromptSession:
        def __init__(self, *a, **k) -> None:
            pass

        def prompt(self, *a, **k):  # noqa: ANN002 ANN003
            try:
                return next(inputs)
            except StopIteration:
                raise EOFError  # clean REPL exit

    monkeypatch.setattr(cli, "PromptSession", _FakePromptSession)
    # Silence the startup banner sleeps (print_banner uses time.sleep).
    monkeypatch.setattr(cli.time, "sleep", lambda *_a, **_k: None)

    # The REPL runs the turn on a background daemon thread via vgg_execute_async;
    # capture the completed trace via the on_complete callback by spying on the
    # SYNC core (vgg_execute) the async path calls. A threading.Event lets us wait
    # for the background completion deterministically.
    import threading

    from vector_os_nano.vcli.engine import VectorEngine

    executed: dict[str, object] = {}
    done = threading.Event()
    real_execute = VectorEngine.vgg_execute

    def _spy_execute(self, goal_tree):  # noqa: ANN001
        trace = real_execute(self, goal_tree)
        executed["trace"] = trace
        done.set()
        return trace

    monkeypatch.setattr(VectorEngine, "vgg_execute", _spy_execute)

    cli.main(["--no-permission"])

    # The REPL must have decomposed + executed the turn (a trace was produced),
    # proving the interactive loop is unregressed by the run_one_turn refactor.
    assert done.wait(timeout=15.0), "REPL did not execute the VGG turn (loop regressed?)"
    assert executed["trace"].goal_tree.goal == "repl turn writes a marker"
