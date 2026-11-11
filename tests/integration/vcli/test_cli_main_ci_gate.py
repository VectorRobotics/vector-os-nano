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

2. REPL UNREGRESSED — the REPL and ``-p`` share their setup. A smoke test runs
   ``main`` with a stubbed prompt session feeding one input then EOF, and asserts
   the REPL starts, runs the turn through the native producer, and exits cleanly.
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
    """The REPL path (no -p) runs a turn through the native producer and exits cleanly.

    Drives the REAL ``main()`` with a fake PromptSession that yields one input then
    EOF (clean exit). The VECTOR_FAKE_LLM_TOOLS seam scripts the model: write a
    marker, verify it, finish. The written marker and the GROUNDED trace prove the
    interactive loop reached the producer.
    """
    import threading

    from vector_os_nano.vcli import cli
    from vector_os_nano.vcli.engine import VectorEngine

    # Run the REPL IN tmp_path so the dev-world file oracle (cwd-scoped) reads the
    # marker the actor writes; isolate HOME (sessions, history).
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    marker = "repl_marker.txt"
    script = {"turns": [
        {"tool_calls": [{"name": "file_write", "input": {"file_path": marker, "content": "ok\n"}}]},
        {"tool_calls": [{"name": "verify", "input": {"expr": f"path_contains({marker!r}, 'ok')"}}]},
        {"tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]}
    script_file = tmp_path / "script.json"
    script_file.write_text(json.dumps(script), encoding="utf-8")
    monkeypatch.setenv("VECTOR_FAKE_LLM_TOOLS", str(script_file))
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

    # The REPL runs the turn on its worker thread (cli._run_turn_busy); capture the
    # trace by spying on the producer.
    executed: dict[str, object] = {}
    done = threading.Event()
    real_native = VectorEngine.run_turn_native

    def _spy_native(self, *a, **k):  # noqa: ANN001 ANN002 ANN003
        trace = real_native(self, *a, **k)
        executed["trace"] = trace
        done.set()
        return trace

    monkeypatch.setattr(VectorEngine, "run_turn_native", _spy_native)

    cli.main(["--no-permission"])

    assert done.wait(timeout=15.0), "REPL did not run the native turn (loop regressed?)"
    trace = executed["trace"]
    assert trace.goal_tree.goal == "write the marker then verify it"
    assert [s.strategy for s in trace.steps] == ["file_write"] and trace.steps[0].verify_result
    assert (tmp_path / marker).read_text() == "ok\n"


def test_cli_help_exits_zero() -> None:
    """main(['--help']) prints usage and exits with code 0 (moved from test_end_to_end)."""
    from vector_os_nano.vcli.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
