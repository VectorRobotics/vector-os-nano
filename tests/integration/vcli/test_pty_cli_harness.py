# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""R2a PART C — acceptance through the REAL cli.main, driven by a stdlib PTY.

These are the project's FIRST honest capability tests: each spawns the actual
``python -m vector_os_nano.vcli.cli -p <prompt> --json`` entrypoint under a PTY
(``tests/harness/pty_cli.run_cli_turn``), reads the machine ``VECTOR_VERDICT``
line, and asserts the verdict + exit code. No ``~/sandbox`` bypass, no engine
poking — the product is the system under test.

Cases (RED -> GREEN):
  * NO-OP / staged 'done' (verify='True')  -> verified False / exit 2 (RAN)
  * GROUNDED (a file_write a real oracle reads back)
                                           -> verified True  / exit 0
  * two different prompts                  -> different VerdictReport.goal
                                              (no stale reuse across runs)
  * empty-oracle / non-oracle verify       -> fail-closed (verified False)
  * a chat turn (no action, no verify)     -> exit 0, acted false, no claim

All runs are deterministic via the ``VECTOR_FAKE_LLM_TOOLS`` seam (the model's tool
calls are scripted, no live LLM); the dev world is the default (no MuJoCo). Marked
``cli_main`` + ``capability`` so the CI gate counts them.
"""
from __future__ import annotations

import uuid

import pytest

from tests.harness.pty_cli import run_cli_turn


def _unique(name: str) -> str:
    """A per-run unique filename so concurrent/repeat runs never collide."""
    return f"{name}_{uuid.uuid4().hex[:8]}.txt"


# ---------------------------------------------------------------------------
# NO-OP / staged 'done' — verify='True' -> RAN -> verified False / exit 2
# ---------------------------------------------------------------------------


def _write_then_verify(fname: str, content: str, verify: str) -> dict:
    return {"turns": [
        {"tool_calls": [{"name": "file_write", "input": {"file_path": fname, "content": content}}]},
        {"tool_calls": [{"name": "verify", "input": {"expr": verify}}]},
        {"tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]}


@pytest.mark.cli_main
@pytest.mark.capability
def test_noop_staged_done_is_not_verified(tmp_path) -> None:
    fname = _unique("noop")
    script = _write_then_verify(fname, "x\n", "True")  # sentinel — no grounded evidence
    result = run_cli_turn("stage a done with no evidence then finish", tool_script=script, cwd=tmp_path)
    assert result.verified is False
    assert result.exit_code == 2
    assert result.evidence == "RAN"
    assert result.verdict["acted"] is True


# ---------------------------------------------------------------------------
# GROUNDED — the actor really writes; a real oracle reads real post-state
#            -> GROUNDED -> verified True / exit 0
# ---------------------------------------------------------------------------


@pytest.mark.cli_main
@pytest.mark.capability
def test_grounded_turn_is_verified(tmp_path) -> None:
    # The dev-world file oracle (`path_contains`) only reads WITHIN cwd, so run
    # the child IN tmp_path and use a relative filename.
    fname = _unique("marker")
    script = _write_then_verify(fname, "ready\n", f"path_contains({fname!r}, 'ready')")
    result = run_cli_turn("create the marker file then verify it", tool_script=script, cwd=tmp_path)
    assert result.verified is True
    assert result.exit_code == 0
    assert result.evidence == "GROUNDED"
    assert result.verdict["n_grounded"] == 1
    # The actor really wrote the file (the oracle did not lie).
    assert (tmp_path / fname).exists()


# ---------------------------------------------------------------------------
# Two different prompts -> different goal (no stale verdict reuse)
# ---------------------------------------------------------------------------


@pytest.mark.cli_main
@pytest.mark.capability
def test_two_prompts_yield_distinct_goals(tmp_path) -> None:
    f1, f2 = _unique("alpha"), _unique("beta")
    ra = run_cli_turn("GOAL_ALPHA write the alpha file then verify it",
                      tool_script=_write_then_verify(f1, "A\n", f"path_contains({f1!r}, 'A')"), cwd=tmp_path)
    rb = run_cli_turn("GOAL_BETA write the beta file then verify it",
                      tool_script=_write_then_verify(f2, "B\n", f"path_contains({f2!r}, 'B')"), cwd=tmp_path)
    assert ra.goal == "GOAL_ALPHA write the alpha file then verify it"
    assert rb.goal == "GOAL_BETA write the beta file then verify it"
    assert ra.verified is True and rb.verified is True


# ---------------------------------------------------------------------------
# Empty / non-oracle verify -> fail-closed (a phantom predicate cannot ground)
# ---------------------------------------------------------------------------


@pytest.mark.cli_main
@pytest.mark.capability
def test_non_oracle_verify_fails_closed(tmp_path) -> None:
    # A verify that calls a function NOT in the dev oracle set classifies RAN — an
    # absent oracle can only make the gate stricter. The file is written, but the
    # verdict is NOT verified.
    fname = _unique("phantom")
    script = _write_then_verify(fname, "z\n", "totally_made_up_oracle() == 1")
    result = run_cli_turn("verify with a phantom predicate then finish", tool_script=script, cwd=tmp_path)
    assert result.verified is False
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# A chat turn: the model answers, nothing acts or claims -> exit 0, acted false
# ---------------------------------------------------------------------------


@pytest.mark.cli_main
@pytest.mark.capability
def test_chat_turn_exits_0_with_acted_false(tmp_path) -> None:
    script = {"turns": [{"text": "我是 V，一个机器人运行时。", "tool_calls": [], "stop_reason": "end_turn"}]}
    result = run_cli_turn("你是谁？", tool_script=script, cwd=tmp_path)
    assert result.exit_code == 0
    assert result.verdict["acted"] is False and result.verified is False
    assert result.evidence == "NO_TRACE" and result.verdict["error"] == ""
    assert "我是 V" in result.raw_output
