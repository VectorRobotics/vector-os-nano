# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""REPL CUTOVER (2026-06-19) — native-attempt-then-fallback as the REPL's DEFAULT path.

The interactive ``vector-cli`` REPL now ATTEMPTS the native tool-use producer first
for an action-shaped turn, then FALLS BACK to the legacy planner when native took no
action — so bare ``vector-cli`` + natural language exercises the redesign (CLAUDE.md
North Star "Acceptance interface"). These are DETERMINISTIC unit proofs of the three
REPL helpers (no sim, no LLM); the real acceptance is driving the actual REPL by NL.

  - ``_repl_native_enabled`` — default ON; VECTOR_REPL_NATIVE in {0,false,off,no} forces
    the pure-legacy REPL (reversible escape hatch).
  - ``_repl_attempt_native`` — runs on the REAL session; acted trace -> returns True and
    renders the honest verdict; no-action / raised -> returns False (caller falls back
    to legacy) and the native entries are removed again (no session pollution).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from vector_os_nano.vcli import cli
from vector_os_nano.vcli.session import Session
from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused
from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)


# ---------------------------------------------------------------------------
# Minimal trace builders (shaped like native_loop produces).
# ---------------------------------------------------------------------------


def _acted_trace(goal: str, *, strategy: str, verify: str, verified_pose: bool) -> ExecutionTrace:
    sub = SubGoal(name="native_step_0", description="d", verify=verify, strategy=strategy)
    step = StepRecord(
        sub_goal_name="native_step_0",
        strategy=strategy,
        success=True,
        verify_result=verified_pose,
        duration_sec=0.0,
        actor_caused=ActorCaused.CAUSED,
    )
    tree = GoalTree(goal=goal, sub_goals=(sub,))
    return ExecutionTrace(goal_tree=tree, steps=(step,), success=True, total_duration_sec=0.0)


def _noaction_trace(goal: str) -> ExecutionTrace:
    sub = SubGoal(name="native_step_0", description="d", verify="at_position(10.0, 3.0)", strategy="")
    step = StepRecord(
        sub_goal_name="native_step_0",
        strategy="",
        success=True,
        verify_result=True,
        duration_sec=0.0,
        actor_caused=ActorCaused.UNCAUSED,
    )
    tree = GoalTree(goal=goal, sub_goals=(sub,))
    return ExecutionTrace(goal_tree=tree, steps=(step,), success=True, total_duration_sec=0.0)


# ---------------------------------------------------------------------------
# Test doubles — console / session / engine.
# ---------------------------------------------------------------------------


class _FakeStatus:
    def __enter__(self) -> "_FakeStatus":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class _FakeConsole:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def status(self, *a: object, **k: object) -> _FakeStatus:
        return _FakeStatus()

    def print(self, *a: object, **k: object) -> None:
        self.lines.append(" ".join(str(x) for x in a))

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _FakeSession() -> Session:
    """A real, in-memory Session (never saved by these tests)."""
    return Session(session_id="t", created_at="", updated_at="", path=Path("/nonexistent/t.jsonl"))


class _FakeEngine:
    def __init__(self, trace: object, *, answer: str = "") -> None:
        self._vgg_agent = None
        self._trace = trace
        self._answer = answer
        self.native_calls = 0
        self.sessions: list[object] = []

    def run_turn_native(self, user_message, agent=None, session=None, app_state=None, on_progress=None, ask_permission=None):  # noqa: ANN001
        # ``on_progress`` mirrors the REAL VectorEngine.run_turn_native signature —
        # _repl_attempt_native passes it for the live spinner; a fake omitting it made
        # the call raise TypeError (caught as "no action"), masking the cutover path.
        self.native_calls += 1
        self.sessions.append(session)
        session.append_user(user_message)  # like the real loop: the turn is on the session
        if isinstance(self._trace, Exception):
            raise self._trace
        session.append_assistant(self._answer)
        return self._trace


def _stub_oracle(monkeypatch) -> None:
    monkeypatch.setattr(
        "vector_os_nano.vcli.cognitive.trace_store.verify_oracle_names",
        lambda agent, engine: frozenset({"at_position"}),
    )


# ---------------------------------------------------------------------------
# _repl_native_enabled — default ON, env disables.
# ---------------------------------------------------------------------------


def test_repl_native_enabled_default_on(monkeypatch) -> None:
    monkeypatch.delenv("VECTOR_REPL_NATIVE", raising=False)
    assert cli._repl_native_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "off", "no", "False", "OFF", "No"])
def test_repl_native_enabled_disabled(monkeypatch, val: str) -> None:
    monkeypatch.setenv("VECTOR_REPL_NATIVE", val)
    assert cli._repl_native_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "anything"])
def test_repl_native_enabled_on_for_truthy(monkeypatch, val: str) -> None:
    monkeypatch.setenv("VECTOR_REPL_NATIVE", val)
    assert cli._repl_native_enabled() is True


# ---------------------------------------------------------------------------
# _repl_attempt_native — acted -> True on the real session; no-action / raised ->
# False and the session is as before.
# ---------------------------------------------------------------------------


def test_repl_attempt_native_true_and_records_when_acted(monkeypatch) -> None:
    _stub_oracle(monkeypatch)
    trace = _acted_trace("g", strategy="walk", verify="at_position(11.0, 3.0)", verified_pose=True)
    engine = _FakeEngine(trace, answer="到了")
    session = _FakeSession()
    console = _FakeConsole()

    acted = cli._repl_attempt_native(engine, "走到坐标 (11,3)", session, {}, console)

    assert acted is True
    assert engine.native_calls == 1
    # Honest verdict rendered (GROUNDED + CAUSED + at_position True -> verified True).
    assert "verified=True" in console.text
    assert "walk" in console.text and "到了" in console.text
    # cli-entry:native-scratch-session-no-memory: the turn ran on the REAL session.
    assert engine.sessions == [session]
    assert [e.get("content") for e in session._entries if e["type"] == "user"] == ["走到坐标 (11,3)"]


def test_repl_attempt_native_false_and_clean_when_no_action(monkeypatch) -> None:
    _stub_oracle(monkeypatch)
    engine = _FakeEngine(_noaction_trace("g"), answer="你好")
    session = _FakeSession()
    session.append_user("earlier turn")
    console = _FakeConsole()

    acted = cli._repl_attempt_native(engine, "你好", session, {}, console)

    assert acted is False, "verify-only / no-action native trace must fall back to legacy"
    assert engine.native_calls == 1
    # The native entries are gone again: the legacy path records the turn itself.
    assert [e.get("content") for e in session._entries] == ["earlier turn"]


def test_repl_attempt_native_false_when_native_raises(monkeypatch) -> None:
    _stub_oracle(monkeypatch)
    engine = _FakeEngine(RuntimeError("native blew up"))
    session = _FakeSession()
    console = _FakeConsole()

    acted = cli._repl_attempt_native(engine, "走到 (11,3)", session, {}, console)

    assert acted is False, "a raised native attempt must fall back to legacy, not crash the REPL"
    assert session._entries == []


def test_repl_attempt_native_prints_the_lifecycle_result_and_no_empty_verdict(monkeypatch) -> None:
    """A turn that only started a sim: its readiness line shows, no 0/0 verdict claim."""
    _stub_oracle(monkeypatch)
    started = ExecutionTrace(goal_tree=GoalTree(goal="g", sub_goals=()), steps=(), success=False,
                             total_duration_sec=0.0, actions_dispatched=1)

    class _Launch(_FakeEngine):
        def run_turn_native(self, user_message, session=None, **_kw):  # noqa: ANN001
            session.append_user(user_message)
            session.append_assistant("", [{"id": "c1", "name": "start_simulation", "input": {}, "type": "tool_use"}])
            session.append_tool_results([{"type": "tool_result", "tool_use_id": "c1",
                                          "content": "Started go2_piper simulation, ready in 9.1s: ...\nmore"}])
            session.append_assistant("go2 仿真已启动")
            return started

    console = _FakeConsole()
    assert cli._repl_attempt_native(_Launch(started), "启动 go2 仿真", _FakeSession(), {}, console) is True
    assert "Started go2_piper simulation, ready in 9.1s" in console.text
    assert "go2 仿真已启动" in console.text and "verdict" not in console.text


def test_repl_attempt_native_false_verdict_when_verify_fails(monkeypatch) -> None:
    # Native ACTED but the deterministic verify FAILED -> native still owns the turn
    # (acted True) and the rendered verdict is honestly NOT verified.
    _stub_oracle(monkeypatch)
    trace = _acted_trace("g", strategy="walk", verify="at_position(99.0, 99.0)", verified_pose=False)
    engine = _FakeEngine(trace)
    session = _FakeSession()
    console = _FakeConsole()

    acted = cli._repl_attempt_native(engine, "走到 (99,99)", session, {}, console)

    assert acted is True, "native dispatched an action -> it owns the turn"
    assert "verified=False" in console.text, "a failed verify must surface verified=False (honest)"
