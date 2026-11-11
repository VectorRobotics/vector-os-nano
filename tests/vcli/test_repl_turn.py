# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The REPL turn (``cli._repl_turn``): every turn is the native producer's, rendered.

Deterministic unit proofs (no sim, no LLM; the bare-REPL PTY tests drive the real
thing). Moved and rewritten from tests/vcli/test_repl_native_cutover.py when the
legacy fallback was deleted:

  - a turn that claimed something (acted, verified, refused, stopped, lost its LLM)
    renders the model's words, its steps and the honest verdict;
  - a chat turn renders the model's words only: no step line, no verdict claim, and
    the turn stays on the session (nothing re-plans it, nothing is removed);
  - a native turn that raises is not swallowed (the REPL's busy loop prints it).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from vector_os_nano.vcli import cli
from vector_os_nano.vcli.runtime import Runtime
from vector_os_nano.vcli.session import Session
from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused
from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)


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


def _rt(engine: object, session: Session | None = None) -> Runtime:
    return Runtime(engine=engine, session=session or _FakeSession(), app_state={})


class _FakeEngine:
    def __init__(self, trace: object, *, answer: str = "") -> None:
        self._agent = None
        self._trace = trace
        self._answer = answer
        self.native_calls = 0
        self.sessions: list[object] = []

    def run_turn_native(self, user_message, agent=None, session=None, app_state=None, on_progress=None, ask_permission=None):  # noqa: ANN001
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


def _chat_trace(goal: str) -> ExecutionTrace:
    return ExecutionTrace(goal_tree=GoalTree(goal=goal, sub_goals=()), steps=(), success=False,
                          total_duration_sec=0.0)


def test_acted_turn_renders_the_verdict_on_the_real_session(monkeypatch) -> None:
    _stub_oracle(monkeypatch)
    trace = _acted_trace("g", strategy="walk", verify="at_position(11.0, 3.0)", verified_pose=True)
    engine = _FakeEngine(trace, answer="到了")
    session = _FakeSession()
    console = _FakeConsole()

    cli._repl_turn(_rt(engine, session), "走到坐标 (11,3)", console)

    assert engine.native_calls == 1
    assert "verified=True" in console.text
    assert "walk" in console.text and "到了" in console.text
    # cli-entry:native-scratch-session-no-memory: the turn ran on the REAL session.
    assert engine.sessions == [session]
    assert [e.get("content") for e in session._entries if e["type"] == "user"] == ["走到坐标 (11,3)"]


def test_chat_turn_renders_the_models_text_and_no_claim(monkeypatch) -> None:
    """legacy-cognitive:native-decline-falls-to-legacy-producer: a turn without an
    action is the model's answer, kept on the session; nothing else runs for it."""
    _stub_oracle(monkeypatch)
    engine = _FakeEngine(_chat_trace("g"), answer="I see no green bottle; explore first.")
    session = _FakeSession()
    session.append_user("earlier turn")
    console = _FakeConsole()

    cli._repl_turn(_rt(engine, session), "拿绿色的瓶子", console)

    assert engine.native_calls == 1
    assert "I see no green bottle; explore first." in console.text
    assert "verdict" not in console.text and "▸" not in console.text
    assert [e.get("content") or e.get("text") for e in session._entries] == [
        "earlier turn", "拿绿色的瓶子", "I see no green bottle; explore first."]
    assert cli._last_response == "I see no green bottle; explore first."


def test_a_task_sentence_the_model_only_answered_is_a_goal_not_met(monkeypatch) -> None:
    """A sentence that compiled a task contract is a task, never a chat: the model's
    words alone show the verdict with the goal not met (charter 3.5)."""
    _stub_oracle(monkeypatch)
    goal = "走到坐标 (3.0, 7.5)"
    trace = ExecutionTrace(goal_tree=GoalTree(goal=goal, sub_goals=()), steps=(), success=False,
                           total_duration_sec=0.0, contract=("at_position(3.0, 7.5)",),
                           contract_start=(False,), contract_end=(False,))
    console = _FakeConsole()
    cli._repl_turn(_rt(_FakeEngine(trace, answer="好的，我这就过去。")), goal, console)
    assert "好的，我这就过去。" in console.text
    assert "verified=False" in console.text and "goal=failed" in console.text
    assert "goal not met:" in console.text


def test_a_verify_without_an_action_is_a_claim_and_grades_honestly(monkeypatch) -> None:
    """A verify-only turn claims something: its step and an unverified verdict show."""
    _stub_oracle(monkeypatch)
    console = _FakeConsole()
    cli._repl_turn(_rt(_FakeEngine(_noaction_trace("g"), answer="已经在了")), "去 (10,3)", console)
    assert "verify at_position(10.0, 3.0)" in console.text
    assert "verified=False" in console.text and "RAN" in console.text


def test_a_raising_native_turn_is_not_swallowed(monkeypatch) -> None:
    _stub_oracle(monkeypatch)
    with pytest.raises(RuntimeError, match="native blew up"):
        cli._repl_turn(_rt(_FakeEngine(RuntimeError("native blew up"))), "走到 (11,3)", _FakeConsole())


def test_a_lifecycle_only_turn_prints_its_result_and_no_empty_verdict(monkeypatch) -> None:
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
    cli._repl_turn(_rt(_Launch(started)), "启动 go2 仿真", console)
    assert "Started go2_piper simulation, ready in 9.1s" in console.text
    assert "go2 仿真已启动" in console.text and "verdict" not in console.text


def test_a_failed_verify_renders_an_unverified_verdict(monkeypatch) -> None:
    _stub_oracle(monkeypatch)
    trace = _acted_trace("g", strategy="walk", verify="at_position(99.0, 99.0)", verified_pose=False)
    console = _FakeConsole()
    cli._repl_turn(_rt(_FakeEngine(trace)), "走到 (99,99)", console)
    assert "verified=False" in console.text, "a failed verify must surface verified=False (honest)"
