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
  - ``_intent_actionable`` — the classify_intent OPTIMIZATION hint (use_vgg -> attempt
    native; else go straight to tool_use); fail-OPEN to native on a classify error.
  - ``_repl_attempt_native`` — acted trace -> returns True, renders the honest verdict,
    records a clean session summary; no-action / raised -> returns False (caller falls
    back to legacy), and NO session pollution.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vector_os_nano.vcli import cli
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


class _FakeSession:
    def __init__(self) -> None:
        self.user: list[str] = []
        self.asst: list[str] = []

    def append_user(self, t: str) -> None:
        self.user.append(t)

    def append_assistant(self, t: str, *a: object, **k: object) -> None:
        self.asst.append(t)


class _FakeEngine:
    def __init__(self, trace: object, *, use_vgg: bool = True, raise_classify: bool = False) -> None:
        self._vgg_agent = None
        self._trace = trace
        self._use_vgg = use_vgg
        self._raise = raise_classify
        self.native_calls = 0

    def classify_intent(self, text: str) -> SimpleNamespace:
        if self._raise:
            raise RuntimeError("classify boom")
        return SimpleNamespace(use_vgg=self._use_vgg)

    def run_turn_native(self, user_message, agent=None, session=None, app_state=None, on_progress=None, ask_permission=None):  # noqa: ANN001
        # ``on_progress`` mirrors the REAL VectorEngine.run_turn_native signature —
        # _repl_attempt_native passes it for the live spinner; a fake omitting it made
        # the call raise TypeError (caught as "no action"), masking the cutover path.
        self.native_calls += 1
        if isinstance(self._trace, Exception):
            raise self._trace
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
# _intent_actionable — classify hint, fail-open.
# ---------------------------------------------------------------------------


def test_intent_actionable_true_when_use_vgg() -> None:
    assert cli._intent_actionable(_FakeEngine(None, use_vgg=True), "走到 (11,3)") is True


def test_intent_actionable_false_when_not_use_vgg() -> None:
    assert cli._intent_actionable(_FakeEngine(None, use_vgg=False), "你好") is False


def test_intent_actionable_fail_open_on_classify_error() -> None:
    # A classify failure must NOT silently skip the redesign — attempt native.
    assert cli._intent_actionable(_FakeEngine(None, raise_classify=True), "x") is True


# ---------------------------------------------------------------------------
# _repl_attempt_native — acted -> True+record; no-action / raised -> False+clean.
# ---------------------------------------------------------------------------


def test_repl_attempt_native_true_and_records_when_acted(monkeypatch) -> None:
    _stub_oracle(monkeypatch)
    trace = _acted_trace("g", strategy="walk", verify="at_position(11.0, 3.0)", verified_pose=True)
    engine = _FakeEngine(trace)
    session = _FakeSession()
    console = _FakeConsole()

    acted = cli._repl_attempt_native(engine, "走到坐标 (11,3)", session, {}, console)

    assert acted is True
    assert engine.native_calls == 1
    # Honest verdict rendered (GROUNDED + CAUSED + at_position True -> verified True).
    assert "verified=True" in console.text
    assert "walk" in console.text
    # Clean session summary recorded for follow-up context.
    assert session.user == ["走到坐标 (11,3)"]
    assert len(session.asst) == 1 and "native executed" in session.asst[0]


def test_repl_attempt_native_false_and_clean_when_no_action(monkeypatch) -> None:
    _stub_oracle(monkeypatch)
    engine = _FakeEngine(_noaction_trace("g"))
    session = _FakeSession()
    console = _FakeConsole()

    acted = cli._repl_attempt_native(engine, "启动 go2 仿真", session, {}, console)

    assert acted is False, "verify-only / no-action native trace must fall back to legacy"
    assert engine.native_calls == 1
    # NO session pollution — the legacy path will record the turn instead.
    assert session.user == [] and session.asst == []


def test_repl_attempt_native_false_when_native_raises(monkeypatch) -> None:
    _stub_oracle(monkeypatch)
    engine = _FakeEngine(RuntimeError("native blew up"))
    session = _FakeSession()
    console = _FakeConsole()

    acted = cli._repl_attempt_native(engine, "走到 (11,3)", session, {}, console)

    assert acted is False, "a raised native attempt must fall back to legacy, not crash the REPL"
    assert session.user == [] and session.asst == []


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
    assert "Verified: False" in session.asst[0]
