# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""D9 #2 latency: batching the action + its verify into ONE response cuts native
round-trips. The user's "load 好几秒" is the SYNCHRONOUS LLM round-trips (per-turn
setup is ~0.04ms — negligible). The runner already dispatches multiple tool calls
in one turn (native_loop.py:528-543); the system prompt now instructs the model to
emit (action, verify) together. This pins that batching => fewer backend.call
round-trips, with an identical honest trace (one action->verify step).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


class _CountingBackend:
    """Wraps a scripted backend, counting .call() round-trips."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def call(self, **kwargs):
        self.calls += 1
        return self._inner.call(**kwargs)


def _run(tool_script, on_progress=None, skill=None):
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.core.types import SkillResult
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.permissions import PermissionContext
    from vector_os_nano.vcli.session import Session
    from vector_os_nano.vcli.tools.base import CategorizedToolRegistry
    from vector_os_nano.vcli.worlds.dev import DevWorld
    from tests.harness.fake_backend import FakeToolScriptBackend

    class _WriteFileSkill:
        name = "write_file"
        description = "TEST: write text to a file (a dev action)."
        parameters = {"file_path": {"type": "string", "required": True},
                      "content": {"type": "string", "required": True}}
        preconditions: list = []
        effects: dict = {}

        def execute(self, params, context):
            Path(params["file_path"]).write_text(str(params.get("content", "")), encoding="utf-8")
            return SkillResult(success=True, result_data={"file_path": params.get("file_path")})

    work = tempfile.mkdtemp(prefix="batch_rt_")
    prev = os.getcwd()
    os.chdir(work)
    try:
        agent = Agent(config={})
        agent._skill_registry.register(skill if skill is not None else _WriteFileSkill())
        backend = _CountingBackend(FakeToolScriptBackend.from_tool_script(tool_script))
        eng = VectorEngine(backend=backend, registry=CategorizedToolRegistry(),
                           permissions=PermissionContext(no_permission=True))  # harness grants asks
        eng._world = DevWorld()
        eng.init_vgg(agent=agent, skill_registry=agent._skill_registry, world=DevWorld())
        eng._vgg_agent = agent
        eng._backend = backend
        session = Session(session_id="batch", created_at="t", updated_at="t",
                          path=Path(work) / "s.jsonl")
        trace = eng.run_turn_native("create out.txt containing ready", session=session,
                                    on_progress=on_progress)
        return backend.calls, trace
    finally:
        os.chdir(prev)


def test_batched_action_verify_uses_fewer_roundtrips():
    from tests.harness.fake_backend import tool_turn

    write = ("write_file", {"file_path": "out.txt", "content": "ready\n"})
    verify = ("verify", {"expr": "path_contains('out.txt', 'ready')"})

    # Sequential: action / verify / finish — 3 round-trips.
    seq_calls, seq_trace = _run([tool_turn(write), tool_turn(verify), tool_turn(end=True)])
    # Batched: (action, verify) in ONE response / finish — 2 round-trips.
    batch_calls, batch_trace = _run([tool_turn(write, verify), tool_turn(end=True)])

    assert seq_calls == 3
    assert batch_calls == 2
    assert batch_calls < seq_calls  # the latency win is realized by the runner
    # Same honest trace shape: exactly one action->verify step either way.
    assert len(seq_trace.steps) == 1
    assert len(batch_trace.steps) == 1


def test_on_progress_streams_steps():
    """D9 #2 perceived latency: on_progress fires per tool call (+ model text), so
    the REPL spinner shows live activity instead of a frozen 'load 好几秒'."""
    from tests.harness.fake_backend import tool_turn

    msgs: list[str] = []
    _run(
        [tool_turn(("write_file", {"file_path": "out.txt", "content": "ready\n"}),
                   ("verify", {"expr": "path_contains('out.txt', 'ready')"})),
         tool_turn(end=True)],
        on_progress=msgs.append,
    )
    assert msgs, "on_progress never fired — the wait stays opaque"
    blob = " | ".join(msgs)
    assert "write_file" in blob          # the action streamed
    assert "verify" in blob              # the verify streamed


# --- D23: verify-compliance — force a model-authored verify before finish ---

_WRITE = ("write_file", {"file_path": "out.txt", "content": "ready\n"})
_VERIFY = ("verify", {"expr": "path_contains('out.txt', 'ready')"})
_FINISH = ("finish", {})


def test_verify_skip_is_nudged_then_recorded():
    """Model runs the action then STOPS without verifying -> the runner nudges, and
    once the model verifies, the step is recorded. Without the nudge the turn would
    break at the empty end_turn and the trace would be empty (steps=0)."""
    from tests.harness.fake_backend import tool_turn

    msgs: list[str] = []
    _calls, trace = _run(
        [tool_turn(_WRITE), tool_turn(end=True), tool_turn(_VERIFY)],
        on_progress=msgs.append,
    )
    assert any("re-prompt" in m.lower() or "verify before" in m.lower() for m in msgs), \
        f"expected a verify nudge; got {msgs}"
    assert len(trace.steps) == 1  # the nudge let the model verify -> one graded step


def test_stubborn_no_verify_terminates_bounded_and_honest():
    """A model that NEVER verifies (only an action, then narration) still terminates
    (bounded nudges, not max_turns), and the action stays ON the record as an unproven
    step (never silently dropped, which let the caller re-run it) — never a false green."""
    from tests.harness.fake_backend import tool_turn
    from vector_os_nano.vcli.verdict import VerdictReport

    calls, trace = _run([tool_turn(_WRITE)])  # then script exhausts -> end_turn forever
    assert [(s.strategy, s.error) for s in trace.steps] == [("write_file", "not verified")]
    assert trace.actions_dispatched == 1
    report = VerdictReport.from_trace(trace, frozenset({"path_contains"}))
    assert not report.verified and report.evidence == "RAN"
    assert calls <= 6                     # bounded by _MAX_VERIFY_NUDGES, NOT max_turns=24


def test_compliant_finish_after_verify_not_nudged():
    """Action+verify then finish: finish is accepted with no nudge (one graded step)."""
    from tests.harness.fake_backend import tool_turn

    msgs: list[str] = []
    _calls, trace = _run(
        [tool_turn(_WRITE, _VERIFY), tool_turn(_FINISH)],
        on_progress=msgs.append,
    )
    assert len(trace.steps) == 1
    assert not any("re-prompt" in m.lower() or "verify required" in m.lower() for m in msgs)


def test_recover_after_failed_verify_records_fail_then_pass():
    """RECOVER pillar (north-star #4): a FAILed verify -> re-act -> verify PASS records
    [FAIL, PASS]. The moat keeps the FAIL row (D10: a recovered turn is verified=False
    via the all-GROUNDED gate — never a false green). Real-LLM confirmed (R9): haiku
    re-acts after FAIL until PASS; this guards the path with a scripted retry."""
    from pathlib import Path as _P
    from vector_os_nano.core.types import SkillResult
    from tests.harness.fake_backend import tool_turn

    class _FlakyWrite:
        name = "write_file"; description = "write text to a file"
        parameters = {"file_path": {"type": "string", "required": True},
                      "content": {"type": "string", "required": True}}
        preconditions: list = []; effects: dict = {}
        def __init__(self): self.calls = 0
        def execute(self, params, context):
            self.calls += 1
            if self.calls >= 2:  # writes only on the retry
                _P(params["file_path"]).write_text(str(params.get("content", "")), encoding="utf-8")
            return SkillResult(success=True, result_data={"attempt": self.calls})

    flaky = _FlakyWrite()
    _calls, trace = _run(
        [tool_turn(_WRITE, _VERIFY),   # attempt 1 -> verify FAIL (no-op write)
         tool_turn(_WRITE, _VERIFY),   # retry -> verify PASS (writes)
         tool_turn(_FINISH)],
        skill=flaky,
    )
    results = [bool(s.verify_result) for s in trace.steps]
    assert results == [False, True]    # recorded FAIL then PASS (recovered)
    assert flaky.calls == 2            # the model re-acted (didn't give up)


def test_multistep_plan_records_one_step_per_subgoal():
    """PLAN/ROUTE (north-star #1/#2): a 2-step task decomposes into 2 (action->verify)
    steps, both GROUNDED. Real-LLM confirmed (R9/D27): haiku decomposes + routes +
    verifies EACH step (out.txt='a', done.txt='b'); this guards the path scripted."""
    from tests.harness.fake_backend import tool_turn

    w1 = ("write_file", {"file_path": "out.txt", "content": "a"})
    v1 = ("verify", {"expr": "path_contains('out.txt', 'a')"})
    w2 = ("write_file", {"file_path": "done.txt", "content": "b"})
    v2 = ("verify", {"expr": "path_contains('done.txt', 'b')"})
    _calls, trace = _run([tool_turn(w1, v1), tool_turn(w2, v2), tool_turn(_FINISH)])
    assert [bool(s.verify_result) for s in trace.steps] == [True, True]  # both subgoals graded
