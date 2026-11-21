# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""WP05 regressions: native producer safety and ownership.

Each test names the ledger bug it pins and fails on the pre-WP05 base. Deterministic,
no MuJoCo, no network: scripted backends drive the REAL ``run_turn_native`` /
``run_one_turn`` / REPL busy loop.
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vector_os_nano.vcli.backends.fake import FakeToolScriptBackend, tool_turn
from tests.unit.vcli.test_native_loop import _make_agent, _make_engine, _session
from rich.console import Console

from vector_os_nano.vcli import cli, render, repl
from vector_os_nano.vcli.cognitive import abort
from vector_os_nano.vcli.cognitive.trace_store import verify_oracle_names
from vector_os_nano.vcli.permissions import PermissionContext
from vector_os_nano.vcli.tools.base import CategorizedToolRegistry, PermissionResult, ToolResult
from vector_os_nano.vcli.verdict import VerdictReport


def _dev_engine(backend: Any, *tools: Any, permissions: Any = None) -> Any:
    """A dev-world engine (no agent) whose registry 'code' category holds *tools*."""
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.worlds.dev import DevWorld

    reg = CategorizedToolRegistry()
    for t in tools:
        reg.register(t, category="code")
    eng = VectorEngine(backend=backend, registry=reg, permissions=permissions or PermissionContext())
    eng.bind(None, DevWorld())
    return eng


class _SpyTool:
    """A side-effecting code tool with a scripted permission decision and a call spy."""

    def __init__(self, name: str, behavior: str) -> None:
        self.name, self.description = name, f"spy {name}"
        self.input_schema = {"type": "object", "properties": {"command": {"type": "string"}}}
        self._behavior = behavior
        self.calls = 0

    def check_permissions(self, params: dict, ctx: Any) -> PermissionResult:
        return PermissionResult(self._behavior, "scripted")

    def is_read_only(self, params: dict) -> bool:
        return False

    def execute(self, params: dict, ctx: Any) -> ToolResult:
        self.calls += 1
        return ToolResult(content="ran")


def _script(*turns: Any) -> FakeToolScriptBackend:
    return FakeToolScriptBackend.from_tool_script(list(turns))


# ---------------------------------------------------------------------------
# Permission gate on every native dispatch (producers:native-permission-bypass,
# cli-entry:native-bypasses-permissions, tools-worlds:native-loop-no-permissions)
# ---------------------------------------------------------------------------


def test_a_denied_tool_is_never_executed() -> None:
    spy = _SpyTool("bash", "deny")
    eng = _dev_engine(_script(tool_turn(("bash", {"command": "echo hi"})), tool_turn(end=True)), spy)
    trace = eng.run_turn_native("run it", session=_session())
    assert spy.calls == 0
    assert trace.refused == ("bash: scripted",) and trace.actions_dispatched == 0
    assert not render._chat_turn(trace)  # a refusal is shown with its verdict, never re-run


def test_rm_rf_root_is_blocked_by_the_bash_deny_list_on_the_native_path(monkeypatch) -> None:
    from vector_os_nano.vcli.tools import bash_tool

    ran: list[Any] = []
    monkeypatch.setattr(bash_tool.subprocess, "Popen", lambda *a, **k: ran.append(a))
    eng = _dev_engine(
        _script(tool_turn(("bash", {"command": "rm -rf /"})), tool_turn(end=True)),
        bash_tool.BashTool(),
        permissions=PermissionContext(no_permission=True),  # the deny-list beats even this
    )
    eng.run_turn_native("clean everything", session=_session())
    assert ran == []


def test_headless_denies_an_ask_tool_unless_granted(tmp_path, monkeypatch) -> None:
    from vector_os_nano.vcli.tools.file_tools import FileWriteTool

    monkeypatch.chdir(tmp_path)
    target = tmp_path / "out.txt"

    def turn(**kw: Any) -> None:
        eng = _dev_engine(
            _script(tool_turn(("file_write", {"file_path": str(target), "content": "x"})), tool_turn(end=True)),
            FileWriteTool(), permissions=kw.pop("permissions", None),
        )
        eng.run_turn_native("write it", session=_session(), **kw)

    turn()  # -p / no resolver: an 'ask' is a deny
    assert not target.exists()
    turn(ask_permission=lambda n, p: "n")
    assert not target.exists()
    turn(ask_permission=lambda n, p: "y")
    assert target.read_text() == "x"
    target.unlink()
    turn(permissions=PermissionContext(no_permission=True))  # --no-permission
    assert target.read_text() == "x"


def test_a_side_effecting_capability_is_surfaced_and_asks_first() -> None:
    """S6: every registered capability is a native tool; one that acts asks first."""
    from vector_os_nano.vcli.cognitive.capabilities.registry import CapabilityRegistry
    from vector_os_nano.vcli.native_tools import _build_motor_tools

    calls: list[dict] = []
    cap = SimpleNamespace(
        name="vla_grasp", kind="vla", side_effecting=True, description="grasp with a VLA",
        input_schema={"type": "object", "properties": {"target": {"type": "string"}}},
        invoke=lambda payload, ctx: calls.append(payload) or SimpleNamespace(success=True, output={}),
    )
    eng = _dev_engine(_script(tool_turn(("vla_grasp", {"target": "can"})), tool_turn(end=True)))
    eng._capability_registry = CapabilityRegistry()
    eng._capability_registry.register(cap)
    assert "vla_grasp" in _build_motor_tools(None, eng)
    eng.run_turn_native("grab the can", session=_session())
    assert calls == []  # headless: the ask was denied, the VLA never ran


# ---------------------------------------------------------------------------
# Per-run CancelToken (producers:sticky-abort-native, skills-nav:stale-global-abort-native,
# spine:native-ignores-abort, legacy-cognitive:B23-abort-cleared, spine:missed1)
# ---------------------------------------------------------------------------


def test_a_stop_mid_turn_records_an_aborted_step_and_calls_the_model_no_more() -> None:
    agent, base = _make_agent()
    real_walk = base.walk

    def walk_then_stop(*a: Any) -> bool:
        ok = real_walk(*a)
        abort.request_abort()  # the user's stop lands while the skill runs
        return ok

    base.walk = walk_then_stop
    backend = _script(
        tool_turn(("walk", {"distance": 1.0})),
        tool_turn(("walk", {"distance": 1.0})),  # must never be reached
        tool_turn(end=True),
    )
    trace = _make_engine(agent, backend).run_turn_native("walk", session=_session())
    assert backend._cursor == 1  # no model call after the stop
    assert trace.steps[-1].error == "aborted" and trace.steps[-1].success is False
    assert trace.actions_dispatched == 1


def test_a_stop_step_the_model_takes_does_not_abort_its_own_turn() -> None:
    """'走两步然后停下': the model's own stop call halts the base, and the turn goes on."""
    from vector_os_nano.skills.go2.stop import StopSkill

    agent, base = _make_agent()
    base.set_velocity = lambda *v: None
    agent._skill_registry.register(StopSkill())
    trace = _make_engine(agent, _script(
        tool_turn(("walk", {"distance": 2.0, "speed": 1.0})),
        tool_turn(("stop", {})),
        tool_turn(("verify", {"expr": "at_position(2.0, 0.0, 1.0)"})),
        tool_turn(end=True),
    )).run_turn_native("走两步然后停下", session=_session())
    assert not trace.cancelled and [s.error for s in trace.steps] == [""]
    assert trace.steps[0].verify_result


def test_a_previous_stop_does_not_abort_the_next_native_turn() -> None:
    agent, base = _make_agent()

    def navigate_to(x: float, y: float, timeout: float = 60.0, **kwargs):
        from vector_os_nano.sim.nav.follow import NavResult
        if abort.is_abort_requested():  # every real nav loop polls this
            return NavResult(reached=False, reason='aborted')
        base._x, base._y = x, y
        return NavResult(reached=True, reason='arrived', target=[x, y])

    base.navigate_to = navigate_to
    abort.request_abort()  # a "停" from an earlier turn
    eng = _make_engine(agent, _script(tool_turn(("navigate", {"x": 2.0, "y": 0.0})), tool_turn(end=True)))
    eng.run_turn_native("go to (2, 0)", session=_session())
    assert (base._x, base._y) == (2.0, 0.0)


def test_starting_a_new_run_never_clears_another_runs_abort() -> None:
    seen: list[bool] = []
    ready, check = threading.Event(), threading.Event()

    def background_run() -> None:
        abort.clear_abort()  # the background plan's own start
        ready.set()
        check.wait(5)
        seen.append(abort.is_abort_requested())

    t = threading.Thread(target=background_run)
    t.start()
    ready.wait(5)
    abort.request_abort()  # stop
    abort.clear_abort()    # a chat turn starts on this thread right after
    check.set()
    t.join(5)
    assert seen == [True]


def test_the_answer_wrap_never_clears_a_stop_on_a_background_run() -> None:
    """producers:answer-wrap-clears-abort: chat during a plan must not un-stop it."""
    from vector_os_nano.vcli.backends.types import LLMResponse
    from vector_os_nano.vcli.session import TokenUsage

    class _Chat:
        def call(self, **kw: Any) -> LLMResponse:
            return LLMResponse(text="hi", tool_calls=[], stop_reason="end_turn", usage=TokenUsage())

    eng = _dev_engine(_Chat())
    seen: list[bool] = []
    ready, check = threading.Event(), threading.Event()

    def background_plan() -> None:
        abort.clear_abort()
        ready.set()
        check.wait(5)
        seen.append(abort.is_abort_requested())

    t = threading.Thread(target=background_plan)
    t.start()
    ready.wait(5)
    eng._emergency_stop("停", _session())
    eng.run_turn_native("你好", session=_session())  # a chat turn while the plan unwinds
    check.set()
    t.join(5)
    assert seen == [True]


def test_the_p0_stop_runs_the_stop_skill_and_never_homes_the_arm() -> None:
    """agent.execute_skill('stop') appended a 'home' step: arm moves, gripper opens."""
    from vector_os_nano.core.skill import SkillRegistry
    from vector_os_nano.vcli.engine import VectorEngine

    ran: list[str] = []
    reg = SkillRegistry()
    reg.register(SimpleNamespace(name="stop", execute=lambda p, c: ran.append("stop")))
    agent = SimpleNamespace(_skill_registry=reg, _build_context=lambda: None,
                            execute_skill=lambda *a, **k: ran.append("execute_skill(stop, then home)"))
    eng = VectorEngine(backend=None)
    eng._agent = agent
    eng._emergency_stop("停", None)
    assert ran == ["stop"]


def test_engine_runs_are_single_flight_and_preempt_the_previous_run() -> None:
    eng = _dev_engine(_script())
    first_token: list[Any] = []
    entered, released = threading.Event(), threading.Event()

    def first() -> None:
        with eng.actuating_run() as tok:
            first_token.append(tok)
            entered.set()
            tok.wait(5)  # runs until cancelled
        released.set()

    t = threading.Thread(target=first)
    t.start()
    entered.wait(5)
    assert eng.busy
    with eng.actuating_run() as second:
        assert first_token[0].cancelled  # busy -> the previous run was cancelled
        assert released.is_set()          # ...and had let go before this one ran
        assert not second.cancelled
    t.join(5)
    assert not eng.busy


def test_a_walk_ends_on_a_stop() -> None:
    """A stop during a long open-loop walk ends it now, not when the timer runs out: the
    base polls the run's stop signal the session hands it (is_abort_requested)."""
    import numpy as np

    from vector_os_nano.sim.drivers.base import LeggedBase, _Abort

    def wait_sim(seconds: float, cancel: Any = None, hold: bool = False) -> bool:  # a wall-clock stand-in
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if cancel is not None and cancel.is_set():
                return False
            time.sleep(0.01)
        return True

    base = LeggedBase.__new__(LeggedBase)
    base._lock, base._owners, base._cmd, base._hold, base._cmd_motion = threading.Lock(), [], (0.0,) * 3, None, 0.0
    base._stops = 0
    base._cos_fall = 0.5
    base._spec = SimpleNamespace(limits=(0.8, 0.4, 4.0), fall_z=0.15)
    base._rt = SimpleNamespace(wait_sim=wait_sim, release_clock=lambda: None,
                               snapshot=lambda: SimpleNamespace(q=np.array([0, 0, 0.3, 1.0, 0, 0, 0])))
    base._fq, base._cancel = slice(0, 7), _Abort(abort.is_abort_requested)
    abort.clear_abort()
    threading.Timer(0.2, abort.request_abort).start()
    t0 = time.monotonic()
    assert base.walk(0.3, 0.0, 0.0, 5.0) is False
    assert time.monotonic() - t0 < 1.5 and base._cmd == (0.0, 0.0, 0.0) and not base._owners
    abort.clear_abort()


class _StopLandsMidCall:
    """The owner's stop lands while the model is thinking; the reply then asks for a tool."""

    def __init__(self, reply: Any) -> None:
        self._reply, self.calls = reply, 0

    def call(self, **kw: Any) -> Any:
        self.calls += 1
        abort.request_abort()
        return self._reply


def test_a_stop_during_the_llm_call_owns_the_turn_and_runs_nothing() -> None:
    """The stop cancels native before it dispatched anything: no tool runs, and the
    stopped turn is graded (its verdict shows the stop; nothing re-runs it)."""
    agent, base = _make_agent()
    walks: list[Any] = []
    base.walk = lambda *a: walks.append(a) or True
    backend = _StopLandsMidCall(tool_turn(("walk", {"distance": 1.0})))
    trace = _make_engine(agent, backend).run_turn_native("往前走一米", session=_session())
    assert walks == [] and backend.calls == 1
    assert trace.cancelled and trace.actions_dispatched == 0
    assert not render._chat_turn(trace)


def test_a_stop_anywhere_in_a_turn_stops_every_run_the_turn_starts_later() -> None:
    """A stop landing between two runs of one turn still stops the later run; the
    next turn starts clean."""
    eng = _dev_engine(_script())
    with abort.turn():
        with eng.actuating_run():
            pass  # the first run: nothing dispatched
        abort.request_abort()
        with eng.actuating_run() as later:
            assert later.cancelled and abort.is_abort_requested()
    with abort.turn():
        with eng.actuating_run() as nxt:
            assert not nxt.cancelled


def test_a_stop_from_a_tool_runs_none_of_the_rest_of_its_batch() -> None:
    """The REPL turn runs on a worker, so a stop never raises inside the native
    dispatch loop: the batch itself must stop at the first tool after the stop."""

    class _StopsTheTurn(_SpyTool):
        def execute(self, params: dict, ctx: Any) -> ToolResult:
            abort.request_abort()  # the owner's stop lands while A runs
            return super().execute(params, ctx)

    a, b = _StopsTheTurn("bash", "allow"), _SpyTool("file_write", "allow")
    eng = _dev_engine(_script(tool_turn(("bash", {"command": "A"}), ("file_write", {"command": "B"}))), a, b)
    s = _session()
    with abort.turn():
        trace = eng.run_turn_native("A then B", session=s)
    assert (a.calls, b.calls) == (1, 0) and trace.cancelled
    results = [r["content"] for e in s._entries if e["type"] == "tool_result" for r in e["results"]]
    assert results == ["ran", "Stopped: not run."]
    _assert_well_formed(s.to_messages())


# ---------------------------------------------------------------------------
# No double execution (cli-entry:native-exception-double-action,
# legacy-cognitive:B4-native-exception-double-actuation, producers:native-double-execution)
# ---------------------------------------------------------------------------


class _DiesAfter:
    """A backend that issues *turns*, then raises (an LLM outage mid-turn)."""

    def __init__(self, *turns: Any) -> None:
        self._turns, self.calls, self._model = list(turns), 0, "fake-model-1"

    def call(self, **kw: Any) -> Any:
        self.calls += 1
        if self._turns:
            return self._turns.pop(0)
        raise RuntimeError("503 upstream unavailable")


def test_an_llm_failure_after_an_action_returns_the_partial_trace() -> None:
    agent, base = _make_agent()
    walks: list[float] = []
    real_walk = base.walk
    base.walk = lambda *a: walks.append(a[0]) or real_walk(*a)
    backend = _DiesAfter(tool_turn(("walk", {"distance": 1.0})))
    trace = _make_engine(agent, backend).run_turn_native("walk 1 m", session=_session())

    assert len(walks) == 1  # the base walked once
    assert trace.actions_dispatched == 1
    assert "503" in trace.llm_error and trace.model == "fake-model-1"
    assert trace.steps[-1].strategy == "walk" and trace.steps[-1].success is False
    assert not render._chat_turn(trace)  # graded, never re-run
    report = VerdictReport.from_trace(trace, verify_oracle_names(agent, None))
    assert report.llm_unavailable and report.model == "fake-model-1"
    assert report.exit_code() == 1 and not report.verified


def test_an_llm_failure_after_a_verified_step_never_verifies_the_cut_short_plan() -> None:
    agent, _base = _make_agent()
    eng = _make_engine(agent, _DiesAfter(
        tool_turn(("walk", {"distance": 2.0, "speed": 1.0})),
        tool_turn(("verify", {"expr": "at_position(2.0, 0.0, 1.0)"})),
    ))
    trace = eng.run_turn_native("go to (2, 0) and then to (4, 0)", session=_session())
    assert trace.steps[0].verify_result  # the first leg did ground...
    report = VerdictReport.from_trace(trace, verify_oracle_names(agent, eng))
    assert not report.verified and report.evidence == "FAILED"  # ...the plan did not
    assert report.llm_unavailable and report.exit_code() == 1


def _p_turn(monkeypatch, capsys, trace: Any, answer: str = "") -> tuple[int, dict]:
    """Run cli.run_one_turn --json over a fake engine returning *trace*; (exit, verdict)."""
    import json

    seen: dict[str, Any] = {}

    def run_turn_native(prompt: str, **kw: Any) -> Any:
        seen.update(kw)
        kw["session"].append_user(prompt)
        kw["session"].append_assistant(answer)
        return trace

    from vector_os_nano.vcli.runtime import Runtime

    engine = SimpleNamespace(_agent=None, _backend=object(), is_stop=lambda t: False,
                             run_turn_native=run_turn_native)
    monkeypatch.setattr(cli, "_build",
                        lambda args, face: Runtime(engine=engine, session=_session(), app_state={}))
    monkeypatch.setattr(render.console, "file", open(os.devnull, "w"))
    code = cli.run_one_turn(SimpleNamespace(print_prompt="walk 1 m", json=True))
    line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("VECTOR_VERDICT "))
    assert seen["ask_permission"] is None  # cli-entry:missed3: -p never prompts on stdin
    return code, json.loads(line.split(" ", 1)[1])


@pytest.mark.parametrize("owned", ["acted", "cancelled"])
def test_p_path_grades_an_acted_or_stopped_turn(monkeypatch, capsys, owned) -> None:
    from vector_os_nano.vcli.cognitive.types import ExecutionTrace, GoalTree

    trace = ExecutionTrace(GoalTree(goal="g", sub_goals=()), (), False, 0.0,
                           actions_dispatched=int(owned == "acted"), cancelled=owned == "cancelled")
    code, verdict = _p_turn(monkeypatch, capsys, trace)
    assert code != 0 and verdict["verified"] is False
    assert verdict["acted"] is (owned == "acted")


def test_p_path_chat_turn_exits_0_with_acted_false(monkeypatch, capsys) -> None:
    """A -p turn with no action and no verify is the model's answer: exit 0, no claim."""
    from vector_os_nano.vcli.cognitive.types import ExecutionTrace, GoalTree

    trace = ExecutionTrace(GoalTree(goal="g", sub_goals=()), (), False, 0.0)
    code, verdict = _p_turn(monkeypatch, capsys, trace, answer="你好！")
    assert code == 0
    assert verdict["acted"] is False and verdict["verified"] is False
    assert verdict["evidence"] == "NO_TRACE" and verdict["error"] == "" and verdict["n_steps"] == 0


def test_p_path_a_task_sentence_without_an_action_is_a_failed_goal_not_chat(monkeypatch, capsys) -> None:
    """A goal that compiled a task contract claims an end state: a turn where the model
    only talks is graded (goal failed, exit 2), never exit 0 as a chat."""
    from vector_os_nano.vcli.cognitive.types import ExecutionTrace, GoalTree

    trace = ExecutionTrace(GoalTree(goal="走到坐标 (1.0, 0.0)", sub_goals=()), (), False, 0.0,
                           contract=("at_position(1.0, 0.0)",), contract_start=(False,), contract_end=(False,))
    code, verdict = _p_turn(monkeypatch, capsys, trace, answer="好的")
    assert code == 2 and verdict["verified"] is False and verdict["acted"] is False
    assert verdict["goal_check"] == "failed" and verdict["contract"] == ["at_position(1.0, 0.0)"]


def test_p_path_a_verify_without_an_action_is_graded_not_chat(monkeypatch, capsys) -> None:
    """A verify the model made is a claim: graded (RAN, exit 2), never exit 0 as a chat."""
    from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused
    from vector_os_nano.vcli.cognitive.types import ExecutionTrace, GoalTree, StepRecord, SubGoal

    sg = SubGoal(name="native_step_0", description="d", verify="at_position(1.0, 2.0)")
    step = StepRecord("native_step_0", "", True, True, 0.0, actor_caused=ActorCaused.UNCAUSED)
    trace = ExecutionTrace(GoalTree(goal="g", sub_goals=(sg,)), (step,), True, 0.0)
    code, verdict = _p_turn(monkeypatch, capsys, trace)
    assert code == 2 and verdict["acted"] is False and verdict["evidence"] == "RAN"


@pytest.mark.parametrize("owned", ["acted", "cancelled"])
def test_repl_turn_renders_the_verdict_of_an_unverified_acted_turn(owned) -> None:
    import io

    from vector_os_nano.vcli.cognitive.types import ExecutionTrace, GoalTree

    from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused
    from vector_os_nano.vcli.cognitive.types import StepRecord, SubGoal
    from vector_os_nano.vcli.runtime import Runtime

    # R1V-4: the acted case carries a real step (walk, then a FAILED verify).
    sub = SubGoal(name="native_step_0", description="d", verify="at_position(9.0, 9.0)", strategy="walk")
    step = StepRecord("native_step_0", "walk", True, False, 0.0, actor_caused=ActorCaused.CAUSED)
    acted = ExecutionTrace(GoalTree(goal="g", sub_goals=(sub,) if owned == "acted" else ()),
                           (step,) if owned == "acted" else (), False, 0.0,
                           actions_dispatched=int(owned == "acted"), cancelled=owned == "cancelled")
    engine = SimpleNamespace(_agent=None, run_turn_native=lambda *a, **k: acted)
    out = io.StringIO()
    repl._repl_turn(Runtime(engine=engine, session=_session(), app_state={}), "walk",
                   Console(file=out, width=200))
    assert "verified=False" in out.getvalue(), out.getvalue()
    if owned == "acted":
        assert "walk → verify at_position(9.0, 9.0)" in out.getvalue()


@pytest.mark.parametrize("tool", ["file_read", "bash"])
def test_a_dev_turn_native_owns_shows_the_models_answer(tmp_path, monkeypatch, tool) -> None:
    """'总结一下 README.md' / '运行测试' in the bare REPL: native reads (or runs), the
    model answers, and the answer is printed. A read is never nudged for a verify."""
    import io

    from vector_os_nano.vcli.tools import bash_tool
    from vector_os_nano.vcli.tools.file_tools import FileReadTool

    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("Vector OS: an agent runtime.\n")
    call = ("file_read", {"file_path": str(tmp_path / "README.md")}) if tool == "file_read" \
        else ("bash", {"command": "echo '3 passed'"})
    backend = _script(tool_turn(call), tool_turn(text="README 说这是一个 agent runtime。", end=True))
    eng = _dev_engine(backend, FileReadTool(), bash_tool.BashTool(),
                      permissions=PermissionContext(no_permission=True))
    from vector_os_nano.vcli.runtime import Runtime

    out = io.StringIO()
    repl._repl_turn(Runtime(engine=eng, session=_session(), app_state={}), "总结一下 README.md 的内容",
                   Console(file=out, width=200))
    assert "README 说这是一个 agent runtime。" in out.getvalue(), out.getvalue()
    if tool == "file_read":
        assert backend._cursor == 2  # the read, the answer: no 'verify' nudges


# ---------------------------------------------------------------------------
# Step honesty (docs-vs-code:missed2, producers:missed1)
# ---------------------------------------------------------------------------


def test_a_failed_skill_followed_by_a_true_verify_is_not_grounded(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pre.txt").write_text("seed")
    spy = _SpyTool("deploy", "allow")
    spy.execute = lambda p, c: ToolResult(content="deploy failed", is_error=True)  # type: ignore[method-assign]
    eng = _dev_engine(_script(
        tool_turn(("deploy", {})),
        tool_turn(("verify", {"expr": "path_contains('pre.txt', 'seed')"})),
        tool_turn(end=True),
    ), spy)
    trace = eng.run_turn_native("deploy", session=_session())
    assert trace.steps[0].success is False
    assert VerdictReport.from_trace(trace, verify_oracle_names(None, eng)).evidence != "GROUNDED"


def test_a_read_only_call_does_not_ground_a_verify_of_preexisting_state(tmp_path, monkeypatch) -> None:
    from vector_os_nano.vcli.tools.file_tools import FileReadTool

    monkeypatch.chdir(tmp_path)
    (tmp_path / "pre.txt").write_text("seed")
    eng = _dev_engine(_script(
        tool_turn(("file_read", {"file_path": str(tmp_path / "pre.txt")})),
        tool_turn(("verify", {"expr": "path_contains('pre.txt', 'seed')"})),
        tool_turn(end=True),
    ), FileReadTool())
    trace = eng.run_turn_native("check pre.txt", session=_session())
    assert trace.actions_dispatched == 0
    assert VerdictReport.from_trace(trace, verify_oracle_names(None, eng)).evidence == "RAN"


# ---------------------------------------------------------------------------
# LLM infrastructure policy (graft c)
# ---------------------------------------------------------------------------


def test_an_unavailable_llm_falls_back_to_openrouter_when_opted_in(monkeypatch) -> None:
    from vector_os_nano.vcli import backends

    agent, _base = _make_agent()
    made: list[tuple] = []
    fallback = _script(tool_turn(end=True))
    fallback._model = "anthropic/claude-sonnet-4-6"
    monkeypatch.setattr(backends, "create_backend", lambda *a, **k: made.append(a) or fallback)
    monkeypatch.setenv("VECTOR_LLM_FALLBACK", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    trace = _make_engine(agent, _DiesAfter()).run_turn_native("hi", session=_session())
    assert made and made[0][0] == "openrouter"
    assert trace.llm_error == "" and trace.model == "anthropic/claude-sonnet-4-6"


def test_without_the_opt_in_an_llm_outage_is_an_infrastructure_verdict(monkeypatch) -> None:
    monkeypatch.delenv("VECTOR_LLM_FALLBACK", raising=False)
    agent, _base = _make_agent()
    trace = _make_engine(agent, _DiesAfter()).run_turn_native("hi", session=_session())
    report = VerdictReport.from_trace(trace, frozenset())
    assert report.llm_unavailable and report.evidence == "NO_TRACE" and report.exit_code() == 1
    assert not render._chat_turn(trace)  # shown as an infrastructure verdict, not as a chat


# ---------------------------------------------------------------------------
# REPL: stop / Ctrl-C mid-turn, permission prompts from the worker
# (cli-entry:ctrlc-native-exits-repl, producers:ctrl-c-kills-repl-native,
#  cli-entry:stop-not-p0-in-repl, producers:stop-not-p0, cli-entry:missed1)
# ---------------------------------------------------------------------------


class _ReplEngine:
    def __init__(self) -> None:
        self.stops: list[str] = []
        self.stopped = threading.Event()

    def is_stop(self, text: str) -> bool:
        return text.strip() in ("停", "stop")

    def _emergency_stop(self, text: str, session: Any, agent: Any = None, app_state: Any = None) -> None:
        self.stops.append(text)
        self.stopped.set()

    def _has_live_viewer(self) -> bool:
        return False


@pytest.fixture
def typed(monkeypatch):
    """Replace stdin with a pipe; returns a function that 'types' a line."""
    r, w = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(r, "r"))
    monkeypatch.setattr(render.console, "file", open(os.devnull, "w"))
    yield lambda line: os.write(w, (line + "\n").encode())
    os.close(w)


def test_a_typed_stop_mid_turn_stops_the_robot_without_the_llm(typed) -> None:
    eng = _ReplEngine()
    threading.Timer(0.2, typed, args=("停",)).start()
    t0 = time.monotonic()
    nxt = repl._run_turn_busy(eng, lambda: eng.stopped.wait(5), {})
    assert eng.stops == ["停"] and nxt is None
    assert time.monotonic() - t0 < 2.0


def test_another_command_mid_turn_cancels_then_runs_next(typed) -> None:
    eng = _ReplEngine()
    threading.Timer(0.2, typed, args=("往后走",)).start()
    assert repl._run_turn_busy(eng, lambda: eng.stopped.wait(5), {}) == "往后走"
    assert eng.stops == ["往后走"]  # the running turn was stopped first


def test_a_stop_typed_after_another_command_drops_that_command(typed) -> None:
    """'往后走' then '停' while the turn unwinds: the robot stays stopped — the queued
    command never runs after the owner's later stop."""
    eng = _ReplEngine()
    threading.Timer(0.1, typed, args=("往后走",)).start()
    threading.Timer(0.3, typed, args=("停",)).start()
    assert repl._run_turn_busy(eng, lambda: time.sleep(0.6), {}) is None
    assert eng.stops == ["往后走", "停"]


@pytest.fixture
def sigints():
    """Count the SIGINTs the main thread has HANDLED (each still raises
    KeyboardInterrupt), so a turn can wait for every Ctrl-C to land inside it: a
    late timer must never interrupt pytest itself."""
    handled: list[int] = []

    def _handler(signum: int, frame: Any) -> None:
        handled.append(signum)
        raise KeyboardInterrupt

    prev = signal.signal(signal.SIGINT, _handler)
    yield handled
    signal.signal(signal.SIGINT, prev)


def _until(cond: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond() and time.monotonic() < deadline:
        time.sleep(0.01)


@pytest.mark.own_process
@pytest.mark.parametrize("stdin_idle", [False, True])
def test_ctrl_c_mid_turn_stops_and_never_exits(typed, monkeypatch, sigints, stdin_idle) -> None:
    """A Ctrl-C lands wherever the REPL thread is — reading stdin, or waiting on the
    turn (stdin_idle: the loop then lives in the wait) — and never escapes."""
    if stdin_idle:
        monkeypatch.setattr(repl, "_read_line", lambda timeout: None)
    eng = _ReplEngine()
    for at in (0.2, 0.35):  # a panicked double Ctrl-C
        threading.Timer(at, os.kill, args=(os.getpid(), signal.SIGINT)).start()
    try:
        nxt = repl._run_turn_busy(eng, lambda: _until(lambda: len(sigints) == 2), {})
    except KeyboardInterrupt:
        pytest.fail("Ctrl-C escaped the REPL turn loop (the REPL would exit)")
    assert nxt is None and len(sigints) == 2
    assert eng.stops and set(eng.stops) == {"interrupt"}


@pytest.mark.own_process
def test_ctrl_c_during_a_stops_speed_report_never_exits(typed, sigints) -> None:
    """The REPL runs a P0 stop like any turn: a panicked Ctrl-C while the stop measures
    the base speed (~1 sim-s) stops again and never escapes the REPL."""
    eng = _ReplEngine()
    base = SimpleNamespace(get_velocity=lambda: (0.0, 0.0, 0.0), get_sim_time=time.monotonic)
    app = {"agent": SimpleNamespace(_base=base)}
    threading.Timer(0.3, os.kill, args=(os.getpid(), signal.SIGINT)).start()

    def stop_turn() -> None:
        repl._p0_stop(eng, "停下", app)
        _until(lambda: len(sigints) == 1)  # a late Ctrl-C still lands inside the turn

    try:
        repl._run_turn_busy(eng, stop_turn, app)
    except KeyboardInterrupt:
        pytest.fail("Ctrl-C during the stop report escaped the REPL")
    assert eng.stops == ["停下", "interrupt"]


def test_the_stop_report_shows_a_lurch_between_its_marks() -> None:
    """Two samples can miss the body moving again between them: the report carries
    the peak from 0.2 to 1.0 sim-s (a fake clock: 0.01 sim-s per read)."""
    clock = iter(i * 0.01 for i in range(10_000))
    now = [0.0]

    def sim_time() -> float:
        now[0] = next(clock)
        return now[0]

    def velocity() -> tuple[float, float, float]:
        t = now[0]
        return (0.3 if t < 0.1 else 0.12 if 0.4 <= t < 0.5 else 0.02, 0.0, 0.0)

    base = SimpleNamespace(get_velocity=velocity, get_sim_time=sim_time)
    assert repl._stop_report(SimpleNamespace(_base=base)) == (
        "base 0.02 m/s at 0.2 sim-s, peak 0.12 m/s to 1.0 sim-s, 0.02 m/s at 1.0 sim-s"
    )


def test_a_stopped_turns_worker_never_prompts(typed, monkeypatch) -> None:
    """A stopped (stale) turn's permission ask is denied, never shown to the owner."""
    monkeypatch.setattr(repl.Prompt, "ask", lambda *a, **k: pytest.fail("a stopped turn prompted"))
    answers: list[str] = []

    def turn() -> None:
        abort.request_abort()
        answers.append(repl.ask_permission("bash", {"command": "rm -rf build"}))

    repl._run_turn_busy(_ReplEngine(), turn, {})
    assert answers == ["n"]


def test_a_permission_prompt_from_the_turn_worker_is_answered_by_the_repl_thread(typed, monkeypatch) -> None:
    eng = _ReplEngine()
    asked_on: list[str] = []
    monkeypatch.setattr(repl.Prompt, "ask", lambda *a, **k: asked_on.append(threading.current_thread().name) or "y")
    answers: list[str] = []
    repl._run_turn_busy(eng, lambda: answers.append(repl.ask_permission("file_write", {"p": 1})), {})
    assert answers == ["y"] and asked_on == ["MainThread"]
    # Off the main thread with no REPL loop to answer, an ask fails closed.
    out: list[str] = []
    t = threading.Thread(target=lambda: out.append(repl.ask_permission("bash", {})))
    t.start()
    t.join(5)
    assert out == ["n"]


# ---------------------------------------------------------------------------
# A dangling tool_use never bricks the session (cli-entry/producers:
# dangling-tool-use-bricks-session)
# ---------------------------------------------------------------------------


def _assert_well_formed(messages: list[dict]) -> None:
    """Every assistant tool_use is answered by tool_results in the NEXT message."""
    for i, m in enumerate(messages):
        if m["role"] != "assistant" or not isinstance(m["content"], list):
            continue
        ids = {b["id"] for b in m["content"] if b.get("type") == "tool_use"}
        if not ids:
            continue
        nxt = messages[i + 1] if i + 1 < len(messages) else {"content": []}
        answered = {b.get("tool_use_id") for b in nxt["content"] if isinstance(nxt["content"], list)}
        assert ids <= answered, f"tool_use {ids - answered} unanswered in {nxt}"


class _StrictBackend:
    """Like the real APIs: a malformed message list is a hard error (a 400)."""

    def call(self, messages: list[dict], **kw: Any) -> Any:
        from vector_os_nano.vcli.backends.types import LLMResponse
        from vector_os_nano.vcli.session import TokenUsage

        _assert_well_formed(messages)
        return LLMResponse(text="ok", tool_calls=[], stop_reason="end_turn", usage=TokenUsage())


def test_a_session_with_a_dangling_tool_use_resumes(tmp_path) -> None:
    from vector_os_nano.vcli.engine import VectorEngine

    s = _session()
    s.append_user("walk forward")
    s.append_assistant("", [{"type": "tool_use", "id": "t1", "name": "walk", "input": {}},
                            {"type": "tool_use", "id": "t2", "name": "turn", "input": {}}])
    # Ctrl-C landed mid-dispatch: no tool_result was ever appended.
    VectorEngine(backend=_StrictBackend()).run_turn_native("what happened?", session=s)
    assert s.last_assistant_text() == "ok"
    msgs = s.to_messages()
    _assert_well_formed(msgs)
    synthetic = [b for b in msgs[2]["content"] if b.get("tool_use_id") in ("t1", "t2")]
    assert len(synthetic) == 2 and all(b["is_error"] for b in synthetic)


def test_a_partly_answered_tool_use_is_completed_in_the_same_message() -> None:
    s = _session()
    s.append_user("go")
    s.append_assistant("", [{"type": "tool_use", "id": "a", "name": "walk", "input": {}},
                            {"type": "tool_use", "id": "b", "name": "turn", "input": {}}])
    s.append_tool_results([{"tool_use_id": "a", "content": "done"}])
    msgs = s.to_messages()
    _assert_well_formed(msgs)
    assert [b["tool_use_id"] for b in msgs[-1]["content"]] == ["a", "b"]


def test_navigate_is_not_rerun_when_the_base_raises_a_type_error_mid_way() -> None:
    """producers:missed4: ANY TypeError re-ran navigate_to without a timeout."""
    from vector_os_nano.vcli.native_tools import _NativeBaseNavigateTool

    calls: list[tuple] = []

    def navigate_to(x: float, y: float, timeout: float = 60.0, **kwargs):
        calls.append((x, y))
        raise TypeError("unsupported operand mid-navigation")

    agent, base = _make_agent()
    base.navigate_to = navigate_to
    ctx = SimpleNamespace(agent=agent, app_state=None, abort=threading.Event())
    result = _NativeBaseNavigateTool().execute({"x": 1.0, "y": 2.0}, ctx)
    assert result.is_error and calls == [(1.0, 2.0)]


def test_the_dev_world_can_launch_a_sim() -> None:
    """producers:keyword-gate-default-producer: no keyword router decides anything; the
    dev world's native toolset holds the sim lifecycle, so '启动 go2 仿真' is the
    model's to route, like a dev write."""
    from vector_os_nano.vcli.native_tools import _build_motor_tools
    from vector_os_nano.vcli.runtime import build_registry

    dev = SimpleNamespace(_registry=build_registry())
    assert {"start_simulation", "stop_simulation"} <= set(_build_motor_tools(None, dev))


def test_the_chat_llm_key_is_never_handed_to_the_vlm(monkeypatch) -> None:
    """cli-entry:missed4: --api-key (Anthropic/DeepSeek) went to the OpenRouter VLM."""
    from vector_os_nano.sim import session as sim_session

    seen: dict[str, Any] = {}

    def fake_start(spec: Any, **kw: Any) -> Any:
        seen.update(kw)
        return SimpleNamespace(summary=lambda: "ok")

    monkeypatch.setattr(sim_session.SimSession, "start", staticmethod(fake_start))
    monkeypatch.setattr(render.console, "file", open(os.devnull, "w"))
    # Spelled so tests/conftest.py's sim-tier source regex keeps this file hermetic.
    from vector_os_nano.vcli import runtime

    start = runtime.start_sim
    start("go2", gui=False, notice=lambda msg: None)
    assert seen.get("vlm_api_key") in (None, "")
