# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""STEP 5 — NATIVE-ATTEMPT-THEN-FALLBACK routing spike (flag-gated, default OFF).

The prep toward the CUTOVER: a SEPARATE additive mode ``--native-first`` (env
``VECTOR_NATIVE_FIRST=1``) that, in the -p/--print path ONLY, ATTEMPTS the native
tool-use producer first and FALLS BACK to the legacy decompose+execute plan when
native took NO action (could not route the goal — no suitable tool, or a dev world
with no agent). The always-native ``--native-loop`` flag is UNCHANGED; native-first
is additive. Every existing path is byte-identical when neither flag is set.

Three proofs, no LLM required for (a)/(b):

(a) UNIT — ``_native_trace_acted``: a trace with a step strategy='walk' -> True; a
    trace with only an empty-strategy (verify-only) step -> False; an empty trace ->
    False.
(b) FALLBACK WIRING (deterministic, no sim) — a stub engine whose
    ``run_turn_native`` returns a NO-ACTION trace and whose ``vgg_decompose`` /
    ``vgg_execute`` return a GROUNDED trace: native-first ON -> the LEGACY verdict is
    emitted (fallback fired), exit matches the legacy trace. Second case: an ACTED
    grounded native trace -> the NATIVE verdict is emitted (no fallback;
    ``vgg_decompose`` NOT called — proven by a call-spy).
(c) COVERED-GO2 ON REAL SIM (@sim, scripted) — native-first ON + a walk tool_script
    through the REAL ``cli.main -p`` -> native acts -> native verdict GROUNDED /
    verified True / exit 0. Mirrors the trichotomy honest-walk case but routes via
    the native-first flag instead of --native-loop, proving a covered goal routes to
    native UNDER native-first on the real sim.
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
# Trace builders — minimal ExecutionTraces shaped like native_loop produces.
# ---------------------------------------------------------------------------


def _acted_trace(goal: str, *, strategy: str, verify: str, verified_pose: bool) -> ExecutionTrace:
    """A trace with ONE acted step (non-empty strategy) — native routed the goal.

    ``verified_pose`` toggles whether the deterministic verify passed; with a
    graded-CAUSED step + a passing robot predicate it classifies GROUNDED.
    """
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
    """A trace with ONE verify-only step (empty strategy) — native did NOT act.

    This is exactly what NativeStepRunner records for a verify with no preceding
    skill dispatch (``strategy = self._chain[-1] if self._chain else ""``).
    """
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


def _empty_trace(goal: str) -> ExecutionTrace:
    """A trace with NO steps — native never even verified (no backend / no tools)."""
    tree = GoalTree(goal=goal, sub_goals=())
    return ExecutionTrace(goal_tree=tree, steps=(), success=False, total_duration_sec=0.0)


# ---------------------------------------------------------------------------
# (a) UNIT — _native_trace_acted
# ---------------------------------------------------------------------------


def test_native_trace_acted_true_for_walk_step() -> None:
    trace = _acted_trace("g", strategy="walk", verify="at_position(9.0, 3.0)", verified_pose=True)
    assert cli._native_trace_acted(trace) is True


def test_native_trace_acted_false_for_verify_only_step() -> None:
    assert cli._native_trace_acted(_noaction_trace("g")) is False


def test_native_trace_acted_false_for_empty_trace() -> None:
    assert cli._native_trace_acted(_empty_trace("g")) is False


def test_native_trace_acted_false_for_none() -> None:
    # Defensive: a None trace (native raised) is not "acted".
    assert cli._native_trace_acted(None) is False


def test_native_trace_acted_true_when_any_step_acted() -> None:
    # A mixed trace (one verify-only, one acted) still counts as acted.
    sg0 = SubGoal(name="s0", description="d", verify="x", strategy="")
    sg1 = SubGoal(name="s1", description="d", verify="at_position(1,1)", strategy="walk")
    st0 = StepRecord(sub_goal_name="s0", strategy="", success=True, verify_result=True, duration_sec=0.0)
    st1 = StepRecord(sub_goal_name="s1", strategy="walk", success=True, verify_result=True, duration_sec=0.0)
    tree = GoalTree(goal="g", sub_goals=(sg0, sg1))
    trace = ExecutionTrace(goal_tree=tree, steps=(st0, st1), success=True, total_duration_sec=0.0)
    assert cli._native_trace_acted(trace) is True


# ---------------------------------------------------------------------------
# (a') UNIT — the flag readers (default OFF, flag OR env)
# ---------------------------------------------------------------------------


def test_native_first_enabled_default_off(monkeypatch) -> None:
    monkeypatch.delenv("VECTOR_NATIVE_FIRST", raising=False)
    assert cli._native_first_enabled(SimpleNamespace(native_first=None)) is False


def test_native_first_enabled_via_flag(monkeypatch) -> None:
    monkeypatch.delenv("VECTOR_NATIVE_FIRST", raising=False)
    assert cli._native_first_enabled(SimpleNamespace(native_first=True)) is True


def test_native_first_enabled_via_env(monkeypatch) -> None:
    monkeypatch.setenv("VECTOR_NATIVE_FIRST", "1")
    assert cli._native_first_enabled(SimpleNamespace(native_first=None)) is True


def test_native_first_does_not_enable_native_loop(monkeypatch) -> None:
    # native-first is a SEPARATE mode: it must NOT flip _native_loop_enabled.
    monkeypatch.delenv("VECTOR_NATIVE_LOOP", raising=False)
    assert cli._native_loop_enabled(SimpleNamespace(native_loop=None, native_first=True)) is False


# ---------------------------------------------------------------------------
# (b) FALLBACK WIRING — deterministic, no sim, no LLM. A stub engine + a call-spy.
# ---------------------------------------------------------------------------


class _SpyEngine:
    """A stub engine wired into run_one_turn via a monkeypatched _build_turn_context.

    Records whether ``vgg_decompose`` was called (the fallback tripwire) and returns
    canned traces for ``run_turn_native`` / ``vgg_execute``.
    """

    def __init__(self, native_trace: ExecutionTrace, legacy_trace: ExecutionTrace) -> None:
        self._native_trace = native_trace
        self._legacy_trace = legacy_trace
        self._vgg_agent = None
        self.decompose_calls = 0
        self.native_calls = 0
        self.last_decompose_prompt: str | None = None

    def run_turn_native(self, prompt, agent=None, session=None):  # noqa: ANN001
        self.native_calls += 1
        return self._native_trace

    def is_stop(self, text):  # noqa: ANN001 — no stop skill registered
        return False

    def vgg_decompose(self, prompt):  # noqa: ANN001
        self.decompose_calls += 1
        self.last_decompose_prompt = prompt
        return self._legacy_trace.goal_tree

    def vgg_execute(self, goal_tree):  # noqa: ANN001
        return self._legacy_trace


def _wire_stub_engine(monkeypatch, engine: _SpyEngine) -> None:
    """Patch _build_turn_context so run_one_turn uses the stub engine + a session.

    verify_oracle_names is also stubbed to a fixed oracle set so the verdict gate
    runs deterministically without a real agent/world (the GROUNDED classification
    only needs at_position to be a known oracle for the acted-native case).
    """
    ctx = SimpleNamespace(engine=engine, session=None)
    monkeypatch.setattr(cli, "_build_turn_context", lambda args: ctx)
    monkeypatch.setattr(
        "vector_os_nano.vcli.cognitive.trace_store.verify_oracle_names",
        lambda agent, engine: frozenset({"at_position"}),
    )


def test_native_first_falls_back_to_legacy_when_native_no_action(monkeypatch) -> None:
    monkeypatch.delenv("VECTOR_NATIVE_LOOP", raising=False)
    native = _noaction_trace("legacy goal")  # native could not route -> no action
    legacy = _acted_trace(
        "legacy goal", strategy="walk", verify="at_position(2.0, 0.0)", verified_pose=True
    )
    engine = _SpyEngine(native_trace=native, legacy_trace=legacy)
    _wire_stub_engine(monkeypatch, engine)

    args = SimpleNamespace(
        print_prompt="legacy goal", json=False, native_first=True, native_loop=None
    )
    code = cli.run_one_turn(args)

    # Fallback fired: vgg_decompose ran on the RAW prompt, native verdict was NOT used.
    assert engine.native_calls == 1, "native-first must ATTEMPT native first"
    assert engine.decompose_calls == 1, "fallback must call legacy vgg_decompose"
    assert engine.last_decompose_prompt == "legacy goal", "fallback decompose uses the RAW prompt"
    # The emitted verdict is the LEGACY grounded trace -> verified -> exit 0.
    assert code == 0, f"legacy grounded fallback must exit 0; got {code}"


def test_native_first_uses_native_when_native_acted(monkeypatch) -> None:
    monkeypatch.delenv("VECTOR_NATIVE_LOOP", raising=False)
    native = _acted_trace(
        "covered goal", strategy="walk", verify="at_position(9.0, 3.0)", verified_pose=True
    )
    # A legacy trace that, if (wrongly) used, would FAIL — so a mis-route is caught.
    legacy = _acted_trace(
        "covered goal", strategy="walk", verify="at_position(9.0, 3.0)", verified_pose=False
    )
    engine = _SpyEngine(native_trace=native, legacy_trace=legacy)
    _wire_stub_engine(monkeypatch, engine)

    args = SimpleNamespace(
        print_prompt="covered goal", json=False, native_first=True, native_loop=None
    )
    code = cli.run_one_turn(args)

    # Native handled it: NO fallback, vgg_decompose NEVER called.
    assert engine.native_calls == 1
    assert engine.decompose_calls == 0, "native acted -> legacy decompose must NOT run"
    # The emitted verdict is the NATIVE grounded trace -> verified -> exit 0.
    assert code == 0, f"native grounded must exit 0; got {code}"


def test_print_native_escape_hatch_never_attempts_native(monkeypatch) -> None:
    # S5b: the -p path now DEFAULTS to native-first (the print cutover). The reversible
    # escape hatch VECTOR_PRINT_NATIVE=0 forces the pure-legacy path — byte-identical to
    # the pre-cutover behavior (NO native attempt; legacy decompose+execute runs).
    monkeypatch.delenv("VECTOR_NATIVE_LOOP", raising=False)
    monkeypatch.delenv("VECTOR_NATIVE_FIRST", raising=False)
    monkeypatch.setenv("VECTOR_PRINT_NATIVE", "0")
    native = _acted_trace("g", strategy="walk", verify="at_position(1,1)", verified_pose=True)
    legacy = _acted_trace("g", strategy="walk", verify="at_position(2.0, 0.0)", verified_pose=True)
    engine = _SpyEngine(native_trace=native, legacy_trace=legacy)
    _wire_stub_engine(monkeypatch, engine)

    args = SimpleNamespace(print_prompt="g", json=False, native_first=None, native_loop=None)
    code = cli.run_one_turn(args)

    assert engine.native_calls == 0, "VECTOR_PRINT_NATIVE=0 must NEVER attempt native"
    assert engine.decompose_calls == 1, "the escape hatch runs the legacy path"
    assert code == 0


def test_print_native_default_attempts_native_then_falls_back(monkeypatch) -> None:
    # S5b: with NO flags and NO escape hatch, the -p path DEFAULTS to native-first —
    # native is attempted; a no-action native trace falls THROUGH to legacy (additive).
    monkeypatch.delenv("VECTOR_NATIVE_LOOP", raising=False)
    monkeypatch.delenv("VECTOR_NATIVE_FIRST", raising=False)
    monkeypatch.delenv("VECTOR_PRINT_NATIVE", raising=False)
    native = _noaction_trace("g")  # native could not route -> falls back to legacy
    legacy = _acted_trace("g", strategy="walk", verify="at_position(2.0, 0.0)", verified_pose=True)
    engine = _SpyEngine(native_trace=native, legacy_trace=legacy)
    _wire_stub_engine(monkeypatch, engine)

    args = SimpleNamespace(print_prompt="g", json=False, native_first=None, native_loop=None)
    code = cli.run_one_turn(args)

    assert engine.native_calls == 1, "default (no escape hatch) must ATTEMPT native"
    assert engine.decompose_calls == 1, "native no-action falls THROUGH to legacy"
    assert code == 0


def test_print_native_default_does_not_preempt_explicit_native_loop(monkeypatch) -> None:
    # The default-ON print cutover must NOT pre-empt an EXPLICIT --native-loop (pure
    # native, no fallback) — block 1 is skipped so the --native-loop block owns the turn.
    monkeypatch.delenv("VECTOR_NATIVE_LOOP", raising=False)
    monkeypatch.delenv("VECTOR_NATIVE_FIRST", raising=False)
    monkeypatch.delenv("VECTOR_PRINT_NATIVE", raising=False)
    native = _acted_trace("g", strategy="walk", verify="at_position(9.0, 3.0)", verified_pose=True)
    legacy = _acted_trace("g", strategy="walk", verify="at_position(2.0, 0.0)", verified_pose=False)
    engine = _SpyEngine(native_trace=native, legacy_trace=legacy)
    _wire_stub_engine(monkeypatch, engine)

    args = SimpleNamespace(print_prompt="g", json=False, native_first=None, native_loop=True)
    code = cli.run_one_turn(args)

    # --native-loop owns it (pure native, no fallback): native ran, legacy never did.
    assert engine.native_calls == 1
    assert engine.decompose_calls == 0, "explicit --native-loop must run the pure-native block"
    assert code == 0


# ---------------------------------------------------------------------------
# (c) COVERED-GO2 ON REAL SIM — native-first routes a covered goal to native.
# ---------------------------------------------------------------------------

_SIM_TIMEOUT_SEC = 240.0

# go2 starts at (10, 3); a 1.0 m backward walk reaches within at_position tol of
# (9, 3) (mirrors the trichotomy honest-walk script).
_HONEST_SCRIPT = {
    "turns": [
        {"tool_calls": [
            {"name": "walk", "input": {"direction": "backward", "distance": 1.0, "speed": 0.3}}
        ]},
        {"tool_calls": [{"name": "verify", "input": {"expr": "at_position(9.0, 3.0)"}}]},
        {"tool_calls": [{"name": "finish", "input": {}}], "stop_reason": "end_turn"},
    ]
}


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.capability
def test_native_first_covered_go2_routes_to_native(sim_cleanup) -> None:
    """A covered walk goal under native-first ON routes to native on the REAL sim.

    Same honest-walk case as the trichotomy, but selected via --native-first (the
    native-attempt-then-fallback mode) instead of --native-loop. Native acts ->
    native verdict GROUNDED / verified True / exit 0.
    """
    pytest.importorskip("mujoco")
    from tests.harness.pty_cli import run_cli_turn

    r = run_cli_turn(
        "走到坐标 (9.0,3.0)",
        sim_go2=True,
        timeout_sec=_SIM_TIMEOUT_SEC,
        extra_args=["--headless", "--native-first"],
        tool_script=_HONEST_SCRIPT,
    )
    assert r.verified is True, f"native-first covered walk should verify; got {r.verdict}"
    assert r.exit_code == 0, f"verified covered walk must exit 0; got {r.exit_code}"
    assert r.evidence == "GROUNDED", f"got evidence={r.evidence}"
    step = r.verdict["per_step"][0]
    assert step["evidence"] == "GROUNDED"
    assert step["success"] is True and step["verify_result"] is True
    # ROUTING: native acted with the walk skill (proves native, not legacy, handled it).
    assert step["strategy"] == "walk", f"per-step strategy must be walk; got {step['strategy']}"
    assert step["verify"].startswith("at_position"), f"got verify={step['verify']}"

# NOTE (S5b): a no-flag "default cutover routes to native on the REAL sim" e2e is NOT
# added here because run_cli_turn FORCES VECTOR_NATIVE_LOOP=1 for a tool_script without
# --native-first (pty_cli.py), so it can't exercise the default _print_native_enabled
# path. The default cutover IS e2e-verified by a raw -p run on the go2 sim (DECISIONS
# D73): NO native flag -> native OWNS the turn (strategy=walk) -> honest verdict via the
# frozen spine. Test (c) above is the committed sim proof of block-1's mechanism; the
# deterministic unit tests pin that the DEFAULT triggers block 1.
