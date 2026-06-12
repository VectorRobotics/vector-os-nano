# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Level 64 — Phase B.1.2: verify-as-eval, evidence gate, and vector-eval.

Acceptance criteria (docs/agent-kernel-phase-b-plan.md, B-2):
- trace save/load round-trips; replay reproduces pass/fail.
- a verify="True"-only (sentinel) trace is NOT counted as verified evidence;
  robot traces bypass the gate.
- EvalRunner returns correct green/red and a non-zero exit on any red, no robot.

Pure kernel logic — no robot, no network, no mujoco fixtures.
"""
from __future__ import annotations

from pathlib import Path

from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.trace_store import (
    evidence_passed,
    load_trace,
    replay,
    save_trace,
    step_evidence_ok,
)
from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)
from vector_os_nano.vcli.eval_runner import EvalRunner, main as eval_main
from vector_os_nano.vcli.worlds.dev import dev_verify_namespace


# ---------------------------------------------------------------------------
# Trace builders
# ---------------------------------------------------------------------------


def _trace(
    *,
    verify: str = "file_exists('a.txt')",
    success: bool = True,
    verify_result: bool = True,
    goal: str = "make a.txt",
) -> ExecutionTrace:
    sg = SubGoal(
        name="step1",
        description="create a.txt",
        verify=verify,
        strategy="tool_call",
        strategy_params={"tool": "file_write", "args": {"file_path": "a.txt", "content": "x"}},
        depends_on=(),
    )
    step = StepRecord(
        sub_goal_name="step1",
        strategy="tool_call",
        success=success,
        verify_result=verify_result,
        duration_sec=0.05,
    )
    return ExecutionTrace(
        goal_tree=GoalTree(goal=goal, sub_goals=(sg,)),
        steps=(step,),
        success=success,
        total_duration_sec=0.05,
    )


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path) -> None:
    tr = _trace()
    path = save_trace(tr, tmp_path / "t.json")
    assert path.exists()

    reloaded = load_trace(path)
    assert reloaded.goal_tree.goal == tr.goal_tree.goal
    assert reloaded.success is True
    assert isinstance(reloaded.goal_tree.sub_goals, tuple)
    assert isinstance(reloaded.steps, tuple)
    assert reloaded.goal_tree.sub_goals[0].verify == "file_exists('a.txt')"
    assert reloaded.goal_tree.sub_goals[0].strategy_params["tool"] == "file_write"
    assert reloaded.steps[0].verify_result is True
    # depends_on must survive as a tuple (JSON list -> tuple)
    assert isinstance(reloaded.goal_tree.sub_goals[0].depends_on, tuple)


def test_save_default_path_under_dot_vector(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    # re-evaluate the module default by writing with path=None
    import importlib

    import vector_os_nano.vcli.cognitive.trace_store as ts

    importlib.reload(ts)
    path = ts.save_trace(_trace())
    assert path.parent == tmp_path / ".vector" / "traces"
    assert path.exists()
    importlib.reload(ts)  # restore default for other tests


def test_load_tolerates_minimal_dict(tmp_path: Path) -> None:
    p = tmp_path / "min.json"
    p.write_text('{"goal_tree": {"goal": "g"}, "success": false}')
    tr = load_trace(p)
    assert tr.goal_tree.goal == "g"
    assert tr.success is False
    assert tr.steps == ()


# ---------------------------------------------------------------------------
# Replay reproduces pass/fail
# ---------------------------------------------------------------------------


def test_replay_reproduces_pass(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("hello")
    verifier = GoalVerifier(dev_verify_namespace())
    assert replay(_trace(verify="file_exists('a.txt')"), verifier) is True


def test_replay_reproduces_fail(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # a.txt does NOT exist
    verifier = GoalVerifier(dev_verify_namespace())
    assert replay(_trace(verify="file_exists('a.txt')"), verifier) is False


def test_replay_sentinel_only_is_not_replayable() -> None:
    verifier = GoalVerifier(dev_verify_namespace())
    # Only a "True" sentinel verify -> nothing deterministic to replay -> False.
    assert replay(_trace(verify="True"), verifier) is False


# ---------------------------------------------------------------------------
# Evidence gate
# ---------------------------------------------------------------------------


def test_evidence_passed_real_predicate() -> None:
    assert evidence_passed(_trace(verify="file_exists('a.txt')"), is_robot=False) is True


def test_evidence_rejects_true_sentinel() -> None:
    assert evidence_passed(_trace(verify="True"), is_robot=False) is False


def test_evidence_rejects_empty_verify() -> None:
    assert evidence_passed(_trace(verify=""), is_robot=False) is False


def test_evidence_no_world_level_bypass() -> None:
    # Invariant II (design review 2026-06-12 #4): is_robot no longer bypasses
    # the gate — a sentinel verify is not evidence in ANY world. Async motor
    # skills are excused per-step via the world-declared exemption set.
    assert evidence_passed(_trace(verify="True"), is_robot=True) is False
    assert evidence_passed(
        _trace(verify="True"),
        exempt_strategies=frozenset({_trace(verify="True").steps[0].strategy}),
    ) is True


def test_evidence_requires_verify_result_true() -> None:
    tr = _trace(verify="file_exists('a.txt')", success=True, verify_result=False)
    assert evidence_passed(tr, is_robot=False) is False


# ---------------------------------------------------------------------------
# EvalRunner green / red + exit semantics
# ---------------------------------------------------------------------------


def test_eval_runner_green_on_evidenced_success() -> None:
    runner = EvalRunner(run_task=lambda _t: _trace(verify="file_exists('a.txt')"))
    res = runner.run_case({"task": "make a.txt"})
    assert res.passed is True


def test_eval_runner_red_on_sentinel_only() -> None:
    runner = EvalRunner(run_task=lambda _t: _trace(verify="True"))
    res = runner.run_case({"task": "do nothing real"})
    assert res.passed is False
    assert "evidence" in res.detail


def test_eval_runner_red_on_execution_failure() -> None:
    runner = EvalRunner(run_task=lambda _t: _trace(success=False, verify_result=False))
    res = runner.run_case({"task": "fail"})
    assert res.passed is False
    assert "failed" in res.detail


def test_eval_runner_red_on_no_trace() -> None:
    runner = EvalRunner(run_task=lambda _t: None)
    res = runner.run_case({"task": "declined"})
    assert res.passed is False
    assert "no trace" in res.detail


def test_eval_runner_expect_predicate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("ready")
    verifier = GoalVerifier(dev_verify_namespace())
    runner = EvalRunner(run_task=lambda _t: _trace(), verifier=verifier)

    ok = runner.run_case({"task": "t", "expect": "path_contains('a.txt', 'ready')"})
    assert ok.passed is True

    bad = runner.run_case({"task": "t", "expect": "path_contains('a.txt', 'MISSING')"})
    assert bad.passed is False
    assert "expect failed" in bad.detail


def test_eval_report_aggregate_and_exit_code() -> None:
    def run_task(task: str):
        return _trace(verify="file_exists('a.txt')") if task == "good" else _trace(verify="True")

    runner = EvalRunner(run_task=run_task)
    report = runner.run([{"task": "good"}, {"task": "bad"}])
    assert report.total == 2
    assert report.green == 1
    assert report.all_passed is False


def test_eval_main_exit_code(tmp_path: Path, monkeypatch, capsys) -> None:
    """main() returns 0 only when every case is green; non-zero otherwise."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("hi")

    # Inject a fake engine so main() does no network / no real backend.
    import vector_os_nano.vcli.eval_runner as er

    monkeypatch.setattr(er, "build_dev_engine", lambda allow_ask=False: object())
    monkeypatch.setattr(
        er, "_engine_run_task",
        lambda _engine: (lambda task: _trace(verify="file_exists('a.txt')")
                         if task == "good" else _trace(verify="True")),
    )

    green_file = tmp_path / "green.json"
    green_file.write_text('[{"task": "good"}]')
    assert eval_main([str(green_file)]) == 0

    mixed_file = tmp_path / "mixed.json"
    mixed_file.write_text('[{"task": "good"}, {"task": "bad"}]')
    assert eval_main([str(mixed_file)]) == 1


# ---------------------------------------------------------------------------
# W1.1 — per-step learning-tier reward gate (step_evidence_ok)
#
# step_evidence_ok itself is the predicate (T1/T2/T3 below pin it directly). The
# COMPOSITION (success=step.success AND step_evidence_ok) and is_robot now live
# in ONE place: GoalExecutor._record_strategy_stats. The discriminating tests
# (further down) drive that helper with a stub StrategyStats and assert the
# captured ``success`` — so a visual-override / sentinel "success" cannot train
# the strategy bandit, while robot learning stays exactly == step.success.
# ---------------------------------------------------------------------------


def _step_and_goal(
    *,
    verify: str = "file_exists('a.txt')",
    success: bool = True,
    verify_result: bool = True,
    visual_override: bool = False,
    answer_only: bool = False,
    strategy: str = "tool_call",
) -> tuple[StepRecord, SubGoal]:
    sg = SubGoal(
        name="step1",
        description="do a thing",
        verify=verify,
        strategy=strategy,
        answer_only=answer_only,
    )
    step = StepRecord(
        sub_goal_name="step1",
        strategy=strategy,
        success=success,
        verify_result=verify_result,
        duration_sec=0.05,
        visual_override=visual_override,
    )
    return step, sg


def test_w1_1_sentinel_verify_blocks_evidence_dev() -> None:
    # T1: sub_goal.verify is the sentinel 'True' -> no deterministic evidence.
    step, sg = _step_and_goal(verify="True", success=True, verify_result=True)
    assert step_evidence_ok(step, sg, False) is False


def test_w1_1_visual_override_blocks_evidence_dev() -> None:
    # T2: a VLM visual override is not deterministic evidence.
    step, sg = _step_and_goal(
        verify="file_exists('a.txt')", success=True, verify_result=True, visual_override=True
    )
    assert step_evidence_ok(step, sg, False) is False


def test_w1_1_real_predicate_has_evidence_dev() -> None:
    # T3: a real predicate + verify_result True + no visual override -> evidence.
    step, sg = _step_and_goal(
        verify="file_exists('a.txt')", success=True, verify_result=True, visual_override=False
    )
    assert step_evidence_ok(step, sg, False) is True


def test_w1_1_no_world_level_bypass_only_exempt_strategies() -> None:
    # Invariant II: is_robot no longer bypasses; the world's bounded
    # per-step exemption set is the only excuse for a sentinel verify.
    passing, sg = _step_and_goal(verify="True", success=True, verify_result=True)
    assert step_evidence_ok(passing, sg, True) is False
    exempt = frozenset({passing.strategy.removesuffix("_skill")})
    assert step_evidence_ok(passing, sg, exempt_strategies=exempt) is True
    # Exemption never launders a failed verify (gate, not reward).
    failed, sg_f = _step_and_goal(verify="True", success=False, verify_result=False)
    assert step_evidence_ok(failed, sg_f, True, exempt_strategies=exempt) is False


# ---------------------------------------------------------------------------
# W1.1 — the reward gate is CENTRALIZED on GoalExecutor._record_strategy_stats.
# These tests are the discriminating ones: they capture the recorded ``success``
# via a stub StrategyStats and prove the gate fires (and that it would NOT have
# fired under the old plain ``success=step.success`` code).
# ---------------------------------------------------------------------------


class _CapturingStats:
    """Minimal StrategyStats stub: captures the recorded ``success`` value."""

    def __init__(self) -> None:
        self.recorded: list[bool] = []

    def record(self, *, strategy_name, sub_goal_name, success, duration_sec) -> None:
        self.recorded.append(success)


def _executor(*, is_robot: bool, stats, evidence_exempt=frozenset()):
    from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor

    return GoalExecutor(
        strategy_selector=None,
        verifier=None,
        stats=stats,
        is_robot=is_robot,
        evidence_exempt=evidence_exempt,
    )


def test_w1_1_record_gate_blocks_sentinel_success_dev() -> None:
    # GAP regression (foreach/fallback bypass): a sentinel verify='True' step that
    # "succeeded" must record success=False — even though step.success is True.
    stats = _CapturingStats()
    step, sg = _step_and_goal(verify="True", success=True, verify_result=True)
    _executor(is_robot=False, stats=stats)._record_strategy_stats(step, sg)
    assert stats.recorded == [False]
    # Discriminating vs the old code, which recorded plain step.success:
    assert step.success is True


def test_w1_1_record_gate_blocks_visual_override_dev() -> None:
    stats = _CapturingStats()
    step, sg = _step_and_goal(
        verify="file_exists('a.txt')", success=True, verify_result=True, visual_override=True
    )
    _executor(is_robot=False, stats=stats)._record_strategy_stats(step, sg)
    assert stats.recorded == [False]
    assert step.success is True  # would have been True under plain step.success


def test_w1_1_record_gate_rewards_real_evidence_dev() -> None:
    stats = _CapturingStats()
    step, sg = _step_and_goal(verify="file_exists('a.txt')", success=True, verify_result=True)
    _executor(is_robot=False, stats=stats)._record_strategy_stats(step, sg)
    assert stats.recorded == [True]


def test_w1_1_record_gate_exempt_strategy_keeps_learning_signal() -> None:
    # Invariant II: learning on a sentinel motor step survives only through
    # the world-declared exemption set — not a world-level bypass.
    stats = _CapturingStats()
    passing, sg = _step_and_goal(verify="True", success=True, verify_result=True)
    exempt = frozenset({passing.strategy.removesuffix("_skill")})
    _executor(is_robot=True, stats=stats,
              evidence_exempt=exempt)._record_strategy_stats(passing, sg)
    assert stats.recorded == [True]
    # A NON-exempt sentinel step records False now (the moat tightened)...
    stats2 = _CapturingStats()
    _executor(is_robot=True, stats=stats2)._record_strategy_stats(passing, sg)
    assert stats2.recorded == [False]
    # ...and a FAILED step records False regardless (AND step.success holds).
    stats3 = _CapturingStats()
    failed, sg_f = _step_and_goal(verify="True", success=False, verify_result=False)
    _executor(is_robot=True, stats=stats3,
              evidence_exempt=exempt)._record_strategy_stats(failed, sg_f)
    assert stats3.recorded == [False]


def test_w1_1_record_gate_noops_without_stats() -> None:
    # No stats attached -> the chokepoint is a safe no-op.
    step, sg = _step_and_goal(verify="True", success=True, verify_result=True)
    _executor(is_robot=False, stats=None)._record_strategy_stats(step, sg)  # no raise
