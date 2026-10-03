# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Level 64 — Phase B.1.2: verify-as-eval, evidence gate, and vector-eval.

Acceptance criteria (docs/agent-kernel-phase-b-plan.md, B-2):
- trace save/load round-trips; the evidence gate grades pass/fail.
- a verify="True"-only (sentinel) trace is NOT counted as verified evidence;
  R1: robot traces NO LONGER bypass the gate — they too require deterministic
  evidence (the old ``if is_robot: return True`` short-circuit is deleted).
- EvalRunner returns correct green/red and a non-zero exit on any red, no robot.

Pure kernel logic — no robot, no network, no mujoco fixtures.
"""
from __future__ import annotations

from pathlib import Path

from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.trace_store import (
    evidence_passed,
    load_trace,
    save_trace,
    step_evidence_ok,
)
from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)
from vector_os_nano.vcli.cognitive.oracle_registry import oracle
from vector_os_nano.vcli.eval_runner import EvalRunner, main as eval_main
from vector_os_nano.vcli.worlds.dev import dev_verify_namespace


# The live verify-namespace callable names a dev+robot world exposes, passed to
# the R1 evidence gate (replaces the old ``is_robot`` flag). file_exists /
# path_contains anchor the dev predicates; the robot oracles (at_position,
# arm_at_home, ...) anchor the sim path. Mirrors verify_oracle_names(agent, engine).
ORACLES = frozenset({
    "at_position", "facing", "visited", "holding_object", "arm_at_home",
    "file_exists", "path_contains", "get_position", "get_heading",
    "describe_scene", "detect_objects", "placed_count", "nearest_room",
    "objects_in_room", "find_object", "room_coverage",
})


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
# Evidence gate: pass/fail over deterministic predicates
# ---------------------------------------------------------------------------


def test_evidence_passed_real_predicate() -> None:
    assert evidence_passed(_trace(verify="file_exists('a.txt')"), ORACLES) is True


def test_evidence_rejects_true_sentinel() -> None:
    assert evidence_passed(_trace(verify="True"), ORACLES) is False


def test_evidence_rejects_empty_verify() -> None:
    assert evidence_passed(_trace(verify=""), ORACLES) is False


def test_evidence_robot_world_now_requires_evidence() -> None:
    # R1 FLIP: the old ``if is_robot: return True`` bypass is GONE. A robot motor
    # step with the sentinel verify="True" now classifies RAN, not GROUNDED, so it
    # no longer auto-passes the done-gate — every world must back success with
    # deterministic evidence.
    assert evidence_passed(_trace(verify="True"), ORACLES) is False


def test_evidence_requires_verify_result_true() -> None:
    tr = _trace(verify="file_exists('a.txt')", success=True, verify_result=False)
    assert evidence_passed(tr, ORACLES) is False


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
# COMPOSITION (success=step.success AND step_evidence_ok) now lives in ONE place:
# GoalExecutor._record_strategy_stats (R1 — no is_robot). The discriminating tests
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
    assert step_evidence_ok(step, sg, ORACLES) is False


def test_w1_1_visual_override_blocks_evidence_dev() -> None:
    # T2: a VLM visual override is not deterministic evidence.
    step, sg = _step_and_goal(
        verify="file_exists('a.txt')", success=True, verify_result=True, visual_override=True
    )
    assert step_evidence_ok(step, sg, ORACLES) is False


def test_w1_1_real_predicate_has_evidence_dev() -> None:
    # T3: a real predicate + verify_result True + no visual override -> evidence.
    step, sg = _step_and_goal(
        verify="file_exists('a.txt')", success=True, verify_result=True, visual_override=False
    )
    assert step_evidence_ok(step, sg, ORACLES) is True


def test_w1_1_robot_world_now_requires_evidence_gate() -> None:
    # R1 FLIP: the robot bypass is GONE. A robot motor step with verify="True" now
    # classifies RAN, not GROUNDED — step_evidence_ok is False for BOTH the passing
    # sentinel step and the failed one (FAILED is also not GROUNDED).
    passing, sg = _step_and_goal(verify="True", success=True, verify_result=True)
    assert step_evidence_ok(passing, sg, ORACLES) is False
    failed, sg_f = _step_and_goal(verify="True", success=False, verify_result=False)
    assert step_evidence_ok(failed, sg_f, ORACLES) is False


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


def _executor(*, stats):
    from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor

    # GoalExecutor no longer takes is_robot (R1). With verifier=None there is no
    # live verify namespace, so _verify_oracle_names() is the empty set (fail
    # closed) — every step classifies RAN unless its evidence is independently
    # GROUNDED. The dev tests below drive sentinel/visual-override/real-predicate
    # cases; with an empty namespace a bare file_exists() would NOT be GROUNDED, so
    # the "rewards real evidence" test attaches a stub verifier exposing file_exists.
    return GoalExecutor(
        strategy_selector=None,
        verifier=stats and _Verifier(),
        stats=stats,
    )


class _Verifier:
    """Stub GoalVerifier exposing the live verify namespace the executor reads via
    ``_namespace`` (the SAME keys verify_oracle_names single-sources)."""

    def __init__(self) -> None:
        # WP02: only @oracle-tagged callables count as evidence. Tag each stub with
        # its scope (dev file predicates vs robot GT) so the reward gate sees them.
        _dev = {"file_exists", "grep_count", "path_contains"}
        self._namespace = {
            name: oracle(scope="dev" if name in _dev else "robot")(lambda *a, **k: False)
            for name in ORACLES
        }


def test_w1_1_record_gate_blocks_sentinel_success_dev() -> None:
    # GAP regression (foreach/fallback bypass): a sentinel verify='True' step that
    # "succeeded" must record success=False — even though step.success is True.
    stats = _CapturingStats()
    step, sg = _step_and_goal(verify="True", success=True, verify_result=True)
    _executor(stats=stats)._record_strategy_stats(step, sg)
    assert stats.recorded == [False]
    # Discriminating vs the old code, which recorded plain step.success:
    assert step.success is True


def test_w1_1_record_gate_blocks_visual_override_dev() -> None:
    stats = _CapturingStats()
    step, sg = _step_and_goal(
        verify="file_exists('a.txt')", success=True, verify_result=True, visual_override=True
    )
    _executor(stats=stats)._record_strategy_stats(step, sg)
    assert stats.recorded == [False]
    assert step.success is True  # would have been True under plain step.success


def test_w1_1_record_gate_rewards_real_evidence_dev() -> None:
    stats = _CapturingStats()
    step, sg = _step_and_goal(verify="file_exists('a.txt')", success=True, verify_result=True)
    _executor(stats=stats)._record_strategy_stats(step, sg)
    assert stats.recorded == [True]


def test_w1_1_record_gate_robot_no_longer_collapses_to_step_success() -> None:
    # R1 FLIP: with the is_robot bypass GONE, a robot sentinel motor step (verify=
    # "True") now classifies RAN, so the recorded reward is False even though
    # step.success is True — reward parity with the done-gate is intentional.
    stats = _CapturingStats()
    passing, sg = _step_and_goal(verify="True", success=True, verify_result=True)
    _executor(stats=stats)._record_strategy_stats(passing, sg)
    assert stats.recorded == [False]
    assert passing.success is True  # step.success was True; the gate still blocked it
    # A FAILED motor step still records False (FAILED is not GROUNDED).
    stats2 = _CapturingStats()
    failed, sg_f = _step_and_goal(verify="True", success=False, verify_result=False)
    _executor(stats=stats2)._record_strategy_stats(failed, sg_f)
    assert stats2.recorded == [False]


def test_w1_1_record_gate_noops_without_stats() -> None:
    # No stats attached -> the chokepoint is a safe no-op.
    step, sg = _step_and_goal(verify="True", success=True, verify_result=True)
    _executor(stats=None)._record_strategy_stats(step, sg)  # no raise
