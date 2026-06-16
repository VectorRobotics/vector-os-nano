# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Invariant I — pre-execution verify baseline (design review 2026-06-12, #1).

The kernel evaluated verify only AFTER execution, so 'was already true' and
'became true' were indistinguishable — any goal whose predicate held in the
initial state degenerated to a silent no-op PASS ('走到厨房' seed). The
executor now evaluates every NON-TRIVIAL predicate BEFORE dispatching the
strategy and records it on StepRecord.pre_satisfied (additive, rule 6); the
step view surfaces it honestly.
"""
from __future__ import annotations

from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.strategy_selector import StrategyResult
from vector_os_nano.vcli.cognitive.types import StepRecord, SubGoal


class _FixedSelector:
    def select(self, sub_goal):
        return StrategyResult("primitive", "noop", {})


def _executor(ns: dict, noop=lambda: True) -> GoalExecutor:
    return GoalExecutor(
        strategy_selector=_FixedSelector(),
        verifier=GoalVerifier(ns),
        primitives={"noop": noop},
    )


def _sg(verify: str) -> SubGoal:
    return SubGoal(name="nav", description="nav", verify=verify,
                   strategy="noop", strategy_params={})


class TestPreSatisfiedField:
    def test_additive_default_false(self):
        # rule 6: existing constructors unaffected.
        r = StepRecord("s", "x", True, True, 0.0)
        assert r.pre_satisfied is False


class TestPreSatisfiedEvaluation:
    def test_pre_true_nontrivial_flagged(self):
        ex = _executor({"at_position": lambda x, y, tol=0.2: True})
        rec = ex._execute_sub_goal(_sg("at_position(1.0, 2.0, 1.6)"))
        assert rec.success is True
        assert rec.pre_satisfied is True

    def test_pre_false_becomes_true_not_flagged(self):
        # The predicate flips during execution — genuinely earned PASS.
        state = {"there": False}

        def at_position(x, y, tol=0.2):
            return state["there"]

        ex = _executor(
            {"at_position": at_position},
            noop=lambda: state.__setitem__("there", True) or True,
        )
        rec = ex._execute_sub_goal(_sg("at_position(1.0, 2.0)"))
        assert rec.success is True
        assert rec.pre_satisfied is False

    def test_trivial_true_sentinel_never_flagged(self):
        # 'True' carries no evidence — pre-eval on it is meaningless noise.
        ex = _executor({})
        rec = ex._execute_sub_goal(_sg("True"))
        assert rec.success is True
        assert rec.pre_satisfied is False

    def test_pre_eval_error_is_failsafe_false(self):
        # Predicates referencing step_output() cannot evaluate pre-exec —
        # must not flag, must not break execution.
        ex = _executor({"check": lambda v: v > 0})
        rec = ex._execute_sub_goal(_sg("check(step_output()['n'])"))
        assert rec.pre_satisfied is False

    def test_pre_eval_step_output_does_not_namerror(self, caplog):
        """R11 regression — step_output is a kernel contract the executor must
        ALWAYS inject (Rule 4). A step_output()-verify evaluated pre-exec
        (exec_output is None) must NOT raise 'name step_output is not defined'
        and be swallowed to a false False (Rule 5). It evaluates HONESTLY
        against the bound-None output instead.
        """
        import logging

        ex = _executor({})
        with caplog.at_level(logging.WARNING):
            rec = ex._execute_sub_goal(_sg("len(step_output('objects')) > 0"))
        # Honest baseline: None has no len -> False (step observed nothing),
        # NOT a NameError. The exact R11 string must never appear.
        assert rec.pre_satisfied is False
        assert "name 'step_output' is not defined" not in caplog.text

    def test_verify_and_value_none_path_injects_step_output(self):
        """The gate itself: exec_output=None still binds step_output (kernel
        contract), so step_output() resolves to None — not a NameError."""
        ex = _executor({})
        ok, val = ex._verify_and_value("step_output() is None", None)
        assert ok is True and val is True  # bound to None, honestly evaluated

    def test_verify_and_value_binds_real_output_when_present(self):
        """Regression: a non-None exec_output still flows into the verify —
        the close-the-loop path (Rule 4) is unchanged."""
        ex = _executor({})
        ok, val = ex._verify_and_value(
            "len(step_output('objects')) > 0", {"objects": [1, 2]}
        )
        assert ok is True and val is True

    def test_pre_satisfied_never_passes_a_step_output_verify(self):
        """Moat guard (Rule 5): step_output is THIS step's own output, which
        does not exist before the step runs, so the PRE-exec baseline must
        ALWAYS be False for any step_output verify. Without this guard, a
        clearing/removal predicate like ``not step_output('items')`` evaluates
        True against bound-None pre-output and the step is SKIPPED — a
        campaign-#3 false-PASS. These must never pre-satisfy."""
        ex = _executor({})
        for expr in ("not step_output('items')",
                     "step_output() is None",
                     "step_output('count') == 0",
                     "len(step_output('objects')) > 0"):
            assert ex._pre_satisfied(expr) is False, expr
            # and end-to-end: the step is NOT skipped as pre-satisfied
            rec = ex._execute_sub_goal(_sg(expr))
            assert rec.pre_satisfied is False, expr

    def test_verify_fail_record_carries_pre_satisfied(self):
        # pre-true but post-false (state regressed mid-step): the failed
        # record still carries the baseline for the replan context.
        state = {"ok": True}

        def pred():
            return state["ok"]

        ex = _executor(
            {"pred": pred},
            noop=lambda: state.__setitem__("ok", False) or True,
        )
        rec = ex._execute_sub_goal(_sg("pred()"))
        assert rec.success is False
        assert rec.pre_satisfied is True


class TestStepViewSurface:
    def test_view_carries_and_renders_pre_satisfied(self):
        from vector_os_nano.vcli.cognitive.observation import (
            render_step_view,
            step_view as step_export_view,
        )

        rec = StepRecord("nav", "navigate_to", True, True, 0.0,
                         pre_satisfied=True)
        view = step_export_view(rec)
        assert view["pre_satisfied"] is True
        line = render_step_view(view, "at_position(1, 2, 1.6)")
        assert "already satisfied" in line

    def test_unflagged_step_renders_clean(self):
        from vector_os_nano.vcli.cognitive.observation import (
            render_step_view,
            step_view as step_export_view,
        )

        rec = StepRecord("nav", "navigate_to", True, True, 0.0)
        line = render_step_view(step_export_view(rec))
        assert "already satisfied" not in line
