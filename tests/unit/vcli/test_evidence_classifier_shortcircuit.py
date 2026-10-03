# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Regression — the truthy-constant SHORT-CIRCUIT moat hole (2026-06-19 review).

A milestone-review skeptic found that a verify expr which short-circuits the oracle
with a truthy constant — ``at_position(99,99) or True``, ``holding_object('x') or 1``
— classified GROUNDED (the old test merely checked a predicate oracle was CALLED
somewhere in the AST) AND evaluates True in the GoalVerifier regardless of real world
state. Net: a step could earn ``verified=True`` with the robot nowhere near the goal,
just by appending ``or True`` to the verify string.

The fix (``_is_grounded_node``) requires the oracle's RESULT to GATE the verdict:
every truth-bearing leaf of the boolean expr must be an oracle term. These tests pin
that the bypass is RAN, that legitimate boolean combinations stay GROUNDED, and that
the spine evidence gate (``classify_step_evidence``) flips a short-circuit step to RAN
even when it was actor-CAUSED. The guard may only ever get STRICTER (rule 5).
"""
from __future__ import annotations

import pytest

from vector_os_nano.vcli.cognitive.evidence_classifier import classify_verify_expr

# A representative live oracle namespace (predicate + state oracles).
_ORACLES = frozenset(
    {
        "at_position", "facing", "visited", "holding_object", "arm_at_home",
        "file_exists", "path_contains", "placed_count", "describe_scene",
        "get_position", "room_coverage",
    }
)


@pytest.mark.parametrize(
    "expr",
    [
        "at_position(99.0, 99.0) or True",
        "at_position(99, 99) or 1",
        "holding_object('banana') or 1",
        "facing(0.0) or 'anything'",
        "at_position(11, 3) and True",          # dead-weight constant in an AND
        "at_position(11, 3) or (1 == 1)",       # constant-true compare disjunct
        "visited('kitchen') or not False",
        "not holding_object('banana')",         # WP02: negation of a rise predicate
        "'mug' not in describe_scene()",        # WP02: `not in` is author-satisfiable
    ],
)
def test_truthy_constant_shortcircuit_is_ran(expr: str) -> None:
    """An oracle OR'd/AND'd with a constant that can satisfy the verdict -> RAN.

    The oracle's result no longer GATES the boolean, so it proves nothing.
    """
    assert classify_verify_expr(expr, _ORACLES) == "RAN", expr


@pytest.mark.parametrize(
    "expr",
    [
        "at_position(11.0, 3.0)",
        "holding_object('banana')",
        "arm_at_home()",
        "facing(1.5708)",
        "path_contains('out.txt', 'ready')",
        "file_exists('out.txt')",
        "at_position(11, 3) or at_position(12, 3)",   # every operand oracle-gated
        "at_position(11, 3) and facing(0.0)",
        "placed_count() == 2",
        "'table' in describe_scene()",
    ],
)
def test_legitimate_oracle_gated_expr_stays_grounded(expr: str) -> None:
    """A real fix must NOT over-reject: every truth-bearing leaf is an oracle term."""
    assert classify_verify_expr(expr, _ORACLES) == "GROUNDED", expr


def test_spine_gate_flips_shortcircuit_to_ran_even_when_caused() -> None:
    """End-to-end at the spine: a CAUSED step whose verify short-circuits -> RAN.

    A genuine walk advances the actor-causation counter (CAUSED), so the R2b
    downgrade does NOT fire — the ONLY thing that must stop a ``... or True`` step
    from grading GROUNDED is the honest classifier. classify_step_evidence is the
    single gate the verdict reads; it must report RAN.
    """
    from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused
    from vector_os_nano.vcli.cognitive.trace_store import classify_step_evidence
    from vector_os_nano.vcli.cognitive.types import StepRecord, SubGoal

    sub = SubGoal(name="s0", description="walk then fake-verify", verify="at_position(99, 99) or True", strategy="walk")
    step = StepRecord(
        sub_goal_name="s0", strategy="walk", success=True,
        verify_result=True, duration_sec=0.0, actor_caused=ActorCaused.CAUSED,
    )
    assert classify_step_evidence(step, sub, _ORACLES) == "RAN"

    # Sanity: the same step with an honest bare-oracle verify IS grounded.
    honest = SubGoal(name="s0", description="walk", verify="at_position(11, 3)", strategy="walk")
    assert classify_step_evidence(step, honest, _ORACLES) == "GROUNDED"


# ---------------------------------------------------------------------------
# STEP-12 review — the Compare-MEMBERSHIP short-circuit (a constant-literal
# container satisfies `in` while the oracle is dead weight). The STEP-8 fix
# tightened or/and but never descended the Compare branch.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        "True in (at_position(99, 99), True)",     # the constant member alone satisfies it
        "1 in (at_position(99, 99), 1)",
        "at_position(99, 99) in (True, False)",    # a bool is ALWAYS in (True, False)
        "at_position(99, 99) in [False]",
        "'x' in ('x', describe_scene())",          # oracle buried in a literal tuple
        "holding_object('banana') in {True, False}",
    ],
)
def test_membership_constant_container_shortcircuit_is_ran(expr: str) -> None:
    """``<oracle> in <constant-literal container>`` is the ``... or True`` short-
    circuit hidden in an ``in`` node — the container's constants gate the verdict and
    the oracle is dead weight, so it proves nothing about the goal -> RAN (rule 5)."""
    assert classify_verify_expr(expr, _ORACLES) == "RAN", expr


@pytest.mark.parametrize(
    "expr",
    [
        "'table' in describe_scene()",     # the documented LEGIT form
    ],
)
def test_membership_against_oracle_container_stays_grounded(expr: str) -> None:
    """The legit membership form — searching an ORACLE-derived collection — is
    preserved: the oracle's returned collection gates the verdict."""
    assert classify_verify_expr(expr, _ORACLES) == "GROUNDED", expr


def test_spine_gate_flips_membership_shortcircuit_to_ran_even_when_caused() -> None:
    """End-to-end at the spine: a CAUSED walk + a membership short-circuit -> RAN."""
    from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused
    from vector_os_nano.vcli.cognitive.trace_store import classify_step_evidence
    from vector_os_nano.vcli.cognitive.types import StepRecord, SubGoal

    sub = SubGoal(name="s0", description="walk then fake-verify",
                  verify="True in (at_position(99, 99), True)", strategy="walk")
    step = StepRecord(
        sub_goal_name="s0", strategy="walk", success=True,
        verify_result=True, duration_sec=0.0, actor_caused=ActorCaused.CAUSED,
    )
    assert classify_step_evidence(step, sub, _ORACLES) == "RAN"


# ---------------------------------------------------------------------------
# STEP-14 review — the CALLABLE-wrapper bypass of the STEP-12 fix: an oracle buried
# inside a builtin container constructor (tuple()/list()/set()) or a shape reduction
# (len()/bool()) is dead weight — its value never reaches the verdict. The STEP-12 fix
# only rejected LITERAL containers.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        "True in tuple((at_position(2, 0), True))",       # builtin container constructor
        "1 in list((at_position(2, 0), 1))",
        "True in set((at_position(2, 0), True))",
        "len(tuple((at_position(2, 0), True))) == 2",     # len() discards the oracle value
        "bool(tuple((at_position(2, 0), True))) == True",
    ],
)
def test_callable_wrapper_and_reduction_shortcircuit_is_ran(expr: str) -> None:
    """An oracle BURIED inside a builtin container constructor (tuple()/list()/set()) or
    reduced over such a literal container (len()/bool() over a constant tuple) is dead
    weight — its value never gates the verdict. The STEP-12 fix only rejected LITERAL
    containers; STEP-14 closes the callable-wrapper family via the truth-bearing check —
    while a reduction DIRECTLY over an oracle (len(detect_objects())) stays honest.
    RAN (rule 5 stricter-only)."""
    assert classify_verify_expr(expr, _ORACLES) == "RAN", expr


@pytest.mark.parametrize(
    "expr",
    [
        "get_position()[0] < 5.0",          # an index over a state oracle preserves its value
        "placed_count() == 2",
    ],
)
def test_truth_bearing_oracle_compare_stays_grounded(expr: str) -> None:
    """A compare whose operand value comes DIRECTLY off a state oracle (the call, a
    constant index, or len() of it) stays GROUNDED. WP02 drops arithmetic (BinOp) from
    the verify grammar entirely, so a value-absorbing `*0`/`-1.0` can never reach the
    verdict — it does not even evaluate."""
    assert classify_verify_expr(expr, _ORACLES) == "GROUNDED", expr
