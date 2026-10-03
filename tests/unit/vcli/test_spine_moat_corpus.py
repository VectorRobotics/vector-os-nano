# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""WP02 acceptance — the honest-verify moat as a table-driven adversarial corpus,
a mutation-sensitivity check, and the GoalVerifier sandbox-escape regression.

This is THE moat test: every FG-1..FG-4 false-green from spine.md and every ledger
false-green must classify RAN; the previously-honest predicates must stay GROUNDED;
and loosening any one guard must let the corpus detect the regression (so a future
edit that re-opens a hole fails here, not silently in production).
"""
from __future__ import annotations

import pytest

from vector_os_nano.vcli.cognitive import evidence_classifier as ec
from vector_os_nano.vcli.cognitive.evidence_classifier import classify_verify_expr
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier

# A robot+arm world's EVIDENCE oracle set (what verify_oracle_names returns there).
# Dev file predicates are DELIBERATELY absent — they are dev-scope, never evidence in
# a robot world (FG-4 / dev-oracles-in-robot-world).
ROBOT_ORACLES = frozenset({
    "at_position", "facing", "visited", "holding_object", "arm_at_home",
    "placed_count", "detect_objects", "describe_scene", "get_position", "get_heading",
})

# ---------------------------------------------------------------------------
# The adversarial corpus — every one MUST classify RAN (>= 40 expressions).
# ---------------------------------------------------------------------------

CORPUS_RAN: list[str] = [
    # FG-1 value-absorbing constructs (BinOp / max / min / abs / *0 / bitwise)
    "(at_position(99, 99) | True) == True",
    "at_position(99, 99) * 0 == 0",
    "at_position(99, 99) & False == False",
    "max(at_position(99, 99), 1) == 1",
    "min(at_position(99, 99), 0) == 0",
    "abs(at_position(99, 99)) >= 0",
    "at_position(99, 99) + 1 == 1",
    "len(tuple((at_position(2, 0), True))) == 2",
    "bool(tuple((at_position(2, 0), True))) == True",
    "True in (at_position(99, 99), True)",
    "1 in [at_position(99, 99), 1]",
    "at_position(99, 99) in (True, False)",
    "True in tuple((at_position(2, 0), True))",
    # FG-2 vacuous predicates / negations / tautologies
    "not at_position(99, 99)",
    "not holding_object('banana')",
    "not holding_object()",
    "at_position(99, 99) < 2",
    "at_position(99, 99) >= 0",
    "get_position() is not None",
    "get_position() is None",
    "holding_object('x') is not None",
    "len(detect_objects()) >= 0",
    "len(detect_objects()) > -1",
    "placed_count() >= 0",
    "placed_count() != -1",
    "describe_scene() != ''",
    "'' in describe_scene()",
    "'mug' not in describe_scene()",
    "get_position() == get_position()",
    "at_position(99, 99) == at_position(0, 0)",
    "len(get_position()) == len(get_position())",
    # FG-3 tolerance inflation (classifier sees the literal tol; oracle clamps it,
    # but the classifier also refuses an over-cap literal at_position).
    # (The oracle-clamp half is covered in test_tolerance_is_clamped_in_oracle.)
    # FG-4 dev predicates in a robot world (not in ROBOT_ORACLES -> no oracle call)
    "file_exists('.')",
    "file_exists('README.md')",
    "path_contains('README.md', 'x')",
    "grep_count('import', '.') > 0",
    "tests_pass()",
    # no-oracle / constant-only
    "",
    "True",
    "1 == 1",
    "len([1, 2, 3]) == 3",
    "2 > 1",
    "'a' in 'abc'",
    # bare state oracle (always truthy) / oracle-arg-not-literal
    "get_position()",
    "describe_scene()",
    "at_position(get_position()[0], get_position()[1])",
    "holding_object(describe_scene())",
    # syntax error
    "this is not valid python (",
    # sandbox-escape shaped (also blocked by the verifier grammar)
    "().__class__.__mro__",
]

# ---------------------------------------------------------------------------
# The honest set — real predicates that MUST stay GROUNDED (byte-for-byte).
# ---------------------------------------------------------------------------

HONEST_GROUNDED: list[str] = [
    "at_position(11.0, 3.0)",                       # go2 walk
    "at_position(2.0, 0.0)",
    "facing(1.5708)",                               # go2 turn
    "facing(0.0)",
    "visited('kitchen')",
    "at_position(11, 3) and facing(0.0)",
    "at_position(11, 3) or at_position(12, 3)",     # every disjunct oracle-gated
    "holding_object('banana')",                     # arm grasp
    "holding_object()",
    "arm_at_home()",
    "holding_object('banana') and arm_at_home()",
    "placed_count() == 2",
    "placed_count((0.0, 0.0, 0.5, 0.5)) >= 1",
    "len(detect_objects()) > 0",
    "len(detect_objects()) >= 1",
    "'table' in describe_scene()",
    "get_position()[0] < 5.0",
    "at_position(11, 3) == True",
]


@pytest.mark.parametrize("expr", CORPUS_RAN)
def test_adversarial_corpus_never_grounded(expr: str) -> None:
    assert classify_verify_expr(expr, ROBOT_ORACLES) == "RAN", expr


@pytest.mark.parametrize("expr", HONEST_GROUNDED)
def test_honest_predicates_stay_grounded(expr: str) -> None:
    assert classify_verify_expr(expr, ROBOT_ORACLES) == "GROUNDED", expr


def test_corpus_is_large_enough() -> None:
    # The acceptance requires >= 40 adversarial expressions.
    assert len(CORPUS_RAN) >= 40


# ---------------------------------------------------------------------------
# Mutation sensitivity — loosen ONE guard at a time; a corpus entry must flip to
# GROUNDED, proving the corpus can detect a regression in that guard.
# ---------------------------------------------------------------------------


def test_mutation_absorption_guard(monkeypatch) -> None:
    """Loosen the absorption/short-circuit guard (a Compare is grounded if an oracle
    is called ANYWHERE in it — the pre-WP structural check). A value-absorbing entry
    must flip to GROUNDED."""
    import ast

    def _loose(node, oracle_names):  # noqa: ANN001
        if isinstance(node, ast.BoolOp):
            return all(_loose(v, oracle_names) for v in node.values)
        if isinstance(node, ast.Compare):
            return any(ec._oracle_call(c, oracle_names) for c in ast.walk(node))
        return ec._predicate_call(node, oracle_names)

    monkeypatch.setattr(ec, "_is_grounded_node", _loose)
    flipped = [e for e in CORPUS_RAN if classify_verify_expr(e, ROBOT_ORACLES) == "GROUNDED"]
    assert "(at_position(99, 99) | True) == True" in flipped


def test_mutation_vacuous_compare_guard(monkeypatch) -> None:
    """Loosen the vacuous-compare guard (accept ANY compare with a state term on one
    side, regardless of op / bound). A ``>= 0`` / ``is not None`` entry must flip."""
    import ast

    def _loose_compare(node, oracle_names):  # noqa: ANN001
        operands = [node.left, *node.comparators]
        return any(ec._state_term(o, oracle_names) for o in operands)

    monkeypatch.setattr(ec, "_grounded_compare", _loose_compare)
    flipped = [e for e in CORPUS_RAN if classify_verify_expr(e, ROBOT_ORACLES) == "GROUNDED"]
    assert "len(detect_objects()) >= 0" in flipped


def test_mutation_scope_guard() -> None:
    """Loosen the scope guard (a dev predicate leaks into the robot oracle set). The
    dev file predicate must then flip to GROUNDED — this is what verify_oracle_names'
    scope filter prevents."""
    leaked = ROBOT_ORACLES | {"file_exists"}
    assert classify_verify_expr("file_exists('.')", ROBOT_ORACLES) == "RAN"
    assert classify_verify_expr("file_exists('.')", leaked) == "GROUNDED"


# ---------------------------------------------------------------------------
# GoalVerifier sandbox-escape regression (ledger legacy-cognitive:missed0).
# ---------------------------------------------------------------------------

_ESCAPES = [
    # Reach the real builtins through a generator frame (walrus + comprehension).
    '(g := (g.gi_frame.f_back.f_back.f_builtins for _ in [0])) and '
    '[y for y in g][0]["open"]("/etc/hostname")',
    # Attribute access to the object graph.
    "().__class__.__mro__",
    "[].__class__",
    # Lambda / comprehension were evaluation vectors.
    "lambda: 1",
    "[x for x in range(3)]",
    # Arithmetic allocation bomb shape.
    "'x' * 10",
]


@pytest.mark.parametrize("expr", _ESCAPES)
def test_goal_verifier_rejects_sandbox_escape(expr: str) -> None:
    """Every escape-shaped expression is refused by the AST allowlist -> False, and
    never reaches a real builtin. ``evaluate`` returns (False, None)."""
    v = GoalVerifier({})
    ok, raw = v.evaluate(expr)
    assert ok is False and raw is None, expr


def test_goal_verifier_timeout_poisons_then_refuses(monkeypatch) -> None:
    """A thread-path evaluation that outlives its timeout marks the verifier poisoned;
    the next evaluate refuses rather than running concurrently with a runaway
    (spine:thread-timeout-runaway). The runaway is released in finally — no leak."""
    import threading

    import vector_os_nano.vcli.cognitive.goal_verifier as gv

    release = threading.Event()

    def _slow(**_):  # noqa: ANN001, ANN202
        release.wait(10)
        return True

    v = GoalVerifier({"slow": _slow})
    # Force the watchdog path (non-signal) and a tiny timeout.
    monkeypatch.setattr(gv.GoalVerifier, "_eval_with_timeout", gv.GoalVerifier._eval_thread_timeout)
    monkeypatch.setattr(gv, "_TIMEOUT_SECONDS", 1)
    try:
        assert v.evaluate("slow()")[0] is False          # timed out -> False
        assert v._runaway is not None and v._runaway.is_alive()
        # A runaway is still running -> the next evaluate refuses (never concurrent).
        assert v.evaluate("slow()")[0] is False
    finally:
        release.set()
        if v._runaway is not None:
            v._runaway.join(timeout=5)


# ---------------------------------------------------------------------------
# FG-3 — tolerance inflation is clamped INSIDE the oracle (every goal, not only
# coordinate goals): at_position(0, 0, 1e6) / facing(0, 7) cannot verify a far pose.
# ---------------------------------------------------------------------------


def test_tolerance_is_clamped_in_oracle() -> None:
    import math
    from types import SimpleNamespace

    from vector_os_nano.vcli.worlds.go2_sim_oracle import (
        _MAX_ARRIVAL_TOL_M,
        make_at_position,
        make_facing,
    )

    # A base parked far from the origin and facing +x.
    base = SimpleNamespace(
        get_position=lambda: (100.0, 100.0, 0.0),
        get_heading=lambda: 0.0,
        _connected=True,
    )
    agent = SimpleNamespace(_base=base)
    at_position = make_at_position(agent)
    facing = make_facing(agent)

    # An inflated tol is clamped to the cap -> cannot span 140 m to the origin.
    assert at_position(0.0, 0.0, 1e6) is False
    assert at_position(0.0, 0.0, 999.0) is False
    # Within the clamped cap it still answers honestly at the real pose.
    assert at_position(100.0, 100.0, _MAX_ARRIVAL_TOL_M) is True
    # facing: a 7 rad tol is clamped, so a 180-deg-off heading is NOT "facing".
    assert facing(math.pi, 7.0) is False
    assert facing(0.0, 7.0) is True  # actually facing +x; clamp doesn't make it lie
