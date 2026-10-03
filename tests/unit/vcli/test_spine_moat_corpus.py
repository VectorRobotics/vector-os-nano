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
    "detection_matches_gt",  # g1 perception bool oracle
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
    # a bool oracle ordered / negated / used as a state term (detection_matches_gt
    # used to get the state rules, so `< 2` and `== False` graded GROUNDED)
    "detection_matches_gt('red') < 2",
    "detection_matches_gt('red') >= 0",
    "detection_matches_gt('red') == False",
    "not detection_matches_gt('red')",
    "len(detection_matches_gt('red')) == 0",
    # an empty container is no goal: `detect_objects() == []` holds on the arm oracle's
    # fail-safe value (RAN on base, which demanded a Constant in the compare)
    "detect_objects() == []",
    "detect_objects() == ()",
    "get_position() == []",
    "get_position() == ()",
    "get_heading() == ()",
    "[] == describe_scene()",
    "placed_count() == []",
    "[[]] == placed_count()",
    "placed_count() == ((),)",
    "[] in detect_objects()",
    "(detect_objects() == []) and (get_position() == ())",
    "at_position(1, 2) or detect_objects() == []",
    # FG-3 tolerance inflation: a literal tol above the cap (2 m / 45 deg) is vacuous
    # (the oracle also clamps it at runtime, see test_tolerance_is_clamped_in_oracle)
    "at_position(0, 0, 1e6)",
    "at_position(0, 0, tol=99)",
    "at_position(x=0, y=0, tol=2.01)",
    "at_position(0, 0, 1e6) == True",
    "at_position(0, 0, 'big')",
    "facing(0, 7)",
    "facing(0.0, tol=3.2)",
    "facing(0.0, 0.8) and at_position(11, 3)",
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
    "at_position(10.0, 3.0, 2.0)",                  # tol exactly at the cap
    "at_position(x=11, y=3, tol=0.3)",
    "facing(0.52, 0.78)",                           # just under 45 deg
    "facing(0.0, tol=0.5)",
    "detect_objects() == ['banana']",
    "detection_matches_gt('red') == True",  # g1 (R7); its bare call stays RAN
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


def test_scope_filter_keeps_dev_predicates_out_of_robot_evidence() -> None:
    """evidence_names' scope filter: a dev-tagged predicate in the namespace is not
    robot evidence (so ``file_exists('.')`` grades RAN), and an untagged callable is
    evidence in no scope."""
    from vector_os_nano.vcli.cognitive.oracle_registry import evidence_names, oracle

    ns = {
        "file_exists": oracle(scope="dev")(lambda path: True),
        "at_position": oracle(scope="robot")(lambda x, y, tol=0.5: True),
        "nearest_room": lambda: "kitchen",
    }
    robot = evidence_names(ns, "robot")
    assert robot == {"at_position"}
    assert evidence_names(ns, "dev") == {"file_exists"}
    # No scope (the reward gate) on a namespace mixing scopes fails closed.
    assert evidence_names(ns) == frozenset()
    assert evidence_names({k: ns[k] for k in ("at_position", "nearest_room")}) == {"at_position"}
    assert classify_verify_expr("file_exists('.')", robot) == "RAN"
    assert classify_verify_expr("file_exists('.')", robot | {"file_exists"}) == "GROUNDED"


def _tagged_bool_oracles() -> set[str]:
    """Names of every ``@oracle(...)``-decorated ``def f(...) -> bool`` under vcli/."""
    import ast
    from pathlib import Path

    names: set[str] = set()
    for path in Path(ec.__file__).resolve().parents[1].rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.FunctionDef)
                and isinstance(node.returns, ast.Name)
                and node.returns.id == "bool"
                and any(
                    isinstance(d, ast.Call) and getattr(d.func, "id", None) == "oracle"
                    for d in node.decorator_list
                )
            ):
                names.add(node.name)
    return names


def test_every_capped_tol_oracle_is_capped_in_the_classifier() -> None:
    """Every ``@oracle(max_tol=...)`` must have the SAME cap in the classifier's
    _TOL_ARG, else an over-cap literal tol grades GROUNDED."""
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    from vector_os_nano.vcli.worlds import go2_sim_oracle as g

    capped = {
        node.name
        for path in Path(ec.__file__).resolve().parents[1].rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(d, ast.Call) and getattr(d.func, "id", None) == "oracle"
            and any(k.arg == "max_tol" for k in d.keywords)
            for d in node.decorator_list
        )
    }
    assert capped == set(ec._TOL_ARG), sorted(capped ^ set(ec._TOL_ARG))
    agent = SimpleNamespace(_base=None)
    tagged = {f.__name__: f.__oracle__.max_tol for f in (g.make_at_position(agent), g.make_facing(agent))}
    assert tagged == {name: cap for name, (_i, cap) in ec._TOL_ARG.items()}


def test_every_bool_oracle_gets_the_bool_rules() -> None:
    """A tagged bool oracle missing from _BOOL_ORACLES falls under the state rules,
    where ``pred(...) < 2`` / ``== False`` ground (always true / a negation)."""
    found = _tagged_bool_oracles()
    assert {"at_position", "holding_object", "file_exists", "detection_matches_gt"} <= found
    assert found <= ec._BOOL_ORACLES, sorted(found - ec._BOOL_ORACLES)


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


def test_goal_verifier_ambiguous_truth_fails_closed() -> None:
    """bool-outside-try: an oracle value with no truth value (a numpy array) grades
    False instead of raising out of the verify sandbox."""

    class _NoTruth:
        def __bool__(self) -> bool:
            raise ValueError("truth value of an array is ambiguous")

    v = GoalVerifier({"scan": lambda: _NoTruth()})
    assert v.evaluate("scan()") == (False, None)
    assert v.verify("scan()") is False


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


# ---------------------------------------------------------------------------
# evidence_passed whole-trace guards (B1-foreach-evidence-blind / foreach-unchecked,
# spine.md FG-8): the gate grades every step or none. Each case isolates one guard.
# ---------------------------------------------------------------------------

_DETECT_ONLY = frozenset({"detect_objects"})


def _detect_trace(extra_sub_goals=(), extra_steps=(), success=True):  # noqa: ANN001, ANN202
    from vector_os_nano.vcli.cognitive.types import ExecutionTrace, GoalTree, StepRecord, SubGoal

    detect = SubGoal(name="detect", description="detect", verify="len(detect_objects()) > 0")
    return ExecutionTrace(
        goal_tree=GoalTree(goal="list the objects on the table", sub_goals=(detect, *extra_sub_goals)),
        steps=(StepRecord("detect", "detect_skill", True, True, 0.1), *extra_steps),
        success=success,
        total_duration_sec=0.1,
    )


def _pick_each():  # noqa: ANN202
    from vector_os_nano.vcli.cognitive.types import ForEachSpec, SubGoal

    tmpl = SubGoal(name="tmpl", description="pick ${item.name}", verify="True", strategy="pick")
    return SubGoal(
        name="pick_each", description="pick each", verify="",
        foreach=ForEachSpec(source_step="detect", source_path="objects", body=(tmpl,)),
    )


def _child(success: bool):  # noqa: ANN202
    from vector_os_nano.vcli.cognitive.types import StepRecord

    return StepRecord("pick_each[0].tmpl", "pick", success, success, 0.1)


def test_evidence_passed_control_grounded_trace_passes() -> None:
    from vector_os_nano.vcli.cognitive.trace_store import evidence_passed

    assert evidence_passed(_detect_trace(), _DETECT_ONLY) is True


@pytest.mark.parametrize(
    "trace_kwargs",
    [
        pytest.param({"success": False}, id="trace-not-success"),
        pytest.param({"extra_sub_goals": (_pick_each(),)}, id="zero-iteration-foreach"),
        pytest.param({"extra_steps": (_child(True),)}, id="unmapped-child-verify-True"),
        pytest.param({"extra_steps": (_child(False),)}, id="unmapped-child-FAILED"),
        pytest.param(
            {"extra_sub_goals": (_pick_each(),), "extra_steps": (_child(False),)},
            id="FG-8-foreach-with-failed-child",
        ),
    ],
)
def test_evidence_passed_whole_trace_guards(trace_kwargs) -> None:  # noqa: ANN001
    from vector_os_nano.vcli.cognitive.trace_store import evidence_passed

    assert evidence_passed(_detect_trace(**trace_kwargs), _DETECT_ONLY) is False


# ---------------------------------------------------------------------------
# Harvested (WP04 dead-track sweep): the deleted ROS2-proxy tests pinned a base
# whose pose was a PLACEHOLDER — Go2ROS2Proxy started at (0, 0, 0.28), on the
# living_room box's corner, before any odometry arrived — so visited('living_room')
# and at_position(0, 0) graded True with no sim running (periphery B-5). A base
# that cannot vouch for its pose must fail every base oracle closed, wherever the
# placeholder happens to sit.
# ---------------------------------------------------------------------------


class _UnvouchedBase:
    """A base parked exactly on the goal — vouched for only when connected + readable."""

    def __init__(self, *, connected=True, pose=(0.0, 0.0, 0.28), heading=0.0, raises=False):  # noqa: ANN001
        self._connected = connected
        self._pose, self._heading, self._raises = pose, heading, raises

    def get_position(self):  # noqa: ANN201
        if self._raises:
            raise RuntimeError("no odometry received yet")
        return self._pose

    def get_heading(self):  # noqa: ANN201
        if self._raises:
            raise RuntimeError("no odometry received yet")
        return self._heading


def _base_oracle_verdicts(base) -> tuple[bool, bool, bool]:  # noqa: ANN001
    from types import SimpleNamespace

    from vector_os_nano.vcli.worlds.go2_sim_oracle import make_at_position, make_facing, make_visited

    agent = SimpleNamespace(_base=base)
    return (
        make_at_position(agent)(0.0, 0.0),
        make_facing(agent)(0.0),
        make_visited(agent, {"living_room": (0.0, 0.0, 6.0, 5.0)})("living_room"),
    )


@pytest.mark.parametrize(
    "base",
    [
        pytest.param(_UnvouchedBase(connected=False), id="disconnected-at-placeholder"),
        pytest.param(_UnvouchedBase(raises=True), id="no-odom-yet-raises"),
        pytest.param(_UnvouchedBase(pose=None, heading=None), id="pose-none"),
        pytest.param(_UnvouchedBase(pose=(float("nan"),) * 3, heading=float("nan")), id="pose-nan"),
    ],
)
def test_base_oracles_fail_closed_on_an_unvouched_pose(base) -> None:  # noqa: ANN001
    assert _base_oracle_verdicts(base) == (False, False, False)


def test_base_oracles_ground_the_same_pose_when_vouched() -> None:
    """Control: the identical pose on a connected, readable base DOES ground, so the
    fail-closed cases above fail for the right reason."""
    assert _base_oracle_verdicts(_UnvouchedBase()) == (True, True, True)
