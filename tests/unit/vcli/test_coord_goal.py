# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""STEP 13 — goal-authenticity: the verify CONSTANT must match the user's coordinate goal.

A model that does a REAL actor-caused walk to the WRONG place then verifies
``at_position(<its own landing>)`` used to grade verified=True (the moat never compared
the verify constant to the user's commanded target). These tests pin the pure helpers
and the spine downgrade: a goal-mismatched coordinate verify -> RAN; an honest one ->
GROUNDED; every non-coordinate turn fails OPEN (unchanged). Stricter-only (rule 5)."""
from __future__ import annotations

import pytest

from vector_os_nano.vcli.cognitive.coord_goal import (
    coord_goal_mismatch,
    has_necessary_at_position,
    parse_goal_coord,
)

_ORACLES = frozenset(
    {"at_position", "facing", "visited", "holding_object", "arm_at_home",
     "file_exists", "path_contains"}
)


@pytest.mark.parametrize(
    "goal, expected",
    [
        ("走到坐标 (11.0,3.0)", (11.0, 3.0)),
        ("go to (11,3)", (11.0, 3.0)),
        ("走到坐标 (-2.5, 0)", (-2.5, 0.0)),
        ("create out.txt containing ready", None),
        ("把香蕉抓起来拿在手里", None),
        ("(1,2) then (3,4)", None),           # ambiguous -> fail open
        ("到 (5,5) 再回 (5,5)", (5.0, 5.0)),   # same coord twice -> unambiguous
        ("", None),
        (None, None),
    ],
)
def test_parse_goal_coord(goal, expected) -> None:
    assert parse_goal_coord(goal) == expected


@pytest.mark.parametrize(
    "expr, present",
    [
        ("at_position(11.0, 3.0)", True),
        ("at_position(11, 3, 0.5)", True),              # tol == oracle tol -> kept
        ("at_position(-2, 0)", True),
        ("facing(1.5708)", False),
        ("holding_object('banana')", False),
        ("path_contains('out.txt', 'ready')", False),
        ("at_position(get_x(), 3)", False),    # non-literal target -> not a gating const
        ("len(get_position()) == 3", False),
        ("", False),
        # STEP-15 hardening (via _at_position_call_xy, still used here):
        ("at_position(x=11, y=3)", True),               # kwarg form read
        ("at_position(x=11, y=3, tol=0.3)", True),      # tighter tol -> kept
        ("at_position(11, 3, 99)", False),              # TOL-INFLATION (positional) -> rejected
        ("at_position(11, 3, tol=99)", False),          # TOL-INFLATION (kwarg) -> rejected
        ("at_position(11, 3, get_tol())", False),       # non-literal tol -> rejected
        ("at_position(11, 3) or visited('x')", False),  # non-necessary (OR decoy)
    ],
)
def test_has_necessary_at_position(expr, present) -> None:
    assert has_necessary_at_position(expr) is present


def test_coord_goal_mismatch() -> None:
    # mismatch: a real walk to the WRONG place (dist 1.0 > 0.5 tol)
    assert coord_goal_mismatch("走到坐标 (11.0,3.0)", "at_position(10.0, 3.0)", 0.5) is True
    # match: at the goal
    assert coord_goal_mismatch("走到坐标 (11.0,3.0)", "at_position(11.0, 3.0)", 0.5) is False
    # within tol -> no mismatch
    assert coord_goal_mismatch("走到坐标 (11.0,3.0)", "at_position(11.3, 3.0)", 0.5) is False
    # fail-open: no coordinate goal
    assert coord_goal_mismatch("把香蕉抓起来", "at_position(99.0, 99.0)", 0.5) is False
    # fail-open: non-coordinate verify
    assert coord_goal_mismatch("走到坐标 (11.0,3.0)", "facing(0.0)", 0.5) is False
    # default tol re-reads the oracle's _AT_POSITION_TOL_M (0.5) -> still a mismatch
    assert coord_goal_mismatch("走到坐标 (11.0,3.0)", "at_position(8.0, 3.0)") is True


# ---------------------------------------------------------------------------
# Spine: classify_step_evidence downgrades a goal-mismatched CAUSED at_position
# to RAN, keeps an honest one GROUNDED, and fails OPEN for everything else.
# ---------------------------------------------------------------------------


def _caused_step(verify: str):
    from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused
    from vector_os_nano.vcli.cognitive.types import StepRecord, SubGoal

    sg = SubGoal(name="s0", description="walk", verify=verify, strategy="walk")
    step = StepRecord(
        sub_goal_name="s0", strategy="walk", success=True,
        verify_result=True, duration_sec=0.1, actor_caused=ActorCaused.CAUSED,
    )
    return step, sg


def test_spine_wrong_place_caused_walk_downgrades_to_ran() -> None:
    from vector_os_nano.vcli.cognitive.trace_store import classify_step_evidence

    step, sg = _caused_step("at_position(8.0, 3.0)")  # verified its OWN landing
    # No goal threaded -> unchanged GROUNDED (the pre-STEP-13 residual).
    assert classify_step_evidence(step, sg, _ORACLES) == "GROUNDED"
    # With the user's real goal (11,3): the constant (8,3) mismatches -> RAN.
    assert classify_step_evidence(step, sg, _ORACLES, "走到坐标 (11.0,3.0)") == "RAN"


def test_spine_honest_coordinate_stays_grounded() -> None:
    from vector_os_nano.vcli.cognitive.trace_store import classify_step_evidence

    step, sg = _caused_step("at_position(11.0, 3.0)")
    assert classify_step_evidence(step, sg, _ORACLES, "走到坐标 (11.0,3.0)") == "GROUNDED"


@pytest.mark.parametrize(
    "verify, goal",
    [
        ("holding_object('banana')", "把香蕉抓起来拿在手里"),        # non-coord verify + goal
        ("path_contains('out.txt', 'ready')", "create out.txt containing ready"),
        ("facing(0.0)", "走到坐标 (11.0,3.0)，然后转向"),            # coord goal but facing verify
        ("at_position(8.0, 3.0)", "向前走两米"),                     # at_position verify, no coord goal
    ],
)
def test_spine_non_coordinate_fails_open_stays_grounded(verify, goal) -> None:
    from vector_os_nano.vcli.cognitive.trace_store import classify_step_evidence

    step, sg = _caused_step(verify)
    assert classify_step_evidence(step, sg, _ORACLES, goal) == "GROUNDED"


# ---------------------------------------------------------------------------
# STEP-15 — positive dual helper + coordinate-intent detector.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr, goal_xy, expected",
    [
        ("at_position(11, 3)", (11.0, 3.0), True),
        ("at_position(11.2, 3.1)", (11.0, 3.0), True),    # within 0.5 tol
        ("at_position(x=11, y=3)", (11.0, 3.0), True),    # kwarg honest
        ("at_position(5, 5)", (11.0, 3.0), False),        # wrong coord
        ("at_position(11, 3, 99)", (11.0, 3.0), False),   # tol-inflated -> not honest
        ("facing(0.5, 0.5)", (11.0, 3.0), False),         # non-at_position
        ("len(get_position()) == 3", (11.0, 3.0), False),
        ("", (11.0, 3.0), False),
        # STEP-16 BOOLEAN-NECESSITY: the matching at_position must be a NECESSARY
        # conjunct, not merely PRESENT. A disjunction / negation / arithmetic burial
        # lets the verify be True while the robot is NOT at the coord -> not honest.
        ("at_position(11, 3) or visited(99, 99)", (11.0, 3.0), False),
        ("at_position(11, 3) or facing(0, 0)", (11.0, 3.0), False),
        ("at_position(11, 3) or arm_at_home()", (11.0, 3.0), False),
        ("visited(99, 99) or at_position(11, 3)", (11.0, 3.0), False),  # decoy first
        ("at_position(11, 3) or at_position(99, 99)", (11.0, 3.0), False),
        ("not at_position(11, 3) or visited(99, 99)", (11.0, 3.0), False),
        ("(at_position(11, 3) or visited(0, 0)) and facing(1, 0)", (11.0, 3.0), False),
        ("at_position(11, 3) == False or facing(9, 9)", (11.0, 3.0), False),
        ("at_position(11, 3) + at_position(99, 99) >= 1", (11.0, 3.0), False),
        ("at_position(11, 3) * 1 + visited(99, 99) >= 1", (11.0, 3.0), False),
        # HONEST conjunctions: the matching at_position IS necessary -> stay True.
        ("at_position(11, 3) and facing(1, 0)", (11.0, 3.0), True),
        ("at_position(11, 3) and at_position(99, 99)", (11.0, 3.0), True),
        ("at_position(11, 3) and visited('kitchen')", (11.0, 3.0), True),
        ("at_position(11, 3) and (visited(0, 0) or facing(1, 0))", (11.0, 3.0), True),
    ],
)
def test_at_position_const_matches(expr, goal_xy, expected) -> None:
    from vector_os_nano.vcli.cognitive.coord_goal import at_position_const_matches

    assert at_position_const_matches(expr, goal_xy) is expected


@pytest.mark.parametrize(
    "goal, expected",
    [
        ("走到坐标 (11,3)", True),
        ("导航到 11,3", True),          # paren-less
        ("go to 11, 3", True),
        ("先到 (11,3) 再到 (12,3)", True),  # multi-coord
        ("探索房间", False),
        ("turn around", False),
        ("go to the kitchen", False),
        ("pick up the cup", False),
        ("", False),
        (None, False),
    ],
)
def test_goal_has_coordinate_intent(goal, expected) -> None:
    from vector_os_nano.vcli.cognitive.coord_goal import goal_has_coordinate_intent

    assert goal_has_coordinate_intent(goal) is expected


# ---------------------------------------------------------------------------
# STEP-15 — TURN-LEVEL coordinate gate in evidence_passed: a coordinate goal must be
# VERIFIED with a GROUNDED at_position matching the commanded coord. Closes the
# wrong-predicate-type hole (coord goal verified entirely with facing / len()==3) while
# keeping every honest case + all non-go2 worlds green. Stricter-only (rule 5).
# ---------------------------------------------------------------------------

_GO2 = frozenset({"at_position", "facing", "visited", "get_position", "get_heading"})
_DEV = frozenset({"file_exists", "path_contains"})
_ARM = frozenset({"holding_object", "arm_at_home"})


def _trace(goal: str, steps):
    """steps = list of (verify, strategy, verify_result, ActorCaused)."""
    from vector_os_nano.vcli.cognitive.types import (
        ExecutionTrace,
        GoalTree,
        StepRecord,
        SubGoal,
    )

    sgs = tuple(
        SubGoal(name=f"s{i}", description="x", verify=v, strategy=st)
        for i, (v, st, _vr, _ac) in enumerate(steps)
    )
    strs = tuple(
        StepRecord(
            sub_goal_name=f"s{i}", strategy=st, success=True, verify_result=vr,
            duration_sec=0.1, actor_caused=ac,
        )
        for i, (_v, st, vr, ac) in enumerate(steps)
    )
    return ExecutionTrace(
        goal_tree=GoalTree(goal=goal, sub_goals=sgs), steps=strs,
        success=True, total_duration_sec=1.0,
    )


def _CA():
    from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused

    return ActorCaused.CAUSED


def test_turn_gate_blocks_wrong_predicate_type() -> None:
    """A COORDINATE goal verified ENTIRELY with a non-at_position predicate (the
    confirmed STEP-14 hole) must NOT verify, even though every step is GROUNDED."""
    from vector_os_nano.vcli.cognitive.trace_store import evidence_passed

    # HOLE A: facing under a coordinate goal (CAUSED -> per-step GROUNDED).
    assert evidence_passed(_trace("走到坐标 (11,3)", [("facing(0.5,0.5)", "turn", True, _CA())]), _GO2) is False
    # HOLE B: shape-trivial len()==3 under a coordinate goal.
    assert evidence_passed(_trace("走到坐标 (11,3)", [("len(get_position())==3", "walk", True, _CA())]), _GO2) is False
    # TOL-INFLATION: matching constant but a vacuous 99 m tolerance.
    assert evidence_passed(_trace("走到坐标 (11,3)", [("at_position(11,3,99)", "walk", True, _CA())]), _GO2) is False
    # KWARG wrong-coord: at_position(x=99,y=99) used to extract None -> fail open.
    assert evidence_passed(_trace("走到坐标 (11,3)", [("at_position(x=99,y=99)", "walk", True, _CA())]), _GO2) is False
    # PARSE-EVASION: paren-less + multi-coord phrasings still require honest at_position.
    assert evidence_passed(_trace("导航到 11,3", [("facing(0.5,0.5)", "turn", True, _CA())]), _GO2) is False
    assert evidence_passed(_trace("先到 (11,3) 再到 (12,3)", [("facing(0.5,0.5)", "turn", True, _CA())]), _GO2) is False


def test_turn_gate_keeps_honest_coordinate_turns() -> None:
    from vector_os_nano.vcli.cognitive.trace_store import evidence_passed

    # Single honest at_position matching the goal.
    assert evidence_passed(_trace("走到坐标 (11,3)", [("at_position(11,3)", "walk", True, _CA())]), _GO2) is True
    # Honest kwarg + tolerance-edge landing.
    assert evidence_passed(_trace("走到坐标 (11,3)", [("at_position(x=11,y=3)", "walk", True, _CA())]), _GO2) is True
    assert evidence_passed(_trace("走到坐标 (11,3)", [("at_position(11.2,3.1)", "walk", True, _CA())]), _GO2) is True
    # Multiskill: the at_position leg satisfies the requirement; facing leg rides along.
    assert evidence_passed(_trace("走到坐标 (11,3) 然后转向", [
        ("at_position(11,3)", "walk", True, _CA()),
        ("facing(0.5,0.5)", "turn", True, _CA()),
    ]), _GO2) is True
    # Parse-evasion honest: multi-coord goal WITH at_position legs.
    assert evidence_passed(_trace("先到 (11,3) 再到 (12,3)", [
        ("at_position(11,3)", "walk", True, _CA()),
        ("at_position(12,3)", "walk", True, _CA()),
    ]), _GO2) is True


def test_turn_gate_non_coordinate_go2_turns_fail_open() -> None:
    from vector_os_nano.vcli.cognitive.trace_store import evidence_passed

    # A coordinate-less go2 turn (turn / explore) is untouched.
    assert evidence_passed(_trace("转向看桌子", [("facing(0.5,0.5)", "turn", True, _CA())]), _GO2) is True
    assert evidence_passed(_trace("探索房间", [("visited('kitchen')", "explore", True, _CA())]), _GO2) is True


def test_turn_gate_world_blind_non_go2_worlds_untouched() -> None:
    """dev / arm / answer worlds carry no ``at_position`` oracle, so an incidental (x,y)
    in the goal text must NEVER trigger the coordinate requirement (world-blind fix)."""
    from vector_os_nano.vcli.cognitive.trace_store import evidence_passed

    assert evidence_passed(_trace("write (3,4) to file out.txt", [
        ("path_contains('out.txt','x')", "tool", True, _CA()),
    ]), _DEV) is True
    assert evidence_passed(_trace("place the cube at (0.5,0.3)", [
        ("holding_object('cube')", "pick", True, _CA()),
    ]), _ARM) is True


def test_turn_gate_dummy_at_position_must_be_grounded() -> None:
    """The gate cannot be gamed by a dummy at_position step the robot didn't reach: an
    UNCAUSED (teleport / no-op) at_position is downgraded to RAN per-step, so it fails
    the all-GROUNDED rule before the coordinate clause is even reached."""
    from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused
    from vector_os_nano.vcli.cognitive.trace_store import evidence_passed

    assert evidence_passed(
        _trace("走到坐标 (11,3)", [("at_position(11,3)", "walk", True, ActorCaused.UNCAUSED)]), _GO2
    ) is False


# ---------------------------------------------------------------------------
# STEP-16 — BOOLEAN-NECESSITY (4th moat review, loop-until-dry). The coordinate
# goal-authenticity layer used to check that the matching at_position constant was
# PRESENT (first ast.walk hit), not that it was NECESSARY for the verify's truth. A
# real CAUSED walk to the WRONG place + an OR/NOT/arithmetic decoy carrying a
# goal-matching at_position constant graded verified=True while the robot stood
# elsewhere. The fix makes the matching at_position a NECESSARY conjunct; these pin
# the whole family RAN end-to-end through evidence_passed, honest conjunctions GREEN.
# Stricter-only (rule 5).
# ---------------------------------------------------------------------------

_NECESSITY_FALSE_GREENS = [
    "at_position(11,3) or visited(99,99)",
    "at_position(11,3) or facing(0,0) or visited(99,99)",
    "at_position(11,3) or arm_at_home()",
    "visited(99,99) or at_position(11,3)",
    "at_position(11,3) or at_position(99,99)",
    "not at_position(11,3) or visited(99,99)",
    "(at_position(11,3) or visited(0,0)) and facing(1,0)",
    "at_position(11,3) == False or facing(9,9)",
    "at_position(11,3) + at_position(99,99) >= 1",
    "at_position(11,3) * 1 + visited(99,99) >= 1",
]


@pytest.mark.parametrize("verify", _NECESSITY_FALSE_GREENS)
def test_turn_gate_blocks_boolean_necessity_decoys(verify) -> None:
    """Every boolean-necessity decoy (a CAUSED walk + a verify whose goal-matching
    at_position is not NECESSARY) must grade the turn NOT verified (RAN)."""
    from vector_os_nano.vcli.cognitive.trace_store import evidence_passed

    assert evidence_passed(_trace("走到坐标 (11,3)", [(verify, "walk", True, _CA())]), _GO2) is False


def test_turn_gate_keeps_honest_necessary_conjunctions() -> None:
    """A matching at_position under a (possibly nested) conjunction is NECESSARY -> the
    turn stays verified. The fix only rejects disjunction/negation/arithmetic burial."""
    from vector_os_nano.vcli.cognitive.trace_store import evidence_passed

    for verify in (
        "at_position(11,3)",
        "at_position(11,3) and facing(1,0)",
        "at_position(11,3) and at_position(99,99)",
        "at_position(11,3) and visited('kitchen')",
        "at_position(11,3) and (visited(0,0) or facing(1,0))",
        "at_position(x=11, y=3)",
    ):
        assert evidence_passed(_trace("走到坐标 (11,3)", [(verify, "walk", True, _CA())]), _GO2) is True, verify
