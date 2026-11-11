# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""D17 — goal-authenticity for OBJECT (grasp/pick) goals.

The fakeable-grasp false-green: a bare ``vector-cli`` + NL "抓前面的东西" let the model
satisfy a PHYSICAL grasp by ``file_write('grabbed.txt')`` then verifying
``file_exists('grabbed.txt')``. ``file_exists`` is a PREDICATE oracle (classifies
GROUNDED) AND a non-robot predicate (never actor-causation-graded), so every prior gate
failed OPEN and the turn graded GROUNDED — a physical grasp satisfied by touching a file.

These tests pin the turn-level gate (the object-goal analogue of the D15/D16 coordinate
turn gate): a grasp goal grades GROUNDED ONLY when a GT manipulation oracle
(``holding_object`` / ``placed_count``) NECESSARILY gates a GROUNDED step; a fabricated
grasp verified by ``file_exists`` / ``describe_scene`` / ``path_contains`` grades RAN.
Stricter-only (rule 5): real grasp -> GROUNDED, fake grasp -> RAN, every non-grasp turn
(coordinate nav / place-only / dev / g1-detect) UNCHANGED."""
from __future__ import annotations

import pytest

from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused
from vector_os_nano.vcli.cognitive.object_goal import goal_intents, has_necessary_manip_oracle
from vector_os_nano.vcli.cognitive.trace_store import evidence_passed
from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)

# A go2+arm (manipulation) world's live oracle set — ``holding_object`` present.
_ARM_ORACLES = frozenset(
    {
        "holding_object", "at_position", "facing", "arm_at_home", "detect_objects",
        "describe_scene", "placed_count", "file_exists", "path_contains",
        "lifted", "kept_holding", "near_object", "object_on", "released",
    }
)
# A go2-only (coordinate) world — NO arm, so the object-goal gate must never fire.
_GO2_ORACLES = frozenset({"at_position", "facing", "visited", "file_exists", "path_contains"})
# A dev world — code tools only.
_DEV_ORACLES = frozenset({"file_exists", "path_contains"})


# ---------------------------------------------------------------------------
# Pure helpers — intent detection + manipulation-oracle necessity.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "goal, expected",
    [
        ("抓前面的东西", True),
        ("把香蕉抓起来", True),
        ("pick up the banana", True),
        ("grasp the red cube", True),
        ("grab it", True),
        ("拿起杯子", True),
        ("把香蕉抓起来放到盒子里", True),     # pick-and-place still has grasp intent
        # non-grasp goals fail OPEN:
        ("走到坐标 (11,3)", False),
        ("导航到客厅", False),
        ("explore the room", False),
        ("detect the red object", False),
        ("把两个放到盒子里", True),            # place is a manipulation intent too (WP07)
        ("turn left", False),
        ("", False),
        (None, False),
    ],
)
def test_goal_intents_name_a_manipulation(goal, expected) -> None:
    assert bool(goal_intents(goal)) is expected


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("holding_object('banana')", True),
        ("holding_object()", True),
        ("holding_object('banana') and arm_at_home()", True),  # and-conjunct gates
        ("placed_count() == 1", False),                         # a placement never proves a grasp
        ("released('banana')", False),
        ("object_on('banana', 'tray')", False),
        ("lifted('banana')", True),                             # held clear of its support
        ("kept_holding('banana') and near_object('user')", True),
        ("not holding_object()", True),                         # release-completion gates
        # NOT a manipulation oracle -> the fabrication vectors:
        ("file_exists('grabbed.txt')", False),
        ("len(describe_scene()) > 0", False),
        ("path_contains('grabbed.txt', 'x')", False),
        ("len(read_file('grabbed.txt')) > 0", False),
        # present-but-not-necessary decoy:
        ("holding_object('banana') or True", False),
        ("at_position(1, 2)", False),
        ("", False),
    ],
)
def test_has_necessary_manip_oracle(expr, expected) -> None:
    assert has_necessary_manip_oracle(expr) is expected


# ---------------------------------------------------------------------------
# Turn-level gate — the un-fakeable contract.
# ---------------------------------------------------------------------------


def _step(name: str, verify: str, strategy: str, actor=ActorCaused.NOT_GRADED, vr=True):
    sg = SubGoal(name=name, description=name, verify=verify, strategy=strategy)
    rec = StepRecord(
        sub_goal_name=name, strategy=strategy, success=True,
        verify_result=vr, duration_sec=0.0, actor_caused=actor,
    )
    return sg, rec


def _trace(goal: str, *steps):
    return ExecutionTrace(
        goal_tree=GoalTree(goal=goal, sub_goals=tuple(s[0] for s in steps)),
        steps=tuple(s[1] for s in steps),
        success=True,
        total_duration_sec=0.0,
    )


def test_fake_grasp_via_file_exists_grades_ran() -> None:
    """THE regression: a "抓" goal verified by file_exists(...) must grade RAN, not GROUNDED."""
    sg, rec = _step("s0", "file_exists('grabbed.txt')", "file_write")
    trace = _trace("抓前面的东西", (sg, rec))
    assert evidence_passed(trace, _ARM_ORACLES) is False


@pytest.mark.parametrize(
    "verify",
    [
        "file_exists('grabbed.txt')",
        "len(describe_scene()) > 0",
        "path_contains('grabbed.txt', 'grabbed')",
    ],
)
def test_fake_grasp_other_self_authored_verifies_grade_ran(verify) -> None:
    sg, rec = _step("s0", verify, "file_write")
    trace = _trace("grab the banana", (sg, rec))
    assert evidence_passed(trace, _ARM_ORACLES) is False


def test_real_grasp_via_holding_object_stays_grounded() -> None:
    """A real perception_grasp -> holding_object weld (CAUSED) must stay GROUNDED."""
    sg, rec = _step("s0", "holding_object('banana')", "perception_grasp", ActorCaused.CAUSED)
    trace = _trace("抓前面的东西", (sg, rec))
    assert evidence_passed(trace, _ARM_ORACLES) is True


@pytest.mark.parametrize(
    "goal, verify, strategy, actor",
    [
        # generic pick proven by holding anything
        ("pick up the cube", "holding_object()", "pick", ActorCaused.CAUSED),
        # grasp + return-home chain
        ("抓香蕉回家", "holding_object('banana') and arm_at_home()", "perception_grasp", ActorCaused.CAUSED),
    ],
)
def test_honest_manipulation_goals_stay_grounded(goal, verify, strategy, actor) -> None:
    sg, rec = _step("s0", verify, strategy, actor)
    trace = _trace(goal, (sg, rec))
    assert evidence_passed(trace, _ARM_ORACLES) is True


def test_pick_and_place_needs_the_grasp_proven_not_only_the_placement() -> None:
    """WP10: a pick-and-place turn verified ONLY by where the object ended up is not a
    proven grasp (it may have been resting there all along); with the grasp step's
    holding_object GROUNDED alongside the place, it is."""
    place = _step("s1", "object_on('banana', 'tray')", "place", ActorCaused.CAUSED)
    assert evidence_passed(_trace("把香蕉抓起来放到盒子里", place), _ARM_ORACLES) is False
    grasp = _step("s0", "holding_object('banana')", "pick", ActorCaused.CAUSED)
    assert evidence_passed(_trace("把香蕉抓起来放到盒子里", grasp, place), _ARM_ORACLES) is True


def test_release_negation_is_not_grounded() -> None:
    """WP02 (stricter-only): a release terminal written as ``not holding_object()`` —
    the negation of a RISE predicate — is RAN, not GROUNDED. A release re-homes to a
    positive fall-transition oracle (``released()``), not ``not`` of a rise predicate;
    and at runtime this step was already RAN (release never grades CAUSED). The
    object-goal gate still SEES a necessary manip oracle, so this is the classifier's
    negation rule, not the gate opening a hole."""
    sg, rec = _step("s0", "not holding_object()", "place", ActorCaused.NOT_GRADED)
    trace = _trace("抓起香蕉放下", (sg, rec))
    assert evidence_passed(trace, _ARM_ORACLES) is False
    assert has_necessary_manip_oracle("not holding_object()") is True


@pytest.mark.parametrize(
    "goal, verify, strategy, actor, oracles",
    [
        # place-only goal proven by a placement (D69: a placement proves a place)
        ("把两个放到盒子里", "object_on('banana', 'tray')", "place", ActorCaused.CAUSED, _ARM_ORACLES),
        # coordinate nav (go2, no holding_object oracle) — gate never fires
        ("走到坐标 (11,3)", "at_position(11, 3)", "navigate", ActorCaused.CAUSED, _GO2_ORACLES),
        # dev file task — gate never fires (no holding_object oracle)
        ("create out.txt containing ready", "path_contains('out.txt', 'ready')", "file_write", ActorCaused.NOT_GRADED, _DEV_ORACLES),
    ],
)
def test_non_grasp_turns_unchanged(goal, verify, strategy, actor, oracles) -> None:
    sg, rec = _step("s0", verify, strategy, actor)
    trace = _trace(goal, (sg, rec))
    assert evidence_passed(trace, oracles) is True
