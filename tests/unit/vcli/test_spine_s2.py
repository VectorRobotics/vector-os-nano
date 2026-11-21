# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""WP07 spine S2 — attempt semantics, per-channel causation, the task contract and the
manipulation vocabulary, as hand-built traces through the REAL gate (no sim).

Every case pins one rule of ``trace_store.turn_reject_reason`` (the gate
``evidence_passed`` and the verdict share) or of ``actor_causation.grade_channels``;
the mutation-sensitivity block at the end switches each new guard off and shows an
adversarial case the suite would then pass, so a regression cannot slip through.
"""
from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vector_os_nano.vcli.cognitive import actor_causation as ac
from vector_os_nano.vcli.cognitive import trace_store as ts
from vector_os_nano.vcli.cognitive.actor_causation import ActorBaseline, ActorCaused, grade_channels
from vector_os_nano.vcli.cognitive.object_goal import goal_intents, task_contract
from vector_os_nano.vcli.cognitive.oracle_registry import OracleTag
from vector_os_nano.vcli.cognitive.trace_store import evidence_passed, turn_reject_reason
from vector_os_nano.vcli.cognitive.types import ExecutionTrace, GoalTree, StepRecord, SubGoal
from vector_os_nano.vcli.verdict import VerdictReport

_BASE = OracleTag(scope="robot", channel="base", transition="rise")
TAGS = {
    "at_position": _BASE, "facing": _BASE,
    "holding_object": OracleTag(scope="robot", channel="gripper", transition="rise"),
    "released": OracleTag(scope="robot", channel="gripper", transition="fall"),
    "object_on": OracleTag(scope="robot", channel="gripper", transition="fall"),
    "arm_at_home": OracleTag(scope="robot", channel="arm", transition="rise"),
    "in_view": _BASE,
}
MANIP = frozenset(TAGS)                       # go2 + arm world
GO2 = frozenset({"at_position", "facing"})    # base-only world
ARM_ONLY = MANIP - {"at_position", "facing"}  # SO-101 tabletop

C, U = ActorCaused.CAUSED, ActorCaused.UNCAUSED


def _row(verify: str, *, ok: bool = True, vr: bool = True, caused=C, attempt: str = "native",
         channels: tuple[str, ...] = (), housekeeping: bool = False):
    return verify, StepRecord("", "skill", ok, vr, 0.0, actor_caused=caused, attempt_of=attempt,
                              channels=channels, housekeeping=housekeeping)


# I4d: a grasp contract binds through a world manifest (no manifest compiles none).
GRAB, HOLD = "拿起绿色的瓶子", "holding_object('pickable_bottle_green')"


def _trace(goal: str, *rows, contract=(), start=(), end=(), intents=(), success=True, world="") -> ExecutionTrace:
    sub_goals, steps = [], []
    for i, (verify, rec) in enumerate(rows):
        name = f"native_step_{i}"
        sub_goals.append(SubGoal(name=name, description=name, verify=verify))
        steps.append(StepRecord(name, rec.strategy, rec.success, rec.verify_result, 0.0,
                                actor_caused=rec.actor_caused, attempt_of=rec.attempt_of,
                                channels=rec.channels, housekeeping=rec.housekeeping))
    return ExecutionTrace(GoalTree(goal, tuple(sub_goals)), tuple(steps), success, 0.0,
                          contract=tuple(contract), contract_start=tuple(start),
                          contract_end=tuple(end), intents=tuple(intents), world=world)


# ---------------------------------------------------------------------------
# Attempt semantics (spine:retry-poisons-turn, producers:native-recovery-never-verified)
# ---------------------------------------------------------------------------


def test_fail_then_caused_pass_of_the_same_predicate_verifies() -> None:
    t = _trace("走到坐标 (9.0,3.0)",
               _row("at_position(9.0, 3.0)", vr=False),
               _row("at_position(9, 3)"),           # 9 == 9.0: the same predicate
               contract=("at_position(9.0, 3.0)",), start=(False,), end=(True,))
    assert evidence_passed(t, GO2), turn_reject_reason(t, GO2)
    report = VerdictReport.from_trace(t, GO2)
    assert [s.superseded for s in report.per_step] == [True, False]
    assert report.per_step[0].reject_reason == "verify FAIL"  # kept as audit
    assert (report.n_steps, report.n_grounded) == (1, 1)


def test_fail_then_uncaused_pass_is_not_verified() -> None:
    t = _trace("走到坐标 (9.0,3.0)",
               _row("at_position(9.0, 3.0)", vr=False),
               _row("at_position(9.0, 3.0)", caused=U))
    assert not evidence_passed(t, GO2)
    assert "UNCAUSED" in turn_reject_reason(t, GO2)


@pytest.mark.parametrize("easier", [
    "at_position(9.0, 3.0, 2.0)",       # a wider tolerance is a different predicate
    "facing(0.0)",
    "in_view('can')",
])
def test_a_pass_of_a_different_easier_predicate_does_not_supersede(easier: str) -> None:
    t = _trace("走到坐标 (9.0,3.0)", _row("at_position(9.0, 3.0)", vr=False), _row(easier))
    assert not evidence_passed(t, MANIP)
    assert ts.graded_indices(t) == [0, 1]


def test_a_producer_cannot_merge_two_predicates_into_one_group() -> None:
    """attempt_of only OPTS IN: rows sharing it still need the same normalized verify."""
    t = _trace("go there", _row("at_position(5.0, 0.0)", vr=False, attempt="g"),
               _row("at_position(1.0, 0.0)", attempt="g"))
    assert ts.graded_indices(t) == [0, 1] and not evidence_passed(t, GO2)


def test_rows_without_attempt_of_are_never_superseded() -> None:
    """A legacy / hand-built trace (no attempt_of) grades exactly as before WP07."""
    t = _trace("go", _row("at_position(1.0, 0.0)", vr=False, attempt=""),
               _row("at_position(1.0, 0.0)", attempt=""))
    assert ts.graded_indices(t) == [0, 1] and not evidence_passed(t, GO2)


def test_a_failed_action_superseded_by_a_successful_retry_verifies() -> None:
    t = _trace("走到坐标 (9.0,3.0)", _row("at_position(9.0, 3.0)", ok=False, vr=False),
               _row("at_position(9.0, 3.0)"), contract=("at_position(9.0, 3.0)",), start=(False,), end=(True,))
    assert evidence_passed(t, GO2)



# ---------------------------------------------------------------------------
# Independent coordinate + manipulation gates, the vocabulary, D69
# ---------------------------------------------------------------------------


def test_go_to_and_grab_with_a_vacuous_grasp_verify_is_not_verified() -> None:
    """spine:object-gate-skipped (FG-5): the coordinate branch no longer returns
    before the manipulation gate."""
    # I4d: a seek clause is no closed form, so this goal has no contract; the gates judge it
    t = _trace("go to (2,3) and grab the can",
               _row("at_position(2, 3)"), _row("in_view('can')", caused=ActorCaused.NOT_GRADED))
    assert not evidence_passed(t, MANIP)
    assert "holding_object" in turn_reject_reason(t, MANIP)
    honest = _trace("go to (2,3) and grab the can", _row("at_position(2, 3)"), _row("holding_object('can')"))
    assert evidence_passed(honest, MANIP)


def test_arm_only_world_runs_the_grasp_gate() -> None:
    """FG-5 second half: a world with holding_object but no at_position."""
    t = _trace("grab the banana", _row("in_view('can')", caused=ActorCaused.NOT_GRADED))
    assert not evidence_passed(t, ARM_ONLY)


@pytest.mark.parametrize("goal", [
    "bring me the red can", "fetch the bottle", "把绿色瓶子拿过来", "拿给我那个杯子",
    "carry the box to the kitchen", "deliver the mug", "get me a can", "递给我",
])
def test_fetch_goal_verified_only_by_detect_is_not_verified(goal: str) -> None:
    """d69-intent-regex-misses-fetch / grasp-regex-narrow (FG-6)."""
    t = _trace(goal, _row("in_view('can')", caused=ActorCaused.NOT_GRADED))
    assert "fetch" in goal_intents(goal) or "grasp" in goal_intents(goal)
    assert not evidence_passed(t, MANIP)


def test_a_placement_never_satisfies_a_grasp_goal() -> None:
    """docs-vs-code:placed-count-satisfies-grasp-gate (D69): a placement (object_on,
    placed_count's successor) proves a place, never a grasp."""
    t = _trace("pick up the banana", _row("object_on('banana', 'tray')"))
    assert not evidence_passed(t, MANIP)
    place = _trace("put the banana in the box", _row("object_on('banana', 'tray')"))
    assert evidence_passed(place, MANIP)


def test_a_dispatched_gripper_skill_is_a_manipulation_intent() -> None:
    """No manipulation verb in the goal, but the producer dispatched a grasp skill."""
    t = _trace("那个绿色的东西", _row("in_view('can')", caused=ActorCaused.NOT_GRADED),
               intents=("grasp",))
    assert not evidence_passed(t, MANIP)
    assert evidence_passed(_trace("那个绿色的东西", _row("in_view('can')",
                                                    caused=ActorCaused.NOT_GRADED)), MANIP)


@pytest.mark.parametrize("goal, intents", [
    ("拿起杯子", {"grasp"}), ("把绿色瓶子拿过来", {"grasp", "fetch"}), ("取一下杯子", {"grasp"}),
    ("搬到厨房", {"fetch"}), ("送到餐厅", {"fetch"}), ("放到桌子上", {"place"}),
    ("take the cup", {"grasp"}), ("put it down", {"place"}), ("place it on the table", {"place"}),
    # no manipulation: launch phrasing, navigation, cancellation, reading, a look
    ("启动 go2 仿真 带机械臂", set()), ("去厨房", set()), ("取消", set()), ("读取文件", set()),
    ("take a look around", set()), ("get to the kitchen", set()), ("放松一下", set()),
    ("go to the place by the door", set()), ("发送消息", set()), ("拿手好戏", set()),
])
def test_manipulation_vocabulary(goal: str, intents: set[str]) -> None:
    assert set(goal_intents(goal)) == intents


def test_every_gripper_skill_alias_is_in_the_vocabulary() -> None:
    """The vocabulary is derived from the manipulation skills' aliases: every alias
    of a skill whose verify_hint names a gripper oracle reads as a manipulation, so
    a new alias can never fail the gate open."""
    from vector_os_nano.skills.grasp import GraspSkill
    from vector_os_nano.skills.place import PlaceSkill

    for cls in (GraspSkill, PlaceSkill):
        assert any(o in cls.verify_hint for o in ("holding_object", "released", "object_at", "object_on"))
        for alias in cls.__skill_aliases__:
            assert goal_intents(alias), f"{cls.__name__} alias {alias!r} is not a manipulation"


# ---------------------------------------------------------------------------
# Per-channel causation (multi-channel-first-match, release-never-caused, F->T)
# ---------------------------------------------------------------------------


def _b(**kw) -> ActorBaseline:
    return ActorBaseline(**kw)


def test_holding_and_at_position_after_an_earlier_stationary_grasp_is_ran() -> None:
    before = _b(base_cmd_motion=0.0, base_pos=(1.0, 0.0, 0.3), base_heading=0.0,
                gripper_welds={"can": True})
    after = before  # nothing moved, the weld was already on
    caused, detail = grade_channels(before, after, "holding_object('can') and at_position(1.0, 0.0)", TAGS)
    assert caused is U and detail == ("gripper:holding_object:UNCAUSED", "base:at_position:UNCAUSED")
    t = _trace("hold it here", _row("holding_object('can') and at_position(1.0, 0.0)", caused=caused))
    assert ts.classify_step_evidence(t.steps[0], t.goal_tree.sub_goals[0], MANIP) == "RAN"


def test_every_channel_must_be_caused_no_first_match() -> None:
    """A real walk does not launder a gripper conjunct whose grasp happened earlier."""
    before = _b(base_cmd_motion=0.0, base_pos=(0.0, 0.0, 0.3), base_heading=0.0, gripper_welds={"can": True})
    after = _b(base_cmd_motion=2.0, base_pos=(2.0, 0.0, 0.3), base_heading=0.0, gripper_welds={"can": True})
    caused, detail = grade_channels(before, after, "at_position(2.0, 0.0) and holding_object('can')", TAGS)
    assert caused is U and detail == ("base:at_position:CAUSED", "gripper:holding_object:UNCAUSED")


def test_open_after_a_grasp_then_not_holding_is_caused() -> None:
    before = _b(gripper_welds={"x": True})
    after = _b(gripper_welds={"x": False})
    assert grade_channels(before, after, "not holding_object('x')", TAGS)[0] is C
    assert grade_channels(before, after, "released('x')", TAGS)[0] is C
    # ... but a release is never caused by a grasp, nor by nothing.
    assert grade_channels(after, before, "released('x')", TAGS)[0] is U
    assert grade_channels(after, after, "released()", TAGS)[0] is U


def test_gripper_causation_is_target_aware() -> None:
    before = _b(gripper_welds={"banana": False, "mug": False})
    after = _b(gripper_welds={"banana": True, "mug": False})
    assert grade_channels(before, after, "holding_object('banana')", TAGS)[0] is C
    assert grade_channels(before, after, "holding_object('mug')", TAGS)[0] is U


@pytest.mark.parametrize("verify", ["at_position(0.0, 0.0)", "at_position(0.5, 0.0, 1.0)"])
def test_a_base_target_already_met_at_the_baseline_is_uncaused(verify: str) -> None:
    """F->T: a wiggle near a target the robot already stood on proves nothing."""
    before = _b(base_cmd_motion=0.0, base_pos=(0.0, 0.0, 0.3), base_heading=0.0)
    after = _b(base_cmd_motion=1.0, base_pos=(0.1, 0.0, 0.3), base_heading=0.0)
    assert grade_channels(before, after, verify, TAGS)[0] is U
    far = _b(base_cmd_motion=0.0, base_pos=(-3.0, 0.0, 0.3), base_heading=0.0)
    assert grade_channels(far, after, verify, TAGS)[0] is C


def test_facing_already_met_at_the_baseline_is_uncaused() -> None:
    before = _b(base_cmd_motion=0.0, base_pos=(0.0, 0.0, 0.3), base_heading=0.0)
    turned = _b(base_cmd_motion=1.0, base_pos=(0.0, 0.0, 0.3), base_heading=math.radians(40))
    assert grade_channels(before, turned, "facing(0.52, 0.78)", TAGS)[0] is U   # true at 0 rad
    assert grade_channels(before, turned, "facing(0.70, 0.30)", TAGS)[0] is C


# ---------------------------------------------------------------------------
# The task contract (graft a) and the housekeeping-step rule (graft i)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("goal, names, contract", [
    ("走到坐标 (9.0,3.0)", GO2, ("at_position(9.0, 3.0)",)),
    # I4d: no world manifest here, so no manipulation contract (test_task_contract_binding
    # has the bound ones); the coordinate of a manipulation goal is a waypoint
    ("拿起绿色的瓶子", MANIP, ()),
    ("把绿色瓶子拿过来", MANIP, ()),
    ("bring me the red can", MANIP, ()),
    ("go to (2,3) and grab the can", MANIP, ()),
    ("go to (3,0) and grab the red can", GO2, ()),   # no grasp proof here: never a nav contract
    ("走到 (3,0) 把罐子拿过来", GO2, ()),
    ("把香蕉放到盒子里", MANIP, ()),                       # place: no manifest surface named
    ("去厨房", MANIP, ()),
    ("bring me the README", frozenset({"file_exists"}), ()),           # dev world: no contract
])
def test_task_contract_compiles_from_the_goal(goal, names, contract) -> None:
    assert task_contract(goal, names) == contract


def test_already_satisfied_at_turn_start_is_never_grounded() -> None:
    t = _trace(GRAB, _row("holding_object('green bottle')"),
               contract=(HOLD,), start=(True,), end=(True,), world="apartment")
    assert turn_reject_reason(t, MANIP) == f"already satisfied at turn start: {HOLD}"
    assert ts.goal_check(t, MANIP) == ("failed", f"already satisfied at turn start: {HOLD}")


def test_contract_not_met_at_turn_end_is_not_verified() -> None:
    t = _trace(GRAB, _row("holding_object('green bottle')"),
               contract=(HOLD,), start=(False,), end=(False,), world="apartment")
    assert "not met at turn end" in turn_reject_reason(t, MANIP)


def test_without_a_manifest_a_fetch_is_unchecked_and_the_gates_judge_it() -> None:
    """I4d: with no world manifest a fetch compiles no contract (it was kept_holding(None),
    which no turn could ever meet): the goal is unchecked, never proven, and the fetch
    gate still needs an actor-caused hold."""
    t = _trace("把绿色瓶子拿过来", _row("holding_object('b')"))
    assert task_contract(t.goal_tree.goal, MANIP) == ()
    assert ts.goal_check(t, MANIP) == ("unchecked", "")
    assert "holding_object" in turn_reject_reason(_trace("把绿色瓶子拿过来", _row("arm_at_home()")), MANIP)


def test_a_met_contract_needs_a_caused_step_that_asserts_it() -> None:
    t = _trace(GRAB, _row("at_position(1.0, 1.0)"),
               _row("holding_object('green bottle')", caused=U, housekeeping=True),
               contract=(HOLD,), start=(False,), end=(True,), world="apartment")
    assert turn_reject_reason(t, MANIP) == f"no actor-caused step proves {HOLD}"
    assert ts.goal_check(t, MANIP)[0] == "failed"


def test_the_wp03_explore_grasp_stop_trace_verifies_under_its_contract() -> None:
    """Graft (i), the live WP03 trace: explore -> grasp -> verify holding
    (CAUSED) -> stop -> verify holding (UNCAUSED). Before WP07 the trailing stop step
    graded RAN and the turn RAN (1/3). The contract decides: the grasp is GROUNDED and
    the RAN housekeeping row neither FAILed nor contradicted it."""
    rows = (_row("holding_object('pickable_bottle_green')", attempt=""),
            _row("holding_object('pickable_bottle_green')", caused=U, attempt="", housekeeping=True))
    old = _trace("拿起绿色的瓶子", *rows)
    # I4 G4: a housekeeping row never downgrades a turn another row proved, contract or
    # not (a legacy trace carries none); it alone never proves one (below).
    assert evidence_passed(old, MANIP), turn_reject_reason(old, MANIP)
    only = _trace("拿起绿色的瓶子", rows[1])
    assert turn_reject_reason(only, MANIP) == "no checked step"
    t = _trace(GRAB, *rows, contract=(HOLD,), start=(False,), end=(True,), world="apartment")
    assert evidence_passed(t, MANIP), turn_reject_reason(t, MANIP)
    assert ts.goal_check(t, MANIP) == ("proven", "")


@pytest.mark.parametrize("goal, contract, last", [
    # an unproven actuating step the contract does not cover (an extra turn; K1: "... then
    # turn left" compiles no contract at all)
    ("走到坐标 (3,0)", ("at_position(3.0, 0.0)",), _row("")),
    ("go to (2,3)", ("at_position(2.0, 3.0)",), _row("", ok=True, vr=False)),
    (GRAB, (HOLD,), _row("facing(0.5, 0.5)", caused=U)),
])
def test_an_unproven_actuating_step_fails_a_met_contract(goal, contract, last) -> None:
    """Only a housekeeping row (read-only chain, or a declared stop) is exempt under a
    met contract; an actuating row that did not GROUND still rejects the turn."""
    first = _row(contract[0])
    t = _trace(goal, first, last, contract=contract, start=(False,), end=(True,), world="apartment")
    assert turn_reject_reason(t, GO2 | MANIP) == "native_step_1: " + ts.step_reject_reason(
        t.steps[1], t.goal_tree.sub_goals[1], GO2 | MANIP, goal)
    verify, rec = last
    exempt = _trace(goal, first, (verify, replace(rec, housekeeping=True)),
                    contract=contract, start=(False,), end=(True,), world="apartment")
    assert evidence_passed(exempt, GO2 | MANIP), turn_reject_reason(exempt, GO2 | MANIP)


# Verifier P8's sentence was "... 然后左转90度"; K1 (SP1V) compiles no contract for it, so
# the coordinate contract these traces carry is the plain coordinate goal's.
_P8 = "走到坐标 (9.0,3.0)"


@pytest.mark.parametrize("turn, stop", [
    # verifier P8: the turn skill FAILS, then stop and the same vacuous re-verify
    (_row("True", ok=False), _row("True", housekeeping=True)),
    # verifier P1: an UNCAUSED (teleported) facing PASS on the stop row
    (_row("facing(0.0)", vr=False), _row("facing(0.0)", caused=U, housekeeping=True)),
    # an unproven turn re-verified after the stop
    (_row("facing(0.0)", caused=U), _row("facing(0.0)", caused=U, housekeeping=True)),
    # an all-housekeeping group with a FAILED attempt is not exempt either
    (_row("True", ok=False, housekeeping=True), _row("True", housekeeping=True)),
])
def test_a_housekeeping_row_never_launders_an_earlier_attempt(turn, stop) -> None:
    """A stop row joins the attempt group of the same verify text and supersedes
    it; it is exempt only when its WHOLE group is housekeeping and succeeded, so it
    can never stand in for a FAILED or unproven actuating attempt."""
    t = _trace(_P8, _row("at_position(9.0, 3.0)"), turn, stop,
               contract=("at_position(9.0, 3.0)",), start=(False,), end=(True,))
    assert ts.graded_indices(t) == [0, 2]
    assert turn_reject_reason(t, GO2).startswith("native_step_2: ")


def test_a_stop_reverify_caused_against_the_group_baseline_still_verifies() -> None:
    """The legit shape the rule keeps: the turn's verify FAILs mid-motion, the stop's
    re-verify PASSes CAUSED against the group baseline, so it GROUNDs on its own."""
    t = _trace(_P8, _row("at_position(9.0, 3.0)"), _row("facing(1.57)", vr=False),
               _row("facing(1.57)", housekeeping=True),
               contract=("at_position(9.0, 3.0)",), start=(False,), end=(True,))
    assert evidence_passed(t, GO2), turn_reject_reason(t, GO2)


def test_the_spine_recompiles_the_contract_a_producer_supplies() -> None:
    """A producer bug that hands the spine a weaker contract (the go2-only grab as a
    navigation contract, which would also unlock the housekeeping exemption) rejects."""
    t = _trace("go to (3,0) and grab the red can", _row("at_position(3.0, 0.0)"),
               _row("", housekeeping=True),
               contract=("at_position(3.0, 0.0)",), start=(False,), end=(True,))
    assert turn_reject_reason(t, GO2) == "the task contract does not match the goal"


def test_a_housekeeping_row_is_exempt_only_under_a_contract() -> None:
    rows = (_row("at_position(3.0, 0.0)"), _row("", housekeeping=True))
    assert not evidence_passed(_trace("走到坐标 (3,0)", *rows), GO2)
    assert evidence_passed(_trace("走到坐标 (3,0)", *rows, contract=("at_position(3.0, 0.0)",),
                                  start=(False,), end=(True,)), GO2)


def test_a_failing_non_contract_step_still_fails_the_turn() -> None:
    t = _trace(GRAB, _row(HOLD), _row("arm_at_home()", vr=False),
               contract=(HOLD,), start=(False,), end=(True,), world="apartment")
    assert "verify FAIL" in turn_reject_reason(t, MANIP)


# ---------------------------------------------------------------------------
# ActorEpoch: only the actor's own tools count as commanded motion
# ---------------------------------------------------------------------------


def _bare_go2():
    """The generic LeggedBase with just the state set_velocity touches (no sim)."""
    import threading
    from types import SimpleNamespace

    from vector_os_nano.sim.drivers.base import LeggedBase

    go2 = LeggedBase.__new__(LeggedBase)
    go2._spec = SimpleNamespace(limits=(0.8, 0.4, 4.0))
    go2._lock, go2._owners, go2._cmd, go2._hold, go2._cmd_motion = threading.Lock(), [], (0.0,) * 3, None, 0.0
    return go2


def test_commanded_motion_counts_only_inside_a_dispatched_tool() -> None:
    """docs-vs-code:inprocess-nav-caused-counter: a bridge / TARE / background write
    between tools is not the actor's command."""
    from vector_os_nano.sim.runtime import ACTOR_EPOCH

    go2 = _bare_go2()
    go2.set_velocity(0.5, 0.0, 0.0)           # autonomous: no tool is running
    assert go2.cmd_motion() == 0.0
    with ACTOR_EPOCH.tool():
        go2.set_velocity(0.5, 0.0, 0.0)       # the dispatched tool's command
    assert go2.cmd_motion() == pytest.approx(0.5)
    assert not ACTOR_EPOCH.active


# ---------------------------------------------------------------------------
# Mutation sensitivity (graft g): switch one guard off, an adversarial case passes.
# ---------------------------------------------------------------------------


def test_mutation_contract_guard(monkeypatch) -> None:
    t = _trace(GRAB, _row(HOLD), contract=(HOLD,), start=(True,), end=(True,), world="apartment")
    assert not evidence_passed(t, MANIP)
    monkeypatch.setattr(ts, "_contract_reason", lambda *a: "")
    assert evidence_passed(t, MANIP)


def test_mutation_f_to_t_guard(monkeypatch) -> None:
    before = _b(base_cmd_motion=0.0, base_pos=(0.0, 0.0, 0.3), base_heading=0.0)
    after = _b(base_cmd_motion=1.0, base_pos=(0.1, 0.0, 0.3), base_heading=0.0)
    assert grade_channels(before, after, "at_position(0.0, 0.0)", TAGS)[0] is U
    monkeypatch.setattr(ac, "_base_truth_at", lambda *a: None)
    assert grade_channels(before, after, "at_position(0.0, 0.0)", TAGS)[0] is C


def test_mutation_attempt_causation_guard(monkeypatch) -> None:
    t = _trace("走到坐标 (9.0,3.0)", _row("at_position(9.0, 3.0)", vr=False),
               _row("at_position(9.0, 3.0)", caused=U), contract=("at_position(9.0, 3.0)",), start=(False,), end=(True,))
    assert not evidence_passed(t, GO2)
    monkeypatch.setattr(ts, "_actor_uncaused", lambda step: False)
    assert evidence_passed(t, GO2)


def test_mutation_attempt_key_guard(monkeypatch) -> None:
    t = _trace("走到坐标 (9.0,3.0)", _row("at_position(9.0, 3.0)", vr=False),
               _row("at_position(9.0, 3.0, 2.0)"), contract=("at_position(9.0, 3.0)",), start=(False,), end=(True,))
    assert not evidence_passed(t, GO2)
    monkeypatch.setattr(ts, "attempt_key", lambda expr: "")  # every predicate one group
    assert evidence_passed(t, GO2)


def test_mutation_manipulation_gate(monkeypatch) -> None:
    t = _trace("那个绿色的东西", _row("in_view('can')", caused=ActorCaused.NOT_GRADED),
               intents=("grasp",))
    assert not evidence_passed(t, MANIP)
    monkeypatch.setattr(ts, "_manipulation_reason", lambda *a: "")
    assert evidence_passed(t, MANIP)


def test_mutation_actor_epoch(monkeypatch) -> None:
    from vector_os_nano.sim.runtime import ActorEpoch

    go2 = _bare_go2()
    go2.set_velocity(0.5, 0.0, 0.0)
    assert go2.cmd_motion() == 0.0
    monkeypatch.setattr(ActorEpoch, "active", property(lambda self: True))
    go2.set_velocity(0.5, 0.0, 0.0)
    assert go2.cmd_motion() > 0.0


def test_verdict_reports_actor_caused_and_reasons() -> None:
    """tests:negative-tests-cannot-isolate-gate: the verdict says WHY per step."""
    t = _trace("走到坐标 (9.0,3.0)", _row("at_position(9.0, 3.0)", caused=U, channels=("base:at_position:UNCAUSED",)))
    r = VerdictReport.from_trace(t, GO2)
    step = r.to_dict()["per_step"][0]
    assert step["actor_caused"] == "UNCAUSED"
    assert step["reject_reason"] == "UNCAUSED (base:at_position:UNCAUSED)"
    assert r.reject_reason.endswith("UNCAUSED (base:at_position:UNCAUSED)")
    assert SimpleNamespace(**r.to_dict()).acted is False
