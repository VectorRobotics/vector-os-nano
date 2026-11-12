# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""I4 G2 — the task contract binds its object / surface through the world manifest,
and kept_holding is caused only by a grasp in the turn.

Before: the fetch contract was ``kept_holding()`` (no target; the oracle needs one, so
it read False and no fetch could ever verify), a place goal compiled no contract at
all, and a contract compiled against a dev body before ``start_simulation`` was
silently empty. Now: fetch -> kept_holding(<obj>) and near_object('user'); place ->
object_on(<obj>, <surface>); grasp -> holding_object(<obj>); a real fetch GROUNDS,
while an object already held at turn start, or dropped during the carry, never does.
"""
from __future__ import annotations

import pytest

from vector_os_nano.vcli.backends.fake import FakeToolScriptBackend, tool_turn
from tests.unit.vcli.test_native_loop import (
    _NavBase, _Skill, _make_agent, _make_engine, _session, _verdict,
)
from vector_os_nano.vcli.cognitive.actor_causation import ActorBaseline, ActorCaused, grade_channels
from vector_os_nano.vcli.cognitive.object_goal import backs_contract, task_contract
from vector_os_nano.vcli.cognitive.oracle_registry import OracleTag
from vector_os_nano.vcli.cognitive.trace_store import turn_reject_reason
from vector_os_nano.vcli.cognitive.types import ExecutionTrace, GoalTree, StepRecord, SubGoal

APT = frozenset({"at_position", "holding_object", "kept_holding", "near_object", "object_on",
                 "released", "object_at", "lifted"})
GREEN = "pickable_bottle_green"


# ---------------------------------------------------------------------------
# Compile: the manifest binds the object and the surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("goal, contract", [
    ("把绿色瓶子拿过来", (f"kept_holding('{GREEN}')", "near_object('user')")),
    ("bring me the red can", ("kept_holding('pickable_can_red')", "near_object('user')")),
    ("拿起绿色的瓶子", (f"holding_object('{GREEN}')",)),
    ("把绿色瓶子放到餐桌上", (f"object_on('{GREEN}', 'dining_table')",)),
    ("put the yellow cup on the coffee table", ("object_on('pickable_cup_yellow', 'coffee_table')",)),
    ("拿起绿色瓶子放到茶几上", (f"object_on('{GREEN}', 'coffee_table')",)),
    # a bare "table" names none of the four surfaces: the destination binds None and
    # fails closed (I4c; before I4F2 it decayed to released, so a drop anywhere grounded)
    ("把绿色瓶子放到桌子上", (f"object_on('{GREEN}', None)",)),
    # a pure release: the bottle out of the hand at turn end (I4F1-D3)
    ("put the green bottle down", (f"released('{GREEN}')",)),
    ("put the bottle down", ("released(None)",)),
    # REF1: a category alone binds when exactly one object of the world has it (was None)
    ("put the can down", ("released('pickable_can_red')",)),
    ("把那个罐子拿过来", ("kept_holding('pickable_can_red')", "near_object('user')")),
    # unresolvable object: None binds, the item is no evidence (fails closed)
    ("把那个瓶子拿过来", ("kept_holding(None)", "near_object('user')")),
    ("把瓶子放到餐桌上", ("object_on(None, 'dining_table')",)),
    # I4-V2: a grasp binds like a fetch: two bottles bind None
    ("拿起瓶子", ("holding_object(None)",)),
    # I4d: a word the world has no object for fills no object slot: no closed form, so
    # unchecked (the manipulation gate still rejects it: test_a_grasp_of_an_unresolvable_...)
    ("pick up the apple", ()),
])
def test_the_contract_binds_through_the_world_manifest(goal, contract) -> None:
    assert task_contract(goal, APT, "apartment") == contract


def test_without_a_world_no_manipulation_contract_compiles() -> None:
    """I4d (charter 3.5): the slots bind only through a manifest, so without one a
    manipulation goal is unchecked (the fetch was kept_holding(None): never met)."""
    assert task_contract("把绿色瓶子拿过来", APT) == ()
    assert task_contract("把绿色瓶子放到餐桌上", APT) == ()


def test_a_contract_item_is_backed_by_any_name_of_the_same_object() -> None:
    item = f"kept_holding('{GREEN}')"
    assert backs_contract(item, "kept_holding('green bottle') and near_object('user', 1.5)", "apartment")
    assert backs_contract("near_object('user')", "kept_holding('green bottle') and near_object('user', 1.5)")
    assert not backs_contract(item, "kept_holding('red can')", "apartment")         # another object
    assert not backs_contract(item, "kept_holding('green bottle') or True", "apartment")  # not asserted
    assert not backs_contract("near_object('user')", "near_object('green bottle')")


def _hand_trace(goal, verify, *, contract, start, end, world="apartment", attempt="native"):
    sg = SubGoal(name="s0", description="s0", verify=verify)
    st = StepRecord("s0", "skill", True, True, 0.0, actor_caused=ActorCaused.CAUSED, attempt_of=attempt)
    return ExecutionTrace(GoalTree(goal, (sg,)), (st,), True, 0.0, contract=contract,
                          contract_start=start, contract_end=end, world=world)


def test_an_unresolved_object_fails_closed_with_its_reason() -> None:
    t = _hand_trace("把那个瓶子拿过来", "kept_holding('green bottle') and near_object('user')",
                    contract=("kept_holding(None)", "near_object('user')"), start=(False, False), end=(True, True))
    assert "names no object" in turn_reject_reason(t, APT)


def test_a_native_trace_without_its_contract_is_rejected() -> None:
    """The rebind hole: a contract compiled against the dev body (empty) never skips
    the check; the spine recompiles against the body that graded the turn."""
    t = _hand_trace("走到坐标 (3.0,0.0)", "at_position(3.0, 0.0)", contract=(), start=(), end=())
    assert turn_reject_reason(t, APT) == "the task contract does not match the goal"
    legacy = _hand_trace("走到坐标 (3.0,0.0)", "at_position(3.0, 0.0)", contract=(), start=(), end=(), attempt="")
    assert turn_reject_reason(legacy, APT) == ""  # the legacy executor (R1 deletes it)


# ---------------------------------------------------------------------------
# kept_holding: caused only by a grasp in this turn (window="turn")
# ---------------------------------------------------------------------------

_TAGS = {"kept_holding": OracleTag(scope="robot", channel="gripper", transition="rise", window="turn")}


def test_kept_holding_is_judged_from_the_turn_start() -> None:
    from vector_os_nano.vcli.worlds.world_manifest import load_world

    held, empty = ActorBaseline(gripper_welds={GREEN: True}), ActorBaseline(gripper_welds={GREEN: False})
    expr, apt = "kept_holding('green bottle')", load_world("apartment")  # the alias resolves via the manifest
    # the carry step's own baseline already holds it; the grasp was earlier this turn
    assert grade_channels(held, held, expr, _TAGS, turn_baseline=empty, world=apt)[0] is ActorCaused.CAUSED
    # held before the turn began: never the actor's fetch
    assert grade_channels(held, held, expr, _TAGS, turn_baseline=held, world=apt)[0] is ActorCaused.UNCAUSED
    assert grade_channels(held, held, expr, _TAGS, world=apt)[0] is ActorCaused.UNCAUSED  # no turn baseline


# ---------------------------------------------------------------------------
# Through the native runner on the fake apartment (go2_piper oracles)
# ---------------------------------------------------------------------------


class _BottleArm:
    _connected = True

    def __init__(self, held: bool = False) -> None:
        self.held = held

    def get_object_positions(self):
        return {GREEN: [0.0, 0.0, 0.3] if self.held else [10.9, 3.0, 0.02]}

    def get_joint_positions(self):
        return [0.0] * 6


class _Gripper:
    def __init__(self, arm: _BottleArm) -> None:
        self._arm = arm

    def is_holding(self) -> bool:
        return self._arm.held

    def weld_is_active(self) -> dict:
        return {GREEN: self._arm.held}


def _fetch(script, *, held_at_start=False, extra=()):
    from tests.harness.fake_gt import bind_fake_sim, unbind_fake_sim
    from vector_os_nano.sim.session import session_for

    agent, _ = _make_agent(10.0, 3.0)
    agent._base = _NavBase(10.0, 3.0)
    arm = _BottleArm()
    agent._arm, agent._gripper = arm, _Gripper(arm)
    unbind_fake_sim(session_for(agent))
    bind_fake_sim(agent, "go2_piper")
    arm.held = held_at_start
    gt = session_for(agent).gt
    agent._skill_registry.register(
        _Skill("grasp_fake", lambda: setattr(arm, "held", True), verify_hint="holding_object()"))
    for make in extra:
        agent._skill_registry.register(make(arm, gt))
    backend = FakeToolScriptBackend.from_tool_script(
        [*(tool_turn(c) for c in script), tool_turn(("finish", {}))])
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("把绿色瓶子拿过来", session=_session())
    return trace, _verdict(agent, eng, trace)


_GRASP = ("grasp_fake", {"object": "green bottle"})
_TO_USER = ("navigate", {"x": 8.0, "y": 3.0})
_DELIVERED = ("verify", {"expr": "kept_holding('green bottle') and near_object('user')"})


def test_a_real_fetch_grounds() -> None:
    trace, rep = _fetch([_GRASP, _TO_USER, _DELIVERED])
    assert trace.contract == (f"kept_holding('{GREEN}')", "near_object('user')")
    assert (trace.contract_start, trace.contract_end) == ((False, False), (True, True))
    assert trace.world == "apartment"
    assert rep.verified is True, rep.reject_reason


def test_an_object_held_at_turn_start_never_grounds_a_fetch() -> None:
    trace, rep = _fetch([_TO_USER, _DELIVERED], held_at_start=True)
    assert trace.contract_start[0] is True
    assert rep.verified is False
    assert trace.steps[-1].actor_caused is ActorCaused.UNCAUSED
    assert "gripper:kept_holding:UNCAUSED" in trace.steps[-1].channels


def test_a_drop_during_the_carry_never_grounds_a_fetch() -> None:
    def _fumble(arm, gt):  # dropped and picked up again mid-carry
        def run():
            for held in (False, True):
                arm.held = held
                gt.sim_time += 1.0
                gt.attachments()  # the GT attach log records the transition
        return _Skill("fumble", run)

    trace, rep = _fetch([_GRASP, ("fumble", {}), _TO_USER, _DELIVERED], extra=(_fumble,))
    assert trace.contract_end[0] is False  # held at the end, but dropped this turn
    assert rep.verified is False


def test_delivering_without_proving_the_hold_is_not_a_fetch() -> None:
    trace, rep = _fetch([_GRASP, _TO_USER, ("verify", {"expr": "near_object('user')"})])
    assert rep.verified is False
    assert rep.reject_reason.startswith("no actor-caused step proves kept_holding"), rep.reject_reason


def test_the_contract_start_fails_closed_on_an_unevaluable_predicate() -> None:
    """(WP07, moved here) A contract predicate the verifier cannot evaluate at turn
    start reads as already satisfied, never as 'not yet': an arm-only body has no
    near_object, so a fetch there can never be the turn's F->T."""
    from tests.harness.fake_gt import bind_fake_sim, unbind_fake_sim
    from vector_os_nano.sim.session import session_for

    class _BananaArm(_BottleArm):
        def get_object_positions(self):
            return {"banana": [0.0, 0.0, 0.3] if self.held else [0.3, 0.0, 0.02]}

    class _BananaGripper(_Gripper):
        def weld_is_active(self) -> dict:
            return {"banana": self._arm.held}

    agent, _ = _make_agent(0.0, 0.0)
    arm = _BananaArm()
    agent._base, agent._arm, agent._gripper = None, arm, _BananaGripper(arm)
    unbind_fake_sim(session_for(agent))
    bind_fake_sim(agent, "so101")
    agent._skill_registry.register(
        _Skill("grasp_fake", lambda: setattr(arm, "held", True), verify_hint="holding_object()"))
    backend = FakeToolScriptBackend.from_tool_script([tool_turn(_GRASP), tool_turn(("finish", {}))])
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("把香蕉拿过来", session=_session())
    assert trace.contract == ("kept_holding('banana')", "near_object('user')")
    assert trace.contract_start == (False, True)  # near_object is unbound on the arm
    assert _verdict(agent, eng, trace).verified is False


def test_an_unresolvable_fetch_object_fails_closed_through_the_runner() -> None:
    """Replaces WP07's unevaluable-start probe: under the WP10 oracles (fail safe to
    False) the unbound object is caught by the spine, not by the start read."""
    from tests.harness.fake_gt import bind_fake_sim, unbind_fake_sim
    from vector_os_nano.sim.session import session_for

    agent, _ = _make_agent(10.0, 3.0)
    arm = _BottleArm()
    agent._arm, agent._gripper = arm, _Gripper(arm)
    unbind_fake_sim(session_for(agent))
    bind_fake_sim(agent, "go2_piper")
    agent._skill_registry.register(
        _Skill("grasp_fake", lambda: setattr(arm, "held", True), verify_hint="holding_object()"))
    backend = FakeToolScriptBackend.from_tool_script([tool_turn(_GRASP), tool_turn(("finish", {}))])
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("把那个瓶子拿过来", session=_session())
    assert trace.contract == ("kept_holding(None)", "near_object('user')")
    rep = _verdict(agent, eng, trace)
    assert rep.verified is False and "names no object" in rep.reject_reason


def test_a_lifecycle_tool_recompiles_the_contract_for_the_new_body() -> None:
    """A dev-world turn has no at_position, so its contract is empty until
    start_simulation rebinds the body; then the goal's coordinate contract is compiled
    (and printed) against the new body — the one the spine grades the turn on."""
    from tests.unit.vcli.test_native_loop import _FakeStartSim
    from vector_os_nano.vcli.cognitive.trace_store import verify_oracle_names
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.permissions import PermissionContext
    from vector_os_nano.vcli.tools.base import CategorizedToolRegistry
    from vector_os_nano.vcli.verdict import VerdictReport
    from vector_os_nano.vcli.worlds.dev import DevWorld

    backend = FakeToolScriptBackend.from_tool_script([
        tool_turn(("start_simulation", {})),
        tool_turn(("walk", {"distance": 2.0, "speed": 0.3})),
        tool_turn(("verify", {"expr": "at_position(2.0, 0.0, 1.0)"})),
        tool_turn(("finish", {})),
    ])
    ref: list = []
    registry = CategorizedToolRegistry()
    registry.register(_FakeStartSim(ref), category="sim")
    eng = VectorEngine(backend=backend, registry=registry, permissions=PermissionContext(no_permission=True))
    ref.append(eng)
    eng.bind(None, DevWorld())
    app_state: dict = {"agent": None}
    events: list[str] = []
    trace = eng.run_turn_native("启动 go2 然后走到坐标 (2.0,0.0)", session=_session(),
                                app_state=app_state, on_progress=events.append)
    assert trace.contract == ("at_position(2.0, 0.0)",)
    assert (trace.contract_start, trace.contract_end) == ((False,), (True,))
    assert "Task contract: at_position(2.0, 0.0)" in events
    report = VerdictReport.from_trace(trace, verify_oracle_names(app_state["agent"], eng))
    assert report.verified is True, report.reject_reason


# ---------------------------------------------------------------------------
# producers:place-deliver-goals-ungated / spine:no-turn-contract-fetch-place: a place
# or fetch goal never grounds on an unrelated caused predicate, nor on another object
# ---------------------------------------------------------------------------


def _one_row(goal, verify, contract=None, start=None, end=None):
    contract = task_contract(goal, APT, "apartment") if contract is None else contract
    n = len(contract)
    return _hand_trace(goal, verify, contract=contract, start=start or (False,) * n, end=end or (True,) * n)


@pytest.mark.parametrize("goal, verify", [
    ("put the red can on the dining table", "at_position(3.0, 7.5)"),     # walked there, placed nothing
    ("把绿色瓶子拿过来", "near_object('user')"),                          # arrived, holding unproven
    ("拿起绿色的瓶子", "holding_object('red can')"),                       # the wrong object
    ("把绿色瓶子放到餐桌上", "object_on('green bottle', 'coffee_table')"),  # the wrong surface
])
def test_a_goal_never_grounds_on_another_caused_predicate(goal, verify) -> None:
    assert turn_reject_reason(_one_row(goal, verify), APT) != ""


def test_a_place_naming_no_surface_the_world_resolves_fails_closed() -> None:
    """I4F2-D1: "the table" fits four surfaces, so the destination binds None; a release
    (or any other proof) never stands in for the placement."""
    t = _one_row("place the red can on the table", "at_position(10.9, 3.0)")
    assert t.contract == ("object_on('pickable_can_red', None)",)
    assert "names no destination" in turn_reject_reason(t, APT)
    assert "names no destination" in turn_reject_reason(
        _one_row("place the red can on the table", "released('red can')"), APT)
    # "the bottle" names no one object: the object slot is the one reported
    assert "names no object" in turn_reject_reason(
        _one_row("place the bottle on the table", "released('green bottle')"), APT)
    # REF1: "the can" names the one can, so the unbound destination is the one reported
    assert "names no destination" in turn_reject_reason(
        _one_row("place the can on the table", "released('red can')"), APT)


def test_the_right_object_on_the_right_surface_grounds() -> None:
    assert turn_reject_reason(_one_row("把绿色瓶子放到餐桌上", "object_on('green bottle', 'dining_table')"), APT) == ""
    assert turn_reject_reason(_one_row("拿起绿色的瓶子", f"holding_object('{GREEN}')"), APT) == ""


# ---------------------------------------------------------------------------
# I4-V1: a place goal is never proven by a grasp alone (the grasp intent, from the
# goal or from the dispatched grasp skill, adds the hold proof; it never replaces the
# place proof). I4-V2: a grasp naming an object the world cannot resolve fails closed.
# ---------------------------------------------------------------------------


def _rows(goal, verifies, *, intents=()):
    contract = task_contract(goal, APT, "apartment")
    sgs = tuple(SubGoal(name=f"s{i}", description="", verify=v) for i, v in enumerate(verifies))
    sts = tuple(StepRecord(f"s{i}", "skill", True, True, 0.0, actor_caused=ActorCaused.CAUSED, attempt_of="native")
                for i in range(len(verifies)))
    n = len(contract)
    return ExecutionTrace(GoalTree(goal, sgs), sts, True, 0.0, contract=contract, contract_start=(False,) * n,
                          contract_end=(True,) * n, intents=tuple(intents), world="apartment")


def test_pick_and_place_is_not_proven_by_a_hold() -> None:
    goal = "pick up the green bottle and put it down"
    t = _rows(goal, [f"holding_object('{GREEN}')"])
    assert "released" in turn_reject_reason(t, APT), t.contract
    both = _rows(goal, [f"holding_object('{GREEN}')", "released('green bottle')"])
    assert turn_reject_reason(both, APT) == ""


@pytest.mark.parametrize("goal", ["拿起绿色瓶子放到桌子上", "pick up the green bottle and place it on the table"])
def test_a_pick_and_place_to_an_unresolved_destination_is_never_proven(goal) -> None:
    """I4c: the end state of a pick-and-place is object_on; "the table" binds None, so
    neither the hold nor the hold plus a release proves it."""
    assert _rows(goal, []).contract == (f"object_on('{GREEN}', None)",)
    for verifies in ([f"holding_object('{GREEN}')"], [f"holding_object('{GREEN}')", "released('green bottle')"]):
        assert "names no destination" in turn_reject_reason(_rows(goal, verifies), APT)


def test_a_pure_place_with_a_dispatched_grasp_needs_both_proofs() -> None:
    # a pure release (I4c: "on the table" names a destination the world does not resolve)
    hold = _rows("put the red can down", ["holding_object('red can')"], intents=("grasp",))
    assert "released" in turn_reject_reason(hold, APT)
    place = _rows("put the red can down", ["released('red can')"], intents=("grasp",))
    assert "holding_object/kept_holding/lifted" in turn_reject_reason(place, APT)
    both = _rows("put the red can down", ["holding_object('red can')", "released('red can')"],
                 intents=("grasp",))
    assert turn_reject_reason(both, APT) == ""


@pytest.mark.parametrize("goal, contract, why", [
    ("把绿色瓶子放到桌子上", f"object_on('{GREEN}', None)", "names no destination"),  # I4c
    ("put the green bottle down", f"released('{GREEN}')", "released"),
])
def test_a_place_goal_that_only_grasped_is_rejected_through_the_runner(goal, contract, why) -> None:
    from tests.harness.fake_gt import bind_fake_sim, unbind_fake_sim
    from vector_os_nano.sim.session import session_for

    agent, _ = _make_agent(10.0, 3.0)
    agent._base = _NavBase(10.0, 3.0)
    arm = _BottleArm()
    agent._arm, agent._gripper = arm, _Gripper(arm)
    unbind_fake_sim(session_for(agent))
    bind_fake_sim(agent, "go2_piper")
    agent._skill_registry.register(
        _Skill("grasp_fake", lambda: setattr(arm, "held", True), verify_hint="holding_object()"))
    backend = FakeToolScriptBackend.from_tool_script([tool_turn(_GRASP), tool_turn(("finish", {}))])
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native(goal, session=_session())
    rep = _verdict(agent, eng, trace)
    assert trace.contract == (contract,) and "grasp" in trace.intents, (trace.contract, trace.intents)
    assert rep.verified is False and why in rep.reject_reason, rep.reject_reason


@pytest.mark.parametrize("goal, verify", [
    ("pick up the apple", f"holding_object('{GREEN}')"),       # the world has no apple
    ("pick up a bottle", "holding_object('pickable_can_red')"),  # two bottles, and a can
    ("拿起瓶子", f"holding_object('{GREEN}')"),                  # two bottles: which one is unknown
])
def test_a_grasp_of_an_unresolvable_object_fails_closed(goal, verify) -> None:
    assert "names no object" in turn_reject_reason(_one_row(goal, verify), APT)


# ---------------------------------------------------------------------------
# I4F1-D2: every manipulation proof names the goal's object (the wrong object never
# grounds, through an alias either). I4F1-D3: a place ends with the object out of the
# hand (released(obj) at turn end), so proofs out of order never verify.
# ---------------------------------------------------------------------------

RED = "pickable_can_red"


class _TwoArm:
    """The red can and the green bottle; each rises to the EE while held."""

    _connected = True

    def __init__(self) -> None:
        self.held = {RED: False, GREEN: False}

    def get_object_positions(self):
        rest = {RED: [10.6, 3.0, 0.02], GREEN: [10.9, 3.0, 0.02]}
        return {n: [0.0, 0.0, 0.3] if self.held[n] else rest[n] for n in rest}

    def get_joint_positions(self):
        return [0.0] * 6


class _TwoGripper:
    def __init__(self, arm: _TwoArm) -> None:
        self._arm = arm

    def is_holding(self) -> bool:
        return any(self._arm.held.values())

    def weld_is_active(self) -> dict:
        return dict(self._arm.held)


def _two(goal, script, *, held=()):
    from tests.harness.fake_gt import bind_fake_sim, unbind_fake_sim
    from vector_os_nano.sim.session import session_for

    agent, _ = _make_agent(10.0, 3.0)
    agent._base = _NavBase(10.0, 3.0)
    arm = _TwoArm()
    agent._arm, agent._gripper = arm, _TwoGripper(arm)
    unbind_fake_sim(session_for(agent))
    bind_fake_sim(agent, "go2_piper")
    arm.held.update({n: True for n in held})
    for tag, name in (("can", RED), ("green", GREEN)):
        agent._skill_registry.register(_Skill(f"grasp_{tag}", lambda n=name: arm.held.__setitem__(n, True)))
        agent._skill_registry.register(_Skill(f"open_{tag}", lambda n=name: arm.held.__setitem__(n, False)))
    backend = FakeToolScriptBackend.from_tool_script(
        [*(tool_turn(c) for c in script), tool_turn(("finish", {}))])
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native(goal, session=_session())
    return trace, _verdict(agent, eng, trace)


def _v(expr):
    return ("verify", {"expr": expr})


_HOLD_GREEN, _REL_GREEN = _v("holding_object('green bottle')"), _v("released('green bottle')")


def test_releasing_another_object_never_places_the_named_one() -> None:
    """Probe 1: the red can is released, the green bottle never touched."""
    trace, rep = _two("put the green bottle down", [("open_can", {}), _REL_GREEN], held=(RED,))
    assert trace.steps[0].actor_caused is ActorCaused.UNCAUSED, trace.steps[0].channels
    assert rep.verified is False


def test_a_pick_and_place_of_another_object_never_grounds() -> None:
    """Probe 2: the hold proof names the red can; the release alias is not caused by it."""
    trace, rep = _two("pick up the green bottle and put it down",
                      [("grasp_can", {}), _v("holding_object('red can')"), ("open_can", {}), _REL_GREEN])
    assert trace.contract == (f"released('{GREEN}')",)
    assert rep.verified is False


@pytest.mark.parametrize("goal", ["put the green bottle down", "pick up the green bottle and put it down"])
def test_a_place_that_ends_with_the_object_in_hand_never_grounds(goal) -> None:
    """D3: released, then grasped again: both proofs CAUSED, but the bottle ends held."""
    trace, rep = _two(goal, [("open_green", {}), _REL_GREEN, ("grasp_green", {}), _HOLD_GREEN], held=(GREEN,))
    assert [s.actor_caused for s in trace.steps] == [ActorCaused.CAUSED, ActorCaused.CAUSED]
    assert rep.verified is False and "not met at turn end" in rep.reject_reason, rep.reject_reason


def test_a_place_and_a_pick_and_place_of_the_named_object_ground() -> None:
    trace, rep = _two("put the green bottle down", [("open_green", {}), _REL_GREEN], held=(GREEN,))
    assert rep.verified is True, rep.reject_reason
    trace, rep = _two("pick up the green bottle and put it down",
                      [("grasp_green", {}), _HOLD_GREEN, ("open_green", {}), _REL_GREEN])
    assert trace.contract_start == (True,)  # resting at start: exempt only after the grasp
    assert rep.verified is True, rep.reject_reason


def test_a_resting_object_released_before_any_grasp_is_already_satisfied() -> None:
    """The pick-and-place exemption needs the grasp proof FIRST."""
    t = _rows("pick up the green bottle and put it down", ["released('green bottle')", f"holding_object('{GREEN}')"])
    t = ExecutionTrace(t.goal_tree, t.steps, True, 0.0, contract=t.contract, contract_start=(True,),
                       contract_end=(True,), world="apartment")
    assert turn_reject_reason(t, APT).startswith("already satisfied at turn start")


def test_an_alias_target_is_caused_only_by_its_own_weld() -> None:
    from vector_os_nano.vcli.worlds.world_manifest import load_world

    tags = {"released": OracleTag(scope="robot", channel="gripper", transition="fall")}
    before = ActorBaseline(gripper_welds={RED: True, GREEN: False})
    after = ActorBaseline(gripper_welds={RED: False, GREEN: False})
    expr = "released('green bottle')"
    assert grade_channels(before, after, expr, tags, world=load_world("apartment"))[0] is ActorCaused.UNCAUSED
    assert grade_channels(before, after, expr, tags)[0] is ActorCaused.UNCAUSED  # no manifest: not a weld key
    assert grade_channels(before, after, "released('red can')", tags,
                          world=load_world("apartment"))[0] is ActorCaused.CAUSED


@pytest.mark.parametrize("goal, verifies", [
    ("pick up the green bottle and put it down", ["holding_object('red can')", "released('green bottle')"]),
    ("put the green bottle down", ["released('red can')"]),
])
def test_every_manipulation_proof_names_the_goal_object(goal, verifies) -> None:
    assert turn_reject_reason(_rows(goal, verifies), APT) != ""
