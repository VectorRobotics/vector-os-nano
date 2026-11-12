# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""REF1c (charter 3.5, "Manipulation needs a contract"): in a world with a manifest, a
turn that asks for a manipulation or performs one is verified only through a proven
contract.

Before: outside the closed forms the manipulation gate read the goal's object out of
the free sentence (``WorldManifest.mentions``), so "pick up the thing next to the red
can", "the lid of the red can" and "don't pick up the red can" all bound the red can
and the turn verified when the robot held it (the REF1 second verifier: 719 of 3191
goals bound a non-target). A sentence mentions an object for many reasons; only a
closed form says which mention is the goal. Now such a goal compiles no contract, its
goal check is unchecked and the gate rejects it with the forms that can be checked.
"""
from __future__ import annotations

import ast
import dataclasses

import pytest

from tests.unit.vcli.test_goal_check import _v
from tests.unit.vcli.test_native_loop import _CAN, _grasp, _make_arm_agent, _make_engine, _session, _verdict
from vector_os_nano.vcli.backends.fake import FakeToolScriptBackend, tool_turn
from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused
from vector_os_nano.vcli.cognitive.evidence_classifier import classify_verify_expr
from vector_os_nano.vcli.cognitive.object_goal import (
    backs_contract, checkable_goals, named_intents, task_contract, unbound_reason,
)
from vector_os_nano.vcli.cognitive.trace_store import turn_reject_reason
from vector_os_nano.vcli.cognitive.types import ExecutionTrace, GoalTree, StepRecord, SubGoal
from vector_os_nano.vcli.worlds.world_manifest import WORLDS_DIR, load_world

APT = frozenset({"at_position", "holding_object", "kept_holding", "near_object", "object_on",
                 "released", "object_at", "lifted"})
OUTSIDE = "is outside the forms the spine can check"

# The REF1 second verifier's goals whose target is NOT the red can (landmarks, parts,
# exclusions, negations) and the red can under a modifier this world does not have.
NOT_IT = [
    "拿起红罐子旁边的东西", "把红色罐子旁边的东西拿给我", "拿起红色的罐子上面的东西", "拿起红罐子里的东西",
    "拿起红罐子上的标签", "拿起红罐子附近的东西", "拿起红罐子左边的", "拿起红罐子前面的那个",
    "pick up the thing next to the red can", "pick up the lid of the red can", "pick up the red can's lid",
    "grab the label on the red can", "pick up the one beside the red can", "pick up something other than the red can",
    "pick up the red can lid", "bring me the box under the red can",
    "不要拿红罐子", "别拿红罐子", "don't pick up the red can", "do not grab the red can",
    "拿起粉红色的罐子", "拿起红色罐头", "拿起深红色的罐子",
    "pick up the pink red can", "pick up the green red can", "pick up the big red can", "pick up the empty red can",
    "pick up the other red can", "pick up the old red can", "pick up two red cans",
    # REF1F1V-03: an ASCII modifier beside a CJK reference, and English ones
    "拿起pink红罐子", "拿起 pink 红罐子", "pick up the dark red can", "pick up the fake red can",
]
# The closed forms of the same object: still verified, with the goal proven.
IT = ["拿起红罐子", "pick up the red can", "麻烦帮我把红色的罐子拿起来吧", "could you grab the red can for me"]


def _turn(goal: str, verify: str):
    agent, _arm = _make_arm_agent(_grasp)
    backend = FakeToolScriptBackend.from_tool_script(
        [tool_turn(("grasp_fake", {"object": "red can"})), tool_turn(_v(verify)), tool_turn(("finish", {}))])
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native(goal, session=_session())
    return trace, _verdict(agent, eng, trace)


@pytest.mark.parametrize("verify", [f"holding_object('{_CAN}')", "holding_object('red can')"])
@pytest.mark.parametrize("goal", NOT_IT)
def test_holding_the_red_can_never_verifies_a_goal_that_is_not_it(goal, verify) -> None:
    trace, rep = _turn(goal, verify)
    assert rep.verified is False, (goal, rep.goal_check, trace.contract)
    # the grasp step itself is still reported grounded: only the goal is not checked
    assert [s.evidence for s in rep.per_step] == ["GROUNDED"], rep.per_step
    if not trace.contract:  # outside the forms: nothing compiled, nothing failed
        assert rep.goal_check == "unchecked" and OUTSIDE in rep.reject_reason, rep.reject_reason


@pytest.mark.parametrize("goal", IT)
def test_a_closed_form_naming_the_red_can_is_proven(goal) -> None:
    trace, rep = _turn(goal, "holding_object('red can')")
    assert trace.contract == (f"holding_object('{_CAN}')",)
    assert rep.verified is True and rep.goal_check == "proven", (goal, rep.reject_reason)


def test_a_grasp_dispatched_under_a_goal_that_asks_none_is_not_verified() -> None:
    trace, rep = _turn("look around the living room", "holding_object('red can')")
    assert "grasp" in trace.intents and trace.contract == ()
    assert rep.verified is False and rep.goal_check == "unchecked", rep.reject_reason
    assert OUTSIDE in rep.reject_reason and "pick up the" in rep.reject_reason, rep.reject_reason


def _one_row(goal: str, verify: str, world: str) -> ExecutionTrace:
    contract = task_contract(goal, APT, world)
    st = StepRecord("s0", "skill", True, True, 0.0, actor_caused=ActorCaused.CAUSED, attempt_of="native")
    return ExecutionTrace(GoalTree(goal, (SubGoal(name="s0", description="", verify=verify),)), (st,), True, 0.0,
                          contract=contract, contract_start=(False,) * len(contract),
                          contract_end=(True,) * len(contract), world=world)


def test_without_a_manifest_nothing_changes() -> None:
    """Rule 4: no oracle world, no object to bind: the hold proof alone passes the gate."""
    t = _one_row("pick up the thing next to the red can", "holding_object('red can')", "")
    assert turn_reject_reason(t, APT) == ""
    assert OUTSIDE in turn_reject_reason(_one_row(t.goal_tree.goal, "holding_object('red can')", "apartment"), APT)


WORLDS = sorted(p.stem for p in WORLDS_DIR.glob("*.yaml"))


def _can_be_met(item: str, world) -> bool:
    """Every argument of a contract item names a thing *world* has: a pickable, a
    surface, or the user a fetch delivers to."""
    call = ast.parse(item, mode="eval").body
    args = [a.value for a in call.args]
    return all(a in world.pickables or a in world.surfaces or (a == "user" and world.user is not None)
               for a in args)


@pytest.mark.parametrize("world", WORLDS)
def test_every_checkable_goal_the_gate_offers_can_be_met_in_its_world(world) -> None:
    """REF1c-V-03: the gate offers only goals this world can prove: every item of each
    example's contract is a predicate whose every slot names a thing of this world (no
    fetch where there is no user, no place where there is no surface)."""
    w = load_world(world)
    goals = checkable_goals(w, APT)
    assert len(goals) >= 2, goals
    for goal in goals:
        contract = task_contract(goal, APT, world)
        assert contract and all(_can_be_met(i, w) for i in contract), (goal, contract)
        # REF1d-V-02: a place onto the object's own start surface is already true at turn
        # start, and the contract needs it false there
        for i in contract:
            if i.startswith("object_on("):
                obj, dest = (a.value for a in ast.parse(i, mode="eval").body.args)
                assert dest != w.pickables[obj].surface, (goal, i)
    assert any("bring me" in g for g in goals) is (w.user is not None), goals


def test_no_fetch_without_a_user_and_no_place_without_a_surface(monkeypatch) -> None:
    import vector_os_nano.vcli.worlds.world_manifest as wm

    apt = load_world("apartment")
    assert any("bring me" in g for g in checkable_goals(apt, APT))
    assert any(" on the " in g for g in checkable_goals(apt, APT))
    bare = dataclasses.replace(apt, user=None, surfaces={},
                               pickables={k: dataclasses.replace(p, surface="") for k, p in apt.pickables.items()})
    # a body without a base has no near_object: no fetch either
    assert not any("bring me" in g for g in checkable_goals(apt, APT - {"near_object"}))
    monkeypatch.setattr(wm, "load_world", lambda _id: bare)
    goals = checkable_goals(bare, APT)
    assert goals and not any("bring me" in g or "拿过来" in g or " on the" in g for g in goals), goals


@pytest.mark.parametrize("goal", ["把绿色的放到餐桌上", "put the green one on the dining table"])
def test_the_gate_takes_the_object_the_contract_bound(goal) -> None:
    """I4E-VER-1: a colour reference beside a surface word is proven AND verified (the
    gate once resolved the whole sentence, which also held the scenery word)."""
    t = _one_row(goal, "object_on('green bottle', 'dining_table')", "apartment")
    assert t.contract == ("object_on('pickable_bottle_green', 'dining_table')",)
    assert turn_reject_reason(t, APT) == ""
    assert turn_reject_reason(_one_row(goal, "object_on('red can', 'dining_table')", "apartment"), APT) != ""


# REF1c-V-01: a hold the gripper's own close made (gripper_close: its spec names no
# gripper oracle, its driver's close is where the world attaches a gripped body) is a
# manipulation like any grasp: the turn needs a proven contract.
CLOSE_GOALS = ["合上夹爪", "闭合夹爪", "look around the living room", "关闭夹子", "合上夹爪然后回到初始位置"]


def _close_turn(goal: str, verify: str, *, real: bool):
    from types import SimpleNamespace

    from tests.unit.vcli.test_native_loop import _Skill
    from vector_os_nano.core.world_model import WorldModel
    from vector_os_nano.skills.gripper import GripperCloseSkill

    agent, arm = _make_arm_agent()
    if real:  # the shipped skill: the driver's close() grips the red can
        grip = agent._gripper
        grip.close = lambda: setattr(arm, "held", True) or True
        ctx = SimpleNamespace(base=agent._base, arm=arm, gripper=grip, services=None, world_model=WorldModel(),
                              config={"world": "apartment"})
        agent._build_context = lambda: ctx
        agent._skill_registry.register(GripperCloseSkill())
    else:
        agent._skill_registry.register(_Skill("gripper_close", lambda: setattr(arm, "held", True)))
    backend = FakeToolScriptBackend.from_tool_script(
        [tool_turn(("gripper_close", {})), tool_turn(_v(verify)), tool_turn(("finish", {}))])
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native(goal, session=_session())
    return trace, _verdict(agent, eng, trace)


# REF1d-V-01: the literal-left spellings the classifier grounds and the gate once missed.
LITERAL_LEFT = ["True == holding_object('red can')",
                "(True == holding_object('red can')) and (True == lifted('red can'))"]
LITERAL_LEFT_ALONE = [f"True == lifted('{_CAN}')", "True == kept_holding('red can')"]


@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("verify", ["holding_object('red can')", f"lifted('{_CAN}')",
                                    "holding_object('red can') and kept_holding('red can')",
                                    *LITERAL_LEFT, *LITERAL_LEFT_ALONE])
@pytest.mark.parametrize("goal", CLOSE_GOALS)
def test_a_hold_made_by_a_skill_without_a_gripper_hint_needs_a_contract(goal, verify, real) -> None:
    trace, rep = _close_turn(goal, verify, real=real)
    assert "grasp" not in trace.intents  # the dispatch alone does not say it was a grasp
    assert [s.evidence for s in rep.per_step] == ["GROUNDED"], rep.per_step
    assert rep.verified is False and rep.goal_check == "unchecked", (goal, rep.reject_reason)
    assert OUTSIDE in rep.reject_reason, rep.reject_reason


@pytest.mark.parametrize("goal", ["拿起红罐子", "pick up the red can"])
def test_the_same_hold_under_a_closed_form_is_proven(goal) -> None:
    trace, rep = _close_turn(goal, "holding_object('red can')", real=True)
    assert trace.contract == (f"holding_object('{_CAN}')",)
    assert rep.verified is True and rep.goal_check == "proven", rep.reject_reason


@pytest.mark.parametrize("verify", LITERAL_LEFT)
@pytest.mark.parametrize("goal", ["拿起红罐子", "pick up the red can"])
def test_a_literal_left_spelling_never_proves_a_contract(goal, verify) -> None:
    """REF1e-V-01: a contract is proven only by ``<call>`` or ``<call> == True``, as on
    release/1.0; ``True == <call>`` still trips the gate (named_intents) but proves nothing."""
    trace, rep = _close_turn(goal, verify, real=True)
    assert trace.contract and rep.verified is False and rep.goal_check == "failed", rep.reject_reason
    assert not backs_contract(trace.contract[0], verify, "apartment")


@pytest.mark.parametrize("verify", LITERAL_LEFT_ALONE)
def test_a_lift_or_a_kept_hold_alone_never_proves_a_grasp_goal(verify) -> None:
    """The contract is holding_object: a step that proves another oracle backs none of it."""
    trace, rep = _close_turn("pick up the red can", verify, real=True)
    assert rep.verified is False and rep.goal_check == "failed", rep.reject_reason


@pytest.mark.parametrize("verify", ["not holding_object('red can')", "holding_object('red can') or True",
                                    "near_object('red can')"])
def test_a_gripper_turn_that_asserts_no_possession_or_placement_stays_as_it_was(verify) -> None:
    """A verify that is not GROUNDED proves nothing performed: a turn that closes the
    gripper and proves nothing held keeps the verdict the manipulation gate leaves it
    (the same as in a world with no manifest: the gate does not fire)."""
    why = turn_reject_reason(_one_row("合上夹爪", verify, "apartment"), APT)
    assert why == turn_reject_reason(_one_row("合上夹爪", verify, ""), APT) and OUTSIDE not in why, why


@pytest.mark.parametrize("verify", ["not holding_object('red can')",
                                    "released('red can') and not holding_object('red can')"])
def test_not_holding_never_proves_a_release(verify) -> None:
    """``not`` never grounds (the classifier): the step is not evidence, so the turn is
    not verified under the release goal, as on rel/REF1d-fix1; ``released`` alone is."""
    t = _one_row("put the red can down", verify, "apartment")
    assert t.contract == (f"released('{_CAN}')",) and turn_reject_reason(t, APT).startswith("s0: ")
    assert turn_reject_reason(_one_row("put the red can down", "released('red can')", "apartment"), APT) == ""


@pytest.mark.parametrize(("expr", "want"), [
    ("holding_object('x')", {"grasp"}), ("True == lifted('x')", {"grasp"}), ("not kept_holding('x')", {"grasp"}),
    ("object_on('x', 't') or True", {"place"}), ("1 if released('x') else 0", {"place"}),
    ("near_object(object_at('x', 1, 2))", {"place"}), ("holding_object('x') != object_on('x', 't')", {"grasp", "place"}),
    ("near_object('holding_object')", set()), ("at_position(1, 2)", set()), ("holding_object(", set()), ("", set()),
])
def test_named_intents_reads_names_never_what_is_asserted(expr, want) -> None:
    assert named_intents(expr) == frozenset(want)


# REF1e, the point of the package: the gate never re-reads the classifier's grammar. For
# every verify of a generated family, IF the real classifier grades it GROUNDED THEN a
# turn that holds it, in a manifest world, with no compiled contract, is not verified.
_MANIP = ["holding_object('red can')", "lifted('red can')", "kept_holding('red can')", "released('red can')",
          "object_on('red can', 'dining_table')", "object_at('red can', 1.0, 2.0)"]
_OTHER = ["near_object('user')", "at_position(1.0, 2.0)"]
_CONSTS = ["True", "False", "1", "0", "None", "'red can'"]
_OPS = ["==", "!=", "is", "is not", "<", "<=", ">", ">="]


def _shapes(e: str) -> list[str]:
    """*e* under not, a comparison with a constant on either side, a conditional
    expression, a call, parentheses."""
    out = [e, f"({e})", f"not {e}", f"not ({e})", f"{e} if True else False", f"True if {e} else False",
           f"holding_object({e})", f"bool({e})", f"near_object('user', {e})"]
    for c in _CONSTS:
        for op in _OPS:
            out += [f"{e} {op} {c}", f"{c} {op} {e}", f"({c} {op} ({e}))"]
    return out


def _family() -> list[str]:
    one = [x for m in _MANIP for x in _shapes(m)]
    two = [f"{a} {op} {b}" for a in one for b in [*_MANIP, *_OTHER, *_CONSTS] for op in ("and", "or")]
    two += [f"{b} {op} {a}" for a in one for b in [*_MANIP, *_OTHER] for op in ("and", "or")]
    nested = [x for a in _MANIP for b in [*_MANIP, *_OTHER] for x in _shapes(f"{a} and {b}") + _shapes(f"{a} or {b}")]
    return sorted({*one, *two, *nested})


def test_a_grounded_verify_naming_a_manipulation_oracle_never_verifies_without_a_contract() -> None:
    grounded = [e for e in _family() if classify_verify_expr(e, APT) == "GROUNDED"]
    assert len(grounded) > 200, len(grounded)  # the property is not vacuous
    for goal in ("合上夹爪", "look around the living room"):
        for expr in grounded:
            t = _one_row(goal, expr, "apartment")
            assert t.contract == () and OUTSIDE in turn_reject_reason(t, APT), (goal, expr)


# REF1c-V-02 (charter 3.5, a decision): a closed-form slot that is a category word binds
# the world's one object of that category ("pick up the can" in the apartment: the red
# can) and nothing when two fit ("the bottle"), the reason listing the candidates.
def _category_rows() -> list[tuple[str, str, frozenset[str]]]:
    rows = []
    for world in WORLDS:
        w = load_world(world)
        for k in sorted({pk.category for pk in w.pickables.values()}):
            fits = frozenset(n for n, pk in w.pickables.items() if pk.category == k)
            for word in (k, *w.categories.get(k, ())):
                rows += [(world, form, fits) for form in ((word,) if word.isascii() else (word, word + "子"))]
    return rows


def _slots(word: str) -> list[str]:
    if word.isascii():
        return [word, *(f"{d} {word}" for d in ("the", "a", "an", "this", "that", "my", "the the"))]
    return [word, *(f"{d}{word}" for d in ("这个", "那个", "这", "那"))]


def _grasp_goals(slot: str) -> list[str]:
    if slot.isascii():
        return [f"pick up {slot}", f"please grab {slot}", f"pick {slot} up now", f"could you hold {slot} for me"]
    return [f"拿起{slot}", f"请把{slot}拿起来", f"帮我抓住{slot}吧", f"麻烦你捡起{slot}"]


@pytest.mark.parametrize(("world", "word", "fits"), _category_rows())
def test_a_bare_category_binds_the_one_object_of_it_and_nothing_when_two_fit(world, word, fits) -> None:
    w = load_world(world)
    one = next(iter(fits)) if len(fits) == 1 else None
    for slot in _slots(word):
        assert w.resolve(slot) == one, (world, slot)
        for goal in _grasp_goals(slot):
            contract = task_contract(goal, APT, world)
            assert contract == (f"holding_object({one!r})",), (world, goal, contract)
            why = unbound_reason(contract[0], goal, world)
            if one is None:  # the reason quotes the words and lists the candidates
                assert repr(goal) in why and all(w.pickables[n].english in why for n in fits), why
            else:
                assert why == "", why


def test_the_category_table_covers_both_cases_in_both_languages() -> None:
    rows = _category_rows()
    assert {len(f) == 1 for _, _, f in rows} == {True, False}
    assert {w.isascii() for _, w, _ in rows} == {True, False}


# REF1d-V-02 / REF1c-V-03 through the runner: every goal the reject reason offers is proven
# in its world when the scripted body does what it says. Each world's first pickable
# starts where its shipped scene puts it (so101_mujoco.xml: the banana at 0.12, 0.12).
_BODIES = {"apartment": ("go2_piper", (10.9, 3.0, 0.71)), "tabletop": ("so101", (0.12, 0.12, 0.06))}


class _OneArm:
    """The world's first pickable: at an (x, y, z), or "held" 0.24 m above its start."""

    _connected = True

    def __init__(self, name: str, start: tuple[float, float, float], at) -> None:
        self.name, self.start, self.at = name, start, at

    def get_object_positions(self):
        x, y, z = self.start
        return {self.name: [x, y, z + 0.24] if self.at == "held" else list(self.at)}

    def get_joint_positions(self):
        return [0.0] * 6


class _OneGrip:
    def __init__(self, arm: _OneArm) -> None:
        self._arm = arm

    def is_holding(self) -> bool:
        return self._arm.at == "held"

    def weld_is_active(self) -> dict:
        return {self._arm.name: self.is_holding()}


def _offer_run(world: str, goal: str, script: list, at, dest_body: str = "", skills: dict | None = None):
    from tests.harness.fake_gt import bind_fake_sim, unbind_fake_sim
    from tests.unit.vcli.test_native_loop import _NavBase, _Skill, _make_agent
    from vector_os_nano.sim.session import session_for

    body, start = _BODIES[world]
    w = load_world(world)
    agent, _ = _make_agent(10.0, 3.0)
    agent._base = _NavBase(10.0, 3.0)
    arm = _OneArm(next(iter(w.pickables)), start, start if at == "rest" else at)
    agent._arm, agent._gripper = arm, _OneGrip(arm)
    unbind_fake_sim(session_for(agent))
    s = bind_fake_sim(agent, body, support=dest_body or w.surfaces[w.pickables[arm.name].surface].body)
    for name, to in (skills or {}).items():
        agent._skill_registry.register(_Skill(name, lambda to=to: setattr(arm, "at", to)))
    backend = FakeToolScriptBackend.from_tool_script([*(tool_turn(c) for c in script), tool_turn(("finish", {}))])
    eng = _make_engine(agent, backend)
    try:
        trace = eng.run_turn_native(goal, session=_session())
        return trace, _verdict(agent, eng, trace)
    finally:
        unbind_fake_sim(s)


def _offered(world: str) -> list[str]:
    """The goals a free sentence's rejection offers in *world* (its live oracle names)."""
    _, rep = _offer_run(world, "look around", [("grasp", {}), _v("holding_object()")], "rest",
                        skills={"grasp": "held"})
    assert OUTSIDE in rep.reject_reason, rep.reject_reason
    return rep.reject_reason.split("checkable goals: ", 1)[1].split("; ")


@pytest.mark.parametrize("world", sorted(_BODIES))
def test_every_goal_the_reject_reason_offers_is_proven_through_the_runner(world) -> None:
    w = load_world(world)
    pk = next(iter(w.pickables.values()))
    x, y, z = _BODIES[world][1]
    goals = _offered(world)
    assert len(goals) >= 3, goals
    for goal in goals:
        contract = task_contract(goal, APT, world)
        call = ast.parse(contract[0], mode="eval").body
        grasp = [("grasp", {}), _v(f"holding_object({pk.english!r})")]
        at, script, dest, box = "rest", grasp, "", (0.0, 0.0, 0.0, 0.0)
        if call.func.id == "released":
            at, script = "held", [("open", {}), _v(f"released({pk.english!r})")]
        elif call.func.id == "object_on":
            surface = w.surfaces[call.args[1].value]
            dest, box = surface.body, surface.box
            script = [*grasp, ("place", {}), _v(f"object_on({pk.english!r}, {call.args[1].value!r})")]
        elif call.func.id == "kept_holding":
            ux, uy = w.user
            script = [*grasp, ("navigate", {"x": ux, "y": uy}),
                      _v(f"kept_holding({pk.english!r}) and near_object('user')")]
        trace, rep = _offer_run(world, goal, script, at, dest, skills={
            "grasp": "held", "open": (x + 0.05, y, z), "place": ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, z)})
        assert trace.contract == contract and rep.goal_check == "proven", (world, goal, rep.reject_reason)
        assert rep.verified is True, (world, goal, rep.reject_reason)
