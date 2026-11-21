# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""K1 — contract coverage (charter 3.5): the closed whole-sentence forms widened to the
sentences a user plainly says, each with a predicate the world proves.

- a room: "去厨房" / "go to the kitchen" -> in_room(<room>), the room bound through the
  manifest's names and ``vocab.rooms``; a coordinate: "go to (x, y)" / "走到坐标 (x, y)"
  -> at_position(x, y), and ONLY that whole sentence (SP1V: "... then turn left" compiled
  a contract that covered half of it);
- 'it' in a place / release / fetch binds the ONE object held at turn start (the turn
  baseline's grasp attachments), never in a grasp;
- 拿出 / 提起 / X放到S上 / grab me X / get X for me / 拿起X给我 / 把X给我 / 我的 / my /
  拿在手里 / a filler inside a clause / a CHECKED location (取物台上的X);
- a lead clause (a grasp, a room, a coordinate) needs its own grounded step: "pick up X and
  put it down" is no longer proven by a release of an X already held (MR1V).
"""
from __future__ import annotations

import pytest

from tests.unit.vcli.test_native_loop import _make_engine, _nav_agent, _session, _verdict
from tests.unit.vcli.test_task_contract_binding import GREEN, RED, _two, _v
from vector_os_nano.vcli.backends.fake import FakeToolScriptBackend, tool_turn
from vector_os_nano.vcli.cognitive.object_goal import contract_leads, task_contract
from vector_os_nano.vcli.cognitive.trace_store import goal_check
from vector_os_nano.schemas.world import WORLDS_DIR, load_world

BASE = frozenset({"at_position", "facing", "in_room", "near_object"})
ARM = frozenset({"holding_object", "kept_holding", "released", "object_at", "object_on", "lifted"})
WORLDS = sorted(p.parent.name for p in WORLDS_DIR.glob("*/world.yaml"))


def _c(goal: str, names=BASE | ARM, world="apartment", held=()) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return task_contract(goal, names, world, held), contract_leads(goal, names, world, held)


def _goal(trace, rep) -> tuple[str, str]:
    """The goal check and its own reason (the verdict carries the state only)."""
    return goal_check(trace, frozenset(rep.oracle_names))


# ---------------------------------------------------------------------------
# Rooms and coordinates
# ---------------------------------------------------------------------------

_ROOM = [
    ("去厨房", ("in_room('kitchen')",)), ("go to the kitchen", ("in_room('kitchen')",)),
    ("到客厅去", ("in_room('living_room')",)), ("回卧室", ("in_room('master_bedroom')",)),
    ("请去书房吧", ("in_room('study')",)), ("walk into the dining room", ("in_room('dining_room')",)),
    ("导航到厨房", ("in_room('kitchen')",)), ("go back to the living room", ("in_room('living_room')",)),
    ("回到客卧", ("in_room('guest_bedroom')",)), ("enter the bathroom", ("in_room('bathroom')",)),
    # a room word the world has none of: the slot fills and binds nothing (failed, with the rooms)
    ("去房间", ("in_room(None)",)), ("go to the game room", ("in_room(None)",)), ("去会议室", ("in_room(None)",)),
    # outside the forms (unchecked): an unknown place, two rooms, a negation, an extra clause
    ("go to the garage", ()), ("去厨房和客厅", ()), ("go to the kitchen and the living room", ()),
    ("不要去厨房", ()), ("don't go to the kitchen", ()), ("去厨房看看", ()), ("去厨房然后左转", ()),
    ("go to the kitchen then turn left", ()), ("去厨房间", ()), ("去餐桌那里", ()),
]

_COORD = [
    ("走到坐标 (3.0, 2.5)", ("at_position(3.0, 2.5)",)), ("go to (9, 3)", ("at_position(9.0, 3.0)",)),
    ("navigate to (3.0,0.0)", ("at_position(3.0, 0.0)",)), ("走到坐标（11，3）", ("at_position(11.0, 3.0)",)),
    ("请导航到 (-1.5, 2)", ("at_position(-1.5, 2.0)",)), ("move to the point (2,3)", ("at_position(2.0, 3.0)",)),
    # SP1V: a coordinate inside a longer sentence compiles nothing
    ("走到坐标 (9.0,3.0) 然后左转90度", ()), ("go to (9,3) then turn left", ()), ("走到坐标 (3,0) 并停下", ()),
    ("启动 go2 然后走到坐标 (2.0,0.0)", ()), ("go to (3,0), retry if short", ()), ("导航到厨房 (15, 3)", ()),
    ("声称已在 (10.0,3.0)", ()), ("write (3,4) to file out.txt", ()), ("go to (1,2) or (3,4)", ()),
]


@pytest.mark.parametrize("goal, contract", _ROOM + _COORD)
def test_a_room_or_a_coordinate_compiles_only_as_a_whole_sentence(goal, contract) -> None:
    assert _c(goal) == (contract, ())
    assert _c(goal, BASE) == (contract, ())  # go2: no arm is needed to check a room or a point


@pytest.mark.parametrize("world", WORLDS)
@pytest.mark.parametrize("goal", ["去厨房", "go to the kitchen", "走到坐标 (3.0, 2.5)", "go to (9, 3)"])
def test_without_a_base_oracle_a_room_or_a_coordinate_compiles_nothing(world, goal) -> None:
    assert _c(goal, ARM, world) == ((), ())


def test_without_a_manifest_only_a_coordinate_compiles() -> None:
    assert _c("go to (9, 3)", BASE, "") == (("at_position(9.0, 3.0)",), ())
    assert _c("去厨房", BASE, "") == ((), ())


@pytest.mark.parametrize("world", WORLDS)
def test_every_room_word_of_every_world_names_its_room(world) -> None:
    w = load_world(world)
    for word, rooms in w.room_words().items():
        assert w.resolve_room(word) == (next(iter(rooms)) if len(rooms) == 1 else None), word
        if len(rooms) == 1 and not word.isascii():
            assert _c(f"去{word}", BASE, world)[0] == (f"in_room({next(iter(rooms))!r})",), word


def test_a_room_word_under_two_rooms_names_neither() -> None:
    from dataclasses import replace

    w = load_world("apartment")
    two = replace(w, room_aliases={**w.room_aliases, "guest_bedroom": (*w.room_aliases["guest_bedroom"], "卧室")})
    assert two.resolve_room("卧室") is None and two.resolve_room("主卧") == "master_bedroom"


# ---------------------------------------------------------------------------
# 'it': the one object held at turn start (ground truth), never in a grasp
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("goal, contract", [
    ("把它放到餐桌上", "object_on({o}, 'dining_table')"), ("再把它放到餐桌上", "object_on({o}, 'dining_table')"),
    ("put it on the dining table", "object_on({o}, 'dining_table')"), ("放到餐桌上", "object_on({o}, 'dining_table')"),
    ("drop it", "released({o})"), ("把它放下", "released({o})"), ("放下", "released({o})"), ("let go", "released({o})"),
    ("把它拿给我", "kept_holding({o})"), ("bring it to me", "kept_holding({o})"),
])
@pytest.mark.parametrize("held, bound", [((RED,), RED), ((), None), ((RED, GREEN), None)])
def test_it_binds_the_one_object_held_at_turn_start(goal, contract, held, bound) -> None:
    assert _c(goal, held=held)[0][0] == contract.format(o=repr(bound))


@pytest.mark.parametrize("goal", ["pick it up", "拿起它", "grab that", "把它拿起来"])
def test_it_in_a_grasp_names_nothing(goal) -> None:
    assert _c(goal, held=(RED,))[0] == ("holding_object(None)",)


def test_a_named_object_is_never_replaced_by_the_held_one() -> None:
    assert _c("pick up the green bottle and put it down", held=(RED,)) == (
        (f"released('{GREEN}')",), (f"holding_object('{GREEN}')",))
    assert _c("把绿色瓶子放下", held=(RED,))[0] == (f"released('{GREEN}')",)


# ---------------------------------------------------------------------------
# The phrasings the earlier rounds listed (I4D-V3, REF1c-V-04), in every world's words
# ---------------------------------------------------------------------------

def _named(world: str) -> tuple[str, str, str, str, str]:
    """(body, its English name, a CJK alias, its surface's CJK alias, another surface's CJK alias or "")."""
    w = load_world(world)
    body, pk = next((n, p) for n, p in w.pickables.items()
                    if w.resolve(p.english) == n and any(w.resolve(a) == n for a in p.aliases if not a.isascii()))
    zo = next(a for a in pk.aliases if not a.isascii() and w.resolve(a) == body)
    zs = next(a for a in w.surfaces[pk.surface].aliases if not a.isascii())
    other = next((a for n, s in w.surfaces.items() if n != pk.surface
                  for a in s.aliases if not a.isascii() and w.resolve_surface(a) == n), "")
    return body, pk.english, zo, zs, other


_FAMILY = [  # (goal template, last clause items, lead items); {o} the bound body, {s} the other surface
    ("grab me the {en}", ("kept_holding({o})", "near_object('user')"), ()),
    ("get the {en} for me", ("kept_holding({o})", "near_object('user')"), ()),
    ("拿起{zo}给我", ("kept_holding({o})", "near_object('user')"), ()),
    ("把{zo}给我", ("kept_holding({o})", "near_object('user')"), ()),
    ("拿出{zo}", ("holding_object({o})",), ()), ("提起{zo}", ("holding_object({o})",), ()),
    ("把{zo}提起来", ("holding_object({o})",), ()), ("把{zo}拿出来", ("holding_object({o})",), ()),
    ("拿起我的{zo}", ("holding_object({o})",), ()), ("pick up my {en}", ("holding_object({o})",), ()),
    ("把{zo}抓起来拿在手里", ("holding_object({o})",), ()), ("hold the {en} in your hand", ("holding_object({o})",), ()),
    ("pick up the {en} please and bring it to me", ("kept_holding({o})", "near_object('user')"),
     ("holding_object({o})",)),
    ("拿起{zo}，然后把它放下", ("released({o})",), ("holding_object({o})",)),
    ("{zo}放到{zs_other}上", ("object_on({o}, {s})",), ()),
    ("drop the {en} onto the {zs_other}", ("object_on({o}, {s})",), ()),
    ("put down the {en} on the {zs_other}", ("object_on({o}, {s})",), ()),
    # a location is part of the reference only when it is the object's surface in the manifest
    ("把{zs}上的{zo}拿给我", ("kept_holding({o})", "near_object('user')"), ()),
    ("拿起{zs}上面的{zo}", ("holding_object({o})",), ()),
    ("pick up the {en} from the {zs} and put it on the {zs_other}", ("object_on({o}, {s})",), ("holding_object({o})",)),
]


def _cases() -> list[tuple[str, str, tuple[str, ...], tuple[str, ...]]]:
    out = []
    for world in WORLDS:
        body, en, zo, zs, other = _named(world)
        surface = load_world(world).resolve_surface(other) if other else None
        for goal, last, leads in _FAMILY:
            if "{zs_other}" in goal and not other:
                continue
            g = goal.format(en=en, zo=zo, zs=zs, zs_other=other)
            fill = {"o": repr(body), "s": repr(surface)}
            out.append((world, g, tuple(i.format(**fill) for i in last), tuple(i.format(**fill) for i in leads)))
    return out


@pytest.mark.parametrize("world, goal, contract, leads", _cases())
def test_every_listed_phrasing_compiles_in_every_world(world, goal, contract, leads) -> None:
    assert _c(goal, ARM, world) == (contract, leads)


@pytest.mark.parametrize("world", WORLDS)
def test_the_negative_rows_in_every_world(world) -> None:
    body, en, zo, zs, other = _named(world)
    w = load_world(world)
    for goal in (f"拿起{zo}然后跳舞", f"pick up the {en} and dance", f"不要拿起{zo}", f"don't pick up the {en}",
                 f"别把{zo}给我", f"拿起{zo}旁边的东西", f"我要{zo}", f"把{zo}挪到{zs}上"):
        assert _c(goal, ARM, world) == ((), ()), goal
    # a location naming another surface, or furniture that is no surface: the slot binds nothing
    for loc in [other, *(f"{s}子" for s in w.scenery if not s.isascii())][:3]:
        if loc and w.resolve_surface(loc) != w.pickables[body].surface:
            assert _c(f"拿起{loc}上的{zo}", ARM, world)[0] == ("holding_object(None)",), loc


def test_the_apartment_rows_the_rounds_listed() -> None:
    can = repr(RED)
    assert _c("把桌子上的红色罐子拿给我")[0] == ("kept_holding(None)", "near_object('user')")
    assert _c("拿起餐桌上的红罐子")[0] == ("holding_object(None)",)       # it rests on the pick table
    assert _c("pick up the red can on the dining table")[0] == ("holding_object(None)",)
    assert _c("拿起取物台上的红罐子")[0] == (f"holding_object({can})",)
    assert _c("红色的罐子放到餐桌上")[0] == (f"object_on({can}, 'dining_table')",)
    assert _c("把绿色的瓶子放到咖啡桌上")[0] == (f"object_on('{GREEN}', 'coffee_table')",)
    assert _c("我要绿色的瓶子") == ((), ())                                # no end state named


# ---------------------------------------------------------------------------
# A navigation clause before a manipulation clause
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("goal, contract, leads", [
    ("go to the kitchen and pick up the red can", (f"holding_object('{RED}')",), ("in_room('kitchen')",)),
    ("去客厅把黄色杯子拿起来", ("holding_object('pickable_cup_yellow')",), ("in_room('living_room')",)),
    ("去厨房，然后把红罐子拿过来", (f"kept_holding('{RED}')", "near_object('user')"), ("in_room('kitchen')",)),
    ("走到 (10.5, 3) 把红罐子拿起来", (f"holding_object('{RED}')",), ("at_position(10.5, 3.0)",)),
    ("去厨房然后去客厅", ("in_room('living_room')",), ("in_room('kitchen')",)),
    ("pick up the red can and go to the kitchen", ("in_room('kitchen')",), (f"holding_object('{RED}')",)),
    ("去房间把红罐子拿起来", (f"holding_object('{RED}')",), ("in_room(None)",)),
    # a lead the forms do not have: nothing compiles
    ("去餐桌那里把红色罐子拿起来", (), ()), ("look around and pick up the red can", (), ()),
    ("put the red can down and go to the kitchen", (), ()),  # a release never leads
])
def test_a_navigation_clause_leads_a_manipulation(goal, contract, leads) -> None:
    assert _c(goal) == (contract, leads)
    if contract and "room" in str(leads):
        assert _c(goal, ARM) == ((), ())  # an arm without a base cannot check the room


# ---------------------------------------------------------------------------
# Through the runner: proven only with the end state (and each lead's grounded step)
# ---------------------------------------------------------------------------

_KITCHEN = ("navigate", {"x": 17.0, "y": 1.7})


@pytest.mark.parametrize("script, state, why", [
    ([_KITCHEN, _v("in_room('kitchen')")], "proven", ""),
    ([_KITCHEN, _v("in_room('study')")], "failed", "no actor-caused step proves in_room('kitchen')"),
    ([("navigate", {"x": 17.0, "y": 7.5}), _v("in_room('kitchen')")], "failed", "not met at turn end"),
    ([_KITCHEN, _v("at_position(17.0, 1.7)")], "failed", "no actor-caused step proves in_room('kitchen')"),
])
def test_a_room_goal_is_proven_only_in_that_room(script, state, why) -> None:
    agent = _nav_agent()
    agent._base.navigate_to(10.0, 3.0)
    backend = FakeToolScriptBackend.from_tool_script([*(tool_turn(c) for c in script), tool_turn(("finish", {}))])
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("去厨房", session=_session())
    rep = _verdict(agent, eng, trace)
    assert trace.contract == ("in_room('kitchen')",) and trace.contract_start == (False,)
    state_, why_ = _goal(trace, rep)
    assert state_ == state and why in why_, why_
    assert rep.verified is (state == "proven")


_KITCHEN_GOAL = "go to the kitchen and pick up the red can"
_GRAB, _HOLD_RED = ("grasp_can", {}), _v("holding_object('red can')")


@pytest.mark.parametrize("script, state, why", [
    ([_KITCHEN, _v("in_room('kitchen')"), _GRAB, _HOLD_RED], "proven", ""),
    ([_KITCHEN, _GRAB, _HOLD_RED], "failed", "lead clause in_room('kitchen')"),            # never verified
    ([_KITCHEN, _v("in_room('study')"), _GRAB, _HOLD_RED], "failed", "lead clause in_room('kitchen')"),
    ([_GRAB, _HOLD_RED, _KITCHEN, _v("in_room('kitchen')")], "failed", "lead clause in_room('kitchen')"),  # after
    ([_v("in_room('kitchen')"), _GRAB, _HOLD_RED], "failed", "lead clause in_room('kitchen')"),  # a probe, no walk
])
def test_a_lead_room_needs_its_own_grounded_step_before_the_grasp(script, state, why) -> None:
    trace, rep = _two(_KITCHEN_GOAL, script)
    assert trace.contract == (f"holding_object('{RED}')",)
    state_, why_ = _goal(trace, rep)
    assert state_ == state and why in why_, why_
    assert rep.verified is (state == "proven")


def test_pick_up_and_put_down_of_an_object_already_held_needs_the_grasp() -> None:
    """MR1V: the green bottle is in the hand at turn start; a release alone proved
    "pick up the green bottle and put it down". The lead grasp needs its own grounded step."""
    goal = "pick up the green bottle and put it down"
    trace, rep = _two(goal, [("open_green", {}), _v("released('green bottle')")], held=(GREEN,))
    assert trace.contract == (f"released('{GREEN}')",) and trace.held_at_start == (GREEN,)
    state, why = _goal(trace, rep)
    assert state == "failed" and "lead clause holding_object" in why, why
    trace, rep = _two(goal, [("open_green", {}), _v("released('green bottle')"), ("grasp_green", {}),
                             _v("holding_object('green bottle')"), ("open_green", {}),
                             _v("released('green bottle')")], held=(GREEN,))
    assert _goal(trace, rep) == ("proven", ""), _goal(trace, rep)


@pytest.mark.parametrize("held, script, state, why", [
    ((RED,), [("open_can", {}), _v("released('red can')")], "proven", ""),
    ((RED,), [("open_green", {}), _v("released('green bottle')")], "failed", "released"),  # another object
    ((), [("open_can", {}), _v("released('red can')")], "failed", "held: nothing"),
    ((RED, GREEN), [("open_can", {}), _v("released('red can')")], "failed", f"held: {GREEN}, {RED}"),
])
def test_it_is_proven_only_for_the_object_held_at_turn_start(held, script, state, why) -> None:
    trace, rep = _two("把它放下", script, held=held)
    state_, why_ = _goal(trace, rep)
    assert state_ == state and why in why_, why_
    assert rep.verified is (state == "proven")


@pytest.mark.parametrize("goal, state, why", [
    ("拿起取物台上的红罐子", "proven", ""),
    ("拿起餐桌上的红罐子", "failed", "names no object"),    # the can rests on the pick table
    ("提起红罐子", "proven", ""), ("拿出红罐子", "proven", ""), ("拿起我的红罐子", "proven", ""),
])
def test_a_checked_location_and_the_new_grasp_forms_through_the_runner(goal, state, why) -> None:
    trace, rep = _two(goal, [_GRAB, _HOLD_RED])
    state_, why_ = _goal(trace, rep)
    assert state_ == state and why in why_, why_
    assert rep.verified is (state == "proven")
