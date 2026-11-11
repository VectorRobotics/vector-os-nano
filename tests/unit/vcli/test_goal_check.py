# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""I4d — what a verdict claims (charter 3.5): the goal check is separate from step
evidence and ``verified``, and a task contract compiles ONLY from closed whole-sentence
forms whose slots bind through the world manifest.

Five verification rounds each found a sentence the regex-compiled contract got wrong
(failed open, then mis-bound: a question compiled a place sent to the model as its
task, a surface became the object of a release). Now every phrasing those rounds used
compiles either () -- unchecked -- or exactly the end state the sentence names; the
goal is ``proven`` only when that end state came about, through the runner.
"""
from __future__ import annotations

import pytest

from tests.harness.fake_backend import FakeToolScriptBackend, tool_turn
from tests.unit.vcli.test_native_loop import (
    _CAN, _NavBase, _Skill, _make_agent, _make_arm_agent, _make_engine, _session, _verdict,
)
from tests.unit.vcli.test_task_contract_binding import APT, GREEN, RED, _hand_trace
from vector_os_nano.vcli.cognitive.object_goal import task_contract
from vector_os_nano.vcli.cognitive.trace_store import goal_check, turn_reject_reason

_HOLD, _REL = f"holding_object('{GREEN}')", f"released('{GREEN}')"
_FETCH = (f"kept_holding('{GREEN}')", "near_object('user')")


def _on(surface: str | None, obj: str | None = GREEN) -> tuple[str, ...]:
    return (f"object_on({obj!r}, {surface!r})",)


# ---------------------------------------------------------------------------
# Every phrasing of the five verification rounds (results/I4-all.json and the
# verifiers' probes): () or exactly the end state the sentence names
# ---------------------------------------------------------------------------

_COMPILED = {
    # grasp
    "pick up the green bottle": (_HOLD,), "拿起绿瓶子": (_HOLD,), "grab the green bottle": (_HOLD,),
    "拿起绿色的瓶子": (_HOLD,), "take the green bottle": (_HOLD,),
    "拿起红色的罐子": (f"holding_object('{RED}')",),
    "拿起瓶子": ("holding_object(None)",), "pick up a bottle": ("holding_object(None)",),   # two bottles
    "拿起那个罐子": ("holding_object(None)",), "pick it up": ("holding_object(None)",),     # no one object
    # release (optionally after a grasp)
    "put the green bottle down": (_REL,), "put down the green bottle": (_REL,), "drop the green bottle": (_REL,),
    "release the green bottle": (_REL,), "let go of the green bottle": (_REL,), "把绿瓶子放下": (_REL,),
    "松开绿瓶子": (_REL,), "把绿色瓶子放下来": (_REL,), "please set the green bottle down.": (_REL,),
    "put the green bottle down.": (_REL,), "put the green bottle down, please": (_REL,),
    "put the green bottle down now": (_REL,), "just put the green bottle down": (_REL,),
    "can you put the green bottle down": (_REL,), "放下绿瓶子吧": (_REL,), "把绿瓶子放下吧": (_REL,),
    "pick up the green bottle and put it down": (_REL,), "pick up the green bottle, then put it down": (_REL,),
    "grab the green bottle and drop it": (_REL,),
    "drop it": ("released(None)",), "把它放下": ("released(None)",),
    "pick up the green bottle and drop the red can": ("released(None)",),               # two objects named
    # place on a surface (optionally after a grasp)
    "put the green bottle on the dining table": _on("dining_table"), "把绿瓶子放到餐桌上": _on("dining_table"),
    "把绿瓶子搁到餐桌上": _on("dining_table"), "把绿瓶子摆到餐桌上": _on("dining_table"),
    "把绿瓶子搁在餐桌上": _on("dining_table"), "拿起绿瓶子放到饭桌上": _on("dining_table"),
    "put the green bottle onto the dining table": _on("dining_table"),
    "put the green bottle on top of the dining table": _on("dining_table"),
    "put the green bottle down on the dining table": _on("dining_table"),
    "could you put the green bottle on the dining table": _on("dining_table"),
    "pick up the green bottle and place it on the dining table": _on("dining_table"),
    "把绿色瓶子放到桌子上": _on(None), "put the green bottle on the table": _on(None),
    "把绿瓶子放在桌子上": _on(None), "拿起绿瓶子放到桌子上": _on(None),
    "把红色罐子放到桌子上": _on(None, RED), "place the red can on the table": _on(None, RED),
    "put the red can on the shelf": _on(None, RED),
    # fetch to the user
    "把绿瓶子拿过来": _FETCH, "把绿色瓶子拿过来": _FETCH, "再把绿色瓶子拿过来一点": _FETCH, "把绿瓶子拿给我": _FETCH,
    "把绿瓶子拿来给我": _FETCH, "把绿瓶子递过来": _FETCH, "把绿瓶子递给我": _FETCH, "拿给我绿瓶子": _FETCH,
    "给我拿绿瓶子": _FETCH, "bring me the green bottle": _FETCH, "bring me the green bottle please": _FETCH,
    "can you bring me the green bottle": _FETCH, "bring the green bottle to me": _FETCH,
    "bring the green bottle to the user": _FETCH, "bring the green bottle over": _FETCH,
    "hand me the green bottle": _FETCH, "fetch the green bottle": _FETCH,
    "pick up the red can and bring me the green bottle": ("kept_holding(None)", "near_object('user')"),
    "grab the red can then bring me the green bottle": ("kept_holding(None)", "near_object('user')"),
    # I4D-V1: 一点 / 一下 stay fillers after a fetch to the user ("bring it a bit closer")
    "把绿瓶子拿过来一下": _FETCH, "把绿瓶子递过来一点吧": _FETCH, "拿绿瓶子过来一点": _FETCH,
    # I4D-V2: a colour alone names an object only as a reference (the red one, 红色的, 红的, 红色那个)
    "pick up the red one": (f"holding_object('{RED}')",), "拿起红色那个": (f"holding_object('{RED}')",),
    "拿起红的": (f"holding_object('{RED}')",), "拿起那个红色的": (f"holding_object('{RED}')",),
    "bring me the green one": _FETCH, "把绿色的放到餐桌上": _on("dining_table"),
}

_UNCHECKED = [
    # questions and perception: no end state, nothing asked of the model
    "is the green bottle on the dining table?", "绿瓶子在餐桌上吗", "what colour is the green bottle",
    "is the red can still there", "看看红色罐子还在不在", "餐桌上有什么", "describe the dining table",
    "how many bottles are on the dining table", "the dining table", "看一下餐桌", "找到绿色瓶子",
    # navigation
    "go to the kitchen", "move to the green bottle", "get to the dining table",
    "walk to the dining table and look at the green bottle", "go to the green bottle, go to the dining table",
    # a verb or a destination outside the closed forms (was: failed open, or a guessed place)
    "move the green bottle to the dining table", "stick the green bottle on the dining table",
    "leave the green bottle on the dining table", "transfer the green bottle to the dining table",
    "把绿瓶子丢到餐桌上", "把绿瓶子扔到餐桌上", "把绿瓶子移到餐桌上", "把绿瓶子挪到餐桌上", "把绿瓶子拿到餐桌上",
    "拿绿瓶子到餐桌上", "take the green bottle to the dining table", "get the green bottle onto the dining table",
    "pick up the green bottle and take it to the dining table", "pick up the green bottle and move it to the dining table",
    "grab the green bottle and leave it on the dining table", "拿起绿瓶子挪到餐桌上", "拿起绿瓶子移到餐桌上",
    "抓起绿瓶子然后挪到餐桌上", "carry the green bottle to the dining table", "把绿瓶子送到餐桌上",
    "bring the green bottle to the kitchen", "把绿瓶子搬到餐桌上", "deliver the green bottle to the dining table",
    "bring the green bottle to the dining table", "hand the green bottle to the dining table",
    "carry the green bottle", "carry the green bottle over to the dining table", "move the green bottle",
    "move it to the kitchen", "move it over there", "把它挪到厨房", "move the bottle to the table", "挪一下那个瓶子",
    "leave it on the table", "把瓶子移到桌子上", "pick up the green bottle and move it", "给我带绿瓶子",
    "go to the green bottle and bring it to me", "go to the green bottle and move it to the dining table",
    "去绿瓶子那里，挪到餐桌上", "find the green bottle then leave it on the dining table", "找到绿瓶子再挪到餐桌上",
    "bring the green bottle back", "put the green bottle back", "put the green bottle back down",
    "把绿瓶子拿过来放下", "bring me the green bottle and put it on the dining table", "继续拿过来",
    "go to the dining table and put the green bottle down",
    "pick up the green bottle from the pick table and put it down",
    # a destination that is no surface slot (a room, a place word, a preposition not ON)
    "把红色罐子放到厨房", "put the green bottle in the kitchen", "put the green bottle into the box",
    "put the green bottle at the door", "put the green bottle next to the red can",
    "put the green bottle beside the sofa", "put the green bottle under the dining table",
    "put the green bottle over there", "put the green bottle there", "put the green bottle down over there",
    "drop the green bottle on the table", "drop the green bottle on the dining table",
    "drop the green bottle in the kitchen", "drop the green bottle there",
    "release the green bottle onto the coffee table", "let go of the green bottle over the sofa",
    "set the green bottle down by the door", "place the green bottle somewhere safe",
    "put the green bottle on the coffee table near the dining table", "put the green bottle on the red can",
    "put the green bottle down and put it on the dining table", "put the green bottle on the dining-table",
    "把绿瓶子放进盒子里", "把绿瓶子放到厨房", "把绿瓶子放下在桌子上", "把绿瓶子放到那边", "放下绿瓶子到餐桌上",
    "把绿瓶子放在红罐子旁边", "把绿瓶子丢下到地上", "拿起绿瓶子然后放在那里", "拿起绿瓶子然后放到那里",
    "put the green bottle down gently", "把绿瓶子轻轻放下", "绿瓶子放下", "drop the green bottle and the red can",
    # a surface in an object slot (I4CF1-V3), a surface or room as the thing fetched
    "pick up the green bottle and drop the dining table", "抓起绿瓶子放下餐桌", "拿起绿瓶子把餐桌放下",
    "pick up the green bottle and put the dining table down", "put the dining table down",
    "bring me the dining table", "give me the green bottle, the dining table",
    # an object the world has no word for
    "pick up the apple",
    # I4D-V1: 一点 / 一下 after a release, a place or a grasp change the end state
    # (lower it a bit, let go for a moment): never stripped there
    "把绿瓶子放下一点", "放下绿瓶子一点", "把绿瓶子放到餐桌上一点", "把绿瓶子放下一下", "松开绿瓶子一下",
    "拿起绿瓶子一点", "把绿瓶子放下一点吧",
    # I4D-V2: a bare colour word is a noun (the orange is a fruit), never an object reference
    "pick up the orange", "pick up the green", "bring me the orange", "拿起橙色", "拿起绿",
    "put the orange on the dining table", "把橙色放到餐桌上",
]


@pytest.mark.parametrize("goal, contract", list(_COMPILED.items()))
def test_a_closed_form_compiles_exactly_the_end_state_it_names(goal, contract) -> None:
    assert task_contract(goal, APT, "apartment") == contract


@pytest.mark.parametrize("goal", _UNCHECKED)
def test_anything_else_compiles_no_contract(goal) -> None:
    assert task_contract(goal, APT, "apartment") == ()


def test_the_tabletop_binds_its_own_surfaces() -> None:
    assert task_contract("put the banana on the tray", APT, "tabletop") == ("object_on('banana', 'tray')",)
    assert task_contract("把香蕉放到桌子上", APT, "tabletop") == ("object_on('banana', 'table')",)
    assert task_contract("抓起香蕉", APT, "tabletop") == ("holding_object('banana')",)


# ---------------------------------------------------------------------------
# goal_check: proven / failed / unchecked, the one source
# ---------------------------------------------------------------------------


def test_goal_check_states_on_hand_traces() -> None:
    goal = "pick up the green bottle"
    proven = _hand_trace(goal, "holding_object('green bottle')", contract=(_HOLD,), start=(False,), end=(True,))
    assert goal_check(proven, APT) == ("proven", "")
    late = _hand_trace(goal, "holding_object('green bottle')", contract=(_HOLD,), start=(False,), end=(False,))
    assert goal_check(late, APT) == ("failed", f"task contract not met at turn end: {_HOLD}")
    # a producer that carries a contract the sentence does not compile to never proves it
    swapped = _hand_trace("move the green bottle", "holding_object('green bottle')",
                          contract=(_HOLD,), start=(False,), end=(True,))
    assert goal_check(swapped, APT) == ("failed", "the task contract does not match the goal")
    # ... nor does one that drops the contract its goal compiles to
    dropped = _hand_trace(goal, "holding_object('green bottle')", contract=(), start=(), end=())
    assert goal_check(dropped, APT) == ("failed", "the task contract does not match the goal")
    none = _hand_trace("move the green bottle", "holding_object('green bottle')", contract=(), start=(), end=())
    assert goal_check(none, APT) == ("unchecked", "")
    assert turn_reject_reason(none, APT) == ""  # unchecked never rejects: the gates and the steps decide


def test_the_verdict_carries_the_goal_check_and_the_repl_says_it() -> None:
    from vector_os_nano.vcli.render import _goal_line
    from vector_os_nano.vcli.verdict import VerdictReport

    t = _hand_trace("pick up the green bottle", "holding_object('green bottle')",
                    contract=(_HOLD,), start=(False,), end=(True,))
    rep = VerdictReport.from_trace(t, APT)
    assert rep.goal_check == "proven" and rep.to_dict()["goal_check"] == "proven"
    assert '"goal_check": "proven"' in rep.to_sentinel_line()
    assert _goal_line(rep, "") == f"goal proven: {_HOLD}"
    held = _hand_trace("pick up the green bottle", "holding_object('green bottle')",
                       contract=(_HOLD,), start=(True,), end=(True,))
    failed = VerdictReport.from_trace(held, APT)
    assert failed.goal_check == "failed"
    assert _goal_line(failed, goal_check(held, APT)[1]) == f"goal not met: already satisfied at turn start: {_HOLD}"
    un = VerdictReport.from_trace(_hand_trace("move the green bottle", "holding_object('green bottle')",
                                              contract=(), start=(), end=()), APT)
    assert un.goal_check == "unchecked" and un.verified is True
    assert _goal_line(un, "") == ("goal not machine-checked (no task contract for this sentence); "
                                  "proven: holding_object('green bottle')")
    assert VerdictReport.no_trace().goal_check == "unchecked"


# ---------------------------------------------------------------------------
# Through the native runner (fake apartment): proven ONLY when the end state came about
# ---------------------------------------------------------------------------

_POS = {"rest": [10.9, 3.0, 0.71], "held": [10.9, 3.0, 0.95], "dining": [3.0, 7.5, 0.71],
        "floor": [10.0, 3.0, 0.71]}


class _Arm:
    """The green bottle: resting, held, on the dining table, or dropped on the floor."""

    _connected = True

    def __init__(self, state: str) -> None:
        self.state = state

    def get_object_positions(self):
        return {GREEN: _POS[self.state]}

    def get_joint_positions(self):
        return [0.0] * 6


class _Grip:
    def __init__(self, arm: _Arm) -> None:
        self._arm = arm

    def is_holding(self) -> bool:
        return self._arm.state == "held"

    def weld_is_active(self) -> dict:
        return {GREEN: self._arm.state == "held"}


class _Recorder(FakeToolScriptBackend):
    def call(self, **kw):  # type: ignore[override]
        self.seen = [*getattr(self, "seen", []), [b.get("text", "") for b in kw.get("system") or []]]
        return super().call(**kw)


def _run(goal, script, *, start="rest"):
    from tests.harness.fake_gt import bind_fake_sim, unbind_fake_sim
    from vector_os_nano.sim.session import session_for

    agent, _ = _make_agent(10.0, 3.0)
    agent._base = _NavBase(10.0, 3.0)
    arm = _Arm("rest")
    agent._arm, agent._gripper = arm, _Grip(arm)
    unbind_fake_sim(session_for(agent))
    bind_fake_sim(agent, "go2_piper", support="dining_table")
    arm.state = start
    for name, to in (("grasp", "held"), ("open", "floor"), ("place_dining", "dining")):
        agent._skill_registry.register(_Skill(name, lambda to=to: setattr(arm, "state", to)))
    backend = _Recorder.from_tool_script([*(tool_turn(c) for c in script), tool_turn(("finish", {}))])
    eng = _make_engine(agent, backend)
    events: list[str] = []
    trace = eng.run_turn_native(goal, session=_session(), on_progress=events.append)
    return trace, _verdict(agent, eng, trace), backend.seen, events


def _v(expr):
    return ("verify", {"expr": expr})


_GRASP, _OPEN, _PLACE = ("grasp", {}), ("open", {}), ("place_dining", {})
_HOLD_V, _REL_V = _v("holding_object('green bottle')"), _v("released('green bottle')")
_ON_V = _v("object_on('green bottle', 'dining_table')")
_WALK = [("navigate", {"x": 3.0, "y": 7.0}), _v("at_position(3.0, 7.0)")]
_ON_TABLE = "put the green bottle on the dining table"


@pytest.mark.parametrize("goal, script, start", [
    ("pick up the green bottle", [_GRASP, _HOLD_V], "rest"),
    ("put the green bottle down", [_OPEN, _REL_V], "held"),
    ("pick up the green bottle and put it down", [_GRASP, _HOLD_V, _OPEN, _REL_V], "rest"),
    (_ON_TABLE, [_GRASP, _HOLD_V, _PLACE, _ON_V], "rest"),
    ("拿起绿瓶子放到餐桌上", [_GRASP, _HOLD_V, _PLACE, _ON_V], "rest"),
    ("把绿瓶子拿过来", [_GRASP, _HOLD_V, ("navigate", {"x": 8.0, "y": 3.0}),
                  _v("kept_holding('green bottle') and near_object('user')")], "rest"),
])
def test_the_goal_is_proven_when_its_end_state_came_about(goal, script, start) -> None:
    trace, rep, _, events = _run(goal, script, start=start)
    assert trace.contract and rep.goal_check == "proven", (trace.contract, rep.reject_reason)
    assert rep.verified is True, rep.reject_reason
    assert f"Task contract: {' and '.join(trace.contract)}" in events


@pytest.mark.parametrize("goal, script, start", [
    (_ON_TABLE, [_GRASP, _HOLD_V], "rest"),                       # a grasp alone
    (_ON_TABLE, [_OPEN, _REL_V], "held"),                         # a release alone (dropped on the floor)
    (_ON_TABLE, _WALK, "rest"),                                   # a walk alone
    (_ON_TABLE, [_v(f"object_on('{GREEN}', 'dining_table')")], "dining"),   # an echo of the contract
    (_ON_TABLE, [_GRASP, _v(f"object_on('{GREEN}', 'dining_table')")], "rest"),
    ("put the green bottle down", [_GRASP, _HOLD_V], "rest"),     # a grasp alone
    ("pick up the green bottle", [_OPEN, _REL_V], "held"),        # a release alone
    ("pick up the green bottle", _WALK, "rest"),                  # a walk alone
    ("pick up the green bottle", [_v(f"holding_object('{GREEN}')")], "held"),   # an echo
    ("把绿瓶子拿过来", [_GRASP, _HOLD_V], "rest"),                    # a grasp alone
    ("把绿瓶子拿过来", [("navigate", {"x": 8.0, "y": 3.0}), _v("near_object('user')")], "rest"),  # a walk alone
])
def test_a_grasp_a_release_a_walk_or_an_echo_alone_never_proves_the_goal(goal, script, start) -> None:
    trace, rep, _, _ = _run(goal, script, start=start)
    assert trace.contract and rep.goal_check == "failed", (trace.contract, rep.goal_check)
    assert rep.verified is False


@pytest.mark.parametrize("goal", ["move the green bottle to the dining table", "把绿瓶子挪到餐桌上"])
def test_an_uncovered_sentence_is_never_proven(goal) -> None:
    """The end state came about, but the sentence is no closed form: unchecked."""
    trace, rep, seen, _ = _run(goal, [_GRASP, _HOLD_V, _PLACE, _ON_V])
    assert trace.contract == () and rep.goal_check == "unchecked"
    assert not any("Task contract" in t for system in seen for t in system)


@pytest.mark.parametrize("goal", ["is the green bottle on the dining table?", "绿瓶子在餐桌上吗",
                                  "what colour is the green bottle", "看看红色罐子还在不在"])
def test_a_question_is_unchecked_and_the_spine_asks_nothing_of_the_model(goal) -> None:
    trace, rep, seen, events = _run(goal, [_v("object_on('green bottle', 'dining_table')")], start="dining")
    assert trace.contract == () and rep.goal_check == "unchecked"
    assert seen and not any("Task contract" in t or "Unbound" in t for system in seen for t in system)
    assert not any(e.startswith(("Task contract", "Unbound")) for e in events)


# ---------------------------------------------------------------------------
# Which manifest names may reach the model (I4C-V4): the bound items, and the candidates
# of an unbound slot; nothing else of the world
# ---------------------------------------------------------------------------


def _manifest_names() -> set[str]:
    from vector_os_nano.vcli.worlds.world_manifest import load_world

    w = load_world("apartment")
    return ({*w.pickables, *(a for p in w.pickables.values() for a in p.aliases), *w.surfaces,
             *(n.replace("_", " ") for n in w.surfaces), *(a for s in w.surfaces.values() for a in s.aliases)})


@pytest.mark.parametrize("goal, allowed", [
    ("拿起红色的罐子", {_CAN}),
    ("put the red can on the dining table", {_CAN, "dining_table"}),
    # an unbound object: the bound surface and the object candidates (vocabulary())
    ("put it on the dining table", {"dining_table", "green bottle", "blue bottle", "red can", "yellow cup",
                                    "orange box", "purple box"}),
    # an unbound destination: the bound object and the surface candidates
    ("put the red can on the table", {_CAN, "coffee_table", "console_table", "dining_table", "pick_table"}),
])
def test_the_contract_block_names_only_the_bound_items_and_the_candidates(goal, allowed) -> None:
    agent, _arm = _make_arm_agent()
    backend = _Recorder.from_tool_script([tool_turn(("finish", {}))])
    _make_engine(agent, backend).run_turn_native(goal, session=_session())
    block = backend.seen[0][-1]
    assert block.startswith("Task contract: ")
    shown = {n for n in _manifest_names() if n in block.replace(repr(goal), "")}
    assert shown == allowed & _manifest_names(), (block, shown)
