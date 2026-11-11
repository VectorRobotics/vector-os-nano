# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""I4c — a place goal's destination binds fail-closed by default.

Before: a place goal whose destination the manifest could not resolve ("the table" in
a house of four tables, a shelf or a kitchen the world has no surface for) decayed to
``released(obj)``, so a drop anywhere grounded "put the red can on the table"
(I4F2-D1). Now only a PURE release (a closed list of forms) compiles released(obj);
every other place goal compiles object_on(obj, surface) with None for a destination
the world does not resolve, and the reject reason names the slot, quotes the goal and
lists the world's surfaces. The model is told the contract, and a real grasp skill's
auto post-verify binds its object from the call.
"""
from __future__ import annotations

import pytest

from tests.harness.fake_backend import FakeToolScriptBackend, tool_turn
from tests.unit.vcli.test_native_loop import (
    _CAN, _Skill, _make_arm_agent, _make_engine, _session, _verdict,
)
from tests.unit.vcli.test_task_contract_binding import APT, GREEN, RED, _two, _v
from vector_os_nano.vcli.cognitive.object_goal import task_contract, unbound_reason

_SURFACES = "coffee_table, console_table, dining_table, pick_table"


@pytest.mark.parametrize("goal, contract", [
    # pure releases, both languages (optionally after a grasp of the same object)
    ("put the red can down", f"released('{RED}')"),
    ("put down the red can", f"released('{RED}')"),
    ("set the red can down", f"released('{RED}')"),
    ("drop the red can", f"released('{RED}')"),
    ("release the red can", f"released('{RED}')"),
    ("let go of the red can", f"released('{RED}')"),
    ("Please put the red can down!", f"released('{RED}')"),
    ("把红色罐子放下", f"released('{RED}')"),
    ("请把红色的罐子放下来。", f"released('{RED}')"),
    ("放下红色罐子", f"released('{RED}')"),
    ("松开红色罐子", f"released('{RED}')"),
    ("丢下红色罐子", f"released('{RED}')"),
    ("pick up the green bottle and put it down", f"released('{GREEN}')"),
    ("拿起绿色瓶子然后放下", f"released('{GREEN}')"),
    ("拿起绿瓶子放下", f"released('{GREEN}')"),
    ("放下", "released(None)"),       # no object named: fails closed on the object slot
    ("let go", "released(None)"),
    # a destination the world does not resolve: None, whatever words name it
    ("把红色罐子放到桌子上", f"object_on('{RED}', None)"),
    ("place the red can on the table", f"object_on('{RED}', None)"),
    ("put the red can on the shelf", f"object_on('{RED}', None)"),
    ("把红色罐子放到厨房", f"object_on('{RED}', None)"),
    ("drop the red can in the kitchen", f"object_on('{RED}', None)"),
    ("put the red can down beside the sofa", f"object_on('{RED}', None)"),   # no listed preposition
    ("put the red can down please", f"object_on('{RED}', None)"),          # unlisted: fails closed
    ("set the red can down next to the fridge", f"object_on('{RED}', None)"),
    ("把红色罐子放在冰箱旁边", f"object_on('{RED}', None)"),
    ("放下红色罐子在桌子上", f"object_on('{RED}', None)"),
    ("把红色罐子搁在门口然后放下", f"object_on('{RED}', None)"),
    ("put the red can on the coffee table or the dining table", f"object_on('{RED}', None)"),  # two fit
    # a uniquely named surface
    ("put the red can onto the dining table", f"object_on('{RED}', 'dining_table')"),
    ("把红色罐子放在茶几上", f"object_on('{RED}', 'coffee_table')"),
    ("put the red can down on the console table", f"object_on('{RED}', 'console_table')"),
    ("pick up the green bottle and place it on the pick table", f"object_on('{GREEN}', 'pick_table')"),
])
def test_only_a_pure_release_compiles_released(goal, contract) -> None:
    assert task_contract(goal, APT, "apartment") == (contract,)


def test_a_world_without_a_manifest_keeps_no_place_contract() -> None:
    assert task_contract("place the red can on the table", APT) == ()
    assert task_contract("put the red can down", APT) == ()


def test_the_unbound_reason_names_the_slot_quotes_the_goal_and_lists_the_world() -> None:
    goal = "put the red can on the shelf"
    assert unbound_reason(f"object_on('{RED}', None)", goal, "apartment") == (
        f"the goal {goal!r} names no destination the world resolves; surfaces: {_SURFACES}")
    why = unbound_reason("object_on(None, None)", "drop it on the floor", "apartment")
    assert why.startswith("the goal 'drop it on the floor' names no object the world resolves; objects: ")
    assert "red can" in why
    assert unbound_reason(f"object_on('{RED}', 'dining_table')", goal, "apartment") == ""
    assert unbound_reason(f"released('{RED}')", goal, "apartment") == ""


@pytest.mark.parametrize("goal", [
    "把红色罐子放到桌子上",
    "place the red can on the table",
    "put the red can on the shelf",
    "把红色罐子放到厨房",
])
def test_a_drop_anywhere_never_places_on_a_named_destination(goal) -> None:
    """The last verifier's P1 through the runner: the can is only opened out of the
    hand; released('red can') is CAUSED, but the goal names a destination."""
    trace, rep = _two(goal, [("open_can", {}), _v("released('red can')")], held=(RED,))
    assert trace.contract == (f"object_on('{RED}', None)",)
    assert rep.verified is False
    assert rep.reject_reason == (
        f"task contract object_on('{RED}', None): the goal {goal!r} names no destination "
        f"the world resolves; surfaces: {_SURFACES}")


@pytest.mark.parametrize("goal", ["put the red can down", "把红色罐子放下"])
def test_a_pure_release_of_the_named_object_grounds(goal) -> None:
    trace, rep = _two(goal, [("open_can", {}), _v("released('red can')")], held=(RED,))
    assert trace.contract == (f"released('{RED}')",)
    assert rep.verified is True, rep.reject_reason


# ---------------------------------------------------------------------------
# The model is told the contract (the same line the user sees, plus an unbound slot)
# ---------------------------------------------------------------------------


class _Recorder(FakeToolScriptBackend):
    def call(self, **kw):  # type: ignore[override]
        self.seen = [*getattr(self, "seen", []), [b.get("text", "") for b in kw.get("system") or []]]
        return super().call(**kw)


def _systems(goal: str) -> tuple[list[list[str]], list[str]]:
    agent, _arm = _make_arm_agent()
    backend = _Recorder.from_tool_script([tool_turn(("finish", {}))])
    events: list[str] = []
    _make_engine(agent, backend).run_turn_native(goal, session=_session(), on_progress=events.append)
    return backend.seen, events


def test_the_model_receives_the_contract_and_the_unbound_slot() -> None:
    goal = "place the red can on the table"
    seen, events = _systems(goal)
    note = [f"Task contract: object_on('{_CAN}', None)",
            f"Unbound: object_on('{_CAN}', None): the goal {goal!r} names no destination the world "
            f"resolves; surfaces: {_SURFACES}"]
    assert seen and all(system[-1] == "\n".join(note) for system in seen)
    assert all(line in events for line in note)  # the user sees the same lines


def test_the_model_receives_a_bound_contract() -> None:
    seen, events = _systems("拿起红色的罐子")
    assert seen[0][-1] == f"Task contract: holding_object('{_CAN}')"
    assert f"Task contract: holding_object('{_CAN}')" in events


def test_a_turn_without_a_contract_adds_no_block() -> None:
    seen, _ = _systems("你好")
    assert not any("Task contract" in t for system in seen for t in system)


# ---------------------------------------------------------------------------
# A real grasp skill's auto post-verify binds its object from the call
# ---------------------------------------------------------------------------


def _grasp_skills():
    from vector_os_nano.skills.mobile_pick import MobilePickSkill
    from vector_os_nano.skills.perception_grasp import PerceptionGraspSkill
    from vector_os_nano.skills.pick import PickSkill
    from vector_os_nano.skills.pick_top_down import PickTopDownSkill

    return [(PerceptionGraspSkill, "query"), (PickTopDownSkill, "object_label"),
            (MobilePickSkill, "object_label"), (PickSkill, "object_label")]


@pytest.mark.parametrize("cls, field", _grasp_skills(), ids=lambda v: getattr(v, "name", v))
@pytest.mark.parametrize("value, bound", [
    ("red can", "holding_object('red can')"),
    ("红色的罐子", "holding_object('红色的罐子')"),
    ("the can", None),      # the manifest cannot resolve it: no auto-verify, never a guess
    (None, None),           # the parameter is missing
])
def test_a_real_grasp_hint_binds_the_called_object(cls, field, value, bound) -> None:
    def make(arm):
        return _Skill("grasp_real", lambda: setattr(arm, "held", True), verify_hint=cls.verify_hint)

    params = {} if value is None else {field: value}
    agent, _arm = _make_arm_agent(make)
    backend = FakeToolScriptBackend.from_tool_script([tool_turn(("grasp_real", params)), tool_turn(("finish", {}))])
    eng = _make_engine(agent, backend)
    trace = eng.run_turn_native("拿起红色的罐子", session=_session())
    verifies = [sg.verify for sg in trace.goal_tree.sub_goals]
    rep = _verdict(agent, eng, trace)
    if bound is None:
        assert not any(v and v.startswith("holding_object") for v in verifies), verifies
        assert rep.verified is False
    else:
        assert verifies == [bound]
        assert rep.verified is True, rep.reject_reason
