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

import pytest

from tests.unit.vcli.test_goal_check import _v
from tests.unit.vcli.test_native_loop import _CAN, _grasp, _make_arm_agent, _make_engine, _session, _verdict
from vector_os_nano.vcli.backends.fake import FakeToolScriptBackend, tool_turn
from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused
from vector_os_nano.vcli.cognitive.object_goal import checkable_goals, task_contract
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


@pytest.mark.parametrize("world", sorted(p.stem for p in WORLDS_DIR.glob("*.yaml")))
def test_every_checkable_goal_the_gate_offers_compiles_a_bound_contract(world) -> None:
    goals = checkable_goals(load_world(world))
    assert len(goals) >= 4, goals
    for goal in goals:
        contract = task_contract(goal, APT, world)
        assert contract and not any("None" in item for item in contract), (goal, contract)


@pytest.mark.parametrize("goal", ["把绿色的放到餐桌上", "put the green one on the dining table"])
def test_the_gate_takes_the_object_the_contract_bound(goal) -> None:
    """I4E-VER-1: a colour reference beside a surface word is proven AND verified (the
    gate once resolved the whole sentence, which also held the scenery word)."""
    t = _one_row(goal, "object_on('green bottle', 'dining_table')", "apartment")
    assert t.contract == ("object_on('pickable_bottle_green', 'dining_table')",)
    assert turn_reject_reason(t, APT) == ""
    assert turn_reject_reason(_one_row(goal, "object_on('red can', 'dining_table')", "apartment"), APT) != ""
