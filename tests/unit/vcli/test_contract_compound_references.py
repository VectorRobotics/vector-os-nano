# SPDX-License-Identifier: Apache-2.0
"""Regressions found while integrating REF1e and K1, including unsafe alternatives."""
import pytest

from tests.unit.vcli.test_contract_forms import _c
from tests.unit.vcli.test_task_contract_binding import RED, _two, _v


@pytest.mark.parametrize("phrase, obj", [
    ("绿色的瓶子", "pickable_bottle_green"),
    ("蓝色的瓶子", "pickable_bottle_blue"),
    ("紫色的盒子", "pickable_box_purple"),
])
@pytest.mark.parametrize("verb", ["拿起", "抓起"])
def test_glued_place_keeps_the_whole_object_reference(phrase, obj, verb):
    assert _c(f"{verb}{phrase}放到餐桌上") == (
        (f"object_on('{obj}', 'dining_table')",), (f"holding_object('{obj}')",))


@pytest.mark.parametrize("goal", [
    "拿起绿色的瓶子把蓝色的瓶子放到餐桌上",
    "拿起瓶子放到餐桌上",
])
def test_conflicting_or_ambiguous_references_stay_unbound(goal):
    assert _c(goal)[0] == ("object_on(None, 'dining_table')",)


@pytest.mark.parametrize("goal", [
    "不要拿起绿色的瓶子放到餐桌上",
    "拿起绿色的瓶子放到餐桌上然后关灯",
])
def test_extra_or_negated_goals_are_not_partially_compiled(goal):
    assert _c(goal) == ((), ())


def test_take_hold_of_through_the_runner():
    trace, report = _two("take hold of the red can", [
        ("grasp_can", {}), _v("holding_object('red can')")])
    assert trace.contract == (f"holding_object('{RED}')",)
    assert report.goal_check == "proven" and report.verified
