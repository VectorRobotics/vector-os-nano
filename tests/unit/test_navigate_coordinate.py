# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""R38 — NavigateSkill coordinate-goal path (unit, mocked base).

The producer's nav step for "去桌子那里" targets the table by COORDINATE, not a
named room (the SceneGraph has no "table" room). NavigateSkill must accept the
params ``{"x": ..., "y": ...}`` (or ``target=[x, y]``) and drive the base
via ``base.navigate_to(x, y)`` — the FAR planner path — WITHOUT requiring a
SceneGraph room. The existing named-room path is unchanged.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vector_os_nano.core.skill import SkillContext

from vector_os_nano.sim.nav.follow import NavResult
from vector_os_nano.skills.navigate import NavigateSkill


def _ctx(arrived=True, pos=(10.75, 3.05, 0.35), reason=None):
    base = MagicMock()
    base.get_position.return_value = list(pos)
    base.get_heading.return_value = 0.0
    nav = NavResult(reached=arrived, reason=reason or ("arrived" if arrived else "stalled"),
                    target=[10.5, 3.0], pos=list(pos))
    base.navigate_to = MagicMock(return_value=nav)
    # No spatial_memory needed for a coordinate goal.
    return SkillContext(bases={"go2": base}, services={})


def test_navigate_xy_drives_navigate_to_without_room():
    """{x, y} params -> base.navigate_to(x, y); no SceneGraph room required."""
    ctx = _ctx(arrived=True)
    res = NavigateSkill().execute({"x": 10.5, "y": 3.0}, ctx)
    assert res.success is True
    ctx.base.navigate_to.assert_called_once()
    args, kwargs = ctx.base.navigate_to.call_args
    assert float(args[0]) == 10.5
    assert float(args[1]) == 3.0
    assert res.result_data["target"] == [10.5, 3.0]
    assert res.result_data["reason"] == "arrived"


def test_navigate_target_list_form():
    """target=[x, y] is accepted equivalently to {x, y}."""
    ctx = _ctx(arrived=True)
    res = NavigateSkill().execute({"target": [10.5, 3.0]}, ctx)
    assert res.success is True
    args, _ = ctx.base.navigate_to.call_args
    assert (float(args[0]), float(args[1])) == (10.5, 3.0)


def test_navigate_xy_no_base_fails_loud():
    ctx = SkillContext( services={})
    res = NavigateSkill().execute({"x": 10.5, "y": 3.0}, ctx)
    assert res.success is False
    assert res.diagnosis_code == "no_base"


@pytest.mark.parametrize("reason", ["aborted", "fell", "stalled", "unreachable"])
def test_a_failed_planner_run_near_the_goal_is_a_failure(reason):
    """skills-nav:coord-nav-false-success: a run the planner reports as not arrived
    succeeded whenever it stopped within 1.5 m. Dog 0.5 m from the goal."""
    ctx = _ctx(arrived=False, pos=(11.0, 3.0, 0.35), reason=reason)
    res = NavigateSkill().execute({"x": 10.5, "y": 3.0}, ctx)
    ctx.base.navigate_to.assert_called_once()
    assert res.success is False and res.diagnosis_code == "navigation_failed"
    assert res.result_data["reason"] == reason and reason in res.error_message
    assert res.result_data["position"] == [11.0, 3.0]


def test_a_type_error_inside_the_planner_is_not_a_second_navigation():
    """skills-nav:coord-nav-typeerror-rerun: a TypeError re-ran the whole navigation."""
    ctx = _ctx()
    ctx.base.navigate_to.side_effect = TypeError("inside the planner")
    with pytest.raises(TypeError):
        NavigateSkill().execute({"x": 10.5, "y": 3.0}, ctx)
    ctx.base.navigate_to.assert_called_once()


def test_navigate_xy_far_false_and_far_away_fails_loud():
    """FAR False AND the dog far from the goal (couldn't route) -> fail loud.
    Dog at (14.0, 3.0), goal (10.5, 3.0): 3.5 m away, outside the vicinity."""
    ctx = _ctx(arrived=False, pos=(14.0, 3.0, 0.35))
    res = NavigateSkill().execute({"x": 10.5, "y": 3.0}, ctx)
    assert res.success is False
    assert res.diagnosis_code == "navigation_failed"
    assert res.result_data["position"] == [14.0, 3.0]


def test_named_room_path_unchanged_when_no_coordinates():
    """With no x/y/target and no usable SceneGraph, the room path still runs
    (and fails loud about an unknown/unexplored room) — coordinate path is additive."""
    base = MagicMock()
    base.get_position.return_value = [10.0, 3.0, 0.35]
    base.navigate_to = MagicMock(return_value=True)
    ctx = SkillContext(bases={"go2": base}, services={})
    res = NavigateSkill().execute({"room": "nonexistent_room"}, ctx)
    assert res.success is False
    # It did NOT take the coordinate path (no navigate_to call on a bare room name).
    base.navigate_to.assert_not_called()


@pytest.mark.parametrize('params', [{'x': float('nan'), 'y': 2}, {'x': float('inf'), 'y': 2}, {'x': 1}, {'target': ['bad', 2]}])
def test_invalid_coordinate_never_falls_through_to_a_room_or_moves(params):
    ctx = _ctx()
    result = NavigateSkill().execute(dict(params, room='kitchen'), ctx)
    assert not result.success and result.diagnosis_code == 'invalid_target'
    ctx.base.navigate_to.assert_not_called()
