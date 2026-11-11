# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Offline unit tests for NavigateToObjectSkill.

No sim, no nav stack — a real SceneGraph (the object position source of truth)
plus a fake base that records navigate_to calls. The real-sim e2e is covered by
tests/e2e_repl/test_e11_perception_memory.py + the bare-cli fetch flow.
"""
from __future__ import annotations

from typing import Any

import pytest

from vector_os_nano.core.scene_graph import ObjectNode, SceneGraph
from vector_os_nano.core.skill import SkillContext
from vector_os_nano.skills.navigate_to_object import (
    NavigateToObjectSkill,
    _VICINITY_CLEARANCE_M,
)


def _assert_standoff(nav_calls, obj_xy):
    """The single navigate_to target is a standoff at the vicinity clearance from
    the object (never the object centre — that cell is an inflated obstacle)."""
    assert len(nav_calls) == 1
    tx, ty = nav_calls[0]
    d = ((tx - obj_xy[0]) ** 2 + (ty - obj_xy[1]) ** 2) ** 0.5
    assert d == pytest.approx(_VICINITY_CLEARANCE_M, abs=1e-3)


class _FakeBase:
    """Records the last navigate_to target; reports a fixed position."""

    def __init__(self, pos: tuple[float, float, float] = (1.0, 1.0, 0.3)) -> None:
        self._pos = pos
        self.nav_calls: list[tuple[float, float]] = []

    def get_position(self):
        return self._pos

    def get_heading(self):
        return 0.0

    def navigate_to(self, x: float, y: float, timeout: float | None = None,
                    on_progress: Any = None, **kw) -> bool:
        self.nav_calls.append((x, y))
        return True

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        pass


def _sg_with(objects: list[tuple[str, float | None, float | None]]) -> SceneGraph:
    """Build a SceneGraph holding the given (category, x, y) objects."""
    sg = SceneGraph()
    for i, (cat, x, y) in enumerate(objects):
        sg.add_object(ObjectNode(
            object_id=f"obj_{i}", category=cat, room_id="kitchen",
            x=x, y=y, confidence=0.9,
        ))
    return sg


def _ctx(base: _FakeBase, sg: SceneGraph | None) -> SkillContext:
    services = {"spatial_memory": sg} if sg is not None else {}
    return SkillContext(base=base, services=services)


def test_navigates_to_localized_object():
    base = _FakeBase(pos=(10.0, 3.0, 0.3))
    sg = _sg_with([("green bottle", 10.88, 3.00)])
    res = NavigateToObjectSkill().execute({"object": "green bottle"}, _ctx(base, sg))
    assert res.success, res.error_message
    _assert_standoff(base.nav_calls, (10.88, 3.00))
    assert res.result_data["matched_category"] == "green bottle"
    assert res.result_data["object_world"] == [10.88, 3.0]


def test_substring_category_match():
    """'bottle' should match 'green bottle' via substring."""
    base = _FakeBase(pos=(10.0, 3.0, 0.3))
    sg = _sg_with([("green bottle", 10.88, 3.00)])
    res = NavigateToObjectSkill().execute({"object": "bottle"}, _ctx(base, sg))
    assert res.success
    _assert_standoff(base.nav_calls, (10.88, 3.00))


def test_picks_nearest_among_matches():
    base = _FakeBase(pos=(1.0, 1.0, 0.3))
    sg = _sg_with([("green bottle", 5.0, 5.0), ("blue bottle", 1.5, 1.0)])
    res = NavigateToObjectSkill().execute({"object": "bottle"}, _ctx(base, sg))
    assert res.success
    # nearest to (1,1) is the blue bottle at (1.5,1.0); drive to its standoff
    _assert_standoff(base.nav_calls, (1.5, 1.0))
    assert res.result_data["matched_category"] == "blue bottle"


def test_object_not_found_lists_available():
    base = _FakeBase()
    sg = _sg_with([("green bottle", 10.88, 3.00)])
    res = NavigateToObjectSkill().execute({"object": "banana"}, _ctx(base, sg))
    assert not res.success
    assert res.diagnosis_code == "object_not_found"
    assert "green bottle" in res.error_message
    assert base.nav_calls == []


def test_object_known_but_not_localized():
    """A category present but stored at (0,0) => not localized yet."""
    base = _FakeBase()
    sg = _sg_with([("green bottle", None, None)])
    res = NavigateToObjectSkill().execute({"object": "green bottle"}, _ctx(base, sg))
    assert not res.success
    assert res.diagnosis_code == "object_not_localized"
    assert base.nav_calls == []


def test_no_scene_graph():
    base = _FakeBase()
    res = NavigateToObjectSkill().execute({"object": "green bottle"}, _ctx(base, None))
    assert not res.success
    assert res.diagnosis_code == "no_scene_graph"


def test_no_base():
    sg = _sg_with([("green bottle", 10.88, 3.00)])
    ctx = SkillContext(services={"spatial_memory": sg})
    res = NavigateToObjectSkill().execute({"object": "green bottle"}, ctx)
    assert not res.success
    assert res.diagnosis_code == "no_base"


def test_no_object_name():
    base = _FakeBase()
    sg = _sg_with([("green bottle", 10.88, 3.00)])
    res = NavigateToObjectSkill().execute({}, _ctx(base, sg))
    assert not res.success
    assert res.diagnosis_code == "object_not_found"


def test_accepts_query_alias():
    """The object name may arrive under the 'query' alias."""
    base = _FakeBase(pos=(10.0, 3.0, 0.3))
    sg = _sg_with([("red can", 10.90, 3.22)])
    res = NavigateToObjectSkill().execute({"query": "red can"}, _ctx(base, sg))
    assert res.success
    _assert_standoff(base.nav_calls, (10.90, 3.22))
