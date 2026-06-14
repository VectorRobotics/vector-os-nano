# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #5 R3 (workflow-confirmed critical): NavigateSkill._dead_reckoning
must FAIL LOUD on a base with no waypoint navigation (G1MuJoCoBase walks but
has neither go_to_waypoint nor navigate_to), not raise AttributeError on an
unguarded ``base.navigate_to``.
"""
from __future__ import annotations

from vector_os_nano.core.skill import SkillContext
from vector_os_nano.skills.navigate import NavigateSkill


class _WalkOnlyBase:
    """A base that walks but exposes no waypoint service (the G1 shape)."""

    def get_position(self):
        return [0.0, 0.0, 0.0]


class _RoomNode:
    def __init__(self, cx, cy):
        self.center_x = cx
        self.center_y = cy


class _StubSceneGraph:
    """Yields a door chain with one FAR waypoint so the loop reaches the
    waypoint-navigation call (past the already-here short-circuit)."""

    def get_door_chain(self, src, dst):
        return [(10.0, 10.0, "kitchen_door")]

    def get_room(self, key):
        return _RoomNode(20.0, 20.0)   # far from (0,0): not already there


def test_dead_reckoning_no_waypoint_service_fails_loud():
    skill = NavigateSkill()
    ctx = SkillContext(
        base=_WalkOnlyBase(),
        world_model=None,
        services={"spatial_memory": _StubSceneGraph()},
    )
    res = skill._dead_reckoning("kitchen", ctx, total_timeout=10.0)
    assert res.success is False
    assert res.diagnosis_code == "navigation_unsupported"
    assert "go_to_waypoint" in (res.error_message or "")
