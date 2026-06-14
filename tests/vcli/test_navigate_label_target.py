# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #8 R5 — NavigateToPointSkill resolves a labeled target via the
base's list_targets() when the world model has no semantic match.

This is the substrate-agnostic 'go to the target object's point' half of
req #5: in g1_room the targets are GT-known (base.list_targets), so '去蓝色目标'
drives there WITHOUT photoreal recognition (which stays gated by DQ-10). The
world-model path (semantic perception) is unchanged and still wins when present.
"""
from __future__ import annotations

from types import SimpleNamespace

from vector_os_nano.skills.navigate_to_point import NavigateToPointSkill


class _FakeRoomBase:
    """A base exposing list_targets() + navigate_to(), like G1 room mode."""

    def __init__(self):
        self.called = None

    def list_targets(self):
        return {"target_red": (3.7, 0.0), "target_blue": (3.7, 2.0)}

    def navigate_to(self, x, y, tol=0.2):
        self.called = (x, y, tol)
        return {"reached": True, "remaining": 0.1, "moved_m": 4.0,
                "pos": [x, y, 0.77], "transport": "sim_oracle"}


def _ctx(base):
    return SimpleNamespace(base=base, world_model=None, agent=None)


class TestLabelTargetFallback:
    def test_label_resolves_via_list_targets(self):
        base = _FakeRoomBase()
        res = NavigateToPointSkill().execute({"label": "target_blue"}, _ctx(base))
        assert res.success is True
        # routed to the blue target's coordinate via the base
        assert base.called is not None
        assert abs(base.called[0] - 3.7) < 1e-6
        assert abs(base.called[1] - 2.0) < 1e-6

    def test_color_alias_resolves(self):
        base = _FakeRoomBase()
        # 'blue' (without the target_ prefix) must still resolve
        res = NavigateToPointSkill().execute({"label": "blue"}, _ctx(base))
        assert res.success is True
        assert abs(base.called[1] - 2.0) < 1e-6

    def test_compound_and_zh_labels_resolve(self):
        # the planner emits varied shapes — all must reach target_blue
        for lbl in ("blue_target", "蓝色", "蓝色目标", "去蓝色目标"):
            base = _FakeRoomBase()
            res = NavigateToPointSkill().execute({"label": lbl}, _ctx(base))
            assert res.success is True, f"{lbl} failed"
            assert abs(base.called[1] - 2.0) < 1e-6, f"{lbl} wrong target"

    def test_unknown_label_still_fails_loud(self):
        base = _FakeRoomBase()
        res = NavigateToPointSkill().execute({"label": "purple"}, _ctx(base))
        assert res.success is False
        assert "purple" in (res.error_message or "")
