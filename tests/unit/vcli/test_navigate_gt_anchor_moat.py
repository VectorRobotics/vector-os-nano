# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Rule-5 moat: navigate_to(label) must never reach a target via GT.

In a VLN scenario (furnished room) the MEANS is perception. The boot-seeded GT
furniture lives in the world model ONLY as the verify anchor; navigate_to's label
resolver must NOT read it (Path A) and must NOT fall back to base.list_targets()
GT (Path B) — either would "arrive at the chair" with the VLM never invoked. With
the anchor filtered, a GT-only world model yields a loud failure that forces the
planner to recognize_navigate. A genuinely PERCEIVED object still resolves.

The non-VLN color-box room keeps its designed list_targets GT navigation
(Path B fires only when the base is NOT furnished) — covered by
tests/vcli/test_navigate_label_target.py and the last test here.
"""
from __future__ import annotations

from types import SimpleNamespace

from vector_os_nano.core.world_model import ObjectState, WorldModel
from vector_os_nano.skills.navigate_to_point import NavigateToPointSkill


class _Base:
    """navigate_to-capable base; records whether it was driven."""

    def __init__(self, furnished: bool, targets: dict | None = None):
        self._furnished = furnished
        self._targets = targets or {}
        self.called = None

    def list_targets(self):
        return dict(self._targets)

    def navigate_to(self, x, y, tol=0.2):
        self.called = (x, y, tol)
        return {"reached": True, "remaining": 0.1, "moved_m": 3.0,
                "pos": [x, y, 0.77], "transport": "sim_oracle"}


def _ctx(base, wm):
    return SimpleNamespace(base=base, world_model=wm, agent=None)


def _gt_chair():
    return ObjectState(object_id="target_chair", label="chair 椅子", x=3.6, y=0.0,
                       confidence=1.0, state="placed",
                       properties={"source": "g1_room_ground_truth",
                                   "verify_anchor": True})


def _perceived_chair():
    return ObjectState(object_id="chair#live", label="chair", x=2.9, y=0.1,
                       confidence=0.7, state="detected",
                       properties={"source": "vlm_detect"})


class TestPathA_GtSeededWmRead:
    def test_gt_only_world_model_fails_loud_no_teleport(self):
        # VLN room, WM holds ONLY the GT anchor (perception never ran).
        wm = WorldModel()
        wm.add_object(_gt_chair())
        base = _Base(furnished=True, targets={"target_chair": (3.6, 0.0)})
        res = NavigateToPointSkill().execute({"label": "chair"}, _ctx(base, wm))
        assert res.success is False
        assert base.called is None                      # NEVER drove to GT
        assert res.result_data.get("diagnosis") == "object_not_found"

    def test_perceived_object_still_resolves(self):
        # A real detection (source != ground_truth) is a valid means.
        wm = WorldModel()
        wm.add_object(_gt_chair())          # anchor present but must be ignored
        wm.add_object(_perceived_chair())
        base = _Base(furnished=True, targets={"target_chair": (3.6, 0.0)})
        res = NavigateToPointSkill().execute({"label": "chair"}, _ctx(base, wm))
        assert res.success is True
        assert base.called is not None
        assert abs(base.called[0] - 2.9) < 1e-6         # PERCEIVED x, not GT 3.6
        assert abs(base.called[1] - 0.1) < 1e-6


class TestPathB_ListTargetsFallback:
    def test_path_b_gated_off_in_vln_furnished(self):
        # No WM match AND furnished -> must NOT fall back to list_targets GT.
        base = _Base(furnished=True, targets={"target_chair": (3.6, 0.0)})
        res = NavigateToPointSkill().execute({"label": "chair"}, _ctx(base, WorldModel()))
        assert res.success is False
        assert base.called is None                      # GT teleport blocked

    def test_path_b_fires_in_non_vln_colorbox(self):
        # The designed substrate-agnostic coordinate nav (no perception scenario).
        base = _Base(furnished=False, targets={"target_blue": (3.7, 2.0)})
        res = NavigateToPointSkill().execute({"label": "blue"}, _ctx(base, WorldModel()))
        assert res.success is True
        assert base.called is not None
        assert abs(base.called[1] - 2.0) < 1e-6
