# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Rule-5 moat: GT-seeded objects must NOT reach the planner-facing world context.

Boot seeds the room's targets into the world model with their ground-truth
coordinates (source '*_ground_truth') as the deterministic VERIFY anchor. Those
must stay verify-only — if they leak into `_build_world_context`/`_live_objects_line`
the decomposer reads the chair's real (x, y) and navigate_to's it, "finding" the
chair without ever perceiving it (VLN bypassed). Only PERCEIVED objects belong in
the planner's view; filtering GT here only tightens the moat (rule 5).
"""
from __future__ import annotations

from vector_os_nano.core.world_model import ObjectState
from vector_os_nano.vcli.engine import VectorEngine


class _FakeWM:
    def __init__(self, objs):
        self._objs = list(objs)

    def get_objects(self):
        return list(self._objs)


class _FakeAgent:
    def __init__(self, wm):
        self._world_model = wm
        # no _arm/arm -> the GT-name fallback path is not taken


def test_gt_seeded_objects_excluded_from_planner_world_context():
    gt = ObjectState(object_id="target_chair", label="chair 椅子", x=3.6, y=0.0,
                     confidence=1.0, state="placed",
                     properties={"source": "g1_room_ground_truth"})
    perceived = ObjectState(object_id="sofa#1", label="sofa", x=1.0, y=2.0,
                            confidence=0.8, state="detected",
                            properties={"source": "vlm_detect"})
    line = VectorEngine._live_objects_line(_FakeAgent(_FakeWM([gt, perceived])))
    assert "sofa" in line            # perceived object IS surfaced to the planner
    assert "chair" not in line       # GT-seeded object is NOT
    assert "3.6" not in line         # the GT coordinate never leaks


def test_all_gt_yields_empty_planner_context():
    # When only GT seeds exist (nothing perceived yet), the planner sees NO
    # objects — so it must use VLN to locate them, not navigate to GT coords.
    gt = [
        ObjectState(object_id="target_chair", label="chair", x=3.6, y=0.0,
                    confidence=1.0, state="placed",
                    properties={"source": "g1_room_ground_truth"}),
        ObjectState(object_id="target_sofa", label="sofa", x=-2.0, y=1.0,
                    confidence=1.0, state="placed",
                    properties={"source": "go2_room_ground_truth"}),
    ]
    assert VectorEngine._live_objects_line(_FakeAgent(_FakeWM(gt))) == ""


def test_perceived_object_not_shadowed_by_gt_of_same_label():
    # A GT chair (conf 1.0) must not shadow a later PERCEIVED chair (conf 0.7):
    # the GT is filtered first, so the perceived one is what the planner sees.
    gt = ObjectState(object_id="target_chair", label="chair", x=3.6, y=0.0,
                     confidence=1.0, state="placed",
                     properties={"source": "g1_room_ground_truth"})
    perceived = ObjectState(object_id="chair#live", label="chair", x=2.9, y=0.1,
                            confidence=0.7, state="detected",
                            properties={"source": "vlm_detect"})
    line = VectorEngine._live_objects_line(_FakeAgent(_FakeWM([gt, perceived])))
    assert "chair" in line and "x=2.90" in line   # the PERCEIVED position
    assert "3.6" not in line                       # not the GT one
