# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Rule-5 moat: the world_query tool must NOT leak GT-seeded anchors.

WorldQueryTool is read_only + permission="allow", so the planner LLM auto-calls
it. If it serializes the boot-seeded GT objects (label + exact x,y), the LLM
reads the chair's ground-truth coordinate and binds it into navigate_to(x,y),
"finding" the chair with the VLM never invoked — a second prompt-side leak the
``_live_objects_line`` filter never covered (campaign #12 audit). It must surface
only PERCEIVED objects.
"""
from __future__ import annotations

import threading
from pathlib import Path

from vector_os_nano.core.world_model import ObjectState, WorldModel
from vector_os_nano.vcli.tools.base import ToolContext
from vector_os_nano.vcli.tools.robot import WorldQueryTool


class _Agent:
    def __init__(self, wm):
        self._world_model = wm


def _ctx(wm):
    return ToolContext(agent=_Agent(wm), cwd=Path("/tmp"), session=None,
                       permissions=None, abort=threading.Event())


def _build_wm():
    wm = WorldModel()
    wm.add_object(ObjectState(
        object_id="target_chair", label="chair 椅子", x=3.6, y=0.0,
        confidence=1.0, state="placed",
        properties={"source": "g1_room_ground_truth", "verify_anchor": True}))
    wm.add_object(ObjectState(
        object_id="sofa#live", label="sofa", x=1.0, y=2.0,
        confidence=0.8, state="detected", properties={"source": "vlm_detect"}))
    return wm


def test_world_query_hides_gt_anchor_shows_perceived():
    res = WorldQueryTool().execute({}, _ctx(_build_wm()))
    content = res.content
    assert "sofa" in content            # perceived object IS surfaced
    assert "chair" not in content       # GT anchor is NOT
    assert "3.6" not in content         # the GT coordinate never leaks


def test_world_query_all_gt_reports_empty():
    wm = WorldModel()
    wm.add_object(ObjectState(
        object_id="target_chair", label="chair 椅子", x=3.6, y=0.0,
        confidence=1.0, state="placed",
        properties={"source": "g1_room_ground_truth", "verify_anchor": True}))
    res = WorldQueryTool().execute({}, _ctx(wm))
    # nothing perceived yet -> the planner must use VLN, not a GT coordinate
    assert "chair" not in res.content
    assert "3.6" not in res.content
