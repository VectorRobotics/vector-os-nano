# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""skills-manip:detect-zero-sentinel-pollution — DetectSkill stored 2D-only boxes at
(0, 0, 0), and pick then "found" them there."""
from __future__ import annotations

from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.types import Detection
from vector_os_nano.core.world_model import WorldModel
from vector_os_nano.skills.detect import DetectSkill


class _BoxesOnly:
    """A detector without 3D: tracking fails, as on a go2 camera without a tracker."""

    def detect(self, query):
        return [Detection(label="red can", bbox=(10, 10, 40, 60), confidence=0.8)]

    def track(self, detections):
        raise RuntimeError("no depth tracker")


def test_a_box_without_a_3d_pose_never_enters_the_world_model() -> None:
    wm = WorldModel()
    res = DetectSkill().execute({"query": "red can"}, SkillContext(perception=_BoxesOnly(), world_model=wm))
    assert res.success and res.result_data["count"] == 1
    assert res.result_data["objects"][0]["has_3d"] is False
    assert wm.get_objects() == []


def test_a_mobile_base_gets_no_identity_calibration() -> None:
    from vector_os_nano.core.agent import Agent

    class _Base:
        name = "base"

    agent = Agent(base=_Base())
    assert agent._build_context().calibration is None
