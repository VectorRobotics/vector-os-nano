# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Built-in robot skills — all use @skill decorator for routing metadata."""
from __future__ import annotations

from vector_os_nano.skills.describe import DescribeSkill
from vector_os_nano.skills.detect import DetectSkill
from vector_os_nano.skills.go2.stop import StopSkill
from vector_os_nano.skills.gripper import GripperCloseSkill, GripperOpenSkill
from vector_os_nano.skills.handover import HandoverSkill
from vector_os_nano.skills.home import HomeSkill
from vector_os_nano.skills.pick import PickSkill
from vector_os_nano.skills.place import PlaceSkill
from vector_os_nano.skills.scan import ScanSkill
from vector_os_nano.skills.wave import WaveSkill

__all__ = [
    "DescribeSkill",
    "DetectSkill",
    "GripperCloseSkill",
    "GripperOpenSkill",
    "HandoverSkill",
    "HomeSkill",
    "PickSkill",
    "PlaceSkill",
    "ScanSkill",
    "WaveSkill",
    "get_piper_skills",
    "get_so101_skills",
]


def get_so101_skills() -> list:
    """One instance of each SO-101 arm skill (the fixed 5-DoF arm), and its stop."""
    return [
        StopSkill(),
        HomeSkill(),
        ScanSkill(),
        DescribeSkill(),
        DetectSkill(),
        PickSkill(),
        PlaceSkill(),
        HandoverSkill(),
        GripperOpenSkill(),
        GripperCloseSkill(),
        WaveSkill(),
    ]


def get_piper_skills() -> list:
    """The Piper manipulation set of the mobile arm (go2_piper), its gripper included."""
    from vector_os_nano.skills.perception_grasp import PerceptionGraspSkill
    from vector_os_nano.skills.pick_top_down import PickTopDownSkill
    from vector_os_nano.skills.place_top_down import PlaceTopDownSkill

    return [PickTopDownSkill(), PlaceTopDownSkill(), PerceptionGraspSkill(),
            GripperOpenSkill(), GripperCloseSkill()]
