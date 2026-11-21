# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Built-in skills, shared across the configured arms."""
from vector_os_nano.skills.go2.stop import StopSkill
from vector_os_nano.skills.grasp import GraspSkill
from vector_os_nano.skills.gripper import GripperCloseSkill, GripperOpenSkill
from vector_os_nano.skills.home import HomeSkill
from vector_os_nano.skills.place import PlaceSkill
from vector_os_nano.skills.place_top_down import PlaceTopDownSkill

__all__ = ["GraspSkill", "GripperCloseSkill", "GripperOpenSkill", "HomeSkill", "PlaceSkill",
           "get_piper_skills", "get_so101_skills"]


def get_so101_skills():
    return [StopSkill(), HomeSkill(), GraspSkill(), PlaceSkill(), GripperOpenSkill(), GripperCloseSkill()]


def get_piper_skills():
    return [GraspSkill(), PlaceTopDownSkill(), GripperOpenSkill(), GripperCloseSkill()]
