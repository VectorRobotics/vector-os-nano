# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Skills register their own requirements; bodies select by capability."""
import importlib
import pkgutil

from vector_os_nano.core.skill import SKILL_TYPES
from vector_os_nano.schemas.robot import CapabilityProfile
from vector_os_nano.skills.go2.stop import StopSkill
from vector_os_nano.skills.grasp import GraspSkill
from vector_os_nano.skills.gripper import GripperCloseSkill, GripperOpenSkill
from vector_os_nano.skills.home import HomeSkill
from vector_os_nano.skills.place import PlaceSkill
from vector_os_nano.skills.place_top_down import PlaceTopDownSkill

__all__ = ["GraspSkill", "GripperCloseSkill", "GripperOpenSkill", "HomeSkill", "PlaceSkill",
           "get_skills"]


def get_skills(capabilities: CapabilityProfile) -> list:
    for module in pkgutil.walk_packages(__path__, __name__ + "."):
        importlib.import_module(module.name)
    available = {
        name for name, present in (
            ("base", capabilities.has_base), ("arm", capabilities.has_arm),
            ("gripper", capabilities.has_gripper), ("camera", capabilities.camera),
            ("depth", capabilities.depth), ("lidar", capabilities.lidar),
        ) if present
    }
    return [cls() for cls in SKILL_TYPES.values() if cls.__skill_requires__ <= available]
