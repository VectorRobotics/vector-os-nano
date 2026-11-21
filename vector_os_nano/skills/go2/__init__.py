# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Go2 quadruped skill package.

Locomotion and observation skills declare their own capability requirements.
"""
from vector_os_nano.skills.go2.walk import WalkSkill
from vector_os_nano.skills.go2.turn import TurnSkill
from vector_os_nano.skills.go2.stance import StandSkill, SitSkill, LieDownSkill
from vector_os_nano.skills.go2.explore import ExploreSkill
from vector_os_nano.skills.go2.where_am_i import WhereAmISkill
from vector_os_nano.skills.go2.stop import StopSkill
from vector_os_nano.skills.go2.look import LookSkill, DescribeSceneSkill
from vector_os_nano.skills.go2.patrol import PatrolSkill
from vector_os_nano.skills.navigate import NavigateSkill
from vector_os_nano.skills.navigate_to_object import NavigateToObjectSkill


__all__ = [
    "WalkSkill",
    "TurnSkill",
    "StandSkill",
    "SitSkill",
    "LieDownSkill",
    "NavigateSkill",
    "NavigateToObjectSkill",
    "ExploreSkill",
    "WhereAmISkill",
    "StopSkill",
    "LookSkill",
    "DescribeSceneSkill",
    "PatrolSkill",
]
