"""Go2 quadruped skill package.

Exports all Go2 skills and the get_go2_skills() factory.
"""
from vector_os_nano.skills.go2.walk import WalkSkill
from vector_os_nano.skills.go2.turn import TurnSkill
from vector_os_nano.skills.go2.stance import StandSkill, SitSkill, LieDownSkill
from vector_os_nano.skills.navigate import NavigateSkill


def get_go2_skills() -> list:
    """Return one instance of each Go2 skill."""
    return [
        WalkSkill(), TurnSkill(),
        StandSkill(), SitSkill(), LieDownSkill(),
        NavigateSkill(),
    ]


__all__ = [
    "WalkSkill",
    "TurnSkill",
    "StandSkill",
    "SitSkill",
    "LieDownSkill",
    "NavigateSkill",
    "get_go2_skills",
]
