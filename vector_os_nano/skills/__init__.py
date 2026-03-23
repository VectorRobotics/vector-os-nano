"""Built-in robot skills — all use @skill decorator for routing metadata."""
from __future__ import annotations

from vector_os_nano.skills.detect import DetectSkill
from vector_os_nano.skills.gripper import GripperCloseSkill, GripperOpenSkill
from vector_os_nano.skills.home import HomeSkill
from vector_os_nano.skills.pick import PickSkill
from vector_os_nano.skills.place import PlaceSkill
from vector_os_nano.skills.scan import ScanSkill

__all__ = [
    "DetectSkill",
    "GripperCloseSkill",
    "GripperOpenSkill",
    "HomeSkill",
    "PickSkill",
    "PlaceSkill",
    "ScanSkill",
    "get_default_skills",
]


def get_default_skills() -> list:
    """Return one instance of each built-in skill."""
    return [
        HomeSkill(),
        ScanSkill(),
        DetectSkill(),
        PickSkill(),
        PlaceSkill(),
        GripperOpenSkill(),
        GripperCloseSkill(),
    ]
