# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The real arm skills declare a ``verify_hint`` that names tagged GT oracles only.

``Skill.to_schemas()`` surfaces each hint (the MCP tool listing reads it), and the
native loop binds a hint from the call's params as the action's auto post-verify
(``NativeStepRunner._bound_hint``), so a hint naming anything but an ``@oracle`` of
the robot world could never prove the action.
"""

from __future__ import annotations

import re
from typing import Any

from vector_os_nano.core.skill import SkillRegistry
from vector_os_nano.skills.home import HomeSkill
from vector_os_nano.skills.grasp import GraspSkill
from vector_os_nano.skills.place import PlaceSkill


def _arm_schemas() -> list[dict[str, Any]]:
    reg = SkillRegistry()
    for s in (HomeSkill(), GraspSkill(), PlaceSkill()):
        reg.register(s)
    return reg.to_schemas()


def test_to_schemas_includes_verify_hint_per_skill() -> None:
    by_name = {s["name"]: s for s in _arm_schemas()}
    assert by_name["grasp"]["verify_hint"] == "holding_object({query})"
    assert by_name["home"]["verify_hint"] == "arm_at_home()"
    assert by_name["place"]["verify_hint"] == "released()"


def test_verify_hints_name_tagged_robot_oracles_only() -> None:
    from tests.harness.fake_gt import FakeGT, fake_namespace
    from vector_os_nano.vcli.cognitive.oracle_registry import oracle_tags

    gt = FakeGT(None, objects={"pickable_bottle_green": [10.9, 3.0, 0.32]})
    tagged = set(oracle_tags(fake_namespace(None, "go2_piper", gt=gt), "robot"))
    for s in _arm_schemas():
        called = set(re.findall(r"([A-Za-z_]\w*)\(", s["verify_hint"]))
        assert called and called <= tagged, f"{s['name']}: {s['verify_hint']!r} names an untagged predicate"
