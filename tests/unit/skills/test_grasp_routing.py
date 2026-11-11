# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The perception grasp's schema carries its GT verify hint.

(The alias-precedence tests went with SkillRegistry.match: the model routes by the
skill descriptions; no keyword matcher remains.)
"""
from __future__ import annotations

from vector_os_nano.core.skill import SkillRegistry
from vector_os_nano.skills.perception_grasp import PerceptionGraspSkill
from vector_os_nano.skills.pick_top_down import PickTopDownSkill


def _registry_in_sim_order() -> SkillRegistry:
    """Mirror sim_tool _start_go2: PickTopDown first, PerceptionGrasp LAST."""
    reg = SkillRegistry()
    reg.register(PickTopDownSkill())
    reg.register(PerceptionGraspSkill())
    return reg


def test_perception_grasp_has_holding_object_verify_hint():
    schemas = {s["name"]: s for s in _registry_in_sim_order().to_schemas()}
    assert "perception_grasp" in schemas
    assert "holding_object" in schemas["perception_grasp"]["verify_hint"]
