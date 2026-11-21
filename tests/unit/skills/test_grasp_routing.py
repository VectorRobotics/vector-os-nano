# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""One grasp offer per equipped body, with the independent holding predicate."""
import pytest
from vector_os_nano.core.skill import SkillRegistry
from vector_os_nano.skills import get_skills
from vector_os_nano.embodiments.registry import load as load_robot


@pytest.mark.parametrize('body', ['so101', 'piper', 'go2_piper'])
def test_one_grasp_and_its_bound_hint(body):
    reg = SkillRegistry()
    for skill in get_skills(load_robot(body).capabilities):
        reg.register(skill)
    schemas = {s['name']: s for s in reg.to_schemas()}
    assert schemas['grasp']['verify_hint'] == 'holding_object({query})'
    assert not {'pick', 'pick_top_down', 'perception_grasp', 'scan', 'detect', 'describe', 'handover', 'wave'} & schemas.keys()
