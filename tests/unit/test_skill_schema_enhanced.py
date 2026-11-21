# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Enhanced skill schema tests — TDD for failure_modes in Skill protocol + to_schemas()."""
from __future__ import annotations


from vector_os_nano.core.skill import SkillRegistry, Skill


def test_to_schemas_includes_failure_modes():
    """to_schemas() output includes failure_modes when present on skill."""

    class TestSkill:
        name = "test"
        description = "test skill"
        parameters = {}
        preconditions = []
        postconditions = []
        effects = {}
        failure_modes = ["no_arm", "ik_unreachable"]
        __skill_aliases__ = []

        def execute(self, params, context):
            pass

    registry = SkillRegistry()
    registry.register(TestSkill())
    schemas = registry.to_schemas()
    assert len(schemas) == 1
    assert "failure_modes" in schemas[0]
    assert schemas[0]["failure_modes"] == ["no_arm", "ik_unreachable"]


def test_to_schemas_omits_empty_failure_modes():
    """to_schemas() does not include failure_modes key when list is empty."""

    class TestSkill:
        name = "test"
        description = "test"
        parameters = {}
        preconditions = []
        postconditions = []
        effects = {}
        failure_modes = []
        __skill_aliases__ = []

        def execute(self, params, context):
            pass

    registry = SkillRegistry()
    registry.register(TestSkill())
    schemas = registry.to_schemas()
    assert "failure_modes" not in schemas[0]


def test_to_schemas_backward_compat_no_failure_modes():
    """Skills without failure_modes attr still work (getattr default)."""

    class OldSkill:
        name = "old"
        description = "old"
        parameters = {}
        preconditions = []
        postconditions = []
        effects = {}
        # No failure_modes attribute!
        __skill_aliases__ = []

        def execute(self, params, context):
            pass

    registry = SkillRegistry()
    registry.register(OldSkill())
    schemas = registry.to_schemas()
    assert "failure_modes" not in schemas[0]


def test_all_default_skills_have_failure_modes():
    """Every skill from get_skills(load_robot("so101").capabilities) has failure_modes attribute."""
    from vector_os_nano.skills import get_skills
    from vector_os_nano.embodiments.registry import load as load_robot

    for skill in get_skills(load_robot("so101").capabilities):
        assert hasattr(skill, "failure_modes"), f"{skill.name} missing failure_modes"
        assert isinstance(skill.failure_modes, list), (
            f"{skill.name}.failure_modes is not a list"
        )


# ---------------------------------------------------------------------------
# T11: enum and source annotations for pick.py and place.py
# ---------------------------------------------------------------------------





def test_place_schema_requires_a_named_surface_without_a_silent_default():
    from vector_os_nano.skills.place import PlaceSkill
    surface = PlaceSkill.parameters['surface']
    assert surface['type'] == 'string' and surface['required']
    assert 'default' not in surface and 'enum' not in surface
    assert set(PlaceSkill.parameters) == {'surface'}
