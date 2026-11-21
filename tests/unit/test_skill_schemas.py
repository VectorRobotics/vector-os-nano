# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Unit tests for built-in skill schemas and protocol compliance.

Tests verify:
- Each skill has correct name, description, parameters, pre/postconditions, effects
- All skills satisfy the Skill runtime-checkable Protocol
- get_skills(load_robot("so101").capabilities) returns the SO-101 set
"""
from __future__ import annotations

import pytest
from vector_os_nano.embodiments.registry import load as load_robot

from vector_os_nano.core.skill import Skill
from vector_os_nano.skills import (
    HomeSkill,
    PlaceSkill,
    get_skills,
)


# ---------------------------------------------------------------------------
# HomeSkill schema
# ---------------------------------------------------------------------------


class TestHomeSkillSchema:
    def setup_method(self):
        self.skill = HomeSkill()

    def test_name(self):
        assert self.skill.name == "home"

    def test_description_nonempty(self):
        assert isinstance(self.skill.description, str)
        assert len(self.skill.description) > 0

    def test_parameters_empty(self):
        """HomeSkill takes no parameters."""
        assert self.skill.parameters == {}

    def test_preconditions_empty(self):
        """HomeSkill is always executable — no preconditions."""
        assert self.skill.preconditions == []

    def test_home_declares_no_release(self):
        """Home keeps a held object, so it is never read as a place (native loop)."""
        assert "held_object" not in self.skill.effects
        assert "gripper_state" not in self.skill.effects
        assert self.skill.postconditions == []

    def test_effects_is_moving_false(self):
        assert self.skill.effects.get("is_moving") is False

    def test_satisfies_skill_protocol(self):
        assert isinstance(self.skill, Skill)


# ---------------------------------------------------------------------------
# ScanSkill schema
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# DetectSkill schema
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# PickSkill schema
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# PlaceSkill schema
# ---------------------------------------------------------------------------


class TestPlaceSkillSchema:
    def setup_method(self):
        self.skill = PlaceSkill()

    def test_name(self):
        assert self.skill.name == "place"

    def test_description_nonempty(self):
        assert isinstance(self.skill.description, str)
        assert len(self.skill.description) > 0

    def test_parameters_has_xyz(self):
        assert "x" in self.skill.parameters
        assert "y" in self.skill.parameters
        assert "z" in self.skill.parameters

    def test_xyz_not_required(self):
        assert self.skill.parameters["x"]["required"] is False
        assert self.skill.parameters["y"]["required"] is False
        assert self.skill.parameters["z"]["required"] is False

    def test_has_location_param(self):
        assert "location" in self.skill.parameters
        assert self.skill.parameters["location"]["default"] == "front"

    def test_preconditions_gripper_holding_any(self):
        assert "gripper_holding_any" in self.skill.preconditions

    def test_postconditions_gripper_empty(self):
        assert "gripper_empty" in self.skill.postconditions

    def test_effects_gripper_state_open(self):
        assert self.skill.effects.get("gripper_state") == "open"

    def test_effects_held_object_none(self):
        assert self.skill.effects.get("held_object") is None

    def test_satisfies_skill_protocol(self):
        assert isinstance(self.skill, Skill)


# ---------------------------------------------------------------------------
# get_so101_skills
# ---------------------------------------------------------------------------

_SO101_SET = {"stop", "home", "grasp", "place", "gripper_open", "gripper_close", "place_top_down"}


class TestGetSo101Skills:
    def setup_method(self):
        self.skills = get_skills(load_robot("so101").capabilities)

    def test_returns_all_skills(self):
        assert len(self.skills) == len(_SO101_SET)

    def test_all_satisfy_skill_protocol(self):
        for skill in self.skills:
            assert isinstance(skill, Skill), (
                f"{skill.__class__.__name__} does not satisfy Skill protocol"
            )

    def test_all_skill_names_present(self):
        names = {s.name for s in self.skills}
        assert names == _SO101_SET

    def test_all_skills_have_descriptions(self):
        for skill in self.skills:
            assert isinstance(skill.description, str)
            assert len(skill.description) > 0, f"{skill.name} has empty description"

    def test_all_skills_have_parameters_dict(self):
        for skill in self.skills:
            assert isinstance(skill.parameters, dict), (
                f"{skill.name}.parameters must be a dict"
            )

    def test_all_skills_have_preconditions_list(self):
        for skill in self.skills:
            assert isinstance(skill.preconditions, list), (
                f"{skill.name}.preconditions must be a list"
            )

    def test_all_skills_have_postconditions_list(self):
        for skill in self.skills:
            assert isinstance(skill.postconditions, list), (
                f"{skill.name}.postconditions must be a list"
            )

    def test_all_skills_have_effects_dict(self):
        for skill in self.skills:
            assert isinstance(skill.effects, dict), (
                f"{skill.name}.effects must be a dict"
            )

    def test_unique_names(self):
        names = [s.name for s in self.skills]
        assert len(names) == len(set(names)), "Skill names must be unique"


# ---------------------------------------------------------------------------
# test_pick_sampling_from_perception
# ---------------------------------------------------------------------------

