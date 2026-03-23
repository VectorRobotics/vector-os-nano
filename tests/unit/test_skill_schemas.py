"""Unit tests for built-in skill schemas and protocol compliance.

Tests verify:
- Each skill has correct name, description, parameters, pre/postconditions, effects
- All skills satisfy the Skill runtime-checkable Protocol
- get_default_skills() returns all 5 skills
"""
from __future__ import annotations

import pytest

from vector_os_nano.core.skill import Skill
from vector_os_nano.skills import (
    DetectSkill,
    HomeSkill,
    PickSkill,
    PlaceSkill,
    ScanSkill,
    get_default_skills,
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

    def test_postconditions(self):
        assert "gripper_empty" in self.skill.postconditions

    def test_effects_gripper_open(self):
        assert self.skill.effects.get("gripper_state") == "open"

    def test_effects_held_object_none(self):
        assert self.skill.effects.get("held_object") is None

    def test_effects_is_moving_false(self):
        assert self.skill.effects.get("is_moving") is False

    def test_satisfies_skill_protocol(self):
        assert isinstance(self.skill, Skill)


# ---------------------------------------------------------------------------
# ScanSkill schema
# ---------------------------------------------------------------------------


class TestScanSkillSchema:
    def setup_method(self):
        self.skill = ScanSkill()

    def test_name(self):
        assert self.skill.name == "scan"

    def test_description_nonempty(self):
        assert isinstance(self.skill.description, str)
        assert len(self.skill.description) > 0

    def test_parameters_empty(self):
        assert self.skill.parameters == {}

    def test_preconditions_empty(self):
        assert self.skill.preconditions == []

    def test_effects_is_dict(self):
        assert isinstance(self.skill.effects, dict)

    def test_satisfies_skill_protocol(self):
        assert isinstance(self.skill, Skill)


# ---------------------------------------------------------------------------
# DetectSkill schema
# ---------------------------------------------------------------------------


class TestDetectSkillSchema:
    def setup_method(self):
        self.skill = DetectSkill()

    def test_name(self):
        assert self.skill.name == "detect"

    def test_description_nonempty(self):
        assert isinstance(self.skill.description, str)
        assert len(self.skill.description) > 0

    def test_parameters_has_query(self):
        assert "query" in self.skill.parameters

    def test_query_parameter_required(self):
        assert self.skill.parameters["query"]["required"] is True

    def test_query_parameter_type_string(self):
        assert self.skill.parameters["query"]["type"] == "string"

    def test_preconditions_empty(self):
        assert self.skill.preconditions == []

    def test_postconditions_empty(self):
        assert self.skill.postconditions == []

    def test_effects_empty(self):
        assert self.skill.effects == {}

    def test_satisfies_skill_protocol(self):
        assert isinstance(self.skill, Skill)


# ---------------------------------------------------------------------------
# PickSkill schema
# ---------------------------------------------------------------------------


class TestPickSkillSchema:
    def setup_method(self):
        self.skill = PickSkill()

    def test_name(self):
        assert self.skill.name == "pick"

    def test_description_nonempty(self):
        assert isinstance(self.skill.description, str)
        assert len(self.skill.description) > 0

    def test_parameters_has_object_id(self):
        assert "object_id" in self.skill.parameters

    def test_parameters_has_object_label(self):
        assert "object_label" in self.skill.parameters

    def test_object_id_not_required(self):
        assert self.skill.parameters["object_id"]["required"] is False

    def test_object_label_not_required(self):
        assert self.skill.parameters["object_label"]["required"] is False

    def test_preconditions_gripper_empty(self):
        assert "gripper_empty" in self.skill.preconditions

    def test_postconditions_empty(self):
        # Pick ends with drop — no postconditions
        assert self.skill.postconditions == []

    def test_effects_gripper_state_open(self):
        # Pick ends with drop
        assert self.skill.effects.get("gripper_state") == "open"

    def test_satisfies_skill_protocol(self):
        assert isinstance(self.skill, Skill)


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
# get_default_skills
# ---------------------------------------------------------------------------


class TestGetDefaultSkills:
    def setup_method(self):
        self.skills = get_default_skills()

    def test_returns_all_skills(self):
        assert len(self.skills) == 7

    def test_all_satisfy_skill_protocol(self):
        for skill in self.skills:
            assert isinstance(skill, Skill), (
                f"{skill.__class__.__name__} does not satisfy Skill protocol"
            )

    def test_all_skill_names_present(self):
        names = {s.name for s in self.skills}
        assert names == {"home", "scan", "detect", "pick", "place", "gripper_open", "gripper_close"}

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

class TestPickPerceptionSampling:
    """Tests for PickSkill._sample_from_perception() with mock perception."""

    def _make_context(self, perception, world_model=None):
        from vector_os_nano.core.skill import SkillContext
        from vector_os_nano.core.world_model import WorldModel
        from unittest.mock import MagicMock
        import numpy as np
        arm = MagicMock()
        arm.get_joint_positions.return_value = [0.0, 0.0, 0.0, 0.0, 0.0]
        arm.ik.return_value = [0.0, 0.0, 0.0, 0.0, 0.0]
        arm.move_joints.return_value = True
        gripper = MagicMock()
        wm = world_model or WorldModel()
        # identity calibration so camera=base
        cal_matrix = np.eye(4)
        return SkillContext(
            arm=arm,
            gripper=gripper,
            perception=perception,
            world_model=wm,
            calibration={"transform_matrix": cal_matrix.tolist()},
            config={},
        )

    def test_sample_uses_track_for_3d_position(self):
        """_sample_from_perception() calls detect() then track() to get 3D pose."""
        from unittest.mock import MagicMock
        import numpy as np
        from vector_os_nano.core.types import Detection, Pose3D, TrackedObject
        from vector_os_nano.skills.pick import PickSkill

        det = Detection(label="cup", bbox=(10.0, 10.0, 50.0, 50.0), confidence=0.9)
        pose = Pose3D(x=0.25, y=0.0, z=0.22)
        tracked = TrackedObject(track_id=1, label="cup", bbox_2d=(10, 10, 50, 50), pose=pose)

        mock_perception = MagicMock()
        mock_perception.detect.return_value = [det]
        mock_perception.track.return_value = [tracked]
        # update returns empty to limit sampling quickly
        mock_perception.update.return_value = []

        context = self._make_context(mock_perception)
        skill = PickSkill()
        # Override sample_count to 1 so we don't wait
        context = context.__class__(
            arm=context.arm,
            gripper=context.gripper,
            perception=context.perception,
            world_model=context.world_model,
            calibration=context.calibration,
            config={"skills": {"pick": {"sample_count": 1}}},
        )

        result = skill._sample_from_perception({"object_label": "cup"}, context)
        assert result is not None
        # With identity calibration, camera pos == base pos
        assert abs(result[0] - 0.25) < 1e-6
        assert abs(result[2] - 0.22) < 1e-6
        mock_perception.detect.assert_called_once()
        mock_perception.track.assert_called_once()

    def test_sample_returns_none_on_no_detections(self):
        """_sample_from_perception() returns None when detect() finds nothing."""
        from unittest.mock import MagicMock
        from vector_os_nano.skills.pick import PickSkill

        mock_perception = MagicMock()
        mock_perception.detect.return_value = []

        context = self._make_context(mock_perception)
        skill = PickSkill()
        result = skill._sample_from_perception({"object_label": "cup"}, context)
        assert result is None

    def test_sample_returns_none_on_detect_exception(self):
        """_sample_from_perception() returns None when detect() raises."""
        from unittest.mock import MagicMock
        from vector_os_nano.skills.pick import PickSkill

        mock_perception = MagicMock()
        mock_perception.detect.side_effect = RuntimeError("No VLM")

        context = self._make_context(mock_perception)
        skill = PickSkill()
        result = skill._sample_from_perception({"object_label": "cup"}, context)
        assert result is None

    def test_sample_returns_none_when_no_3d_pose(self):
        """_sample_from_perception() returns None if tracked objects have no 3D pose."""
        from unittest.mock import MagicMock
        from vector_os_nano.core.types import Detection, TrackedObject
        from vector_os_nano.skills.pick import PickSkill

        det = Detection(label="cup", bbox=(10, 10, 50, 50))
        # TrackedObject with pose=None (no depth data)
        tracked = TrackedObject(track_id=1, label="cup", bbox_2d=(10, 10, 50, 50), pose=None)

        mock_perception = MagicMock()
        mock_perception.detect.return_value = [det]
        mock_perception.track.return_value = [tracked]
        mock_perception.update.return_value = []

        context = self._make_context(mock_perception)
        skill = PickSkill()
        context = context.__class__(
            arm=context.arm,
            gripper=context.gripper,
            perception=context.perception,
            world_model=context.world_model,
            calibration=context.calibration,
            config={"skills": {"pick": {"sample_count": 1}}},
        )
        result = skill._sample_from_perception({"object_label": "cup"}, context)
        assert result is None
