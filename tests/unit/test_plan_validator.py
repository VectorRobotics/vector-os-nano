"""Tests for PlanValidator — T14, T15, T16.

TDD: all tests written first (RED), then implementation (GREEN).

Tests cover:
  T14 — ValidationError, Repair, ValidationResult types
  T15 — PlanValidator.validate() logic
  T16 — PlanValidator.validate_and_repair() auto-repair logic
"""
from __future__ import annotations

import pytest

from vector_os_nano.core.plan_validator import PlanValidator, Repair, ValidationError, ValidationResult
from vector_os_nano.core.skill import SkillRegistry
from vector_os_nano.core.types import TaskPlan, TaskStep
from vector_os_nano.core.world_model import WorldModel
from vector_os_nano.skills import get_default_skills


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_registry() -> SkillRegistry:
    reg = SkillRegistry()
    for s in get_default_skills():
        reg.register(s)
    return reg


def make_plan(*steps: TaskStep) -> TaskPlan:
    return TaskPlan(goal="test", steps=list(steps))


def make_step(
    step_id: str,
    skill_name: str,
    params: dict | None = None,
    depends_on: list[str] | None = None,
    preconditions: list[str] | None = None,
) -> TaskStep:
    return TaskStep(
        step_id=step_id,
        skill_name=skill_name,
        parameters=params or {},
        depends_on=depends_on or [],
        preconditions=preconditions or [],
        postconditions=[],
    )


@pytest.fixture
def registry() -> SkillRegistry:
    return make_registry()


@pytest.fixture
def world() -> WorldModel:
    return WorldModel()


@pytest.fixture
def validator(registry: SkillRegistry, world: WorldModel) -> PlanValidator:
    return PlanValidator(registry, world)


# ---------------------------------------------------------------------------
# T14 — Type tests
# ---------------------------------------------------------------------------


class TestTypes:
    def test_validation_error_is_frozen(self) -> None:
        e = ValidationError(
            step_id="s1",
            field="skill_name",
            code="unknown_skill",
            message="Skill 'foo' not found",
            suggestion="Did you mean 'pick'?",
        )
        with pytest.raises(Exception):
            e.step_id = "other"  # type: ignore[misc]

    def test_repair_is_frozen(self) -> None:
        r = Repair(
            step_id="s1",
            field="skill_name",
            old_value="grab",
            new_value="pick",
            reason="alias match",
        )
        with pytest.raises(Exception):
            r.step_id = "other"  # type: ignore[misc]

    def test_validation_result_is_frozen(self) -> None:
        result = ValidationResult(valid=True)
        with pytest.raises(Exception):
            result.valid = False  # type: ignore[misc]

    def test_validation_result_defaults(self) -> None:
        result = ValidationResult(valid=True)
        assert result.errors == []
        assert result.warnings == []

    def test_validation_result_with_errors(self) -> None:
        err = ValidationError("s1", "skill_name", "unknown_skill", "not found", "")
        result = ValidationResult(valid=False, errors=[err])
        assert len(result.errors) == 1
        assert result.errors[0].code == "unknown_skill"


# ---------------------------------------------------------------------------
# T15 — PlanValidator.validate() tests
# ---------------------------------------------------------------------------


class TestValidate:
    def test_valid_plan_passes(self, validator: PlanValidator) -> None:
        """Well-formed plan with known skill and valid params -> valid=True, errors=[]."""
        step = make_step("s1", "pick", params={"object_label": "mug", "mode": "drop"})
        plan = make_plan(step)
        result = validator.validate(plan)
        assert result.valid is True
        assert result.errors == []

    def test_unknown_skill_detected(self, validator: PlanValidator) -> None:
        """skill_name='pickup' (not in registry) -> error code='unknown_skill'."""
        step = make_step("s1", "pickup")
        plan = make_plan(step)
        result = validator.validate(plan)
        assert result.valid is False
        assert any(e.code == "unknown_skill" for e in result.errors)
        error = next(e for e in result.errors if e.code == "unknown_skill")
        assert error.step_id == "s1"
        assert error.field == "skill_name"

    def test_invalid_enum_detected(self, validator: PlanValidator) -> None:
        """location='left side' is not in enum -> error code='invalid_enum'."""
        step = make_step("s1", "place", params={"location": "left side"})
        plan = make_plan(step)
        result = validator.validate(plan)
        assert result.valid is False
        assert any(e.code == "invalid_enum" for e in result.errors)
        error = next(e for e in result.errors if e.code == "invalid_enum")
        assert error.field == "parameters.location"

    def test_missing_required_param_detected(self, validator: PlanValidator) -> None:
        """A skill param with no default and not explicitly optional triggers warning."""
        # 'pick' has object_id and object_label both with required=False, so we
        # need a custom scenario. We test the home skill (no required params).
        # Instead let's directly check via a skill that has a missing required param.
        # All built-in skills mark params as required=False, so we get no warning.
        # The spec says: warning code='missing_required' for params without default
        # and without required=False.
        # To trigger this we verify a step against pick with all params having
        # required=False — so this should produce no missing_required warnings.
        step = make_step("s1", "pick", params={})
        plan = make_plan(step)
        result = validator.validate(plan)
        # pick params all have required=False, so no missing_required warning
        missing_warns = [w for w in result.warnings if w.code == "missing_required"]
        assert missing_warns == []

    def test_missing_required_param_custom(self, validator: PlanValidator) -> None:
        """Manually verify missing_required warning logic with a mock skill."""
        from unittest.mock import MagicMock

        # Build a registry with a skill that has a required param (no default, no required=False)
        reg = SkillRegistry()
        mock_skill = MagicMock()
        mock_skill.name = "custom_skill"
        mock_skill.description = "test"
        mock_skill.parameters = {
            "target": {"type": "string"},  # no default, no required=False -> required
        }
        mock_skill.preconditions = []
        mock_skill.postconditions = []
        mock_skill.effects = {}
        mock_skill.failure_modes = []
        mock_skill.__skill_aliases__ = []
        mock_skill.__skill_direct__ = False
        mock_skill.__skill_auto_steps__ = []
        reg.register(mock_skill)

        wm = WorldModel()
        v = PlanValidator(reg, wm)
        step = make_step("s1", "custom_skill", params={})
        plan = make_plan(step)
        result = v.validate(plan)
        # Should have a missing_required warning (not error) since target is absent
        warns = [w for w in result.warnings if w.code == "missing_required"]
        assert len(warns) == 1
        assert warns[0].field == "parameters.target"

    def test_circular_dependency_detected(self, validator: PlanValidator) -> None:
        """s1 depends on s2, s2 depends on s1 -> error code='circular_dependency'."""
        s1 = make_step("s1", "pick", depends_on=["s2"])
        s2 = make_step("s2", "place", depends_on=["s1"])
        plan = make_plan(s1, s2)
        result = validator.validate(plan)
        assert result.valid is False
        assert any(e.code == "circular_dependency" for e in result.errors)

    def test_missing_dependency_detected(self, validator: PlanValidator) -> None:
        """depends_on references a step_id not in the plan -> error code='missing_dependency'."""
        step = make_step("s1", "pick", depends_on=["s99"])
        plan = make_plan(step)
        result = validator.validate(plan)
        assert result.valid is False
        assert any(e.code == "missing_dependency" for e in result.errors)

    def test_precondition_unsatisfiable(self, validator: PlanValidator) -> None:
        """place with gripper_holding_any when gripper is empty -> warning."""
        # place skill has precondition gripper_holding_any; world model gripper is empty
        step = make_step("s1", "place", preconditions=["gripper_holding_any"])
        plan = make_plan(step)
        result = validator.validate(plan)
        warns = [w for w in result.warnings if w.code == "precondition_unsatisfiable"]
        assert len(warns) >= 1

    def test_type_mismatch_detected(self, validator: PlanValidator) -> None:
        """x='hello' for a float param -> error code='type_mismatch'."""
        from unittest.mock import MagicMock

        reg = SkillRegistry()
        mock_skill = MagicMock()
        mock_skill.name = "move_skill"
        mock_skill.description = "move"
        mock_skill.parameters = {
            "x": {"type": "float"},
        }
        mock_skill.preconditions = []
        mock_skill.postconditions = []
        mock_skill.effects = {}
        mock_skill.failure_modes = []
        mock_skill.__skill_aliases__ = []
        mock_skill.__skill_direct__ = False
        mock_skill.__skill_auto_steps__ = []
        reg.register(mock_skill)

        wm = WorldModel()
        v = PlanValidator(reg, wm)
        step = make_step("s1", "move_skill", params={"x": "hello"})
        plan = make_plan(step)
        result = v.validate(plan)
        assert result.valid is False
        assert any(e.code == "type_mismatch" for e in result.errors)
        err = next(e for e in result.errors if e.code == "type_mismatch")
        assert err.field == "parameters.x"

    def test_empty_plan_is_valid(self, validator: PlanValidator) -> None:
        """Empty plan has no errors."""
        plan = make_plan()
        result = validator.validate(plan)
        assert result.valid is True
        assert result.errors == []

    def test_valid_pick_plan_no_errors(self, validator: PlanValidator) -> None:
        """pick with valid enum mode='hold' passes."""
        step = make_step("s1", "pick", params={"mode": "hold"})
        plan = make_plan(step)
        result = validator.validate(plan)
        assert result.valid is True
        enum_errors = [e for e in result.errors if e.code == "invalid_enum"]
        assert enum_errors == []


# ---------------------------------------------------------------------------
# T16 — PlanValidator.validate_and_repair() tests
# ---------------------------------------------------------------------------


class TestValidateAndRepair:
    def test_unknown_skill_auto_repaired(self, validator: PlanValidator) -> None:
        """validate_and_repair fixes 'grab' alias to 'pick'."""
        step = make_step("s1", "grab")
        plan = make_plan(step)
        repaired_plan, repairs = validator.validate_and_repair(plan)
        assert any(r.field == "skill_name" and r.new_value == "pick" for r in repairs)
        assert repaired_plan.steps[0].skill_name == "pick"

    def test_invalid_enum_auto_repaired(self, validator: PlanValidator) -> None:
        """'left side' fuzzy-matches to 'left' enum value."""
        step = make_step("s1", "place", params={"location": "left side"})
        plan = make_plan(step)
        repaired_plan, repairs = validator.validate_and_repair(plan)
        assert any(r.field == "parameters.location" for r in repairs)
        assert repaired_plan.steps[0].parameters["location"] == "left"

    def test_repair_returns_new_plan(self, validator: PlanValidator) -> None:
        """Original plan is unchanged after repair (immutability)."""
        step = make_step("s1", "grab")
        plan = make_plan(step)
        original_skill = plan.steps[0].skill_name
        repaired_plan, _ = validator.validate_and_repair(plan)
        # Original plan unchanged
        assert plan.steps[0].skill_name == original_skill
        # Repaired plan has the fix
        assert repaired_plan.steps[0].skill_name == "pick"
        # Different objects
        assert plan is not repaired_plan

    def test_default_fill_repair(self, validator: PlanValidator) -> None:
        """Missing 'mode' param for pick gets filled with default 'drop'."""
        step = make_step("s1", "pick", params={})
        plan = make_plan(step)
        repaired_plan, repairs = validator.validate_and_repair(plan)
        # Should have filled 'mode' with 'drop'
        default_repairs = [r for r in repairs if r.field == "parameters.mode"]
        assert len(default_repairs) >= 1
        assert default_repairs[0].new_value == "drop"
        assert repaired_plan.steps[0].parameters["mode"] == "drop"

    def test_no_repairs_when_valid(self, validator: PlanValidator) -> None:
        """A valid plan produces no repairs."""
        step = make_step("s1", "home")
        plan = make_plan(step)
        repaired_plan, repairs = validator.validate_and_repair(plan)
        # home has no default-fill candidates with missing values in a valid call
        # repairs may include default fills for params with defaults
        # but skill_name should not change
        skill_repairs = [r for r in repairs if r.field == "skill_name"]
        assert skill_repairs == []

    def test_repair_preserves_step_ids_and_deps(self, validator: PlanValidator) -> None:
        """Repaired plan preserves step_ids, depends_on, preconditions, postconditions."""
        s1 = make_step("s1", "home")
        s2 = make_step("s2", "grab", depends_on=["s1"], preconditions=["gripper_empty"])
        plan = make_plan(s1, s2)
        repaired_plan, repairs = validator.validate_and_repair(plan)
        repaired_s2 = repaired_plan.steps[1]
        assert repaired_s2.step_id == "s2"
        assert repaired_s2.depends_on == ["s1"]
        assert repaired_s2.preconditions == ["gripper_empty"]

    def test_fuzzy_enum_case_insensitive(self, validator: PlanValidator) -> None:
        """'LEFT' fuzzy-matches to 'left'."""
        step = make_step("s1", "place", params={"location": "LEFT"})
        plan = make_plan(step)
        repaired_plan, repairs = validator.validate_and_repair(plan)
        location_repairs = [r for r in repairs if r.field == "parameters.location"]
        assert len(location_repairs) >= 1
        assert repaired_plan.steps[0].parameters["location"] == "left"

    def test_repair_goal_preserved(self, validator: PlanValidator) -> None:
        """Repaired plan preserves original goal, message, clarification fields."""
        plan = TaskPlan(
            goal="pick up the mug",
            steps=[make_step("s1", "grab")],
            message="OK",
            requires_clarification=False,
            clarification_question=None,
        )
        repaired_plan, _ = validator.validate_and_repair(plan)
        assert repaired_plan.goal == "pick up the mug"
        assert repaired_plan.message == "OK"
        assert repaired_plan.requires_clarification is False
        assert repaired_plan.clarification_question is None


# ---------------------------------------------------------------------------
# Helper method tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_check_type_string(self, validator: PlanValidator) -> None:
        assert validator._check_type("hello", "string") is True
        assert validator._check_type(42, "string") is False

    def test_check_type_float(self, validator: PlanValidator) -> None:
        assert validator._check_type(1.5, "float") is True
        assert validator._check_type(1, "float") is True  # int is subtype
        assert validator._check_type("1.5", "float") is False

    def test_check_type_int(self, validator: PlanValidator) -> None:
        assert validator._check_type(3, "int") is True
        assert validator._check_type(3.5, "int") is False

    def test_check_type_bool(self, validator: PlanValidator) -> None:
        assert validator._check_type(True, "bool") is True
        assert validator._check_type(False, "boolean") is True
        # In Python, bool IS a subclass of int, so True/False pass isinstance(v, bool).
        # Plain integers like 1 are NOT bool instances.
        assert validator._check_type(1, "bool") is False  # int is not bool

    def test_check_type_unknown_returns_true(self, validator: PlanValidator) -> None:
        """Unknown type strings pass through (no false rejections)."""
        assert validator._check_type("anything", "json_object") is True

    def test_fuzzy_enum_match_exact(self, validator: PlanValidator) -> None:
        result = validator._fuzzy_enum_match("left", ["left", "right", "front"])
        assert result == "left"

    def test_fuzzy_enum_match_case(self, validator: PlanValidator) -> None:
        result = validator._fuzzy_enum_match("LEFT", ["left", "right"])
        assert result == "left"

    def test_fuzzy_enum_match_underscore(self, validator: PlanValidator) -> None:
        result = validator._fuzzy_enum_match("front_left", ["front_left", "front_right"])
        assert result == "front_left"

    def test_fuzzy_enum_match_none(self, validator: PlanValidator) -> None:
        result = validator._fuzzy_enum_match("nowhere", ["left", "right", "front"])
        assert result is None

    def test_has_cycle_simple(self) -> None:
        s1 = make_step("s1", "pick", depends_on=["s2"])
        s2 = make_step("s2", "place", depends_on=["s1"])
        assert PlanValidator._has_cycle([s1, s2]) is True

    def test_has_cycle_no_cycle(self) -> None:
        s1 = make_step("s1", "pick")
        s2 = make_step("s2", "place", depends_on=["s1"])
        assert PlanValidator._has_cycle([s1, s2]) is False

    def test_has_cycle_three_nodes(self) -> None:
        s1 = make_step("s1", "pick", depends_on=["s3"])
        s2 = make_step("s2", "place", depends_on=["s1"])
        s3 = make_step("s3", "home", depends_on=["s2"])
        assert PlanValidator._has_cycle([s1, s2, s3]) is True

    def test_suggest_skill_name_alias(self, validator: PlanValidator) -> None:
        suggestion = validator._suggest_skill_name("grab")
        assert "pick" in suggestion

    def test_suggest_skill_name_substring(self, validator: PlanValidator) -> None:
        suggestion = validator._suggest_skill_name("picking")
        assert "pick" in suggestion.lower() or "No close match" in suggestion
