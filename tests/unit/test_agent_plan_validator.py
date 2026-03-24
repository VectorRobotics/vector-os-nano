"""Smoke tests for Agent._handle_task PlanValidator integration — T17."""
from __future__ import annotations


def test_plan_validator_import_no_circular():
    """PlanValidator can be imported without circular import errors."""
    from vector_os_nano.core.plan_validator import PlanValidator  # noqa: F401


def test_agent_handle_task_with_validator():
    """Agent._handle_task doesn't crash with PlanValidator in the loop.

    This is a smoke test — full validator tests are in test_plan_validator.py.
    Just verifies the import works and validate_and_repair + validate are callable.
    """
    from vector_os_nano.core.plan_validator import PlanValidator
    from vector_os_nano.core.skill import SkillRegistry
    from vector_os_nano.core.world_model import WorldModel
    from vector_os_nano.core.types import TaskPlan, TaskStep

    registry = SkillRegistry()
    from vector_os_nano.skills import get_default_skills
    for s in get_default_skills():
        registry.register(s)

    world = WorldModel()
    validator = PlanValidator(registry, world)

    plan = TaskPlan(goal="test", steps=[
        TaskStep(step_id="s1", skill_name="home", parameters={})
    ])
    repaired, repairs = validator.validate_and_repair(plan)
    result = validator.validate(repaired)
    assert result.valid
