"""Unit tests for StepTrace.result_data + executor diagnostic data — TDD RED phase.

Tests cover:
- StepTrace.result_data default value
- StepTrace serialization round-trip with result_data
- Executor copies skill result_data into StepTrace on success
- Executor copies skill result_data into StepTrace on failure
- Executor injects diagnosis="skill_not_found" when skill missing
- Executor injects diagnosis="precondition_failed" + predicate when precondition unmet
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_skill(
    name: str,
    success: bool = True,
    result_data: dict[str, Any] | None = None,
    preconditions: list[str] | None = None,
    postconditions: list[str] | None = None,
) -> Any:
    """Create a mock Skill with configurable SkillResult."""
    from vector_os_nano.core.types import SkillResult

    skill = MagicMock()
    skill.name = name
    skill.description = f"Mock {name} skill"
    skill.parameters = {}
    skill.preconditions = preconditions or []
    skill.postconditions = postconditions or []
    skill.effects = {}
    skill.execute.return_value = SkillResult(
        success=success,
        result_data=result_data or {},
    )
    return skill


def make_registry(*skills):
    from vector_os_nano.core.skill import SkillRegistry

    registry = SkillRegistry()
    for skill in skills:
        registry.register(skill)
    return registry


def make_context(world_model):
    from vector_os_nano.core.skill import SkillContext

    return SkillContext(
        arm=MagicMock(),
        gripper=MagicMock(),
        perception=None,
        world_model=world_model,
        calibration=None,
    )


def make_step(step_id, skill_name, params=None, preconditions=None, postconditions=None):
    from vector_os_nano.core.types import TaskStep

    return TaskStep(
        step_id=step_id,
        skill_name=skill_name,
        parameters=params or {},
        depends_on=[],
        preconditions=preconditions or [],
        postconditions=postconditions or [],
    )


def make_plan(steps):
    from vector_os_nano.core.types import TaskPlan

    return TaskPlan(goal="test", steps=steps)


@pytest.fixture
def world_model():
    from vector_os_nano.core.world_model import WorldModel

    return WorldModel()


# ---------------------------------------------------------------------------
# T1 — StepTrace.result_data field
# ---------------------------------------------------------------------------


class TestStepTraceResultDataField:
    def test_step_trace_result_data_default_empty(self):
        """StepTrace() constructed without result_data arg has result_data == {}."""
        from vector_os_nano.core.types import StepTrace

        trace = StepTrace(step_id="s0", skill_name="pick", status="success")
        assert trace.result_data == {}

    def test_step_trace_result_data_accepts_value(self):
        """StepTrace accepts explicit result_data dict."""
        from vector_os_nano.core.types import StepTrace

        trace = StepTrace(
            step_id="s0",
            skill_name="pick",
            status="success",
            result_data={"grasp_x": 0.3, "diagnosis": "ok"},
        )
        assert trace.result_data["grasp_x"] == 0.3
        assert trace.result_data["diagnosis"] == "ok"

    def test_step_trace_serialization_with_result_data(self):
        """Round-trip to_dict / from_dict preserves result_data."""
        from vector_os_nano.core.types import StepTrace

        original = StepTrace(
            step_id="s1",
            skill_name="place",
            status="execution_failed",
            duration_sec=1.5,
            error="something broke",
            result_data={"diagnosis": "exception", "exception_type": "RuntimeError"},
        )
        reconstructed = StepTrace.from_dict(original.to_dict())
        assert reconstructed == original
        assert reconstructed.result_data["diagnosis"] == "exception"
        assert reconstructed.result_data["exception_type"] == "RuntimeError"

    def test_step_trace_serialization_empty_result_data(self):
        """Round-trip preserves empty result_data dict."""
        from vector_os_nano.core.types import StepTrace

        original = StepTrace(step_id="s0", skill_name="home", status="success")
        reconstructed = StepTrace.from_dict(original.to_dict())
        assert reconstructed.result_data == {}

    def test_step_trace_is_still_frozen(self):
        """Adding result_data must not break frozen constraint."""
        import dataclasses

        from vector_os_nano.core.types import StepTrace

        trace = StepTrace(step_id="s0", skill_name="pick", status="success")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            trace.result_data = {"mutated": True}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# T2 — Executor copies result_data into StepTrace
# ---------------------------------------------------------------------------


class TestExecutorResultDataOnSuccess:
    def test_executor_copies_result_data_on_success(self, world_model):
        """Mock skill returns result_data — trace should carry it."""
        from vector_os_nano.core.executor import TaskExecutor

        skill = make_skill(
            "pick",
            success=True,
            result_data={"grasp_x": 0.3, "grasp_y": 0.1, "grasp_z": 0.05},
        )
        registry = make_registry(skill)
        context = make_context(world_model)
        plan = make_plan([make_step("s0", "pick")])

        result = TaskExecutor().execute(plan, registry, context)

        assert result.success is True
        trace = result.trace[0]
        assert trace.result_data["grasp_x"] == 0.3
        assert trace.result_data["grasp_y"] == 0.1
        assert trace.result_data["grasp_z"] == 0.05

    def test_executor_copies_result_data_on_failure(self, world_model):
        """Mock skill fails with result_data — trace should carry it."""
        from vector_os_nano.core.executor import TaskExecutor

        skill = make_skill(
            "pick",
            success=False,
            result_data={"object_detected": False, "confidence": 0.1},
        )
        registry = make_registry(skill)
        context = make_context(world_model)
        plan = make_plan([make_step("s0", "pick")])

        result = TaskExecutor().execute(plan, registry, context)

        assert result.success is False
        trace = result.trace[0]
        assert trace.result_data["object_detected"] is False
        assert trace.result_data["confidence"] == 0.1


class TestExecutorDiagnosticsSkillNotFound:
    def test_executor_skill_not_found_has_diagnosis(self, world_model):
        """Plan referencing unknown skill — trace has diagnosis='skill_not_found'."""
        from vector_os_nano.core.executor import TaskExecutor

        registry = make_registry()  # empty — no skills registered
        context = make_context(world_model)
        plan = make_plan([make_step("s0", "nonexistent_skill")])

        result = TaskExecutor().execute(plan, registry, context)

        assert result.success is False
        assert len(result.trace) == 1
        trace = result.trace[0]
        assert trace.result_data.get("diagnosis") == "skill_not_found"


class TestExecutorDiagnosticsPreconditionFailed:
    def test_executor_precondition_failed_has_diagnosis(self, world_model):
        """Unmet precondition — trace has diagnosis='precondition_failed' and predicate."""
        from vector_os_nano.core.executor import TaskExecutor

        # gripper_holding_any is False in fresh WorldModel
        skill = make_skill("place")
        registry = make_registry(skill)
        context = make_context(world_model)
        step = make_step("s0", "place", preconditions=["gripper_holding_any"])
        plan = make_plan([step])

        result = TaskExecutor().execute(plan, registry, context)

        assert result.success is False
        trace = result.trace[0]
        assert trace.result_data.get("diagnosis") == "precondition_failed"
        assert trace.result_data.get("predicate") == "gripper_holding_any"


class TestExecutorDiagnosticsException:
    def test_executor_exception_has_diagnosis(self, world_model):
        """Skill that raises exception — trace has diagnosis='exception' and exception_type."""
        from vector_os_nano.core.executor import TaskExecutor

        skill = MagicMock()
        skill.name = "bad_skill"
        skill.description = "raises"
        skill.parameters = {}
        skill.preconditions = []
        skill.postconditions = []
        skill.effects = {}
        skill.execute.side_effect = ValueError("boom")

        registry = make_registry(skill)
        context = make_context(world_model)
        plan = make_plan([make_step("s0", "bad_skill")])

        result = TaskExecutor().execute(plan, registry, context)

        assert result.success is False
        trace = result.trace[0]
        assert trace.result_data.get("diagnosis") == "exception"
        assert trace.result_data.get("exception_type") == "ValueError"


class TestExecutorDiagnosticsPostconditionFailed:
    def test_executor_postcondition_failed_has_diagnosis(self, world_model):
        """Postcondition fails — trace has diagnosis, predicate, and original result_data merged."""
        from vector_os_nano.core.executor import TaskExecutor
        from vector_os_nano.core.types import SkillResult

        skill = MagicMock()
        skill.name = "custom_pick"
        skill.description = "pick"
        skill.parameters = {}
        skill.preconditions = []
        skill.postconditions = ["gripper_holding_any"]
        skill.effects = {}
        # Skill succeeds but world model never gets updated → postcondition fails
        skill.execute.return_value = SkillResult(
            success=True,
            result_data={"attempt": 1},
        )

        registry = make_registry(skill)
        context = make_context(world_model)
        step = make_step("s0", "custom_pick")
        plan = make_plan([step])

        result = TaskExecutor().execute(plan, registry, context)

        assert result.success is False
        trace = result.trace[0]
        assert trace.result_data.get("diagnosis") == "postcondition_failed"
        assert trace.result_data.get("predicate") == "gripper_holding_any"
        # Original skill result_data merged in
        assert trace.result_data.get("attempt") == 1
