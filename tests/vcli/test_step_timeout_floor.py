# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""R2-2 regression tests: GoalExecutor real-time timeout floor.

ROOT CAUSE: GoalExecutor compares elapsed wall-clock time against
sub_goal.timeout_sec. Under a live MuJoCo viewer motor skills run in
real-time (arm pick ~20s+), so a LLM-emitted timeout_sec=15 falsely marks
a completed pick as timed-out and triggers a bad replan.

FIX: skills declare typical_duration_sec; the executor floors the effective
timeout at that value so a completed motor action is never falsely failed.

Test strategy: drive GoalExecutor with a fake skill that sleeps ~0.3s in
headless/fast mode so tests are deterministic and complete in < 1s.
"""
from __future__ import annotations

import time
from typing import Any

import pytest

from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal


# ---------------------------------------------------------------------------
# Minimal stubs: selector, verifier, registry, skills
# ---------------------------------------------------------------------------


class _SkillResult:
    def __init__(self) -> None:
        self.success = True
        self.result_data: dict = {}
        self.error_message = ""


class _SlowSkillWithFloor:
    """Sleeps 0.3s and declares typical_duration_sec=1.0.

    Under plan timeout_sec=0.1 and no floor, 0.3s > 0.1s → would be timeout.
    With the floor, effective = max(0.1, 1.0) = 1.0 → 0.3s < 1.0s → NOT timeout.
    """
    name = "slow_motor"
    typical_duration_sec: float = 1.0

    def execute(self, params: dict, context: Any = None) -> _SkillResult:
        time.sleep(0.3)
        return _SkillResult()


class _SlowSkillNoFloor:
    """Sleeps 0.3s but declares NO typical_duration_sec.

    Under plan timeout_sec=0.1, 0.3s > 0.1s → timeout (no floor kicks in).
    This proves the floor is opt-in and the default path is unchanged.
    """
    name = "slow_plain"

    def execute(self, params: dict, context: Any = None) -> _SkillResult:
        time.sleep(0.3)
        return _SkillResult()


class _FastSkillWithFloor:
    """Completes near-instantly but declares typical_duration_sec=1.0.

    Proves the floor is still respected even when the skill is fast.
    """
    name = "fast_floored"
    typical_duration_sec: float = 1.0

    def execute(self, params: dict, context: Any = None) -> _SkillResult:
        return _SkillResult()


class _FakeRegistry:
    """Minimal skill registry: maps skill name → instance."""

    def __init__(self, skills: dict[str, Any]) -> None:
        self._skills = skills

    def get(self, name: str) -> Any | None:
        return self._skills.get(name)

    def list_skills(self) -> list[str]:
        return list(self._skills)


class _StrategyResult:
    def __init__(self, name: str) -> None:
        self.executor_type = "skill"
        self.name = name
        self.params: dict = {}


class _FakeSelector:
    """Returns a fixed StrategyResult for any sub_goal."""

    def __init__(self, skill_name: str) -> None:
        self._skill_name = skill_name

    def select(self, sub_goal: Any) -> _StrategyResult:
        return _StrategyResult(self._skill_name)


class _AlwaysTrueVerifier:
    def verify(self, expression: str) -> bool:
        return True


class _AlwaysFalseVerifier:
    def verify(self, expression: str) -> bool:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_executor(skill_name: str, registry: _FakeRegistry) -> GoalExecutor:
    """Build a GoalExecutor wired with a given skill and registry."""
    return GoalExecutor(
        strategy_selector=_FakeSelector(skill_name),
        verifier=_AlwaysTrueVerifier(),
        skill_registry=registry,
    )


def _make_plan(timeout_sec: float, skill_name: str = "skill") -> GoalTree:
    return GoalTree(
        goal="test",
        sub_goals=(
            SubGoal(
                name="step",
                description="a step",
                verify="True",
                timeout_sec=timeout_sec,
                strategy=f"{skill_name}_skill",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Core floor tests
# ---------------------------------------------------------------------------


class TestTimeoutFloor:
    """effective_timeout = max(plan_timeout, typical_duration_sec or 0)."""

    def test_floor_prevents_false_timeout(self) -> None:
        """A step that takes 0.3s is NOT timed-out when typical_duration_sec=1.0
        floors the limit above 0.3s, even if plan said timeout_sec=0.1.

        This is the R2-2 regression: a live pick (~20s) must not be falsely
        failed just because the LLM emitted timeout_sec=15.
        """
        registry = _FakeRegistry({"slow_motor": _SlowSkillWithFloor()})
        executor = _make_executor("slow_motor", registry)
        plan = _make_plan(timeout_sec=0.1, skill_name="slow_motor")

        trace = executor.execute(plan)

        step = trace.steps[0]
        assert step.success is True, (
            f"step was falsely failed: {step.error!r}; "
            "expected floor to allow 0.3s < max(0.1, 1.0)=1.0"
        )
        assert "timeout" not in step.error.lower()

    def test_no_floor_keeps_timeout_semantics(self) -> None:
        """Without typical_duration_sec, a blown budget still classifies as
        timeout — for a step whose verify does NOT pass (campaign #4 #17:
        a timed-out step that VERIFIES is now an honest PASS with a timing
        warning, so this test pins the unverified branch).

        Proves the floor is opt-in and the default path is unchanged.
        """
        registry = _FakeRegistry({"slow_plain": _SlowSkillNoFloor()})
        executor = GoalExecutor(
            strategy_selector=_FakeSelector("slow_plain"),
            verifier=_AlwaysFalseVerifier(),
            skill_registry=registry,
        )
        plan = _make_plan(timeout_sec=0.1, skill_name="slow_plain")

        trace = executor.execute(plan)

        step = trace.steps[0]
        assert step.success is False, "step should be timed out (no floor)"
        assert "timeout" in step.error.lower(), f"unexpected error: {step.error!r}"


class TestTimeoutFloorNoRegistry:
    """When skill_registry is None, the floor helper must not raise."""

    def test_none_registry_falls_back_to_plan_timeout(self) -> None:
        """GoalExecutor with no registry: floor = plan_timeout (unchanged)."""
        executor = GoalExecutor(
            strategy_selector=_FakeSelector("slow_plain"),
            verifier=_AlwaysTrueVerifier(),
            skill_registry=None,
        )
        # A 0.3s sleep with timeout_sec=0.1 → timed-out (no registry means no floor)
        plan = _make_plan(timeout_sec=0.1, skill_name="slow_plain")

        # We need to wire the skill in the selector's execute path somehow.
        # Since skill_registry=None, _execute_skill returns "Skill not found".
        # That's expected — just prove no crash and the step is not marked timeout
        # for a floor-related reason (it's marked failed for "not found").
        trace = executor.execute(plan)
        step = trace.steps[0]
        # Should fail with "not found", not with an AttributeError from _effective_timeout
        assert step.success is False
        assert "timeout" not in step.error.lower() or "not found" in step.error.lower()


class TestEffectiveTimeoutHelper:
    """Direct unit tests of GoalExecutor._effective_timeout."""

    def _executor_with_registry(self, registry: _FakeRegistry) -> GoalExecutor:
        return GoalExecutor(
            strategy_selector=_FakeSelector("x"),
            verifier=_AlwaysTrueVerifier(),
            skill_registry=registry,
        )

    def test_floor_applied_when_plan_is_lower(self) -> None:
        registry = _FakeRegistry({"slow_motor": _SlowSkillWithFloor()})
        exec_ = self._executor_with_registry(registry)
        result = exec_._effective_timeout(0.5, "slow_motor")
        assert result == 1.0  # typical_duration_sec wins

    def test_plan_wins_when_it_is_higher(self) -> None:
        registry = _FakeRegistry({"slow_motor": _SlowSkillWithFloor()})
        exec_ = self._executor_with_registry(registry)
        result = exec_._effective_timeout(999.0, "slow_motor")
        assert result == 999.0  # plan_timeout wins

    def test_no_floor_attr_unchanged(self) -> None:
        registry = _FakeRegistry({"slow_plain": _SlowSkillNoFloor()})
        exec_ = self._executor_with_registry(registry)
        result = exec_._effective_timeout(5.0, "slow_plain")
        assert result == 5.0  # no attribute → unchanged

    def test_unknown_skill_unchanged(self) -> None:
        registry = _FakeRegistry({})
        exec_ = self._executor_with_registry(registry)
        result = exec_._effective_timeout(5.0, "nonexistent")
        assert result == 5.0

    def test_none_registry_unchanged(self) -> None:
        exec_ = GoalExecutor(
            strategy_selector=_FakeSelector("x"),
            verifier=_AlwaysTrueVerifier(),
            skill_registry=None,
        )
        result = exec_._effective_timeout(5.0, "anything")
        assert result == 5.0

    def test_empty_strategy_name_unchanged(self) -> None:
        registry = _FakeRegistry({"slow_motor": _SlowSkillWithFloor()})
        exec_ = self._executor_with_registry(registry)
        result = exec_._effective_timeout(5.0, "")
        assert result == 5.0


# ---------------------------------------------------------------------------
# Motor skill attribute tests (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("import_path,class_name,expected", [
    ("vector_os_nano.skills.pick", "PickSkill", 45.0),
    ("vector_os_nano.skills.place", "PlaceSkill", 45.0),
    ("vector_os_nano.skills.home", "HomeSkill", 12.0),
    ("vector_os_nano.skills.scan", "ScanSkill", 15.0),
    ("vector_os_nano.skills.wave", "WaveSkill", 15.0),
    ("vector_os_nano.skills.handover", "HandoverSkill", 25.0),
    ("vector_os_nano.skills.go2.walk", "WalkSkill", 30.0),
    ("vector_os_nano.skills.go2.patrol", "PatrolSkill", 90.0),
    ("vector_os_nano.skills.go2.explore", "ExploreSkill", 150.0),
    ("vector_os_nano.skills.go2.turn", "TurnSkill", 10.0),
])
def test_motor_skill_typical_duration_sec(
    import_path: str, class_name: str, expected: float,
) -> None:
    """Each declared motor skill exposes the expected typical_duration_sec."""
    import importlib
    mod = importlib.import_module(import_path)
    cls = getattr(mod, class_name)
    instance = cls()
    val = getattr(instance, "typical_duration_sec", None)
    assert val == expected, (
        f"{class_name}.typical_duration_sec = {val!r}, expected {expected}"
    )


@pytest.mark.parametrize("import_path,class_name", [
    ("vector_os_nano.skills.pick", "PickSkill"),
    ("vector_os_nano.skills.place", "PlaceSkill"),
    ("vector_os_nano.skills.home", "HomeSkill"),
    ("vector_os_nano.skills.scan", "ScanSkill"),
    ("vector_os_nano.skills.wave", "WaveSkill"),
    ("vector_os_nano.skills.handover", "HandoverSkill"),
])
def test_motor_skill_to_schemas_still_works(import_path: str, class_name: str) -> None:
    """Adding typical_duration_sec must not break to_schemas() on the registry.

    The @skill decorator works on plain class attributes, not a rigid dataclass,
    so new attrs are fine — but verify that the registry's to_schemas() still
    produces a non-empty list with the expected skill name.
    """
    import importlib

    from vector_os_nano.core.skill import SkillRegistry

    mod = importlib.import_module(import_path)
    cls = getattr(mod, class_name)
    instance = cls()

    reg = SkillRegistry()
    reg.register(instance)
    schemas = reg.to_schemas()
    assert schemas, f"{class_name}: to_schemas() returned empty"
    names = [s.get("name") for s in schemas]
    assert instance.name in names, (
        f"{class_name}: skill name {instance.name!r} not in schemas {names}"
    )


# ---------------------------------------------------------------------------
# Foreach example timeout check
# ---------------------------------------------------------------------------


def test_foreach_example_body_timeout_is_45() -> None:
    """_FOREACH_EXAMPLE body template must use timeout_sec=45, not 15.

    The 15→45 bump (R2-2 secondary) ensures the LLM learns to emit generous
    timeouts for motor-action loop bodies, not copy a 15s template.
    """
    from vector_os_nano.vcli.cognitive.goal_decomposer import GoalDecomposer

    example = GoalDecomposer._FOREACH_EXAMPLE
    # Extract the JSON object from the example (it contains a JSON block)
    # Find the body's act_item timeout_sec value
    assert '"timeout_sec": 45' in example or '"timeout_sec":45' in example, (
        f"_FOREACH_EXAMPLE body timeout_sec should be 45, got something else.\n"
        f"Excerpt: {example[example.find('timeout_sec') - 10: example.find('timeout_sec') + 30]!r}"
    )
    # Also ensure 15 is NOT the body timeout (it was the old value)
    # (15 may appear elsewhere, so only check the act_item block)
    # Find the body section
    body_start = example.find('"body"')
    assert body_start != -1, "_FOREACH_EXAMPLE has no 'body' key"
    body_section = example[body_start:]
    # The body's timeout_sec must not be 15
    assert '"timeout_sec": 15' not in body_section and '"timeout_sec":15' not in body_section, (
        "body template still uses timeout_sec=15; expected 45"
    )
