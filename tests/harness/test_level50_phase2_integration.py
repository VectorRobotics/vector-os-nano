"""Phase 2 Wave 2 integration tests — AC-13 to AC-15 + stats recording + template matching.

Covers:
- AC-13: Stats-driven selection overrides rule-based when data is sufficient.
- AC-14: Stats with < 3 attempts falls back to rule-based selection.
- AC-15: code_as_policy can be selected as an explicit strategy.
- GoalExecutor records step results to StrategyStats.
- GoalExecutor calls stats.save() after execution.
- GoalDecomposer uses template when available (no LLM call).
- GoalDecomposer falls back to LLM when no template matches.
- End-to-end: decompose → execute → stats recorded.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vector_os_nano.vcli.cognitive.strategy_stats import StrategyStats
from vector_os_nano.vcli.cognitive.strategy_selector import StrategySelector
from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
from vector_os_nano.vcli.cognitive.goal_decomposer import GoalDecomposer
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.cognitive.template_library import TemplateLibrary
from vector_os_nano.vcli.cognitive.experience_compiler import GoalTemplate, SubGoalTemplate
from vector_os_nano.vcli.cognitive.types import SubGoal, GoalTree, StepRecord, ExecutionTrace


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_verifier(always: bool = True) -> GoalVerifier:
    """Return a GoalVerifier that always returns *always*."""
    verifier = MagicMock()
    verifier.verify.return_value = always
    return verifier


def _make_skill_registry(skill_name: str, success: bool = True) -> Any:
    """Return a mock skill registry whose .get() returns a skill that succeeds."""
    skill = MagicMock()
    skill_result = MagicMock()
    skill_result.success = success
    skill_result.error_message = ""
    skill.execute.return_value = skill_result
    registry = MagicMock()
    registry.get.return_value = skill
    return registry


def _make_goal_tree(*sub_goal_names: str) -> GoalTree:
    """Build a simple GoalTree with stub sub-goals."""
    sub_goals = tuple(
        SubGoal(
            name=name,
            description=f"do {name}",
            verify="True",
            strategy="navigate_skill",
        )
        for name in sub_goal_names
    )
    return GoalTree(goal="test_goal", sub_goals=sub_goals)


# ---------------------------------------------------------------------------
# AC-13: Stats-driven selection overrides rule-based choice
# ---------------------------------------------------------------------------

class TestStatsDrivenSelectionOverrides:
    """AC-13: With sufficient data, stats override rule-based choice."""

    def test_stats_driven_selection_overrides_rule(self) -> None:
        """AC-13: code_as_policy with 100% success rate overrides navigate rule."""
        stats = StrategyStats(persist_path=None)

        # code_as_policy: 5/5 = 100% on reach_* pattern
        for _ in range(5):
            stats.record("code_as_policy", "reach_x", True, 10.0)

        # navigate_skill: 5/6 = 83% — good but lower than code_as_policy
        for _ in range(5):
            stats.record("navigate_skill", "reach_x", True, 25.0)
        stats.record("navigate_skill", "reach_x", False, 25.0)

        selector = StrategySelector(stats=stats)
        sg = SubGoal(name="reach_kitchen", description="go to kitchen", verify="True")
        result = selector.select(sg)

        # Stats should override the navigate rule since code_as_policy has higher success
        assert result.name == "code_as_policy"
        assert result.executor_type == "code"

    def test_stats_override_requires_top_strategy_to_differ(self) -> None:
        """Stats do not override when top stats strategy matches rule result."""
        stats = StrategyStats(persist_path=None)

        # navigate has 5/5 = 100% — same as what rule-based would choose
        for _ in range(5):
            stats.record("navigate", "reach_x", True, 15.0)

        selector = StrategySelector(stats=stats)
        sg = SubGoal(name="reach_kitchen", description="go to kitchen", verify="True")
        result = selector.select(sg)

        # Result is still navigate (rule-based and stats agree)
        assert result.name == "navigate"
        assert result.executor_type == "skill"

    def test_stats_override_requires_success_rate_above_threshold(self) -> None:
        """Stats do not override when success_rate <= 0.5."""
        stats = StrategyStats(persist_path=None)

        # code_as_policy: 1/3 = 33% — below 50% threshold
        stats.record("code_as_policy", "reach_x", True, 10.0)
        stats.record("code_as_policy", "reach_x", False, 10.0)
        stats.record("code_as_policy", "reach_x", False, 10.0)

        selector = StrategySelector(stats=stats)
        sg = SubGoal(name="reach_kitchen", description="go to kitchen", verify="True")
        result = selector.select(sg)

        # Falls back to rule-based navigate
        assert result.name == "navigate"


# ---------------------------------------------------------------------------
# AC-14: Insufficient stats → rule-based fallback
# ---------------------------------------------------------------------------

class TestStatsInsufficientDataFallback:
    """AC-14: < 3 attempts → fall back to rule matching."""

    def test_stats_insufficient_data_falls_back(self) -> None:
        """AC-14: Only 1 attempt — rule-based is used."""
        stats = StrategyStats(persist_path=None)
        stats.record("code_as_policy", "reach_x", True, 10.0)  # 1 attempt only

        selector = StrategySelector(stats=stats)
        sg = SubGoal(name="reach_kitchen", description="go to kitchen", verify="True")
        result = selector.select(sg)

        # Rule-based: reach → navigate
        assert result.name == "navigate"
        assert result.executor_type == "skill"

    def test_stats_exactly_two_attempts_falls_back(self) -> None:
        """AC-14: Exactly 2 attempts — still falls back to rule-based."""
        stats = StrategyStats(persist_path=None)
        stats.record("code_as_policy", "reach_x", True, 10.0)
        stats.record("code_as_policy", "reach_x", True, 10.0)  # 2 attempts

        selector = StrategySelector(stats=stats)
        sg = SubGoal(name="reach_kitchen", description="go to kitchen", verify="True")
        result = selector.select(sg)

        assert result.name == "navigate"  # rule-based, not stats

    def test_no_stats_uses_rule(self) -> None:
        """StrategySelector without stats uses pure rule-based matching."""
        selector = StrategySelector()
        sg = SubGoal(name="reach_kitchen", description="go to kitchen", verify="True")
        result = selector.select(sg)

        assert result.name == "navigate"
        assert result.executor_type == "skill"


# ---------------------------------------------------------------------------
# AC-15: code_as_policy can be selected as strategy
# ---------------------------------------------------------------------------

class TestCodeAsPolicyStrategy:
    """AC-15: code_as_policy strategy support."""

    def test_code_as_policy_selectable_via_explicit(self) -> None:
        """AC-15: Explicit strategy='code_as_policy' → executor_type='code'."""
        selector = StrategySelector()
        sg = SubGoal(
            name="complex_task",
            description="do something complex",
            verify="True",
            strategy="code_as_policy",
        )
        result = selector.select(sg)

        assert result.executor_type == "code"
        assert result.name == "code_as_policy"

    def test_code_as_policy_params_passed_through(self) -> None:
        """code_as_policy carries strategy_params unchanged."""
        selector = StrategySelector()
        params = {"code": "print('hello')", "context": "kitchen"}
        sg = SubGoal(
            name="run_code",
            description="execute code",
            verify="True",
            strategy="code_as_policy",
            strategy_params=params,
        )
        result = selector.select(sg)

        assert result.executor_type == "code"
        assert result.params == params

    def test_code_as_policy_via_stats_override(self) -> None:
        """code_as_policy resolved when stats override picks it."""
        stats = StrategyStats(persist_path=None)
        for _ in range(5):
            stats.record("code_as_policy", "detect_x", True, 5.0)

        selector = StrategySelector(stats=stats)
        sg = SubGoal(name="detect_cup", description="find a cup", verify="True")
        result = selector.select(sg)

        assert result.name == "code_as_policy"
        assert result.executor_type == "code"


# ---------------------------------------------------------------------------
# GoalExecutor stats recording
# ---------------------------------------------------------------------------

class TestGoalExecutorStatsRecording:
    """GoalExecutor records step results to StrategyStats."""

    def _make_executor(self, stats: Any = None, success: bool = True) -> GoalExecutor:
        verifier = _make_verifier(always=success)
        selector = StrategySelector()
        registry = _make_skill_registry("navigate", success=success)
        return GoalExecutor(
            strategy_selector=selector,
            verifier=verifier,
            skill_registry=registry,
            stats=stats,
        )

    def test_executor_records_stats_after_step(self) -> None:
        """GoalExecutor records each step result to StrategyStats."""
        stats = StrategyStats(persist_path=None)
        executor = self._make_executor(stats=stats, success=True)

        tree = _make_goal_tree("reach_kitchen")
        executor.execute(tree)

        # Stats should have recorded the step
        pattern = StrategyStats.extract_pattern("reach_kitchen")
        # Some strategy was recorded
        rankings = stats.get_rankings(pattern)
        assert len(rankings) >= 1
        assert rankings[0].total_attempts >= 1

    def test_executor_records_success_true(self) -> None:
        """Stats record success=True when step succeeds."""
        stats = StrategyStats(persist_path=None)
        executor = self._make_executor(stats=stats, success=True)

        tree = _make_goal_tree("reach_bedroom")
        executor.execute(tree)

        pattern = StrategyStats.extract_pattern("reach_bedroom")
        rankings = stats.get_rankings(pattern)
        assert len(rankings) >= 1
        assert rankings[0].successes >= 1

    def test_executor_records_multiple_steps(self) -> None:
        """Stats record all steps in a multi-step GoalTree."""
        stats = StrategyStats(persist_path=None)
        executor = self._make_executor(stats=stats, success=True)

        tree = _make_goal_tree("reach_kitchen", "reach_bedroom")
        executor.execute(tree)

        # Both steps recorded
        for name in ("reach_kitchen", "reach_bedroom"):
            pattern = StrategyStats.extract_pattern(name)
            rankings = stats.get_rankings(pattern)
            assert len(rankings) >= 1, f"No stats for {name}"

    def test_executor_auto_saves_stats(self) -> None:
        """GoalExecutor calls stats.save() after execution."""
        stats = MagicMock(spec=StrategyStats)
        stats.record = MagicMock()
        stats.save = MagicMock()

        verifier = _make_verifier(always=True)
        selector = StrategySelector()
        registry = _make_skill_registry("navigate", success=True)
        executor = GoalExecutor(
            strategy_selector=selector,
            verifier=verifier,
            skill_registry=registry,
            stats=stats,
        )

        tree = _make_goal_tree("reach_kitchen")
        executor.execute(tree)

        stats.save.assert_called_once()

    def test_executor_without_stats_no_error(self) -> None:
        """GoalExecutor with stats=None runs normally."""
        executor = self._make_executor(stats=None, success=True)
        tree = _make_goal_tree("reach_kitchen")
        trace = executor.execute(tree)

        assert isinstance(trace, ExecutionTrace)
        assert trace.success is True


# ---------------------------------------------------------------------------
# GoalDecomposer template matching
# ---------------------------------------------------------------------------

class TestGoalDecomposerTemplateMatching:
    """GoalDecomposer uses template library before calling LLM."""

    def _make_navigate_template(self, room: str = "kitchen") -> GoalTemplate:
        """Build a concrete navigation GoalTemplate."""
        sgt = SubGoalTemplate(
            name_pattern=f"reach_{room}",
            description_pattern=f"Navigate to {room}",
            verify_pattern=f"nearest_room() == '{room}'",
            strategy="navigate_skill",
            timeout_sec=60.0,
        )
        return GoalTemplate(
            name=f"navigate_to_{room}",
            description=f"Go to {room}",
            parameters=(),  # concrete — no placeholders
            sub_goal_templates=(sgt,),
        )

    def _make_parameterized_template(self) -> GoalTemplate:
        """Build a parameterized GoalTemplate with a room placeholder."""
        sgt = SubGoalTemplate(
            name_pattern="reach_${room}",
            description_pattern="Navigate to ${room}",
            verify_pattern="nearest_room() == '${room}'",
            strategy="navigate_skill",
            timeout_sec=60.0,
        )
        return GoalTemplate(
            name="navigate_to_room",
            description="Go to ${room}",
            parameters=("room",),
            sub_goal_templates=(sgt,),
        )

    def test_decomposer_uses_template_when_available(self) -> None:
        """GoalDecomposer returns template match without calling LLM."""
        backend = MagicMock()

        library = TemplateLibrary(persist_path=None)
        template = self._make_navigate_template("kitchen")
        library.add(template)

        decomposer = GoalDecomposer(backend=backend, template_library=library)
        tree = decomposer.decompose("go to kitchen please", "robot is in hallway")

        # LLM backend must NOT have been called
        backend.call.assert_not_called()

        # Returned tree should contain the sub-goal from template
        assert len(tree.sub_goals) == 1
        assert "kitchen" in tree.sub_goals[0].name

    def test_decomposer_falls_back_to_llm_when_no_template(self) -> None:
        """GoalDecomposer calls LLM when no template matches."""
        backend = MagicMock()
        response = MagicMock()
        response.text = '{"goal": "test", "sub_goals": [{"name": "do_it", "description": "do something", "verify": "True"}]}'
        backend.call.return_value = response

        library = TemplateLibrary(persist_path=None)  # empty library
        decomposer = GoalDecomposer(backend=backend, template_library=library)
        tree = decomposer.decompose("make coffee", "kitchen is accessible")

        # LLM backend MUST have been called
        backend.call.assert_called_once()

    def test_decomposer_no_template_library_calls_llm(self) -> None:
        """GoalDecomposer with no template_library always calls LLM."""
        backend = MagicMock()
        response = MagicMock()
        response.text = '{"goal": "test", "sub_goals": [{"name": "do_it", "description": "do something", "verify": "True"}]}'
        backend.call.return_value = response

        decomposer = GoalDecomposer(backend=backend)  # no template_library
        tree = decomposer.decompose("go somewhere", "context")

        backend.call.assert_called_once()

    def test_decomposer_template_returns_correct_structure(self) -> None:
        """Template instantiation returns properly structured GoalTree."""
        backend = MagicMock()

        library = TemplateLibrary(persist_path=None)
        template = self._make_parameterized_template()
        library.add(template)

        decomposer = GoalDecomposer(backend=backend, template_library=library)
        # "kitchen" matches the room parameter
        tree = decomposer.decompose("navigate to kitchen", "context")

        backend.call.assert_not_called()
        assert isinstance(tree, GoalTree)
        assert len(tree.sub_goals) == 1
        sg = tree.sub_goals[0]
        assert "kitchen" in sg.name
        assert sg.strategy == "navigate_skill"


# ---------------------------------------------------------------------------
# End-to-end integration: decompose → execute → stats recorded
# ---------------------------------------------------------------------------

class TestFullPipelineIntegration:
    """End-to-end: decompose → execute → stats recorded."""

    def test_full_pipeline_decompose_execute_record(self) -> None:
        """Full pipeline produces ExecutionTrace and records stats."""
        # --- Decomposer with template ---
        library = TemplateLibrary(persist_path=None)
        sgt = SubGoalTemplate(
            name_pattern="reach_kitchen",
            description_pattern="go to kitchen",
            verify_pattern="nearest_room() == 'kitchen'",
            strategy="navigate_skill",
            timeout_sec=30.0,
        )
        template = GoalTemplate(
            name="go_to_kitchen",
            description="Go to kitchen",
            parameters=(),
            sub_goal_templates=(sgt,),
        )
        library.add(template)

        backend = MagicMock()
        decomposer = GoalDecomposer(backend=backend, template_library=library)

        # --- Executor with stats ---
        stats = StrategyStats(persist_path=None)
        verifier = _make_verifier(always=True)
        selector = StrategySelector(stats=stats)
        registry = _make_skill_registry("navigate", success=True)
        executor = GoalExecutor(
            strategy_selector=selector,
            verifier=verifier,
            skill_registry=registry,
            stats=stats,
        )

        # --- Run pipeline ---
        tree = decomposer.decompose("go to the kitchen", "robot is in hallway")
        trace = executor.execute(tree)

        # LLM not called
        backend.call.assert_not_called()

        # Trace produced
        assert isinstance(trace, ExecutionTrace)
        assert trace.success is True

        # Stats recorded
        pattern = StrategyStats.extract_pattern("reach_kitchen")
        rankings = stats.get_rankings(pattern)
        assert len(rankings) >= 1
        assert rankings[0].total_attempts >= 1
