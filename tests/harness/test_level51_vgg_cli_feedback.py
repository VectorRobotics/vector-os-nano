"""VGG CLI feedback tests — Level 51.

Covers:
- Part A: is_complex() expansion — multi-action verb detection
- Part B: vgg_decompose() / vgg_execute() split on VectorEngine
- Part C: step-callback display format

TDD: tests written before implementation.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from vector_os_nano.vcli.intent_router import IntentRouter
from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_backend() -> MagicMock:
    from vector_os_nano.vcli.backends.types import LLMResponse
    from vector_os_nano.vcli.session import TokenUsage

    backend = MagicMock()
    backend.call.return_value = LLMResponse(
        text="ok",
        tool_calls=[],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )
    return backend


def _make_goal_tree(n: int = 2) -> GoalTree:
    sub_goals = tuple(
        SubGoal(
            name=f"step_{i}",
            description=f"do step {i}",
            verify="True",
            strategy="navigate_to_room",
        )
        for i in range(1, n + 1)
    )
    return GoalTree(goal="test goal", sub_goals=sub_goals)


def _make_trace(goal_tree: GoalTree, success: bool = True) -> ExecutionTrace:
    steps = tuple(
        StepRecord(
            sub_goal_name=sg.name,
            strategy="navigate_to_room",
            success=success,
            verify_result=success,
            duration_sec=0.1,
        )
        for sg in goal_tree.sub_goals
    )
    return ExecutionTrace(
        goal_tree=goal_tree,
        steps=steps,
        success=success,
        total_duration_sec=0.2,
    )


def _make_engine_with_vgg_mocked(
    complex_response: bool = True,
) -> Any:
    """Return a VectorEngine with VGG internals fully mocked."""
    from vector_os_nano.vcli.engine import VectorEngine

    backend = _make_mock_backend()
    router = IntentRouter()
    engine = VectorEngine(backend=backend, intent_router=router)

    goal_tree = _make_goal_tree(2)
    trace = _make_trace(goal_tree)

    mock_decomposer = MagicMock()
    mock_decomposer.decompose.return_value = goal_tree

    mock_executor = MagicMock()
    mock_executor.execute.return_value = trace

    engine._vgg_enabled = True
    engine._goal_decomposer = mock_decomposer
    engine._goal_executor = mock_executor

    return engine, goal_tree, trace


# ===========================================================================
# Part A — is_complex() multi-action detection
# ===========================================================================


class TestIsComplexMultiAction:
    """is_complex() must detect 2+ action verbs as complex."""

    def setup_method(self) -> None:
        self.router = IntentRouter()

    # ---- Chinese comma-separated actions ----

    def test_comma_separated_chinese_complex(self) -> None:
        """'结束探索，开始巡逻' — Chinese comma between two actions → complex."""
        assert self.router.is_complex("结束探索，开始巡逻") is True

    def test_comma_separated_english_complex(self) -> None:
        """'stop exploring, start patrol' — English comma between two actions → complex."""
        assert self.router.is_complex("stop exploring, start patrol") is True

    # ---- Two action verbs without comma ----

    def test_two_action_verbs_no_comma_complex(self) -> None:
        """'停止探索开始巡逻' — no comma, but two distinct action verbs → complex."""
        assert self.router.is_complex("停止探索开始巡逻") is True

    def test_go_and_look_complex(self) -> None:
        """'去厨房看一下' — 去 + 看, two action verbs → complex."""
        assert self.router.is_complex("去厨房看一下") is True

    def test_navigate_and_check_complex(self) -> None:
        """'navigate to kitchen and check for cups' — navigate + check → complex."""
        assert self.router.is_complex("navigate to kitchen and check for cups") is True

    def test_find_and_pick_complex(self) -> None:
        """'找到杯子然后拿起来' uses 找 + 拿, but also 然后 so would already be complex.
        Test directly for multi-verb: '找杯子拿起来'."""
        assert self.router.is_complex("找杯子拿起来") is True

    def test_stop_and_start_english_complex(self) -> None:
        """'stop and start' — two action verbs → complex."""
        assert self.router.is_complex("stop and start") is True

    def test_patrol_and_scan_complex(self) -> None:
        """'patrol the room and scan each area' — patrol + scan → complex."""
        assert self.router.is_complex("patrol the room and scan each area") is True

    # ---- Single action verb — must remain NOT complex ----

    def test_single_action_stand_not_complex(self) -> None:
        """'站起来' — only 站, single action → not complex."""
        assert self.router.is_complex("站起来") is False

    def test_single_action_go_kitchen_not_complex(self) -> None:
        """'去厨房' — only 去, single action → not complex (regression guard)."""
        assert self.router.is_complex("去厨房") is False

    def test_single_action_patrol_not_complex(self) -> None:
        """'开始巡逻' — only 开始 is the trigger; 巡逻 is not in action verbs set → not complex."""
        assert self.router.is_complex("开始巡逻") is False

    def test_single_action_explore_not_complex(self) -> None:
        """'探索房间' — only 探索, single action → not complex."""
        assert self.router.is_complex("探索房间") is False

    def test_single_navigate_english_not_complex(self) -> None:
        """'navigate to the kitchen' — only navigate → not complex."""
        assert self.router.is_complex("navigate to the kitchen") is False

    # ---- Regression: existing tests must still pass ----

    def test_regression_look_look_has_no_complex(self) -> None:
        """'看一看' — same verb repeated, only counts as one → not complex."""
        assert self.router.is_complex("看一看") is False

    def test_regression_sequential_still_works(self) -> None:
        """'探索然后去厨房' — 然后 keyword still triggers complex."""
        assert self.router.is_complex("探索然后去厨房") is True

    def test_regression_conditional_still_works(self) -> None:
        """'如果有杯子' — 如果 keyword still triggers complex."""
        assert self.router.is_complex("如果有杯子拿起来") is True

    def test_regression_scope_still_works(self) -> None:
        """'检查所有房间' — 所有 keyword still triggers complex."""
        assert self.router.is_complex("检查所有房间") is True


# ===========================================================================
# Part B — vgg_decompose / vgg_execute split
# ===========================================================================


class TestVggDecomposeExecuteSplit:
    """VectorEngine must support split decompose / execute phases."""

    def test_vgg_decompose_returns_goal_tree_for_complex(self) -> None:
        """vgg_decompose on a complex message returns a GoalTree."""
        from vector_os_nano.vcli.engine import VectorEngine

        backend = _make_mock_backend()
        router = IntentRouter()
        engine = VectorEngine(backend=backend, intent_router=router)

        goal_tree = _make_goal_tree(2)
        mock_decomposer = MagicMock()
        mock_decomposer.decompose.return_value = goal_tree

        engine._vgg_enabled = True
        engine._vgg_agent = MagicMock()
        engine._vgg_agent._base = MagicMock()
        engine._vgg_agent._skill_registry = None
        engine._goal_decomposer = mock_decomposer
        engine._goal_executor = MagicMock()

        # Use a message that triggers existing rules (看看有没有)
        result = engine.vgg_decompose("去厨房看看有没有杯子")
        assert result is not None
        assert isinstance(result, GoalTree)
        mock_decomposer.decompose.assert_called_once()

    def test_vgg_decompose_returns_none_for_conversation(self) -> None:
        """vgg_decompose on pure conversation returns None."""
        from vector_os_nano.vcli.engine import VectorEngine

        backend = _make_mock_backend()
        router = IntentRouter()
        engine = VectorEngine(backend=backend, intent_router=router)

        mock_decomposer = MagicMock()
        engine._vgg_enabled = True
        engine._goal_decomposer = mock_decomposer
        engine._goal_executor = MagicMock()

        result = engine.vgg_decompose("你好")
        assert result is None

    def test_vgg_decompose_returns_none_when_disabled(self) -> None:
        """vgg_decompose returns None when VGG is disabled."""
        from vector_os_nano.vcli.engine import VectorEngine

        backend = _make_mock_backend()
        engine = VectorEngine(backend=backend)
        # _vgg_enabled defaults to False

        result = engine.vgg_decompose("去厨房看看有没有杯子")
        assert result is None

    def test_vgg_decompose_returns_none_without_router(self) -> None:
        """vgg_decompose returns None when no intent_router is set."""
        from vector_os_nano.vcli.engine import VectorEngine

        backend = _make_mock_backend()
        engine = VectorEngine(backend=backend)  # no router
        engine._vgg_enabled = True
        engine._goal_decomposer = MagicMock()

        result = engine.vgg_decompose("去厨房看看有没有杯子")
        assert result is None

    def test_vgg_decompose_exception_returns_none(self) -> None:
        """vgg_decompose returns None when decomposer raises an exception."""
        from vector_os_nano.vcli.engine import VectorEngine

        backend = _make_mock_backend()
        router = IntentRouter()
        engine = VectorEngine(backend=backend, intent_router=router)

        mock_decomposer = MagicMock()
        mock_decomposer.decompose.side_effect = RuntimeError("LLM unavailable")
        engine._vgg_enabled = True
        engine._goal_decomposer = mock_decomposer
        engine._goal_executor = MagicMock()

        result = engine.vgg_decompose("去厨房看看有没有杯子")
        assert result is None

    def test_vgg_execute_returns_execution_trace(self) -> None:
        """vgg_execute on a GoalTree returns an ExecutionTrace."""
        from vector_os_nano.vcli.engine import VectorEngine

        backend = _make_mock_backend()
        engine = VectorEngine(backend=backend)

        goal_tree = _make_goal_tree(2)
        trace = _make_trace(goal_tree)

        mock_executor = MagicMock()
        mock_executor.execute.return_value = trace
        engine._goal_executor = mock_executor
        engine._vgg_enabled = True

        result = engine.vgg_execute(goal_tree)
        assert isinstance(result, ExecutionTrace)
        assert result.success is True
        mock_executor.execute.assert_called_once_with(goal_tree, on_step=engine._on_vgg_step)

    def test_vgg_execute_passes_on_step_callback(self) -> None:
        """vgg_execute passes _on_vgg_step to the executor."""
        from vector_os_nano.vcli.engine import VectorEngine

        backend = _make_mock_backend()
        engine = VectorEngine(backend=backend)

        goal_tree = _make_goal_tree(1)
        trace = _make_trace(goal_tree)

        mock_executor = MagicMock()
        mock_executor.execute.return_value = trace
        engine._goal_executor = mock_executor
        engine._vgg_enabled = True

        engine.vgg_execute(goal_tree)

        call_kwargs = mock_executor.execute.call_args
        # The on_step kwarg should be engine._on_vgg_step
        assert call_kwargs.kwargs.get("on_step") == engine._on_vgg_step

    def test_decompose_then_execute_full_flow(self) -> None:
        """Full flow: decompose → execute returns consistent GoalTree and trace."""
        from vector_os_nano.vcli.engine import VectorEngine

        backend = _make_mock_backend()
        router = IntentRouter()
        engine = VectorEngine(backend=backend, intent_router=router)

        goal_tree = _make_goal_tree(3)
        trace = _make_trace(goal_tree, success=True)

        mock_decomposer = MagicMock()
        mock_decomposer.decompose.return_value = goal_tree
        mock_executor = MagicMock()
        mock_executor.execute.return_value = trace

        engine._vgg_enabled = True
        engine._vgg_agent = MagicMock()
        engine._vgg_agent._base = MagicMock()
        engine._vgg_agent._skill_registry = None
        engine._goal_decomposer = mock_decomposer
        engine._goal_executor = mock_executor

        # Use multi-verb message to trigger new multi-action rule
        result_tree = engine.vgg_decompose("结束探索，开始巡逻")
        assert result_tree is not None

        result_trace = engine.vgg_execute(result_tree)
        assert isinstance(result_trace, ExecutionTrace)
        assert result_trace.goal_tree is goal_tree
        assert len(result_trace.steps) == 3
