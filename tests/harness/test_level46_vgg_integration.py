"""VGG Integration tests — AC-31 to AC-35.

Tests:
  AC-31  "站起来" → is_complex() False
  AC-32  "去厨房" → is_complex() False
  AC-33  "去厨房看看有没有杯子" → is_complex() True  (看看有没有)
  AC-34  "检查所有房间" → is_complex() True  (所有)
  AC-35  try_vgg with mocked pipeline returns ExecutionTrace for complex task
  +10 additional edge-case / wiring tests
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vector_os_nano.vcli.cognitive.types import (
    ExecutionTrace,
    GoalTree,
    StepRecord,
    SubGoal,
)
from vector_os_nano.vcli.intent_router import IntentRouter


# ---------------------------------------------------------------------------
# Helpers — build a minimal VectorEngine with VGG wired
# ---------------------------------------------------------------------------


def _make_mock_backend() -> MagicMock:
    """Return a mock LLMBackend that returns a text-only response."""
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


def _make_engine_with_vgg(vgg_enabled: bool = True) -> Any:
    """Return a VectorEngine with VGG initialised (or disabled)."""
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.intent_router import IntentRouter

    backend = _make_mock_backend()
    router = IntentRouter()
    engine = VectorEngine(backend=backend, intent_router=router)

    if vgg_enabled:
        engine.init_vgg(backend=backend)
    # If vgg_enabled=False, init_vgg is never called → _vgg_enabled stays False

    return engine


# ---------------------------------------------------------------------------
# AC-31  Simple single-word command → not complex
# ---------------------------------------------------------------------------


class TestIsComplex:
    def setup_method(self) -> None:
        self.router = IntentRouter()

    # AC-31
    def test_simple_stand_not_complex(self) -> None:
        assert self.router.is_complex("站起来") is False

    # AC-32
    def test_simple_navigate_not_complex(self) -> None:
        assert self.router.is_complex("去厨房") is False

    # AC-33
    def test_multi_step_task_complex(self) -> None:
        assert self.router.is_complex("去厨房看看有没有杯子") is True

    # AC-34
    def test_check_all_rooms_complex(self) -> None:
        assert self.router.is_complex("检查所有房间") is True

    # "and then" sequential keyword
    def test_and_then_complex(self) -> None:
        assert self.router.is_complex("if there's a cup, bring it") is True

    # greet / very short → not complex
    def test_hello_not_complex(self) -> None:
        assert self.router.is_complex("hello") is False

    # empty string → not complex
    def test_empty_not_complex(self) -> None:
        assert self.router.is_complex("") is False

    # sequential keyword "然后"
    def test_then_chinese_complex(self) -> None:
        assert self.router.is_complex("探索然后去厨房") is True

    # "check if" perception+judgment
    def test_check_if_complex(self) -> None:
        assert self.router.is_complex("check if the door is open") is True

    # single word (< 5 chars) → always simple
    def test_single_short_word_not_complex(self) -> None:
        assert self.router.is_complex("停") is False

    # "every" scope keyword
    def test_every_room_complex(self) -> None:
        assert self.router.is_complex("go to every room") is True

    # "see if" perception+judgment
    def test_see_if_complex(self) -> None:
        assert self.router.is_complex("see if there's someone in the room") is True

    # "each" scope keyword
    def test_each_complex(self) -> None:
        assert self.router.is_complex("scan each corridor") is True

    # 同时 conjunction
    def test_simultaneous_complex(self) -> None:
        assert self.router.is_complex("走过去同时扫描房间") is True


# ---------------------------------------------------------------------------
# AC-35  VGG pipeline integration
# ---------------------------------------------------------------------------


class TestTryVgg:

    # AC-35: try_vgg returns ExecutionTrace for a complex task (mocked LLM + hardware)
    def test_vgg_pipeline_produces_trace(self) -> None:
        from vector_os_nano.vcli.engine import VectorEngine
        from vector_os_nano.vcli.intent_router import IntentRouter

        backend = _make_mock_backend()
        router = IntentRouter()
        engine = VectorEngine(backend=backend, intent_router=router)

        # Build a mock decomposer that returns a simple GoalTree
        sub_goal = SubGoal(
            name="step_1",
            description="test step",
            verify="world_stats() is not None",
        )
        goal_tree = GoalTree(
            goal="去厨房看看有没有杯子",
            sub_goals=(sub_goal,),
        )
        mock_decomposer = MagicMock()
        mock_decomposer.decompose.return_value = goal_tree

        # Build a mock verifier (always True)
        mock_verifier = MagicMock()
        mock_verifier.verify.return_value = True

        # Build a mock selector
        mock_selector = MagicMock()
        from vector_os_nano.vcli.cognitive.strategy_selector import StrategyResult
        mock_selector.select.return_value = StrategyResult("fallback", "unmatched", {})

        # Build a mock executor that returns a real ExecutionTrace
        step_record = StepRecord(
            sub_goal_name="step_1",
            strategy="unmatched",
            success=True,
            verify_result=True,
            duration_sec=0.01,
        )
        trace = ExecutionTrace(
            goal_tree=goal_tree,
            steps=(step_record,),
            success=True,
            total_duration_sec=0.01,
        )
        mock_executor = MagicMock()
        mock_executor.execute.return_value = trace

        # Inject mocked components directly
        engine._vgg_enabled = True
        engine._goal_decomposer = mock_decomposer
        engine._goal_executor = mock_executor

        result = engine.try_vgg("去厨房看看有没有杯子")

        assert result is not None
        assert isinstance(result, ExecutionTrace)
        assert result.success is True
        mock_decomposer.decompose.assert_called_once()
        mock_executor.execute.assert_called_once()

    def test_try_vgg_disabled_returns_none(self) -> None:
        """When VGG is disabled, try_vgg always returns None."""
        from vector_os_nano.vcli.engine import VectorEngine

        backend = _make_mock_backend()
        engine = VectorEngine(backend=backend)
        # Do NOT call init_vgg → _vgg_enabled = False

        result = engine.try_vgg("去厨房看看有没有杯子")
        assert result is None

    def test_try_vgg_conversation_returns_none(self) -> None:
        """Pure conversation (no action verb) bypasses VGG."""
        from vector_os_nano.vcli.engine import VectorEngine
        from vector_os_nano.vcli.intent_router import IntentRouter

        backend = _make_mock_backend()
        router = IntentRouter()
        engine = VectorEngine(backend=backend, intent_router=router)
        engine._vgg_enabled = True
        engine._goal_decomposer = MagicMock()
        engine._goal_executor = MagicMock()

        result = engine.try_vgg("你好，今天天气怎么样")
        assert result is None

    def test_try_vgg_no_router_returns_none(self) -> None:
        """try_vgg with no intent_router returns None (cannot determine complexity)."""
        from vector_os_nano.vcli.engine import VectorEngine

        backend = _make_mock_backend()
        engine = VectorEngine(backend=backend)  # no intent_router
        engine._vgg_enabled = True
        engine._goal_decomposer = MagicMock()
        engine._goal_executor = MagicMock()

        result = engine.try_vgg("去厨房看看有没有杯子")
        # No router → cannot determine complexity → returns None
        assert result is None

    def test_init_vgg_sets_vgg_enabled(self) -> None:
        """init_vgg() sets _vgg_enabled = True when backend is provided."""
        from vector_os_nano.vcli.engine import VectorEngine

        backend = _make_mock_backend()
        engine = VectorEngine(backend=backend)
        assert engine._vgg_enabled is False

        engine.init_vgg(backend=backend)
        assert engine._vgg_enabled is True

    def test_init_vgg_failure_leaves_engine_functional(self) -> None:
        """If init_vgg fails, _vgg_enabled stays False and engine still runs."""
        from vector_os_nano.vcli.engine import VectorEngine
        from vector_os_nano.vcli.session import Session

        backend = _make_mock_backend()
        engine = VectorEngine(backend=backend)

        # Force GoalDecomposer import to raise (simulate missing dependency)
        with patch(
            "vector_os_nano.vcli.engine.GoalDecomposer",
            side_effect=RuntimeError("unavailable"),
        ):
            engine.init_vgg(backend=backend)

        assert engine._vgg_enabled is False

    def test_run_turn_still_works_simple_task(self) -> None:
        """Engine run_turn unaffected by VGG wiring for simple tasks.

        Uses no intent_router to keep the test focused on the VGG path:
        the engine must still process the turn via tool_use and return a
        TurnResult even when VGG is initialised.
        """
        from pathlib import Path

        from vector_os_nano.vcli.engine import VectorEngine
        from vector_os_nano.vcli.session import Session

        backend = _make_mock_backend()
        # No intent_router — keeps engine in plain tool_use path
        engine = VectorEngine(backend=backend)
        engine.init_vgg(backend=backend)  # VGG enabled but simple task bypasses it

        session = Session(
            session_id="t46-test",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            path=Path("/tmp/t46_session.jsonl"),
        )
        result = engine.run_turn("站起来", session)
        assert result.text == "ok"
        assert result.stop_reason == "end_turn"
