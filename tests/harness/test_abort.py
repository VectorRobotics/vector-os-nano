"""Tests for the global abort signal (Wave 2).

Covers:
- abort module API (request, clear, is_requested, wait_or_abort)
- P0 stop bypass in VectorEngine
- GoalExecutor abort check
- VGGHarness abort check
- StopSkill triggers abort
"""
from __future__ import annotations

import threading
import time

import pytest


# ---------------------------------------------------------------------------
# 1. abort module API
# ---------------------------------------------------------------------------

class TestAbortModule:
    def test_initial_state_not_aborted(self):
        from vector_os_nano.vcli.cognitive.abort import clear_abort, is_abort_requested
        clear_abort()
        assert not is_abort_requested()

    def test_request_abort_sets_flag(self):
        from vector_os_nano.vcli.cognitive.abort import clear_abort, request_abort, is_abort_requested
        clear_abort()
        request_abort()
        assert is_abort_requested()

    def test_clear_abort_resets_flag(self):
        from vector_os_nano.vcli.cognitive.abort import clear_abort, request_abort, is_abort_requested
        request_abort()
        clear_abort()
        assert not is_abort_requested()

    def test_wait_or_abort_returns_true_when_aborted(self):
        from vector_os_nano.vcli.cognitive.abort import clear_abort, request_abort, wait_or_abort
        clear_abort()
        request_abort()
        result = wait_or_abort(5.0)
        assert result is True  # True = abort requested

    def test_wait_or_abort_returns_false_on_timeout(self):
        from vector_os_nano.vcli.cognitive.abort import clear_abort, wait_or_abort
        clear_abort()
        start = time.monotonic()
        result = wait_or_abort(0.05)
        elapsed = time.monotonic() - start
        assert result is False  # False = timed out normally
        assert elapsed < 0.5

    def test_wait_or_abort_wakes_early_on_abort(self):
        from vector_os_nano.vcli.cognitive.abort import clear_abort, request_abort, wait_or_abort
        clear_abort()

        def _abort_later():
            time.sleep(0.05)
            request_abort()

        t = threading.Thread(target=_abort_later)
        t.start()
        start = time.monotonic()
        result = wait_or_abort(5.0)
        elapsed = time.monotonic() - start
        t.join()
        assert result is True
        assert elapsed < 1.0  # woke up well before 5s

    def test_thread_safety(self):
        from vector_os_nano.vcli.cognitive.abort import clear_abort, request_abort, is_abort_requested
        clear_abort()
        errors = []

        def _reader():
            for _ in range(100):
                is_abort_requested()

        def _writer():
            for _ in range(50):
                request_abort()
                clear_abort()

        threads = [threading.Thread(target=_reader) for _ in range(4)]
        threads += [threading.Thread(target=_writer) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # No crash = pass
        clear_abort()


# ---------------------------------------------------------------------------
# 2. P0 stop bypass
# ---------------------------------------------------------------------------

class TestP0StopBypass:
    def test_stop_returns_immediately(self):
        from unittest.mock import MagicMock
        from vector_os_nano.vcli.engine import VectorEngine

        backend = MagicMock()
        engine = VectorEngine(backend=backend)
        session = MagicMock()
        session.append_user = MagicMock()
        session.append_assistant = MagicMock()

        start = time.monotonic()
        result = engine.run_turn("stop", session)
        elapsed = time.monotonic() - start

        assert result.text == "Stopped."
        assert elapsed < 0.5  # must be fast, no LLM call
        backend.call.assert_not_called()  # LLM never called

    def test_chinese_stop_bypasses(self):
        from unittest.mock import MagicMock
        from vector_os_nano.vcli.engine import VectorEngine

        backend = MagicMock()
        engine = VectorEngine(backend=backend)
        session = MagicMock()
        session.append_user = MagicMock()
        session.append_assistant = MagicMock()

        for word in ["停", "停下", "halt", "freeze", "别动", "停止"]:
            result = engine.run_turn(word, session)
            assert result.text == "Stopped.", f"Failed for: {word}"
            backend.call.assert_not_called()

    def test_non_stop_goes_to_llm(self):
        from unittest.mock import MagicMock
        from vector_os_nano.vcli.engine import VectorEngine
        from vector_os_nano.vcli.session import Session

        backend = MagicMock()
        response = MagicMock()
        response.text = "Hello"
        response.tool_calls = []
        response.stop_reason = "end_turn"
        response.usage = MagicMock()
        response.usage.input_tokens = 0
        response.usage.output_tokens = 0
        response.usage.cache_read_tokens = 0
        response.usage.cache_creation_tokens = 0
        backend.call.return_value = response

        engine = VectorEngine(backend=backend)
        session = MagicMock(spec=Session)
        session.append_user = MagicMock()
        session.append_assistant = MagicMock()
        session.append_tool_results = MagicMock()
        session.to_messages = MagicMock(return_value=[])
        session.add_usage = MagicMock()

        engine.run_turn("hello", session)
        backend.call.assert_called_once()  # LLM was called


# ---------------------------------------------------------------------------
# 3. GoalExecutor respects abort
# ---------------------------------------------------------------------------

class TestGoalExecutorAbort:
    def test_aborted_before_first_step(self):
        from unittest.mock import MagicMock
        from vector_os_nano.vcli.cognitive.abort import clear_abort, request_abort
        from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
        from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal

        clear_abort()
        request_abort()

        executor = GoalExecutor(
            strategy_selector=MagicMock(),
            verifier=MagicMock(),
        )
        tree = GoalTree(
            goal="test",
            sub_goals=(
                SubGoal(name="s1", description="step 1", verify="True"),
                SubGoal(name="s2", description="step 2", verify="True"),
            ),
        )
        trace = executor.execute(tree)
        assert not trace.success
        assert len(trace.steps) == 1
        assert trace.steps[0].error == "aborted"
        clear_abort()

    def test_not_aborted_runs_normally(self):
        from unittest.mock import MagicMock
        from vector_os_nano.vcli.cognitive.abort import clear_abort
        from vector_os_nano.vcli.cognitive.goal_executor import GoalExecutor
        from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal

        clear_abort()

        selector = MagicMock()
        result_mock = MagicMock()
        result_mock.executor_type = "skill"
        result_mock.name = "test_skill"
        result_mock.params = {}
        selector.select.return_value = result_mock

        verifier = MagicMock()
        verifier.verify.return_value = True

        skill_mock = MagicMock()
        skill_result = MagicMock()
        skill_result.success = True
        skill_result.error_message = ""
        skill_result.result_data = {}
        skill_mock.execute.return_value = skill_result

        registry = MagicMock()
        registry.get.return_value = skill_mock

        context_mock = MagicMock()

        executor = GoalExecutor(
            strategy_selector=selector,
            verifier=verifier,
            skill_registry=registry,
            build_context=lambda: context_mock,
        )
        tree = GoalTree(
            goal="test",
            sub_goals=(
                SubGoal(name="s1", description="step 1", verify="True"),
            ),
        )
        trace = executor.execute(tree)
        assert trace.success
        clear_abort()


# ---------------------------------------------------------------------------
# 4. VGGHarness respects abort
# ---------------------------------------------------------------------------

class TestVGGHarnessAbort:
    def test_abort_skips_execution(self):
        from unittest.mock import MagicMock
        from vector_os_nano.vcli.cognitive.abort import clear_abort, request_abort
        from vector_os_nano.vcli.cognitive.vgg_harness import VGGHarness
        from vector_os_nano.vcli.cognitive.types import GoalTree, SubGoal

        clear_abort()
        request_abort()

        harness = VGGHarness(
            decomposer=MagicMock(),
            executor=MagicMock(),
        )
        tree = GoalTree(
            goal="test",
            sub_goals=(SubGoal(name="s1", description="step 1", verify="True"),),
        )
        trace = harness.run("test", "", goal_tree=tree)
        assert not trace.success
        clear_abort()


# ---------------------------------------------------------------------------
# 5. StopSkill triggers abort
# ---------------------------------------------------------------------------

class TestStopSkillAbort:
    def test_stop_skill_calls_request_abort(self):
        from unittest.mock import MagicMock, patch
        from vector_os_nano.vcli.cognitive.abort import clear_abort, is_abort_requested
        from vector_os_nano.skills.go2.stop import StopSkill
        from vector_os_nano.core.skill import SkillContext

        clear_abort()
        base = MagicMock()
        ctx = SkillContext(bases={"go2": base})
        skill = StopSkill()
        skill.execute({}, ctx)
        assert is_abort_requested()
        clear_abort()
