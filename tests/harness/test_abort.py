# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tests for the global abort signal (Wave 2).

Covers:
- abort module API (request, clear, is_requested, wait_or_abort)
- P0 stop bypass in VectorEngine (the REPL and -p call it before the producer)
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

def _engine_with_stop_skill(backend):
    """An engine whose agent registers the real StopSkill: the P0 words are its aliases."""
    from types import SimpleNamespace

    from vector_os_nano.core.skill import SkillRegistry
    from vector_os_nano.skills.go2.stop import StopSkill
    from vector_os_nano.vcli.engine import VectorEngine

    reg = SkillRegistry()
    reg.register(StopSkill())
    engine = VectorEngine(backend=backend)
    engine._vgg_agent = SimpleNamespace(_skill_registry=reg)
    return engine


class TestP0StopBypass:
    def test_stop_returns_immediately(self):
        from unittest.mock import MagicMock

        backend = MagicMock()
        engine = _engine_with_stop_skill(backend)
        session = MagicMock()
        session.append_user = MagicMock()
        session.append_assistant = MagicMock()

        start = time.monotonic()
        assert engine.is_stop("stop")
        engine._emergency_stop("stop", session)
        elapsed = time.monotonic() - start

        session.append_assistant.assert_called_once_with("Stopped.", None)
        assert elapsed < 0.5  # must be fast, no LLM call
        backend.call.assert_not_called()  # LLM never called

    def test_chinese_stop_bypasses(self):
        from unittest.mock import MagicMock

        engine = _engine_with_stop_skill(MagicMock())
        for word in ["停", "停下", "halt", "freeze", "别动", "停止", "停！", "Stop!", "停下。"]:
            assert engine.is_stop(word), f"Failed for: {word}"
        assert not engine.is_stop("停？")  # a question about stopping is not a stop
        assert not engine.is_stop("hello")

    def test_stop_words_are_the_registered_stop_skills_aliases(self):
        """Single source (producers:stop-not-p0): no hand-written kernel list."""
        from unittest.mock import MagicMock

        from vector_os_nano.skills.go2.stop import StopSkill
        from vector_os_nano.vcli.engine import VectorEngine

        engine = _engine_with_stop_skill(MagicMock())
        assert engine.stop_words() == frozenset(
            w.lower() for w in ["stop", *StopSkill.__skill_aliases__]
        )
        assert VectorEngine(backend=MagicMock()).stop_words() == frozenset()  # no robot


# ---------------------------------------------------------------------------
# 3. StopSkill triggers abort
# ---------------------------------------------------------------------------

class TestStopSkillAbort:
    def test_stop_skill_stops_every_other_run_but_not_its_own(self):
        """A stop STEP ("walk two steps, then stop") stops background runs (a plan, a
        navigate), never the run it is a step of."""
        import threading
        from unittest.mock import MagicMock
        from vector_os_nano.vcli.cognitive.abort import clear_abort, is_abort_requested
        from vector_os_nano.skills.go2.stop import StopSkill
        from vector_os_nano.core.skill import SkillContext

        other: list[bool] = []
        ready, check = threading.Event(), threading.Event()

        def background_run() -> None:
            clear_abort()
            ready.set()
            check.wait(5)
            other.append(is_abort_requested())

        t = threading.Thread(target=background_run)
        t.start()
        ready.wait(5)
        clear_abort()  # the run this stop is a step of
        StopSkill().execute({}, SkillContext(bases={"go2": MagicMock()}))
        check.set()
        t.join(5)
        assert other == [True]
        assert not is_abort_requested()
