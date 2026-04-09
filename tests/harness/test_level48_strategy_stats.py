"""Level 48 harness tests — StrategyStats persistent strategy success rate tracking.

TDD: Tests written before implementation (RED → GREEN → REFACTOR).

Acceptance Criteria:
  AC-9:  Record 10 navigate_skill/reach_* (8 success) → success_rate = 0.8
  AC-10: get_rankings("reach_*") sorted descending by success_rate
  AC-11: save + load roundtrip preserves all data
  AC-12: No history → get_stats returns None
  + additional edge-case tests
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from vector_os_nano.vcli.cognitive import StrategyRecord, StrategyStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stats(tmp_path: str | None = None) -> tuple[StrategyStats, str]:
    """Return (stats, path) using a temp file."""
    if tmp_path is None:
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, "stats.json")
    return StrategyStats(persist_path=tmp_path), tmp_path


# ---------------------------------------------------------------------------
# extract_pattern
# ---------------------------------------------------------------------------

class TestExtractPattern:
    """Unit tests for StrategyStats.extract_pattern static method."""

    def test_reach_kitchen_yields_reach_star(self) -> None:
        assert StrategyStats.extract_pattern("reach_kitchen") == "reach_*"

    def test_detect_cup_yields_detect_star(self) -> None:
        assert StrategyStats.extract_pattern("detect_cup") == "detect_*"

    def test_observe_living_room_table_yields_observe_star(self) -> None:
        assert StrategyStats.extract_pattern("observe_living_room_table") == "observe_*"

    def test_no_underscore_returns_as_is(self) -> None:
        """Single word with no underscore → keep verbatim."""
        assert StrategyStats.extract_pattern("stand") == "stand"

    def test_leading_underscore_returns_as_is(self) -> None:
        """Leading underscore (idx==0) → keep verbatim."""
        assert StrategyStats.extract_pattern("_private") == "_private"


# ---------------------------------------------------------------------------
# AC-12: No history → get_stats returns None
# ---------------------------------------------------------------------------

class TestNoHistory:
    def test_get_stats_returns_none_for_unknown_strategy(self) -> None:
        stats, _ = _make_stats()
        assert stats.get_stats("foo", "bar_*") is None

    def test_get_rankings_returns_empty_list_for_unknown_pattern(self) -> None:
        stats, _ = _make_stats()
        assert stats.get_rankings("nonexistent_*") == []


# ---------------------------------------------------------------------------
# AC-9: Record 10 executions — 8 success → success_rate = 0.8
# ---------------------------------------------------------------------------

class TestRecordAndStats:
    def test_ac9_success_rate_eight_of_ten(self) -> None:
        stats, _ = _make_stats()
        for i in range(10):
            stats.record("navigate_skill", "reach_kitchen", success=(i < 8), duration_sec=25.0)
        rec = stats.get_stats("navigate_skill", "reach_*")
        assert rec is not None
        assert rec.total_attempts == 10
        assert rec.successes == 8
        assert rec.success_rate == pytest.approx(0.8)

    def test_record_uses_pattern_not_raw_name(self) -> None:
        """record("…", "reach_kitchen") must bucket under reach_*, not reach_kitchen."""
        stats, _ = _make_stats()
        stats.record("navigate_skill", "reach_kitchen", success=True, duration_sec=10.0)
        stats.record("navigate_skill", "reach_office", success=True, duration_sec=10.0)
        rec = stats.get_stats("navigate_skill", "reach_*")
        assert rec is not None
        assert rec.total_attempts == 2

    def test_all_failures_success_rate_zero(self) -> None:
        stats, _ = _make_stats()
        for _ in range(5):
            stats.record("fallback_skill", "reach_door", success=False, duration_sec=5.0)
        rec = stats.get_stats("fallback_skill", "reach_*")
        assert rec is not None
        assert rec.success_rate == pytest.approx(0.0)

    def test_all_successes_success_rate_one(self) -> None:
        stats, _ = _make_stats()
        for _ in range(4):
            stats.record("navigate_skill", "detect_shelf", success=True, duration_sec=12.0)
        rec = stats.get_stats("navigate_skill", "detect_*")
        assert rec is not None
        assert rec.success_rate == pytest.approx(1.0)

    def test_avg_duration_calculated_correctly(self) -> None:
        stats, _ = _make_stats()
        stats.record("navigate_skill", "reach_x", success=True, duration_sec=10.0)
        stats.record("navigate_skill", "reach_x", success=True, duration_sec=30.0)
        rec = stats.get_stats("navigate_skill", "reach_*")
        assert rec is not None
        assert rec.avg_duration == pytest.approx(20.0)

    def test_zero_attempts_properties(self) -> None:
        """StrategyRecord with no attempts: success_rate and avg_duration are 0.0."""
        rec = StrategyRecord(strategy_name="x", sub_goal_pattern="y_*")
        assert rec.success_rate == 0.0
        assert rec.avg_duration == 0.0

    def test_multiple_strategies_tracked_separately(self) -> None:
        stats, _ = _make_stats()
        stats.record("navigate_skill", "reach_kitchen", success=True, duration_sec=20.0)
        stats.record("code_as_policy", "reach_kitchen", success=False, duration_sec=40.0)
        nav_rec = stats.get_stats("navigate_skill", "reach_*")
        code_rec = stats.get_stats("code_as_policy", "reach_*")
        assert nav_rec is not None
        assert code_rec is not None
        assert nav_rec.successes == 1
        assert code_rec.successes == 0
        assert nav_rec.total_attempts == 1
        assert code_rec.total_attempts == 1


# ---------------------------------------------------------------------------
# AC-10: get_rankings sorted descending
# ---------------------------------------------------------------------------

class TestGetRankings:
    def test_ac10_rankings_sorted_descending(self) -> None:
        """navigate_skill 5/5=1.0 > code_as_policy 2/3≈0.667."""
        stats, _ = _make_stats()
        # navigate_skill: 5 successes / 5 attempts
        for _ in range(5):
            stats.record("navigate_skill", "reach_x", success=True, duration_sec=25.0)
        # code_as_policy: 2 successes / 3 attempts
        for i in range(3):
            stats.record("code_as_policy", "reach_x", success=(i < 2), duration_sec=35.0)

        rankings = stats.get_rankings("reach_*")
        assert len(rankings) == 2
        assert rankings[0].strategy_name == "navigate_skill"
        assert rankings[1].strategy_name == "code_as_policy"
        assert rankings[0].success_rate >= rankings[1].success_rate

    def test_rankings_only_includes_matching_pattern(self) -> None:
        stats, _ = _make_stats()
        stats.record("navigate_skill", "reach_kitchen", success=True, duration_sec=10.0)
        stats.record("navigate_skill", "detect_cup", success=True, duration_sec=10.0)
        rankings = stats.get_rankings("reach_*")
        assert len(rankings) == 1
        assert rankings[0].sub_goal_pattern == "reach_*"

    def test_rankings_empty_for_unknown_pattern(self) -> None:
        stats, _ = _make_stats()
        stats.record("navigate_skill", "reach_kitchen", success=True, duration_sec=10.0)
        assert stats.get_rankings("fly_*") == []

    def test_rankings_three_strategies_correct_order(self) -> None:
        stats, _ = _make_stats()
        # A: 1/1 = 1.0
        stats.record("strategy_a", "observe_room", success=True, duration_sec=5.0)
        # B: 0/1 = 0.0
        stats.record("strategy_b", "observe_room", success=False, duration_sec=5.0)
        # C: 1/2 = 0.5
        stats.record("strategy_c", "observe_room", success=True, duration_sec=5.0)
        stats.record("strategy_c", "observe_room", success=False, duration_sec=5.0)

        rankings = stats.get_rankings("observe_*")
        names = [r.strategy_name for r in rankings]
        assert names[0] == "strategy_a"
        assert names[-1] == "strategy_b"


# ---------------------------------------------------------------------------
# AC-11: save + load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_ac11_save_and_load_roundtrip(self) -> None:
        stats, path = _make_stats()
        for i in range(10):
            stats.record("navigate_skill", "reach_kitchen", success=(i < 8), duration_sec=25.0)
        stats.save()

        stats2 = StrategyStats(persist_path=path)
        rec = stats2.get_stats("navigate_skill", "reach_*")
        assert rec is not None
        assert rec.total_attempts == 10
        assert rec.successes == 8
        assert rec.success_rate == pytest.approx(0.8)
        assert rec.total_duration_sec == pytest.approx(250.0)

    def test_save_creates_parent_directory(self) -> None:
        tmp_dir = tempfile.mkdtemp()
        nested_path = os.path.join(tmp_dir, "nested", "deep", "stats.json")
        stats = StrategyStats(persist_path=nested_path)
        stats.record("s", "reach_x", success=True, duration_sec=1.0)
        stats.save()
        assert os.path.exists(nested_path)

    def test_load_nonexistent_file_is_silent(self) -> None:
        """No file → no crash, empty state."""
        tmp_dir = tempfile.mkdtemp()
        path = os.path.join(tmp_dir, "nonexistent.json")
        stats = StrategyStats(persist_path=path)
        assert stats.get_stats("any", "any_*") is None

    def test_corrupt_file_silent_reset(self) -> None:
        """Corrupt JSON → silent reset, no crash."""
        tmp_dir = tempfile.mkdtemp()
        path = os.path.join(tmp_dir, "corrupt.json")
        with open(path, "w") as f:
            f.write("NOT VALID JSON }{")
        stats = StrategyStats(persist_path=path)
        # Should not raise; should have empty state
        assert stats.get_stats("x", "y_*") is None

    def test_corrupt_schema_silent_reset(self) -> None:
        """Valid JSON but wrong schema → silent reset."""
        tmp_dir = tempfile.mkdtemp()
        path = os.path.join(tmp_dir, "bad_schema.json")
        with open(path, "w") as f:
            # Missing required fields
            json.dump([{"unexpected_field": 42}], f)
        stats = StrategyStats(persist_path=path)
        assert stats.get_stats("x", "y_*") is None

    def test_save_multiple_records_and_reload(self) -> None:
        stats, path = _make_stats()
        stats.record("navigate_skill", "reach_kitchen", success=True, duration_sec=20.0)
        stats.record("code_as_policy", "detect_cup", success=False, duration_sec=40.0)
        stats.record("navigate_skill", "detect_cup", success=True, duration_sec=15.0)
        stats.save()

        stats2 = StrategyStats(persist_path=path)
        assert stats2.get_stats("navigate_skill", "reach_*") is not None
        assert stats2.get_stats("code_as_policy", "detect_*") is not None
        assert stats2.get_stats("navigate_skill", "detect_*") is not None
