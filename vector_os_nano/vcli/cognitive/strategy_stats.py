"""StrategyStats — persistent strategy success rate tracking for data-driven selection.

Tracks per-(strategy, sub_goal_pattern) execution outcomes across sessions.
Persists to JSON for cross-session learning.

Example::

    stats = StrategyStats()
    stats.record("navigate_skill", "reach_kitchen", success=True, duration_sec=22.3)
    rec = stats.get_stats("navigate_skill", "reach_*")
    print(rec.success_rate)   # 1.0
    stats.save()
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_PATH = os.path.expanduser("~/.vector_os_nano/strategy_stats.json")


@dataclass
class StrategyRecord:
    """Stats for one strategy on one sub-goal pattern."""

    strategy_name: str
    sub_goal_pattern: str  # "reach_*", "detect_*", etc.
    total_attempts: int = 0
    successes: int = 0
    total_duration_sec: float = 0.0

    @property
    def success_rate(self) -> float:
        """Fraction of successful attempts; 0.0 when no attempts recorded."""
        if self.total_attempts == 0:
            return 0.0
        return self.successes / self.total_attempts

    @property
    def avg_duration(self) -> float:
        """Mean execution duration in seconds; 0.0 when no attempts recorded."""
        if self.total_attempts == 0:
            return 0.0
        return self.total_duration_sec / self.total_attempts


class StrategyStats:
    """Tracks strategy performance for data-driven strategy selection.

    Each record is keyed by (strategy_name, sub_goal_pattern). Sub-goal names
    are automatically bucketed by prefix pattern via ``extract_pattern()``.

    Args:
        persist_path: Path to JSON persistence file.  Defaults to
            ``~/.vector_os_nano/strategy_stats.json``.
    """

    def __init__(self, persist_path: str | None = None) -> None:
        # persist_path=None means in-memory only; no file I/O performed.
        self._path: str | None = persist_path
        self._records: dict[tuple[str, str], StrategyRecord] = {}
        if self._path is not None:
            self.load()

    # ------------------------------------------------------------------
    # Pattern extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_pattern(sub_goal_name: str) -> str:
        """Extract a wildcard pattern from a sub-goal name.

        The first underscore-delimited prefix becomes the pattern stem:

        - ``"reach_kitchen"``          → ``"reach_*"``
        - ``"detect_cup"``             → ``"detect_*"``
        - ``"observe_living_room_table"`` → ``"observe_*"``
        - ``"stand"``                  → ``"stand"``   (no underscore)
        - ``"_private"``               → ``"_private"`` (leading underscore)
        """
        idx = sub_goal_name.find("_")
        if idx > 0:
            return sub_goal_name[:idx] + "_*"
        return sub_goal_name

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def record(
        self,
        strategy_name: str,
        sub_goal_name: str,
        success: bool,
        duration_sec: float,
    ) -> None:
        """Record one strategy execution result.

        Args:
            strategy_name: Identifier of the strategy (e.g. ``"navigate_skill"``).
            sub_goal_name: Raw sub-goal name; bucketed automatically by prefix.
            success: Whether the execution succeeded.
            duration_sec: Wall-clock duration in seconds.
        """
        pattern = self.extract_pattern(sub_goal_name)
        key = (strategy_name, pattern)
        if key not in self._records:
            self._records[key] = StrategyRecord(
                strategy_name=strategy_name,
                sub_goal_pattern=pattern,
            )
        rec = self._records[key]
        rec.total_attempts += 1
        if success:
            rec.successes += 1
        rec.total_duration_sec += duration_sec

    def get_stats(
        self,
        strategy_name: str,
        sub_goal_pattern: str = "*",
    ) -> StrategyRecord | None:
        """Return stats for a specific strategy + pattern.

        Args:
            strategy_name: The strategy identifier.
            sub_goal_pattern: The pre-computed pattern (e.g. ``"reach_*"``).

        Returns:
            :class:`StrategyRecord` if data exists, otherwise ``None``.
        """
        key = (strategy_name, sub_goal_pattern)
        return self._records.get(key)

    def get_rankings(self, sub_goal_pattern: str) -> list[StrategyRecord]:
        """Return all strategies for a pattern, sorted by success_rate descending.

        Args:
            sub_goal_pattern: Pattern to filter by (e.g. ``"reach_*"``).

        Returns:
            List of :class:`StrategyRecord`, best first.  Empty if no data.
        """
        matching = [
            r for r in self._records.values()
            if r.sub_goal_pattern == sub_goal_pattern
        ]
        return sorted(matching, key=lambda r: r.success_rate, reverse=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist all records to the JSON file.

        No-op when persist_path is None (in-memory only mode).
        Creates parent directories as needed.
        """
        if self._path is None:
            return
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "strategy_name": rec.strategy_name,
                "sub_goal_pattern": rec.sub_goal_pattern,
                "total_attempts": rec.total_attempts,
                "successes": rec.successes,
                "total_duration_sec": rec.total_duration_sec,
            }
            for rec in self._records.values()
        ]
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug("StrategyStats saved %d records to %s", len(data), self._path)

    def load(self) -> None:
        """Load records from the JSON file.

        No-op when persist_path is None (in-memory only mode).
        If the file does not exist, state is initialised empty.
        If the file is corrupt or has an unexpected schema, a warning is logged
        and state is reset silently — no exception is raised.
        """
        if self._path is None:
            return
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            records: dict[tuple[str, str], StrategyRecord] = {}
            for item in data:
                rec = StrategyRecord(**item)
                records[(rec.strategy_name, rec.sub_goal_pattern)] = rec
            self._records = records
            logger.debug(
                "StrategyStats loaded %d records from %s", len(records), self._path
            )
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Strategy stats file corrupt, resetting: %s", exc)
            self._records = {}
