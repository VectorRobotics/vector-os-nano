"""StrategySelector — maps SubGoals to execution strategies.

Priority order:
1. sub_goal.strategy is non-empty → resolve explicitly
2. Name/description keyword rules (navigate, observe, detect, etc.)
3. skill_registry.match(description) if a registry is injected
4. Stats override (if stats has sufficient data and a better strategy exists)
5. Fallback result
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vector_os_nano.vcli.cognitive.types import SubGoal


# ---------------------------------------------------------------------------
# Primitive function names (used in _resolve_explicit)
# ---------------------------------------------------------------------------

_PRIMITIVE_NAMES: frozenset[str] = frozenset({
    "walk_forward",
    "turn",
    "scan_360",
    "stop",
})

# Strategy names that end with "_skill" map to skill executor_type.
# All others that appear in KNOWN_STRATEGIES without "_skill" suffix are
# either primitives (above) or treated as skills by default.

_SKILL_SUFFIX = "_skill"

# Minimum attempts required before stats can override rule-based selection.
_STATS_MIN_ATTEMPTS: int = 3

# Minimum success rate required for a stats-driven override.
_STATS_MIN_SUCCESS_RATE: float = 0.5


@dataclass(frozen=True)
class StrategyResult:
    """Result of strategy selection."""

    executor_type: str   # "skill" | "primitive" | "fallback" | "code"
    name: str            # skill name or primitive function name
    params: dict         # parameters to pass


class StrategySelector:
    """Maps SubGoals to execution strategies using rule-based matching.

    Optionally accepts a StrategyStats instance for data-driven selection.
    When stats has sufficient data (>= 3 attempts) and the top strategy has
    a success rate > 50%, it overrides the rule-based choice.
    """

    def __init__(self, skill_registry: Any = None, stats: Any = None) -> None:
        self._skill_registry = skill_registry
        self._stats = stats

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self, sub_goal: SubGoal) -> StrategyResult:
        """Select best execution strategy for a sub-goal.

        Priority:
        1. sub_goal.strategy is non-empty → use directly
        2. Name-based rules (reach→navigate, observe→look, etc.)
        3. Try skill_registry.match(description)
        4. Fallback
        After rule matching produces a result, stats may override if data is
        sufficient (>= 3 attempts, > 50% success rate for the top strategy).
        """
        # Priority 1: Explicit strategy from GoalTree
        if sub_goal.strategy:
            return self._resolve_explicit(sub_goal.strategy, sub_goal.strategy_params)

        # Priority 2: Name/description keyword rules
        name_lower = sub_goal.name.lower()
        desc_lower = sub_goal.description.lower()
        combined = name_lower + " " + desc_lower

        result: StrategyResult | None = None

        # Navigation
        if any(kw in combined for kw in ("reach", "navigate", "go_to", "到", "去")):
            room = sub_goal.strategy_params.get("room", sub_goal.description)
            result = StrategyResult("skill", "navigate", {"room": room})

        # Observation
        elif any(kw in combined for kw in ("observe", "look", "scan", "看", "观察")):
            result = StrategyResult("skill", "look", {})

        # Detection
        elif any(kw in combined for kw in ("detect", "find", "check", "检测", "找", "检查")):
            query = sub_goal.strategy_params.get("query", sub_goal.description)
            result = StrategyResult("skill", "describe_scene", {"query": query})

        # Stance — stand / sit
        elif any(kw in combined for kw in ("stand",)):
            result = StrategyResult("skill", "stand", {})

        elif any(kw in combined for kw in ("sit",)):
            result = StrategyResult("skill", "sit", {})

        # Stop (primitive before walk to avoid 'stop' being caught by nothing)
        elif any(kw in combined for kw in ("stop",)):
            result = StrategyResult("primitive", "stop", {})

        # Movement primitives
        elif any(kw in combined for kw in ("walk", "forward", "前进")):
            dist = sub_goal.strategy_params.get("distance", 1.0)
            result = StrategyResult("primitive", "walk_forward", {"distance_m": dist})

        elif any(kw in combined for kw in ("turn", "rotate", "转")):
            angle = sub_goal.strategy_params.get("angle", 1.57)
            result = StrategyResult("primitive", "turn", {"angle_rad": angle})

        # Priority 3: Skill registry alias match
        elif self._skill_registry is not None:
            match = self._skill_registry.match(sub_goal.description)
            if match is not None:
                result = StrategyResult("skill", match.skill_name, {})

        # Priority 4: Fallback
        if result is None:
            result = StrategyResult("fallback", "unmatched", {"sub_goal": sub_goal.name})

        # Stats override: check if stats has a better-performing strategy
        result = self._maybe_override_with_stats(result, sub_goal)

        return result

    def _maybe_override_with_stats(
        self,
        result: StrategyResult,
        sub_goal: SubGoal,
    ) -> StrategyResult:
        """Optionally replace rule-based result with the top stats-ranked strategy.

        Only overrides when:
        - stats is injected
        - top-ranked strategy has >= _STATS_MIN_ATTEMPTS attempts
        - top-ranked strategy has success_rate > _STATS_MIN_SUCCESS_RATE
        - top-ranked strategy differs from the current result
        """
        if self._stats is None:
            return result

        try:
            from vector_os_nano.vcli.cognitive.strategy_stats import StrategyStats
            pattern = StrategyStats.extract_pattern(sub_goal.name)
            rankings = self._stats.get_rankings(pattern)
        except Exception:  # noqa: BLE001
            return result

        if not rankings:
            return result

        top = rankings[0]
        if (
            top.total_attempts >= _STATS_MIN_ATTEMPTS
            and top.success_rate > _STATS_MIN_SUCCESS_RATE
            and top.strategy_name != result.name
        ):
            return self._resolve_explicit(top.strategy_name, sub_goal.strategy_params)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_explicit(self, strategy: str, params: dict) -> StrategyResult:
        """Resolve an explicit strategy name to a StrategyResult.

        Mapping rules:
        - "code_as_policy" → executor_type="code", name="code_as_policy".
        - Ends with "_skill" → executor_type="skill", name=strategy minus suffix.
        - In _PRIMITIVE_NAMES → executor_type="primitive".
        - Otherwise → executor_type="skill" (assume skill by name convention).
        """
        if strategy == "code_as_policy":
            return StrategyResult("code", "code_as_policy", params)

        if strategy.endswith(_SKILL_SUFFIX):
            skill_name = strategy[: -len(_SKILL_SUFFIX)]
            return StrategyResult("skill", skill_name, params)

        if strategy in _PRIMITIVE_NAMES:
            return StrategyResult("primitive", strategy, params)

        # Treat as a skill with the strategy name as-is
        return StrategyResult("skill", strategy, params)
