# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""StrategySelector — maps SubGoals to execution strategies.

Priority order:
1. sub_goal.strategy is non-empty → resolve explicitly
2. Name/description keyword rules (navigate, observe, detect, etc.)
3. skill_registry.match(description) if a registry is injected
4. Stats override (if stats has sufficient data and a better strategy exists)
5. Fallback result
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from vector_os_nano.vcli.cognitive.types import SubGoal


def _word_match(keywords: tuple[str, ...], text: str) -> bool:
    """Match keywords with word boundaries for ASCII, substring for CJK."""
    for kw in keywords:
        if kw.isascii():
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                return True
        else:
            if kw in text:
                return True
    return False


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

    executor_type: str   # "skill" | "primitive" | "fallback" | "code" | "tool" | "capability" | "answer"
    name: str            # skill name or primitive function name
    params: dict         # parameters to pass


class StrategySelector:
    """Maps SubGoals to execution strategies using rule-based matching.

    Optionally accepts a StrategyStats instance for data-driven selection.
    When stats has sufficient data (>= 3 attempts) and the top strategy has
    a success rate > 50%, it overrides the rule-based choice.
    """

    def __init__(
        self,
        skill_registry: Any = None,
        stats: Any = None,
        capability_names: "frozenset[str] | set[str] | None" = None,
        has_base: bool = True,
    ) -> None:
        self._skill_registry = skill_registry
        self._stats = stats
        # Registered routable-capability names (Phase C). An explicit strategy
        # matching one of these resolves to capability dispatch. Empty => no
        # capability routing (the path stays byte-identical).
        self._capability_names = frozenset(capability_names) if capability_names else frozenset()
        # Whether the connected agent has a mobile base (Stage 2b). When False
        # (an arm-only world) the GO2 locomotion/observation keyword ladder and
        # the base-primitive explicit names are skipped entirely — those routes
        # call _require_base() on the executor side and would raise on an arm.
        # Routing on a baseless world goes through explicit '<skill>_skill'
        # resolution and the skill-registry alias match only. Defaults True so
        # go2/robot behaviour stays byte-identical.
        self._has_base = bool(has_base)

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

        # Priority 1b (fail-loud, rule 8): a step whose explicit strategy was a
        # HALLUCINATION carries the offending name on ``cleared_strategy`` (the
        # decomposer cleared ``strategy`` to "" but stamped the name here). Resolve
        # it to the LOUD ``invalid`` route — a clear, named error with the valid
        # set — instead of letting the cleared (empty) strategy silently re-route
        # through the keyword ladder / registry match to a phantom skill, or fall
        # through to the opaque ``unmatched`` fallback. This fires on EVERY path
        # (first decompose AND replan), since the decomposer stamps the marker on
        # every validation. Registry-independent so even a registry-less world
        # (default go2) fails loud rather than re-deriving a phantom by keyword.
        cleared = getattr(sub_goal, "cleared_strategy", "")
        if cleared:
            valid = self._registered_skill_names()
            return StrategyResult(
                "invalid",
                cleared,
                {
                    "strategy": cleared,
                    "valid_strategies": sorted(valid) if valid is not None else [],
                },
            )

        # Priority 2: Name/description keyword rules
        name_lower = sub_goal.name.lower()
        desc_lower = sub_goal.description.lower()
        combined = name_lower + " " + desc_lower

        result: StrategyResult | None = None

        # GO2 locomotion/observation keyword ladder (base-world routes). These
        # branches resolve to base/locomotion primitives and stance/observation
        # skills that call _require_base() on the executor side; on a baseless
        # world (an arm) they would raise, so the whole ladder is skipped and
        # routing goes through '<skill>_skill' resolution + the registry alias
        # match only. Gated behind has_base so go2 stays byte-identical.
        if self._has_base:
            # Navigation
            if any(kw in combined for kw in ("reach", "navigate", "go_to", "到", "去")):
                room = sub_goal.strategy_params.get("room", sub_goal.description)
                result = self._route("navigate", {"room": room})

            # Observation
            elif any(kw in combined for kw in ("observe", "look", "scan", "看", "观察")):
                result = self._route("look", {})

            # Detection
            elif _word_match(("detect", "find", "check"), combined) or any(
                kw in combined for kw in ("检测", "找", "检查")
            ):
                query = sub_goal.strategy_params.get("query", sub_goal.description)
                result = self._route("detect", {"query": query})

            # Stance — stand / sit
            elif _word_match(("stand",), combined):
                result = StrategyResult("skill", "stand", {})

            elif _word_match(("sit",), combined):
                result = StrategyResult("skill", "sit", {})

            # Stop (primitive before walk to avoid 'stop' being caught by nothing)
            elif _word_match(("stop",), combined):
                result = StrategyResult("primitive", "stop", {})

            # Movement primitives
            elif _word_match(("walk", "forward"), combined) or "前进" in combined:
                dist = sub_goal.strategy_params.get("distance", 1.0)
                result = StrategyResult("primitive", "walk_forward", {"distance_m": dist})

            elif any(kw in combined for kw in ("turn", "rotate", "转")):
                angle = sub_goal.strategy_params.get("angle", 1.57)
                result = StrategyResult("primitive", "turn", {"angle_rad": angle})

        # Priority 3: Skill registry alias match (all worlds)
        if result is None and self._skill_registry is not None:
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
            cand = self._resolve_explicit(top.strategy_name, sub_goal.strategy_params)
            # GUARD (N4 live finding): stats persist across worlds/lives — a
            # go2-era 'navigate' ranking must never override a VALID route in
            # a world where that skill does not exist. A candidate resolving
            # invalid, or to a skill this registry does not have, keeps the
            # rule-based result (never promote a phantom).
            if cand.executor_type == "invalid":
                return result
            if cand.executor_type == "skill":
                valid = self._registered_skill_names()
                if valid is not None and cand.name not in valid:
                    return result
            return cand

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _route(self, name: str, params: dict) -> "StrategyResult | None":
        """Route a keyword-matched perception/planning step (Phase C.2).

        If a routable capability named *name* is registered, dispatch to it (and
        let measured stats later promote a better-performing alternative for the
        same sub-goal pattern); otherwise fall back to the classical skill of the
        same name — PROVIDED this world's registry actually has it. The go2-era
        keyword ladder must never route to a phantom in another world (N4 live
        finding: 'go/到' routed habitat steps to the unregistered 'navigate';
        M5's 'Skill not found: look' was the same hole). Returning None lets
        select() fall through to the registry ALIAS match, which resolves the
        same wording against the real skill set.
        """
        if name in self._capability_names:
            return StrategyResult("capability", name, params)
        valid = self._registered_skill_names()
        if valid is not None and name not in valid:
            return None  # unknown in THIS world — fall through to alias match
        return StrategyResult("skill", name, params)

    def _registered_skill_names(self) -> "frozenset[str] | None":
        """Return the set of registered skill names, or None if undeterminable.

        Returns None when no registry is injected or its skill list cannot be
        read — in that case the selector cannot fail-loud on an unknown skill and
        keeps the by-convention behaviour (treat ``<X>_skill`` as a skill).
        """
        registry = self._skill_registry
        if registry is None:
            return None
        lister = getattr(registry, "list_skills", None)
        if not callable(lister):
            return None
        try:
            return frozenset(str(n) for n in lister())
        except Exception:  # noqa: BLE001
            return None

    def _resolve_explicit(self, strategy: str, params: dict) -> StrategyResult:
        """Resolve an explicit strategy name to a StrategyResult.

        Mapping rules:
        - "code_as_policy" → executor_type="code", name="code_as_policy".
        - "tool_call" → executor_type="tool", name="tool_call" (params carry
          {"tool": <name>, "args": {...}}; dispatched through ToolDispatcher).
        - Ends with "_skill" → executor_type="skill", name=strategy minus suffix.
          When a registry is injected and the bare name is NOT a registered skill
          (and not a known capability/primitive), resolve to executor_type=
          "invalid" so the executor surfaces a clear, named error instead of
          fabricating a phantom skill (Stage 2b fail-loud).
        - In _PRIMITIVE_NAMES (and the agent has a base) → executor_type="primitive".
        - Otherwise → executor_type="skill" (assume skill by name convention).
        """
        if strategy == "code_as_policy":
            return StrategyResult("code", "code_as_policy", params)

        if strategy == "tool_call":
            return StrategyResult("tool", "tool_call", params)

        # Stage 5 (S5.2): an answer-only conversation step. Resolves to a
        # side-effect-free "answer" executor that returns the step's answer text
        # as structured output. The sub-goal carries answer_only=True (the
        # evidence gate's explicit marker); this strategy is its dispatch route.
        if strategy == "answer":
            return StrategyResult("answer", "answer", params)

        # A strategy naming a registered routable capability (Phase C) — the
        # capability name IS the strategy; params is its input payload.
        if strategy in self._capability_names:
            return StrategyResult("capability", strategy, params)

        if strategy.endswith(_SKILL_SUFFIX):
            skill_name = strategy[: -len(_SKILL_SUFFIX)]
            valid = self._registered_skill_names()
            # Fail-loud: a registry is present and the bare name is not a real
            # skill (nor a capability) — do not fabricate a phantom skill. Base
            # primitives are valid base routes only when the agent has a base.
            if (
                valid is not None
                and skill_name not in valid
                and strategy not in self._capability_names
                and not (self._has_base and skill_name in _PRIMITIVE_NAMES)
            ):
                return StrategyResult(
                    "invalid",
                    strategy,
                    {"strategy": strategy, "valid_strategies": sorted(valid)},
                )
            return StrategyResult("skill", skill_name, params)

        # Base locomotion primitives are only routable on a world with a base;
        # on a baseless world they would call _require_base() and raise.
        if self._has_base and strategy in _PRIMITIVE_NAMES:
            # Normalize LLM-generated param names to match primitive signatures
            normalized = dict(params) if params else {}
            if strategy == "walk_forward" and "distance" in normalized:
                normalized["distance_m"] = normalized.pop("distance")
            if strategy == "turn" and "angle" in normalized:
                normalized["angle_rad"] = normalized.pop("angle")
            return StrategyResult("primitive", strategy, normalized)

        # Treat as a skill with the strategy name as-is
        return StrategyResult("skill", strategy, params)
