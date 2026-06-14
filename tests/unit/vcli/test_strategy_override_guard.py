# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""N4 live finding — cross-world contamination guards.

StrategyStats persist across worlds: a go2-era 'navigate' ranking must never
override a valid route in a world whose registry has no such skill (the
phantom override froze the house VLN run). Same family: the engine fast path
must not build navigate_to steps (it cannot bind x/y or the geodesic verify).
"""
from __future__ import annotations

from types import SimpleNamespace

from vector_os_nano.vcli.cognitive.strategy_selector import StrategySelector


class _Reg:
    """Minimal registry: list/get + match-by-alias surface."""

    def __init__(self, names) -> None:
        self._names = list(names)

    def list_skills(self):
        return list(self._names)

    def get(self, name):
        return SimpleNamespace(name=name) if name in self._names else None

    def match(self, text):
        return None


class _Stats:
    def __init__(self, top_name) -> None:
        self._top = top_name

    def get_rankings(self, pattern):
        return [SimpleNamespace(
            strategy_name=self._top, total_attempts=10, success_rate=0.9,
        )]


def _sub_goal(name="navigate_to_goal", strategy="navigate_to_skill"):
    return SimpleNamespace(
        name=name, description="go somewhere", strategy=strategy,
        strategy_params={}, cleared_strategy="",
    )


class _AliasNavReg(_Reg):
    """Alias-matches any text to navigate_to (the habitat wording)."""

    def match(self, text):
        return SimpleNamespace(skill_name="navigate_to", extracted_arg="")


class TestStatsOverrideGuard:
    # NOTE: the stats override applies only to keyword/alias-routed steps —
    # an explicit strategy returns from priority 1 untouched. These steps
    # carry strategy="" so routing (and the override) actually runs.

    def test_phantom_override_refused(self) -> None:
        # go2-era stats rank 'navigate' top — this world has no such skill.
        sel = StrategySelector(
            skill_registry=_AliasNavReg(["walk", "turn", "stop", "navigate_to"]),
            stats=_Stats("navigate"),
        )
        r = sel.select(_sub_goal(strategy=""))
        assert r.executor_type == "skill"
        assert r.name == "navigate_to"  # the valid route survived

    def test_valid_override_still_promotes(self) -> None:
        sel = StrategySelector(
            skill_registry=_AliasNavReg(["walk", "turn", "stop", "navigate_to"]),
            stats=_Stats("walk_skill"),
        )
        r = sel.select(_sub_goal(strategy=""))
        assert r.executor_type == "skill"
        assert r.name == "walk"  # a REAL skill may still be promoted

    def test_invalid_resolution_refused(self) -> None:
        sel = StrategySelector(
            skill_registry=_AliasNavReg(["navigate_to"]),
            stats=_Stats("fly_to_moon_skill"),
        )
        r = sel.select(_sub_goal(strategy=""))
        assert r.name == "navigate_to"


class TestKeywordLadderGuard:
    def test_phantom_ladder_route_falls_to_alias_match(self) -> None:
        # Empty strategy + 'go to' wording: the go2 ladder wants 'navigate',
        # which this world does NOT have — the selector must fall through to
        # the registry alias match (navigate_to) instead of a phantom.
        class _AliasReg(_Reg):
            def match(self, text):
                return SimpleNamespace(skill_name="navigate_to",
                                       extracted_arg="")

        sel = StrategySelector(
            skill_registry=_AliasReg(["walk", "turn", "stop", "navigate_to"]),
            stats=None,
        )
        sg = _sub_goal(strategy="")
        r = sel.select(sg)
        assert (r.executor_type, r.name) == ("skill", "navigate_to")

    def test_registered_ladder_route_unchanged(self) -> None:
        # A world that DOES register 'navigate' keeps the classic go2 route.
        sel = StrategySelector(
            skill_registry=_Reg(["navigate", "walk", "turn"]), stats=None
        )
        r = sel.select(_sub_goal(strategy=""))
        assert (r.executor_type, r.name) == ("skill", "navigate")


class TestFastPathExcludesNavigateTo:
    def test_navigate_to_message_returns_none(self) -> None:
        from vector_os_nano.vcli.engine import VectorEngine

        class _MatchReg(_Reg):
            def match(self, text):
                return SimpleNamespace(
                    skill_name="navigate_to", extracted_arg="(1, 2)"
                )

        engine = VectorEngine.__new__(VectorEngine)  # no backend needed
        tree = engine._try_skill_goal_tree(
            "Go to position (1, 2)", _MatchReg(["navigate_to"])
        )
        assert tree is None  # the LLM planner owns coordinate binding
