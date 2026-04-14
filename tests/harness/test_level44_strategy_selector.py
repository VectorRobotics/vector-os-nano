"""T4 — StrategySelector tests.

AC-21: Explicit strategy in sub_goal.strategy is honoured.
AC-22: Name/description keyword matching selects correct skill.
AC-23: Primitive strategies are resolved to executor_type="primitive".
AC-24: Unmatched sub_goals get fallback result.
"""
from __future__ import annotations

import pytest

from vector_os_nano.vcli.cognitive.types import SubGoal
from vector_os_nano.vcli.cognitive.strategy_selector import StrategySelector, StrategyResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sg(name: str, description: str = "", strategy: str = "",
        strategy_params: dict | None = None) -> SubGoal:
    return SubGoal(
        name=name,
        description=description,
        verify="True",
        strategy=strategy,
        strategy_params=strategy_params or {},
    )


# ---------------------------------------------------------------------------
# AC-21: Explicit strategy
# ---------------------------------------------------------------------------

def test_explicit_navigate_skill():
    """navigate_skill → executor_type=skill, name=navigate."""
    sel = StrategySelector()
    sg = _sg("reach_kitchen", "go to kitchen", strategy="navigate_skill")
    result = sel.select(sg)
    assert result.executor_type == "skill"
    assert result.name == "navigate"


def test_explicit_look_skill():
    """look_skill → executor_type=skill, name=look."""
    sel = StrategySelector()
    sg = _sg("look_around", strategy="look_skill")
    result = sel.select(sg)
    assert result.executor_type == "skill"
    assert result.name == "look"


def test_explicit_describe_scene_skill():
    """describe_scene_skill → executor_type=skill, name=describe_scene."""
    sel = StrategySelector()
    sg = _sg("check_scene", strategy="describe_scene_skill")
    result = sel.select(sg)
    assert result.executor_type == "skill"
    assert result.name == "describe_scene"


def test_explicit_walk_forward_primitive():
    """walk_forward → executor_type=primitive."""
    sel = StrategySelector()
    sg = _sg("step_forward", strategy="walk_forward",
             strategy_params={"distance": 2.0})
    result = sel.select(sg)
    assert result.executor_type == "primitive"
    assert result.name == "walk_forward"


def test_explicit_turn_primitive():
    """turn → executor_type=primitive."""
    sel = StrategySelector()
    sg = _sg("turn_right", strategy="turn", strategy_params={"angle": 1.57})
    result = sel.select(sg)
    assert result.executor_type == "primitive"
    assert result.name == "turn"


def test_explicit_scan_360_primitive():
    """scan_360 → executor_type=primitive."""
    sel = StrategySelector()
    sg = _sg("scan_area", strategy="scan_360")
    result = sel.select(sg)
    assert result.executor_type == "primitive"
    assert result.name == "scan_360"


def test_explicit_stop_skill():
    """stop_skill → executor_type=skill, name=stop."""
    sel = StrategySelector()
    sg = _sg("halt", strategy="stop_skill")
    result = sel.select(sg)
    assert result.executor_type == "skill"
    assert result.name == "stop"


def test_explicit_params_passed_through():
    """strategy_params are forwarded in the StrategyResult."""
    sel = StrategySelector()
    params = {"room": "kitchen", "speed": 0.5}
    sg = _sg("go_kitchen", strategy="navigate_skill", strategy_params=params)
    result = sel.select(sg)
    assert result.params == params


# ---------------------------------------------------------------------------
# AC-22: Name-based keyword matching
# ---------------------------------------------------------------------------

def test_reach_keyword_matches_navigate():
    """'reach' in name → navigate skill."""
    sel = StrategySelector()
    sg = _sg("reach_kitchen", "go to the kitchen")
    result = sel.select(sg)
    assert result.executor_type == "skill"
    assert result.name == "navigate"


def test_observe_keyword_matches_look():
    """'observe' in name → look skill."""
    sel = StrategySelector()
    sg = _sg("observe_table", "look at the table")
    result = sel.select(sg)
    assert result.name in ("look", "describe_scene")


def test_detect_keyword_matches_detect():
    """'detect' in name → detect skill."""
    sel = StrategySelector()
    sg = _sg("detect_cup", "检测杯子是否存在")
    result = sel.select(sg)
    assert result.executor_type == "skill"
    assert result.name == "detect"


def test_stand_keyword_matches_stand():
    """'stand' → stand skill."""
    sel = StrategySelector()
    sg = _sg("stand_up", "stand upright")
    result = sel.select(sg)
    assert result.executor_type == "skill"
    assert result.name == "stand"


def test_sit_keyword_matches_sit():
    """'sit' → sit skill."""
    sel = StrategySelector()
    sg = _sg("sit_down", "sit down now")
    result = sel.select(sg)
    assert result.executor_type == "skill"
    assert result.name == "sit"


def test_stop_keyword_matches_stop_primitive():
    """'stop' in name → stop primitive."""
    sel = StrategySelector()
    sg = _sg("stop_motion", "stop all motion")
    result = sel.select(sg)
    assert result.executor_type == "primitive"
    assert result.name == "stop"


def test_walk_keyword_matches_walk_forward():
    """'walk' in description → walk_forward primitive."""
    sel = StrategySelector()
    sg = _sg("move_ahead", "walk forward 2 metres")
    result = sel.select(sg)
    assert result.executor_type == "primitive"
    assert result.name == "walk_forward"


def test_turn_keyword_matches_turn_primitive():
    """'turn' in description → turn primitive."""
    sel = StrategySelector()
    sg = _sg("rotate_robot", "turn left")
    result = sel.select(sg)
    assert result.executor_type == "primitive"
    assert result.name == "turn"


# ---------------------------------------------------------------------------
# Chinese keyword matching
# ---------------------------------------------------------------------------

def test_chinese_navigate_dao():
    """Chinese '到' keyword → navigate."""
    sel = StrategySelector()
    sg = _sg("到厨房", "到厨房")
    result = sel.select(sg)
    assert result.executor_type == "skill"
    assert result.name == "navigate"


def test_chinese_navigate_qu():
    """Chinese '去' keyword → navigate."""
    sel = StrategySelector()
    sg = _sg("go_kitchen_cn", "去厨房")
    result = sel.select(sg)
    assert result.executor_type == "skill"
    assert result.name == "navigate"


def test_chinese_observe_guan_cha():
    """Chinese '观察' keyword → observe-type result."""
    sel = StrategySelector()
    sg = _sg("observe_cn", "观察桌子")
    result = sel.select(sg)
    assert result.executor_type == "skill"


# ---------------------------------------------------------------------------
# AC-23: strategy_params forwarded for primitives
# ---------------------------------------------------------------------------

def test_walk_forward_distance_param():
    """strategy_params.distance is forwarded."""
    sel = StrategySelector()
    sg = _sg("walk_2m", "walk forward", strategy="walk_forward",
             strategy_params={"distance": 2.0})
    result = sel.select(sg)
    assert result.executor_type == "primitive"
    assert result.name == "walk_forward"
    assert result.params.get("distance_m") == 2.0  # normalized from "distance"


def test_turn_angle_param():
    """strategy_params.angle is normalized to angle_rad."""
    sel = StrategySelector()
    sg = _sg("turn_90", "turn", strategy="turn", strategy_params={"angle": 1.57})
    result = sel.select(sg)
    assert result.params.get("angle_rad") == 1.57  # normalized from "angle"


# ---------------------------------------------------------------------------
# AC-24: Fallback
# ---------------------------------------------------------------------------

def test_unmatched_returns_fallback():
    """Unknown sub_goal with no registry match → fallback."""
    sel = StrategySelector()
    sg = _sg("quantum_teleport", "teleport to mars")
    result = sel.select(sg)
    assert result.executor_type == "fallback"
    assert result.name == "unmatched"
    assert result.params.get("sub_goal") == "quantum_teleport"


def test_skill_registry_match():
    """When registry.match() returns a hit → skill result."""
    class _FakeMatch:
        skill_name = "custom_skill"

    class _FakeRegistry:
        def match(self, description: str):
            return _FakeMatch()

    sel = StrategySelector(skill_registry=_FakeRegistry())
    sg = _sg("do_something_weird", "perform a unique task")
    result = sel.select(sg)
    assert result.executor_type == "skill"
    assert result.name == "custom_skill"


def test_skill_registry_no_match_falls_back():
    """When registry.match() returns None → fallback."""
    class _NoMatchRegistry:
        def match(self, description: str):
            return None

    sel = StrategySelector(skill_registry=_NoMatchRegistry())
    sg = _sg("mystery_op", "do something completely unknown")
    result = sel.select(sg)
    assert result.executor_type == "fallback"


# ---------------------------------------------------------------------------
# StrategyResult immutability
# ---------------------------------------------------------------------------

def test_strategy_result_is_frozen():
    """StrategyResult is a frozen dataclass."""
    r = StrategyResult(executor_type="skill", name="navigate", params={})
    with pytest.raises((AttributeError, TypeError)):
        r.executor_type = "primitive"  # type: ignore[misc]


def test_strategy_result_fields():
    """StrategyResult has correct fields."""
    r = StrategyResult(executor_type="primitive", name="walk_forward",
                       params={"distance_m": 1.5})
    assert r.executor_type == "primitive"
    assert r.name == "walk_forward"
    assert r.params == {"distance_m": 1.5}
