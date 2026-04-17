"""L35: Navigation works without hardcoded coordinates.

Verifies that navigate.py uses SceneGraph exclusively for room/door data
and does not contain _ROOM_CENTERS or _ROOM_DOORS hardcoded dicts.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure repo root is on sys.path
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

import vector_os_nano.skills.navigate as navigate_module
from vector_os_nano.core.scene_graph import RoomNode, SceneGraph
from vector_os_nano.core.types import SkillResult
from vector_os_nano.skills.navigate import (
    NavigateSkill,
    _ROOM_ALIASES,
    _detect_current_room,
    _get_room_center_from_memory,
    _resolve_room,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_explored_sg() -> SceneGraph:
    """Create a SceneGraph pre-populated with rooms and doors (like post-explore)."""
    sg = SceneGraph()
    rooms = {
        "living_room": (3.0, 2.5),
        "kitchen": (17.0, 2.5),
        "hallway": (10.0, 5.0),
        "bathroom": (8.5, 12.0),
    }
    for name, (x, y) in rooms.items():
        for _ in range(5):  # visit_count >= _MIN_VISIT_COUNT (3)
            sg.visit(name, x, y)
    # Add doors
    sg.add_door("living_room", "hallway", 6.5, 3.0)
    sg.add_door("kitchen", "hallway", 13.5, 3.0)
    sg.add_door("bathroom", "hallway", 8.5, 10.5)
    return sg


def _make_mock_base(x: float = 3.0, y: float = 2.5, z: float = 0.3, heading: float = 0.0) -> MagicMock:
    """Mock base with get_position, get_heading, navigate_to, go_to_waypoint, walk."""
    base = MagicMock()
    base.get_position.return_value = [x, y, z]
    base.get_heading.return_value = heading
    base.navigate_to.return_value = True
    base.go_to_waypoint.return_value = True  # _dead_reckoning prefers go_to_waypoint
    base.walk.return_value = None
    return base


def _make_skill_context(base: MagicMock, sg: SceneGraph | None) -> MagicMock:
    """Build a mock SkillContext with the given base and spatial_memory service."""
    ctx = MagicMock()
    ctx.base = base
    services: dict = {}
    if sg is not None:
        services["spatial_memory"] = sg
    # Assign a real dict so .get() works correctly (MagicMock auto-mocks .get)
    ctx.services = services
    return ctx


# ---------------------------------------------------------------------------
# L35-1: No hardcoded dicts in module
# ---------------------------------------------------------------------------


def test_no_hardcoded_room_centers_in_module() -> None:
    """_ROOM_CENTERS must not exist in navigate module after refactor."""
    assert not hasattr(navigate_module, "_ROOM_CENTERS"), (
        "_ROOM_CENTERS still present in navigate.py — hardcoded coordinates not removed"
    )


def test_no_hardcoded_room_doors_in_module() -> None:
    """_ROOM_DOORS must not exist in navigate module after refactor."""
    assert not hasattr(navigate_module, "_ROOM_DOORS"), (
        "_ROOM_DOORS still present in navigate.py — hardcoded door coordinates not removed"
    )


def test_room_aliases_still_present() -> None:
    """_ROOM_ALIASES must still exist — it is language mapping, not coordinates."""
    assert hasattr(navigate_module, "_ROOM_ALIASES")
    assert isinstance(_ROOM_ALIASES, dict)
    assert len(_ROOM_ALIASES) > 10


# ---------------------------------------------------------------------------
# L35-2: _resolve_room with SceneGraph
# ---------------------------------------------------------------------------


def test_resolve_room_with_scenegraph_alias() -> None:
    """Chinese alias '厨房' resolves to 'kitchen' when kitchen is in SceneGraph."""
    sg = _make_explored_sg()
    result = _resolve_room("厨房", sg=sg)
    assert result == "kitchen"


def test_resolve_room_with_scenegraph_direct_name() -> None:
    """Direct name 'kitchen' resolves when kitchen is in SceneGraph."""
    sg = _make_explored_sg()
    result = _resolve_room("kitchen", sg=sg)
    assert result == "kitchen"


def test_resolve_room_unknown_when_not_explored() -> None:
    """Alias resolves but room not in SceneGraph -> None."""
    sg = SceneGraph()  # empty — no rooms visited
    result = _resolve_room("dining room", sg=sg)
    assert result is None


def test_resolve_room_no_sg_returns_canonical_for_alias() -> None:
    """Without SceneGraph, alias match still returns canonical name (backward compat)."""
    result = _resolve_room("kitchen", sg=None)
    assert result == "kitchen"


def test_resolve_room_no_sg_returns_none_for_unknown() -> None:
    """Without SceneGraph, completely unknown name returns None."""
    result = _resolve_room("xyzzy_nonexistent_room", sg=None)
    assert result is None


def test_resolve_room_empty_name() -> None:
    """Empty string returns None."""
    assert _resolve_room("") is None
    assert _resolve_room("", sg=SceneGraph()) is None


# ---------------------------------------------------------------------------
# L35-3: _detect_current_room
# ---------------------------------------------------------------------------


def test_detect_current_room_from_scenegraph() -> None:
    """At (3.0, 2.5) with explored SG -> 'living_room' (nearest room center)."""
    sg = _make_explored_sg()
    result = _detect_current_room(3.0, 2.5, sg=sg)
    assert result == "living_room"


def test_detect_current_room_kitchen() -> None:
    """At (17.0, 2.5) with explored SG -> 'kitchen'."""
    sg = _make_explored_sg()
    result = _detect_current_room(17.0, 2.5, sg=sg)
    assert result == "kitchen"


def test_detect_current_room_empty_sg_returns_unknown() -> None:
    """With empty SceneGraph, returns 'unknown'."""
    sg = SceneGraph()
    result = _detect_current_room(5.0, 5.0, sg=sg)
    assert result == "unknown"


def test_detect_current_room_no_sg_returns_unknown() -> None:
    """Without SceneGraph (sg=None), returns 'unknown'."""
    result = _detect_current_room(5.0, 5.0, sg=None)
    assert result == "unknown"


def test_detect_current_room_backward_compat_no_sg_arg() -> None:
    """Old callers passing just (x, y) still work — returns 'unknown'."""
    result = _detect_current_room(5.0, 5.0)
    assert result == "unknown"


# ---------------------------------------------------------------------------
# L35-4: _get_room_center_from_memory
# ---------------------------------------------------------------------------


def test_get_room_center_returns_position_from_sg() -> None:
    """Returns learned position for room with enough visits."""
    sg = _make_explored_sg()
    result = _get_room_center_from_memory(sg, "kitchen")
    assert result is not None
    x, y = result
    # kitchen visited 5 times at (17.0, 2.5) -> center should be ~(17.0, 2.5)
    assert abs(x - 17.0) < 0.01
    assert abs(y - 2.5) < 0.01


def test_get_room_center_returns_none_for_unknown_room() -> None:
    """Returns None if room not in SceneGraph."""
    sg = _make_explored_sg()
    result = _get_room_center_from_memory(sg, "dining_room")
    assert result is None


def test_get_room_center_returns_none_insufficient_visits() -> None:
    """Returns None if room has never been visited (visit_count < _MIN_VISIT_COUNT=1)."""
    sg = SceneGraph()
    # Room added to graph but never visited — get_room returns node with visit_count=0
    # or get_room returns None entirely. Either way, result must be None.
    result = _get_room_center_from_memory(sg, "never_visited_room")
    assert result is None


def test_get_room_center_no_drift_check() -> None:
    """Position far from any 'expected' center is accepted (no hardcoded drift check)."""
    sg = SceneGraph()
    # Register a room at coordinates very different from the old hardcoded values
    for _ in range(5):
        sg.visit("kitchen", 50.0, 50.0)  # far from old hardcoded (17.0, 2.5)
    result = _get_room_center_from_memory(sg, "kitchen")
    assert result is not None
    x, y = result
    assert abs(x - 50.0) < 0.01
    assert abs(y - 50.0) < 0.01


# ---------------------------------------------------------------------------
# L35-5: NavigateSkill.execute()
# ---------------------------------------------------------------------------


def test_navigate_empty_scenegraph_returns_error() -> None:
    """execute() with empty SceneGraph -> error message contains 'explore'."""
    sg = SceneGraph()
    base = _make_mock_base()
    ctx = _make_skill_context(base, sg)
    skill = NavigateSkill()
    result = skill.execute({"room": "kitchen"}, ctx)
    assert result.success is False
    assert "explore" in result.error_message.lower()


def test_navigate_no_scenegraph_returns_error() -> None:
    """execute() with no spatial_memory service -> error message contains 'explore'."""
    base = _make_mock_base()
    ctx = _make_skill_context(base, None)
    skill = NavigateSkill()
    result = skill.execute({"room": "kitchen"}, ctx)
    assert result.success is False
    assert "explore" in result.error_message.lower()


def test_navigate_unknown_room_in_sg_returns_error() -> None:
    """execute() with explored SG but unknown room -> error with available rooms."""
    sg = _make_explored_sg()
    base = _make_mock_base()
    ctx = _make_skill_context(base, sg)
    skill = NavigateSkill()
    result = skill.execute({"room": "xyzzy_nonexistent"}, ctx)
    assert result.success is False
    assert "unknown" in result.error_message.lower() or "available" in result.error_message.lower()


def test_navigate_after_explore_uses_scenegraph_proxy_mode() -> None:
    """execute() with populated SG and base.navigate_to -> success via proxy mode."""
    sg = _make_explored_sg()
    base = _make_mock_base(x=3.0, y=2.5, z=0.3)
    # base has navigate_to -> triggers proxy mode
    base.navigate_to.return_value = True
    ctx = _make_skill_context(base, sg)

    skill = NavigateSkill()
    result = skill.execute({"room": "kitchen"}, ctx)

    assert result.success is True
    assert result.result_data is not None
    assert result.result_data["room"] == "kitchen"
    assert result.result_data["mode"] == "proxy_nav_stack"


def test_navigate_chinese_alias_resolves_via_sg() -> None:
    """Chinese alias '厨房' navigates successfully to 'kitchen'."""
    sg = _make_explored_sg()
    base = _make_mock_base(x=3.0, y=2.5, z=0.3)
    base.navigate_to.return_value = True
    ctx = _make_skill_context(base, sg)

    skill = NavigateSkill()
    result = skill.execute({"room": "厨房"}, ctx)

    assert result.success is True
    assert result.result_data["room"] == "kitchen"


def test_navigate_no_base_returns_error() -> None:
    """execute() without base -> 'no_base' error."""
    sg = _make_explored_sg()
    ctx = MagicMock()
    ctx.base = None
    ctx.services = {"spatial_memory": sg}

    skill = NavigateSkill()
    result = skill.execute({"room": "kitchen"}, ctx)
    assert result.success is False
    assert result.diagnosis_code == "no_base"


# ---------------------------------------------------------------------------
# L35-6: _dead_reckoning uses SceneGraph door chain
# ---------------------------------------------------------------------------


def test_dead_reckoning_uses_door_chain() -> None:
    """_dead_reckoning builds waypoints from SceneGraph doors, not hardcoded dicts.

    _dead_reckoning now prefers base.go_to_waypoint() over base.navigate_to() to
    avoid a recursive navigate_to -> FAR probe -> door-chain -> navigate_to cascade.
    The fallback is base.navigate_to() when go_to_waypoint is unavailable.
    """
    sg = _make_explored_sg()

    # Place robot at living_room position.
    # base.go_to_waypoint is available — _dead_reckoning uses it in preference to navigate_to.
    base = _make_mock_base(x=3.0, y=2.5, z=0.3)
    base.go_to_waypoint.return_value = True  # nav stack succeeds for each waypoint

    ctx = _make_skill_context(base, sg)

    skill = NavigateSkill()
    result = skill._dead_reckoning("kitchen", ctx)

    # _dead_reckoning must call base.go_to_waypoint at least once (one door in chain)
    # and must succeed overall.
    assert base.go_to_waypoint.call_count >= 1, (
        f"Expected base.go_to_waypoint() calls, got {base.go_to_waypoint.call_count}"
    )
    assert result.success


def test_dead_reckoning_no_sg_returns_error() -> None:
    """_dead_reckoning without SceneGraph returns error about explore."""
    base = _make_mock_base()
    ctx = _make_skill_context(base, None)

    skill = NavigateSkill()
    result = skill._dead_reckoning("kitchen", ctx)

    assert result.success is False
    assert "explore" in result.error_message.lower()


def test_dead_reckoning_no_door_data_returns_error() -> None:
    """_dead_reckoning with SG but no door chain -> returns error."""
    sg = SceneGraph()
    # Add rooms but NO doors (so get_door_chain returns [])
    for _ in range(5):
        sg.visit("living_room", 3.0, 2.5)
    for _ in range(5):
        sg.visit("kitchen", 17.0, 2.5)

    base = _make_mock_base(x=3.0, y=2.5, z=0.3)
    ctx = _make_skill_context(base, sg)

    skill = NavigateSkill()
    result = skill._dead_reckoning("kitchen", ctx)

    assert result.success is False
    assert "explore" in result.error_message.lower()
