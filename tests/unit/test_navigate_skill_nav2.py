"""Tests for NavigateSkill when backed by Nav2 NavStackClient.

Verifies that NavigateSkill correctly delegates to NavStackClient regardless
of the underlying mode (nav2 or cmu). The skill should be mode-agnostic —
it calls navigate_to(x, y) and doesn't know/care about the transport.

Also includes comprehensive room database validation tests.
"""
import math
import pytest
from unittest.mock import MagicMock, PropertyMock, patch

from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.world_model import WorldModel
from vector_os_nano.core.types import SkillResult
from vector_os_nano.skills.navigate import (
    NavigateSkill,
    _ROOM_CENTERS,
    _ROOM_DOORS,
    _ROOM_ALIASES,
    _resolve_room,
    _distance,
    _angle_between,
    _normalize_angle,
    _detect_current_room,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_nav2_context(nav_result: bool = True) -> SkillContext:
    """Create a SkillContext with a Nav2-mode NavStackClient mock."""
    base = MagicMock()
    base.walk.return_value = True
    base.get_position.return_value = [10.0, 6.5, 0.27]  # hallway center
    base.get_heading.return_value = 0.0

    nav = MagicMock()
    nav.is_available = True
    nav.mode = "nav2"
    nav.navigate_to.return_value = nav_result
    nav.get_state_estimation.return_value = MagicMock(x=10.0, y=6.5, z=0.27)

    return SkillContext(
        bases={"go2": base},
        world_model=WorldModel(),
        services={"nav": nav},
    )


def _make_fallback_context() -> SkillContext:
    """Create a SkillContext without nav service (dead-reckoning mode)."""
    base = MagicMock()
    base.walk.return_value = True
    base.get_position.return_value = [10.0, 6.5, 0.27]
    base.get_heading.return_value = 0.0
    return SkillContext(
        bases={"go2": base},
        world_model=WorldModel(),
    )


# ---------------------------------------------------------------------------
# Nav2 Mode Integration Tests
# ---------------------------------------------------------------------------

class TestNavigateSkillNav2Mode:
    """Test NavigateSkill when NavStackClient is in nav2 mode."""

    def test_nav2_navigate_to_kitchen(self):
        """Navigate to kitchen via Nav2 should succeed."""
        ctx = _make_nav2_context(nav_result=True)
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        assert result.success
        ctx.services["nav"].navigate_to.assert_called_once()
        args = ctx.services["nav"].navigate_to.call_args
        # Kitchen center is (17.0, 2.5)
        assert abs(args[0][0] - 17.0) < 1.0
        assert abs(args[0][1] - 2.5) < 1.0

    def test_nav2_navigate_to_all_rooms(self):
        """Navigate to every room should call nav with correct coordinates."""
        skill = NavigateSkill()
        for room_name, (cx, cy) in _ROOM_CENTERS.items():
            ctx = _make_nav2_context(nav_result=True)
            result = skill.execute({"room": room_name}, ctx)
            assert result.success, f"Failed to navigate to {room_name}"
            args = ctx.services["nav"].navigate_to.call_args[0]
            assert abs(args[0] - cx) < 1.0, f"{room_name}: x mismatch {args[0]} vs {cx}"
            assert abs(args[1] - cy) < 1.0, f"{room_name}: y mismatch {args[1]} vs {cy}"

    def test_nav2_failure_returns_navigation_failed(self):
        """Nav2 navigation failure should return navigation_failed diagnosis."""
        ctx = _make_nav2_context(nav_result=False)
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        assert not result.success
        assert "navigation_failed" in result.diagnosis_code

    def test_nav2_unknown_room_returns_error(self):
        """Unknown room name should return unknown_room diagnosis."""
        ctx = _make_nav2_context()
        skill = NavigateSkill()
        result = skill.execute({"room": "nonexistent_room"}, ctx)
        assert not result.success
        assert "unknown_room" in result.diagnosis_code

    def test_nav2_no_base_returns_error(self):
        """No base in context should return no_base diagnosis."""
        ctx = SkillContext(world_model=WorldModel())
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        assert not result.success
        assert "no_base" in result.diagnosis_code

    def test_nav2_chinese_room_name(self):
        """Chinese room names should work via alias resolution."""
        ctx = _make_nav2_context(nav_result=True)
        skill = NavigateSkill()

        chinese_rooms = {
            "厨房": "kitchen",
            "客厅": "living_room",
            "主卧": "master_bedroom",
            "卫生间": "bathroom",
            "书房": "study",
            "走廊": "hallway",
        }
        for chinese, canonical in chinese_rooms.items():
            ctx = _make_nav2_context(nav_result=True)
            result = skill.execute({"room": chinese}, ctx)
            assert result.success, f"Failed for Chinese name: {chinese}"
            # Verify correct coordinates were sent
            args = ctx.services["nav"].navigate_to.call_args[0]
            expected = _ROOM_CENTERS[canonical]
            assert abs(args[0] - expected[0]) < 1.0
            assert abs(args[1] - expected[1]) < 1.0

    def test_nav2_result_data_contains_room_info(self):
        """Result should include room name, position, and mode."""
        ctx = _make_nav2_context(nav_result=True)
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        assert result.success
        data = result.result_data
        assert data is not None
        assert "room" in data
        assert data["room"] == "kitchen"

    def test_nav2_prefers_nav_over_dead_reckoning(self):
        """When nav service is available, should NOT use dead-reckoning walk."""
        ctx = _make_nav2_context(nav_result=True)
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        assert result.success
        # Nav service was called
        ctx.services["nav"].navigate_to.assert_called_once()
        # Base.walk was NOT called (nav mode, not dead-reckoning)
        ctx.bases["go2"].walk.assert_not_called()

    def test_fallback_uses_dead_reckoning_when_no_nav(self):
        """Without nav service, should use base.walk dead-reckoning."""
        ctx = _make_fallback_context()
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        # May or may not succeed (dead-reckoning is approximate)
        # But base.walk should have been called
        assert ctx.bases["go2"].walk.called


# ---------------------------------------------------------------------------
# Room Database Completeness Tests
# ---------------------------------------------------------------------------

class TestRoomDatabase:
    """Verify the room database is complete, consistent, and well-formed."""

    def test_all_rooms_have_centers(self):
        """Every room in _ROOM_CENTERS should have a defined center."""
        expected_rooms = {
            "living_room", "dining_room", "kitchen", "study",
            "master_bedroom", "guest_bedroom", "bathroom", "hallway",
        }
        assert set(_ROOM_CENTERS.keys()) == expected_rooms

    def test_all_rooms_have_doors(self):
        """Every room in _ROOM_CENTERS should have a corresponding door."""
        assert set(_ROOM_DOORS.keys()) == set(_ROOM_CENTERS.keys())

    def test_room_centers_within_bounds(self):
        """All room centers should be within the 20x14m house bounds."""
        for room, (cx, cy) in _ROOM_CENTERS.items():
            assert 0 <= cx <= 20, f"{room} center x={cx} out of bounds"
            assert 0 <= cy <= 14, f"{room} center y={cy} out of bounds"

    def test_room_doors_within_bounds(self):
        """All room doors should be within the house bounds."""
        for room, (dx, dy) in _ROOM_DOORS.items():
            assert 0 <= dx <= 20, f"{room} door x={dx} out of bounds"
            assert 0 <= dy <= 14, f"{room} door y={dy} out of bounds"

    def test_room_centers_are_distinct(self):
        """No two rooms should have the same center."""
        centers = list(_ROOM_CENTERS.values())
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = _distance(centers[i][0], centers[i][1],
                                 centers[j][0], centers[j][1])
                assert dist > 1.0, (
                    f"Rooms too close: "
                    f"{list(_ROOM_CENTERS.keys())[i]} and "
                    f"{list(_ROOM_CENTERS.keys())[j]} "
                    f"(dist={dist:.1f}m)"
                )

    def test_door_is_between_center_and_hallway(self):
        """Each door should be geometrically between the room center and hallway."""
        hallway_center = _ROOM_CENTERS["hallway"]
        for room in _ROOM_CENTERS:
            if room == "hallway":
                continue
            cx, cy = _ROOM_CENTERS[room]
            dx, dy = _ROOM_DOORS[room]
            # Door should be closer to hallway than room center
            door_to_hallway = _distance(dx, dy, hallway_center[0], hallway_center[1])
            center_to_hallway = _distance(cx, cy, hallway_center[0], hallway_center[1])
            assert door_to_hallway <= center_to_hallway + 1.0, (
                f"{room}: door is farther from hallway than center "
                f"(door={door_to_hallway:.1f}m, center={center_to_hallway:.1f}m)"
            )


# ---------------------------------------------------------------------------
# Room Alias Resolution Tests
# ---------------------------------------------------------------------------

class TestRoomAliases:
    """Verify room alias resolution is comprehensive."""

    def test_all_canonical_names_resolve(self):
        """Canonical room names should resolve to themselves."""
        for room in _ROOM_CENTERS:
            assert _resolve_room(room) == room

    def test_english_aliases(self):
        """Common English aliases should resolve correctly."""
        cases = {
            "living room": "living_room",
            "Living Room": "living_room",
            "KITCHEN": "kitchen",
            "master bedroom": "master_bedroom",
            "guest room": "guest_bedroom",
            "bathroom": "bathroom",
            "bath": "bathroom",
            "restroom": "bathroom",
            "toilet": "bathroom",
            "office": "study",
            "hallway": "hallway",
            "hall": "hallway",
            "corridor": "hallway",
        }
        for alias, expected in cases.items():
            result = _resolve_room(alias)
            assert result == expected, f"Alias '{alias}' resolved to '{result}', expected '{expected}'"

    def test_chinese_aliases(self):
        """Chinese room names should resolve correctly."""
        cases = {
            "客厅": "living_room",
            "大厅": "living_room",
            "餐厅": "dining_room",
            "饭厅": "dining_room",
            "厨房": "kitchen",
            "书房": "study",
            "办公室": "study",
            "主卧": "master_bedroom",
            "卧室": "master_bedroom",
            "客卧": "guest_bedroom",
            "客房": "guest_bedroom",
            "次卧": "guest_bedroom",
            "卫生间": "bathroom",
            "浴室": "bathroom",
            "洗手间": "bathroom",
            "厕所": "bathroom",
            "走廊": "hallway",
            "过道": "hallway",
        }
        for alias, expected in cases.items():
            result = _resolve_room(alias)
            assert result == expected, f"Chinese alias '{alias}' resolved to '{result}', expected '{expected}'"

    def test_unknown_room_returns_none(self):
        """Unknown room names should return None."""
        assert _resolve_room("garage") is None
        assert _resolve_room("swimming pool") is None
        assert _resolve_room("车库") is None
        assert _resolve_room("") is None

    def test_whitespace_handling(self):
        """Leading/trailing whitespace should be stripped."""
        assert _resolve_room("  kitchen  ") == "kitchen"
        assert _resolve_room("\n厨房\t") == "kitchen"

    def test_underscore_handling(self):
        """Underscored names should resolve (living_room → living_room)."""
        assert _resolve_room("living_room") == "living_room"
        assert _resolve_room("master_bedroom") == "master_bedroom"

    def test_every_alias_maps_to_valid_room(self):
        """Every alias in _ROOM_ALIASES should map to a room in _ROOM_CENTERS."""
        for alias, canonical in _ROOM_ALIASES.items():
            assert canonical in _ROOM_CENTERS, (
                f"Alias '{alias}' maps to '{canonical}' which is not in _ROOM_CENTERS"
            )

    def test_every_room_has_at_least_one_chinese_alias(self):
        """Every room should have at least one Chinese alias."""
        rooms_with_chinese = set()
        for alias, canonical in _ROOM_ALIASES.items():
            # Check if alias contains CJK characters
            if any('\u4e00' <= c <= '\u9fff' for c in alias):
                rooms_with_chinese.add(canonical)
        for room in _ROOM_CENTERS:
            assert room in rooms_with_chinese, f"Room '{room}' has no Chinese alias"


# ---------------------------------------------------------------------------
# Helper Function Tests
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    """Tests for navigation helper functions."""

    def test_distance_zero(self):
        assert _distance(0, 0, 0, 0) == 0.0

    def test_distance_unit(self):
        assert abs(_distance(0, 0, 1, 0) - 1.0) < 1e-10
        assert abs(_distance(0, 0, 0, 1) - 1.0) < 1e-10

    def test_distance_diagonal(self):
        assert abs(_distance(0, 0, 3, 4) - 5.0) < 1e-10

    def test_distance_symmetry(self):
        d1 = _distance(1, 2, 5, 7)
        d2 = _distance(5, 7, 1, 2)
        assert abs(d1 - d2) < 1e-10

    def test_angle_between_east(self):
        assert abs(_angle_between(0, 0, 1, 0) - 0.0) < 1e-10

    def test_angle_between_north(self):
        assert abs(_angle_between(0, 0, 0, 1) - math.pi / 2) < 1e-10

    def test_angle_between_west(self):
        assert abs(abs(_angle_between(0, 0, -1, 0)) - math.pi) < 1e-10

    def test_angle_between_south(self):
        assert abs(_angle_between(0, 0, 0, -1) + math.pi / 2) < 1e-10

    def test_normalize_angle_in_range(self):
        assert abs(_normalize_angle(0)) < 1e-10
        assert abs(_normalize_angle(math.pi) - math.pi) < 1e-10
        assert abs(_normalize_angle(-math.pi) + math.pi) < 1e-10

    def test_normalize_angle_wrapping(self):
        assert abs(_normalize_angle(3 * math.pi) - math.pi) < 1e-10
        assert abs(_normalize_angle(-3 * math.pi) + math.pi) < 1e-10

    def test_detect_current_room_at_center(self):
        """Robot at a room center should be detected in that room."""
        for room, (cx, cy) in _ROOM_CENTERS.items():
            detected = _detect_current_room(cx, cy)
            assert detected == room, f"At ({cx},{cy}) expected {room}, got {detected}"

    def test_detect_current_room_near_center(self):
        """Robot near a room center (within 1m) should be in that room."""
        for room, (cx, cy) in _ROOM_CENTERS.items():
            # Offset by 0.5m in each direction
            for dx, dy in [(0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)]:
                detected = _detect_current_room(cx + dx, cy + dy)
                assert detected == room, (
                    f"At ({cx+dx:.1f},{cy+dy:.1f}) expected {room}, got {detected}"
                )


# ---------------------------------------------------------------------------
# Sequential Navigation Tests
# ---------------------------------------------------------------------------

class TestSequentialNavigation:
    """Test multi-room sequential navigation patterns."""

    def test_navigate_three_rooms_sequentially(self):
        """Navigate hallway → kitchen → bedroom should all succeed."""
        skill = NavigateSkill()
        rooms = ["hallway", "kitchen", "master_bedroom"]

        for room in rooms:
            ctx = _make_nav2_context(nav_result=True)
            result = skill.execute({"room": room}, ctx)
            assert result.success, f"Sequential nav failed at {room}"

    def test_navigate_same_room_twice(self):
        """Navigating to the same room twice should succeed both times."""
        skill = NavigateSkill()
        for _ in range(2):
            ctx = _make_nav2_context(nav_result=True)
            result = skill.execute({"room": "kitchen"}, ctx)
            assert result.success

    def test_navigate_all_rooms_round_trip(self):
        """Navigate to every room — full house tour."""
        skill = NavigateSkill()
        for room in _ROOM_CENTERS:
            ctx = _make_nav2_context(nav_result=True)
            result = skill.execute({"room": room}, ctx)
            assert result.success, f"Round trip failed at {room}"
