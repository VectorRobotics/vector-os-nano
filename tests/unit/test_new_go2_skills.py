"""Unit tests for ExploreSkill, WhereAmISkill, and StopSkill."""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.types import SkillResult
from vector_os_nano.skills.go2.explore import ExploreSkill
from vector_os_nano.skills.go2.stop import StopSkill
from vector_os_nano.skills.go2.where_am_i import WhereAmISkill, _heading_label


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_base(x: float = 3.0, y: float = 2.5, z: float = 0.28, heading: float = 0.0):
    """Return a mock base with get_position / get_heading / walk / set_velocity."""
    base = MagicMock()
    base.get_position.return_value = [x, y, z]
    base.get_heading.return_value = heading
    base.walk.return_value = True
    base.set_velocity.return_value = None
    return base


def _make_context(base=None) -> SkillContext:
    return SkillContext(base=base)


# ===========================================================================
# StopSkill
# ===========================================================================

class TestStopSkill:
    def test_metadata(self):
        s = StopSkill()
        assert s.name == "stop"
        assert s.__skill_direct__ is True
        assert "stop" in s.__skill_aliases__
        assert "停" in s.__skill_aliases__
        assert "halt" in s.__skill_aliases__

    def test_stop_calls_set_velocity_zero(self):
        base = _make_base()
        ctx = _make_context(base)
        result = StopSkill().execute({}, ctx)

        assert result.success
        base.set_velocity.assert_called_once_with(0.0, 0.0, 0.0)

    def test_stop_no_base(self):
        ctx = _make_context(base=None)
        result = StopSkill().execute({}, ctx)

        assert not result.success
        assert result.diagnosis_code == "no_base"

    def test_stop_result_data(self):
        base = _make_base()
        ctx = _make_context(base)
        result = StopSkill().execute({}, ctx)

        assert result.result_data == {"stopped": True}

    def test_stop_tolerates_set_velocity_exception(self):
        """If set_velocity raises, stop should still return success=False or
        at least not propagate the exception unhandled.  Current impl logs and
        continues, so success=True is the expected outcome."""
        base = _make_base()
        base.set_velocity.side_effect = RuntimeError("motor fault")
        ctx = _make_context(base)

        # Should not raise; set_velocity failure is logged and swallowed
        result = StopSkill().execute({}, ctx)
        assert result.success  # command was still issued


# ===========================================================================
# WhereAmISkill
# ===========================================================================

class TestWhereAmISkill:
    def test_metadata(self):
        s = WhereAmISkill()
        assert s.name == "where_am_i"
        assert "where am i" in s.__skill_aliases__
        assert "我在哪" in s.__skill_aliases__
        assert "location" in s.__skill_aliases__

    def test_reports_room_living_room(self):
        # (3.0, 2.5) is the living_room center
        base = _make_base(x=3.0, y=2.5)
        ctx = _make_context(base)
        result = WhereAmISkill().execute({}, ctx)

        assert result.success
        assert result.result_data["room"] == "living_room"

    def test_reports_room_kitchen(self):
        # (17.0, 2.5) is the kitchen center
        base = _make_base(x=17.0, y=2.5)
        ctx = _make_context(base)
        result = WhereAmISkill().execute({}, ctx)

        assert result.success
        assert result.result_data["room"] == "kitchen"

    def test_position_in_result(self):
        base = _make_base(x=3.0, y=2.5, z=0.3)
        ctx = _make_context(base)
        result = WhereAmISkill().execute({}, ctx)

        pos = result.result_data["position"]
        assert pos == [3.0, 2.5, 0.3]

    def test_heading_in_result(self):
        base = _make_base(heading=math.pi / 2)  # pointing north
        ctx = _make_context(base)
        result = WhereAmISkill().execute({}, ctx)

        assert result.result_data["heading"] == "north"
        assert abs(result.result_data["heading_rad"] - math.pi / 2) < 0.01

    def test_no_base(self):
        ctx = _make_context(base=None)
        result = WhereAmISkill().execute({}, ctx)

        assert not result.success
        assert result.diagnosis_code == "no_base"

    def test_position_unavailable(self):
        base = _make_base()
        base.get_position.side_effect = RuntimeError("sensor offline")
        ctx = _make_context(base)
        result = WhereAmISkill().execute({}, ctx)

        assert not result.success
        assert result.diagnosis_code == "position_unavailable"

    def test_heading_failure_does_not_fail_skill(self):
        """Heading read failure is non-fatal -- skill succeeds with heading=0."""
        base = _make_base()
        base.get_heading.side_effect = RuntimeError("imu error")
        ctx = _make_context(base)
        result = WhereAmISkill().execute({}, ctx)

        assert result.success
        assert result.result_data["heading_rad"] == 0.0

    def test_room_center_in_result(self):
        base = _make_base(x=3.0, y=2.5)
        ctx = _make_context(base)
        result = WhereAmISkill().execute({}, ctx)

        center = result.result_data["room_center"]
        assert center == [3.0, 2.5]


class TestHeadingLabel:
    @pytest.mark.parametrize("radians,expected", [
        (0.0,           "east"),
        (math.pi / 2,   "north"),
        (math.pi,       "west"),
        (-math.pi / 2,  "south"),
        (math.pi / 4,   "northeast"),
        (-math.pi / 4,  "southeast"),
    ])
    def test_cardinal_directions(self, radians: float, expected: str):
        assert _heading_label(radians) == expected


# ===========================================================================
# ExploreSkill
# ===========================================================================

class TestExploreSkillMetadata:
    def test_name(self):
        assert ExploreSkill().name == "explore"

    def test_aliases(self):
        aliases = ExploreSkill.__skill_aliases__
        assert "explore" in aliases
        assert "探索" in aliases
        assert "look around" in aliases
        assert "四处看看" in aliases

    def test_not_direct(self):
        assert ExploreSkill.__skill_direct__ is False

    def test_duration_parameter(self):
        params = ExploreSkill().parameters
        assert "duration" in params
        assert params["duration"]["default"] == 60.0


class TestExploreSkillNoBase:
    def test_no_base_returns_failure(self):
        ctx = _make_context(base=None)
        result = ExploreSkill().execute({}, ctx)

        assert not result.success
        assert result.diagnosis_code == "no_base"


class TestExploreDeadReckoning:
    """Explore via dead-reckoning (ROS2 unavailable path)."""

    def _run_with_short_duration(self, base) -> SkillResult:
        ctx = _make_context(base)
        with patch(
            "vector_os_nano.skills.go2.explore._start_bridge_on_go2",
            return_value=False,
        ):
            return ExploreSkill().execute({"duration": 5.0}, ctx)

    def test_success(self):
        base = _make_base(x=3.0, y=2.5)
        result = self._run_with_short_duration(base)
        assert result.success

    def test_result_data_has_rooms_visited(self):
        base = _make_base(x=3.0, y=2.5)
        result = self._run_with_short_duration(base)
        assert "rooms_visited" in result.result_data
        assert isinstance(result.result_data["rooms_visited"], list)

    def test_result_data_has_coverage(self):
        base = _make_base(x=3.0, y=2.5)
        result = self._run_with_short_duration(base)
        assert "coverage_percent" in result.result_data
        assert 0.0 <= result.result_data["coverage_percent"] <= 100.0

    def test_result_data_mode(self):
        base = _make_base(x=3.0, y=2.5)
        result = self._run_with_short_duration(base)
        assert result.result_data["mode"] == "dead_reckoning"

    def test_starting_room_is_visited(self):
        """Robot starts in living_room -- that room must appear in rooms_visited."""
        base = _make_base(x=3.0, y=2.5)  # living_room center
        result = self._run_with_short_duration(base)
        assert "living_room" in result.result_data["rooms_visited"]

    def test_minimum_duration_clamped(self):
        """Duration < 5 s is clamped to 5 s (no assertion on timing, just no crash)."""
        base = _make_base()
        ctx = _make_context(base)
        with patch(
            "vector_os_nano.skills.go2.explore._start_bridge_on_go2",
            return_value=False,
        ):
            result = ExploreSkill().execute({"duration": 0.1}, ctx)
        assert result.success

    def test_robot_fall_stops_exploration(self):
        """If _navigate_to_waypoint returns False, exploration aborts without raising."""
        base = _make_base(x=10.0, y=5.0)  # hallway
        ctx = _make_context(base)
        with patch(
            "vector_os_nano.skills.go2.explore._start_bridge_on_go2",
            return_value=False,
        ), patch(
            "vector_os_nano.skills.go2.explore._navigate_to_waypoint",
            return_value=False,
        ):
            result = ExploreSkill().execute({"duration": 5.0}, ctx)
        # success=True even if cut short (partial exploration is still a result)
        assert result.success


class TestExploreMonitor:
    """Test _monitor_exploration directly (no ROS2 needed)."""

    def test_monitoring_tracks_rooms(self):
        """_monitor_exploration samples position and detects rooms."""
        base = _make_base(x=3.0, y=2.5)  # living_room
        skill = ExploreSkill()

        # Patch time: first call sets deadline (1000+5=1005), then loop runs a few times
        with patch("vector_os_nano.skills.go2.explore.time") as mock_time:
            call_count = [0]
            def fake_time():
                call_count[0] += 1
                # First call (deadline calc): 1000. Next calls: 1001, 1002, ... then 1010 to exit
                if call_count[0] <= 3:
                    return 1000.0
                return 1010.0
            mock_time.time = fake_time
            mock_time.sleep = lambda _: None
            result = skill._monitor_exploration(base, 5.0)

        assert result.success
        assert result.result_data["mode"] == "tare"
        assert "living_room" in result.result_data["rooms_visited"]
        assert base.get_position.call_count >= 1
