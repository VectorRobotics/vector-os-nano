# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Unit tests for ExploreSkill, WhereAmISkill, and StopSkill."""
from __future__ import annotations

import math
import os
import signal
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
        assert "stop" in s.__skill_aliases__
        assert "停" in s.__skill_aliases__
        assert "halt" in s.__skill_aliases__

    def test_stop_calls_the_ungated_base_stop(self):
        # skills-nav:inprocess-nav-no-abort-stall: set_velocity from the stop's thread
        # is IGNORED while a walk/navigate on another thread holds the skill token.
        base = _make_base()
        ctx = _make_context(base)
        result = StopSkill().execute({}, ctx)

        assert result.success
        base.stop.assert_called_once_with()
        base.set_velocity.assert_not_called()

    def test_stop_falls_back_to_set_velocity_without_stop(self):
        base = MagicMock(spec=["set_velocity", "get_position"])
        result = StopSkill().execute({}, _make_context(base))
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
        """If the base stop raises, stop logs it and does not propagate the
        exception unhandled, so success=True is the expected outcome."""
        base = _make_base()
        base.stop.side_effect = RuntimeError("motor fault")
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

    def test_only_parameter_is_the_backend(self):
        params = ExploreSkill().parameters
        assert list(params) == ["backend"]
        assert params["backend"]["enum"] == ["coverage", "tare"]


class TestExploreSkillNoBase:
    def test_no_base_returns_failure(self):
        ctx = _make_context(base=None)
        result = ExploreSkill().execute({}, ctx)

        assert not result.success
        assert result.diagnosis_code == "no_base"


class TestExploreNonBlocking:
    """Non-blocking explore — starts background thread, returns immediately."""

    def test_no_backend_fails_loud_instead_of_a_false_green(self):
        # skills-nav:explore-false-green-no-tare: with no TARE it returned
        # success 'exploration_started' and did nothing.
        base = MagicMock(spec=["get_position", "get_heading", "walk", "set_velocity", "stop"])
        base.get_position.return_value = [10.0, 5.0, 0.3]
        result = ExploreSkill().execute({}, _make_context(base))
        assert not result.success and result.diagnosis_code == "no_backend"
        from vector_os_nano.skills.go2.explore import is_exploring
        assert not is_exploring()

    def test_tare_requested_but_not_ours_fails_loud(self):
        result = ExploreSkill().execute({"backend": "tare"}, _make_context(_make_base()))
        assert not result.success and result.diagnosis_code == "no_backend"

    def test_unknown_backend_fails_with_the_valid_set(self):
        result = ExploreSkill().execute({"backend": "frontier"}, _make_context(_make_base()))
        assert not result.success and "coverage" in result.error_message

    def test_already_exploring_returns_status(self):
        """Second call while exploring reports current status."""
        from vector_os_nano.skills.go2.explore import (
            _explore_running, cancel_exploration,
        )
        import vector_os_nano.skills.go2.explore as _mod
        _mod._explore_running = True
        _mod._explore_visited = {"hallway"}
        try:
            base = _make_base(x=10.0, y=5.0)
            ctx = _make_context(base)
            result = ExploreSkill().execute({}, ctx)
            assert result.success
            assert result.result_data["status"] == "already_exploring"
            assert "hallway" in result.result_data["rooms_visited"]
        finally:
            _mod._explore_running = False
            _mod._explore_visited.clear()

    def test_cancel_exploration(self):
        """cancel_exploration sets the cancel event."""
        from vector_os_nano.skills.go2.explore import (
            cancel_exploration, _explore_cancel,
        )
        import vector_os_nano.skills.go2.explore as _mod
        _mod._explore_running = True
        cancel_exploration()
        assert _explore_cancel.is_set()
        _mod._explore_running = False
        _explore_cancel.clear()

    def test_tare_stop_is_scoped_to_our_own_descendants(self):
        """stop_tare_only finds TARE by name only below this process: never a
        machine-wide pkill (other projects' planners), and never a no-op either
        (TARE kept publishing /way_point after cancel)."""
        import subprocess
        import sys

        from vector_os_nano.skills.go2.explore import _own_descendants_named

        sleeper = "import time; time.sleep(60)"
        spawner = (
            "import subprocess, sys; "
            f"p = subprocess.Popen([sys.executable, '-c', {sleeper!r}, 'tare_planner_node']); "
            "print(p.pid, flush=True); p.wait()"
        )
        parent = subprocess.Popen([sys.executable, "-c", spawner], stdout=subprocess.PIPE, text=True)
        theirs = subprocess.Popen([sys.executable, "-c", sleeper, "tare_planner_node"])
        ours = int(parent.stdout.readline())
        try:
            assert _own_descendants_named("tare_planner_node", root=parent.pid) == [ours]
            assert {ours, theirs.pid} <= set(_own_descendants_named("tare_planner_node"))
        finally:
            os.kill(ours, signal.SIGKILL)  # the parent's wait() reaps it, then it exits
            parent.wait(timeout=10)
            theirs.kill()
            theirs.wait()

    def test_get_explored_rooms(self):
        """get_explored_rooms returns sorted room list."""
        from vector_os_nano.skills.go2.explore import get_explored_rooms
        import vector_os_nano.skills.go2.explore as _mod
        _mod._explore_visited = {"kitchen", "hallway"}
        assert get_explored_rooms() == ["hallway", "kitchen"]
        _mod._explore_visited.clear()
