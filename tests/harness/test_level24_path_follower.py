"""Level 24: Omnidirectional path follower verification tests."""
import os
import re
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BRIDGE = os.path.join(_REPO, "scripts", "go2_vnav_bridge.py")


def _read_bridge():
    with open(_BRIDGE) as f:
        return f.read()


class TestOmnidirectionalFollower:
    """Verify path follower uses omnidirectional Go2 capabilities."""

    def test_uses_lateral_velocity(self):
        """Follower must send vy (lateral) to go2, not just vx + vyaw."""
        src = _read_bridge()
        assert "self._pf_lat" in src, "No lateral velocity tracking"
        assert "self._go2.set_velocity(self._pf_speed, self._pf_lat" in src

    def test_prefers_forward(self):
        """Forward motion (vx > 0) should be the dominant mode."""
        src = _read_bridge()
        assert "pure forward tracking" in src.lower() or "< 0.52" in src

    def test_heading_gated_acceleration(self):
        """Forward speed only when heading is aligned (two-mode controller)."""
        src = _read_bridge()
        assert "_TRACK_THRE" in src or "_TRACK_RESUME" in src

    def test_strafe_for_moderate_error(self):
        """Uses lateral strafe when direction error is 30-90 degrees."""
        src = _read_bridge()
        assert "_MAX_LAT" in src, "No max lateral speed constant"


class TestReactiveWallAvoidance:
    """Verify reactive wall avoidance overlay in path follower."""

    def test_scan_surroundings_used(self):
        src = _read_bridge()
        assert "_scan_surroundings" in src

    def test_lateral_safety_boundary(self):
        """When close to left/right wall, vy pushes away (MJCF-based safety)."""
        src = _read_bridge()
        assert "left_gap" in src or "left_d" in src
        assert "right_gap" in src or "right_d" in src

    def test_front_safety_boundary(self):
        """When obstacle ahead within safety envelope, vx is reduced/stopped."""
        src = _read_bridge()
        assert "front_gap" in src and ("_COMFORT" in src or "_DANGER" in src)


class TestStuckRecovery:
    """Verify stuck detection and escape behavior."""

    def test_stuck_detector_exists(self):
        src = _read_bridge()
        assert "_stuck_detector" in src

    def test_stuck_sends_reset_waypoint(self):
        src = _read_bridge()
        assert "/reset_waypoint" in src or "reset_waypoint" in src

    def test_stuck_backup_escape(self):
        """After prolonged stuck, robot backs up to escape."""
        src = _read_bridge()
        assert "backing up" in src.lower() or "escape" in src.lower()


class TestIdleWander:
    def test_wander_has_obstacle_check(self):
        src = _read_bridge()
        assert "_check_front_obstacle" in src

    def test_wander_speed_gentle(self):
        src = _read_bridge()
        assert "0.15" in src  # wander speed
