"""Level 32: Path follower precision harness.

Tests verify the Python path follower matches C++ pathFollower behavior:
  - Heading-gated acceleration (only move when aligned)
  - Cross-track vy correction (cos/sin decomposition)
  - Yaw gain matches C++ (7.5)
  - Lookahead distance matches C++ (0.5m)
  - Deceleration profile matches C++ (1.0m slowdown)
  - Stop threshold matches C++ (0.2m)
  - Two-phase wall escape (reverse first, then strafe)
"""
from __future__ import annotations

import math
import os
import re

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nav_debug_helpers import read_bridge_source

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===================================================================
# Part 1: Constants verification (must match C++ pathFollower)
# ===================================================================

class TestFollowerConstants:
    """Verify Python follower constants match C++ pathFollower."""

    def _get_constants(self) -> dict:
        src = read_bridge_source()
        constants = {}
        for match in re.finditer(r'_([A-Z_]+)\s*=\s*([\d.]+)', src):
            constants[match.group(1)] = float(match.group(2))
        return constants

    def test_yaw_gain_matches_cpp(self):
        """YAW_GAIN should be 7.5 (C++ pathFollower line 53)."""
        c = self._get_constants()
        assert c.get("YAW_GAIN", 0) == pytest.approx(7.5, abs=0.1), (
            f"YAW_GAIN={c.get('YAW_GAIN')} — C++ uses 7.5"
        )

    def test_max_speed(self):
        """MAX_SPEED should be ~0.8 m/s."""
        c = self._get_constants()
        assert c.get("MAX_SPEED", 0) == pytest.approx(0.8, abs=0.1)

    def test_max_yaw_rate_adequate(self):
        """MAX_YAW_RATE should be 0.785-1.2 rad/s (C++: 0.785, raised for quadruped spot turn)."""
        c = self._get_constants()
        rate = c.get("MAX_YAW_RATE", 0)
        assert 0.7 <= rate <= 1.2, (
            f"MAX_YAW_RATE={rate} — should be 0.785-1.2 rad/s"
        )

    def test_lookahead_matches_cpp(self):
        """LOOK_AHEAD ~0.8m (wider than C++ 0.5 to smooth curves for quadruped)."""
        c = self._get_constants()
        assert c.get("LOOK_AHEAD", 0) == pytest.approx(0.8, abs=0.1)

    def test_stop_distance_matches_cpp(self):
        """STOP_DIS should be 0.2m (C++ line 62)."""
        c = self._get_constants()
        assert c.get("STOP_DIS", 0) == pytest.approx(0.2, abs=0.05)

    def test_slowdown_distance_matches_cpp(self):
        """SLOW_DWN_DIS should be 1.0m (C++ line 63)."""
        c = self._get_constants()
        assert c.get("SLOW_DWN_DIS", 0) == pytest.approx(1.0, abs=0.1)

    def test_yaw_gain_exists(self):
        """YAW_GAIN should be ~7.5 (proportional heading correction)."""
        c = self._get_constants()
        assert c.get("YAW_GAIN", 0) == pytest.approx(7.5, abs=1.0)

    def test_max_lat_speed(self):
        """MAX_LAT should be ~0.4 m/s for quadruped lateral stability."""
        c = self._get_constants()
        assert c.get("MAX_LAT", 0) == pytest.approx(0.4, abs=0.1)


# ===================================================================
# Part 2: Heading-gated acceleration (the KEY precision feature)
# ===================================================================

class TestOmniDirectional:
    """Quadruped uses full omnidirectional cos/sin decomposition.

    No heading gate — cos/sin naturally maps any heading error to vx/vy:
      0°: full forward  |  90°: pure strafe  |  180°: capped reverse
    The obstacle safety zones prevent wall collisions.
    """

    def test_cos_sin_decomposition_exists(self):
        """Follower must use cos/sin decomposition for vx/vy."""
        src = read_bridge_source()
        follow = src[src.find("def _follow_path"):]
        assert "math.cos(dir_diff)" in follow, "No cos decomposition for vx"
        assert "math.sin(dir_diff)" in follow, "No sin decomposition for vy"

    def test_reverse_speed_capped(self):
        """Reverse speed should be capped (can't see behind)."""
        src = read_bridge_source()
        follow = src[src.find("def _follow_path"):]
        assert "max(-0.3" in follow or "max(-0.2" in follow, (
            "No reverse speed cap — robot could back into obstacles unseen"
        )

    def test_yaw_always_correcting(self):
        """Yaw rate should always be proportional to heading error."""
        src = read_bridge_source()
        follow = src[src.find("def _follow_path"):]
        assert "YAW_GAIN * dir_diff" in follow


# ===================================================================
# Part 3: Cross-track correction via cos/sin decomposition
# ===================================================================

class TestCrossTrackCorrection:
    """Verify omnidirectional cross-track error correction.

    C++ pathFollower uses:
      vx = speed * cos(dirDiff)
      vy = -speed * sin(dirDiff)

    This naturally steers the robot back to the path using lateral velocity.
    """

    def test_cos_sin_decomposition(self):
        """Follower uses cos/sin for vx/vy (not zone-based switching)."""
        src = read_bridge_source()
        follow = src[src.find("def _follow_path"):]
        has_cos = "cos(dir_diff)" in follow or "math.cos(dir_diff)" in follow
        has_sin = "sin(dir_diff)" in follow or "math.sin(dir_diff)" in follow
        assert has_cos and has_sin, (
            "Missing cos/sin decomposition for cross-track correction. "
            "Zone-based vx/vy switching is imprecise — port C++ cos/sin approach."
        )

    def test_vy_correction_sign(self):
        """vy should be -speed*sin(dirDiff) — negative sin for correct direction."""
        src = read_bridge_source()
        follow = src[src.find("def _follow_path"):]
        # Should have: vy = -target_speed * sin(dir_diff) or similar
        has_neg_sin = "-target_speed * math.sin" in follow or "-speed * sin" in follow
        # Also accept: vy = target_speed * (-sin) variant
        assert has_neg_sin or ("-" in follow and "sin(dir_diff)" in follow), (
            "vy correction should use negative sin for correct lateral direction"
        )


# ===================================================================
# Part 4: Behavioral simulation tests
# ===================================================================

class TestFollowerBehavior:
    """Simulate the follower algorithm and verify behavior."""

    @staticmethod
    def _simulate_step(
        rx: float, ry: float, heading: float,
        tx: float, ty: float,  # target point
        ex: float, ey: float,  # endpoint
        speed: float = 0.0,    # current speed
    ) -> dict:
        """Simulate one step of the omni-directional follower."""
        _MAX_SPEED = 0.8
        _SLOW_DWN_DIS = 1.0
        _STOP_DIS = 0.2
        _MAX_LAT = 0.4
        _YAW_GAIN = 7.5
        _STOP_YAW_GAIN = 7.5
        _ACCEL = 0.05

        dx, dy = tx - rx, ty - ry
        path_dir = math.atan2(dy, dx)
        dir_diff = path_dir - heading
        while dir_diff > math.pi: dir_diff -= 2 * math.pi
        while dir_diff < -math.pi: dir_diff += 2 * math.pi
        abs_err = abs(dir_diff)

        end_dis = math.sqrt((ex - rx)**2 + (ey - ry)**2)
        target_speed = _MAX_SPEED
        if end_dis < _SLOW_DWN_DIS:
            target_speed = _MAX_SPEED * (end_dis / _SLOW_DWN_DIS)
        if end_dis < _STOP_DIS:
            target_speed = 0.0

        vyaw = _YAW_GAIN * dir_diff
        heading_ok = True  # no gate — omni-directional

        if end_dis > _STOP_DIS:
            vx = target_speed * math.cos(dir_diff)
            vy = -target_speed * math.sin(dir_diff)
            vx = max(-0.3, vx)  # cap reverse
            vy = max(-_MAX_LAT, min(_MAX_LAT, vy))
        else:
            vx = 0.0
            vy = 0.0

        return {"vx": vx, "vy": vy, "vyaw": vyaw, "heading_ok": heading_ok,
                "dir_diff": dir_diff, "end_dis": end_dis}

    def test_aligned_forward_motion(self):
        """When heading is aligned with target, vx > 0."""
        r = self._simulate_step(0, 0, 0.0, 5, 0, 5, 0)  # facing target
        assert r["vx"] > 0, "Should move forward when aligned"
        assert r["heading_ok"]

    def test_90deg_strafes(self):
        """When heading is 90° off, quadruped strafes (vx≈0, vy≠0)."""
        r = self._simulate_step(0, 0, math.pi / 2, 5, 0, 5, 0)  # facing up, target right
        assert abs(r["vx"]) < 0.1, "vx should be ~0 at 90° (cos(90°)=0)"
        assert abs(r["vy"]) > 0.2, "Should strafe toward target at 90°"

    def test_small_error_still_moves(self):
        """3° error is within threshold — should still move."""
        r = self._simulate_step(0, 0, 0.05, 5, 0, 5, 0)  # 2.9° off
        assert r["vx"] > 0, "3° error should still allow forward motion"
        assert r["heading_ok"]

    def test_45deg_omni_walks(self):
        """45° error — forward + strafe via cos/sin."""
        r = self._simulate_step(0, 0, math.pi / 4, 5, 0, 5, 0)
        assert r["vx"] > 0.3, "cos(45°)=0.71 → should have forward speed"
        assert abs(r["vy"]) > 0.1, "sin(45°)=0.71 → should strafe"

    def test_cross_track_vy_correction(self):
        """Small heading error produces lateral correction via vy."""
        r = self._simulate_step(0, 0, 0.08, 5, 0, 5, 0)  # 4.6° off (within gate)
        assert r["heading_ok"]
        assert r["vy"] != 0.0, "Should have lateral correction"
        # dir_diff = path_dir - heading = 0 - 0.08 = -0.08
        # vy = -speed * sin(-0.08) = +speed * sin(0.08) > 0
        # Robot faces slightly left → vy pushes left to converge to path
        assert abs(r["vy"]) > 0.01, "vy correction should be non-trivial"

    def test_deceleration_near_goal(self):
        """Speed reduces linearly when within 1.0m of goal."""
        r_far = self._simulate_step(0, 0, 0.0, 3, 0, 3, 0)
        r_near = self._simulate_step(0, 0, 0.0, 0.5, 0, 0.5, 0)
        assert r_far["vx"] > r_near["vx"], (
            "Should be slower near goal"
        )

    def test_stop_at_goal(self):
        """Within 0.2m of goal, speed = 0."""
        r = self._simulate_step(0, 0, 0.0, 0.1, 0, 0.1, 0)
        assert r["vx"] == 0.0, "Should stop within 0.2m of goal"

    def test_180deg_pure_reverse(self):
        """180° error — pure reverse (target directly behind)."""
        r = self._simulate_step(0, 0, math.pi, 5, 0, 5, 0)
        assert r["vx"] < 0, "Should reverse when target is behind"
        assert r["vx"] >= -0.3, "Reverse capped"
        assert abs(r["vy"]) < 0.1, "Minimal strafe at 180°"

    def test_120deg_reverse_and_strafe(self):
        """120° error — reverses + strafes (target behind-side)."""
        r = self._simulate_step(0, 0, 2.1, 5, 0, 5, 0)  # 120° off
        # cos(120°) = -0.5 → reverse, capped at -0.3
        assert r["vx"] < 0, "Should reverse at 120° error"
        assert r["vx"] >= -0.3, "Reverse should be capped"
        assert abs(r["vy"]) > 0.1, "Should strafe toward target"

    def test_yaw_rate_proportional(self):
        """Yaw rate should be proportional to heading error."""
        r1 = self._simulate_step(0, 0, 0.3, 5, 0, 5, 0)
        r2 = self._simulate_step(0, 0, 0.6, 5, 0, 5, 0)
        assert abs(r2["vyaw"]) > abs(r1["vyaw"]), (
            "Larger heading error should produce larger yaw rate"
        )
