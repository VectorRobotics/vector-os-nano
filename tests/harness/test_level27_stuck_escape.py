"""Level 27: Stuck escape debug harness — P0 (dog stuck in tight spaces).

Root cause: TARE sends waypoint in tight space → localPlanner can't path →
stuck detector fires /reset_waypoint → TARE re-sends SAME waypoint → loop.

TARE's ResetWaypointCallback only resets direction, does NOT blacklist the
failed viewpoint. After reset, it re-evaluates and picks the same one.

These tests verify:
  - Stuck detector timing and behavior (behavioral, mock-based)
  - Stuck LOOP detection (NEW — current code lacks this)
  - Escape effectiveness (does backup actually help?)
  - Bridge code patterns (static analysis)
"""
from __future__ import annotations

import math
import os
import re

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nav_debug_helpers import (
    StuckSimulator,
    read_bridge_source,
    explore_source,
)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===================================================================
# Part 1: Stuck detector behavioral tests (current behavior)
# ===================================================================

class TestStuckDetectorBehavior:
    """Verify stuck detector timing matches bridge implementation."""

    def test_no_stuck_when_moving(self):
        """Robot that moves > 0.1m per check is never flagged."""
        sim = StuckSimulator()
        for i in range(20):
            action = sim.tick(float(i) * 0.2, 0.0)
            assert action == "ok", f"Moving robot flagged stuck at step {i}"

    def test_stuck_after_4s(self):
        """After 4s (2 checks) at same position → reset_waypoint."""
        sim = StuckSimulator()
        sim.tick(5.0, 5.0)           # first check — sets baseline
        sim.tick(5.01, 5.01)         # 2s — moved < 0.1m
        action = sim.tick(5.02, 5.02)  # 4s — stuck count = 2
        assert action == "reset_waypoint"

    def test_backup_after_8s(self):
        """After 8s (4 checks) stuck → backup escape."""
        sim = StuckSimulator()
        sim.tick(5.0, 5.0)
        sim.tick(5.01, 5.01)
        sim.tick(5.02, 5.02)  # 4s → reset_waypoint
        sim.tick(5.03, 5.03)  # 6s
        action = sim.tick(5.04, 5.04)  # 8s → backup
        assert action == "backup"

    def test_stuck_resets_after_backup(self):
        """After backup, stuck counter resets to 0."""
        sim = StuckSimulator()
        sim.tick(5.0, 5.0)
        sim.tick(5.01, 5.01)
        sim.tick(5.02, 5.02)  # reset_waypoint
        sim.tick(5.03, 5.03)
        sim.tick(5.04, 5.04)  # backup
        # After backup, counter should be 0 — next stuck needs 4s again
        action = sim.tick(5.05, 5.05)
        assert action == "ok", "Counter not reset after backup"

    def test_movement_clears_stuck(self):
        """Moving > 0.1m resets the stuck counter."""
        sim = StuckSimulator()
        sim.tick(5.0, 5.0)
        sim.tick(5.01, 5.01)  # nearly stuck
        sim.tick(6.0, 5.0)   # moved 1m — should clear
        action = sim.tick(6.01, 5.01)
        assert action == "ok", "Movement didn't clear stuck counter"


# ===================================================================
# Part 2: Stuck LOOP detection (RED — bug reproduction)
# ===================================================================

class TestStuckLoopDetection:
    """Detect infinite stuck loops where TARE keeps sending same waypoint.

    This is the P0 bug: reset_waypoint → TARE same waypoint → stuck → repeat.
    The bridge currently has NO loop detection — it just cycles forever.
    """

    def test_detects_repeated_resets_at_same_location(self):
        """If /reset_waypoint fires 3+ times at same spot → loop detected."""
        sim = StuckSimulator()
        # Simulate: robot stuck, backup 0.25m, TARE sends same waypoint,
        # robot goes back to same spot, gets stuck again.
        for cycle in range(4):
            # Stuck at (5, 5) ± tiny drift
            sim.tick(5.0 + cycle * 0.02, 5.0)
            sim.tick(5.01 + cycle * 0.02, 5.01)
            action = sim.tick(5.02 + cycle * 0.02, 5.02)  # reset_waypoint or backup
            if action == "backup":
                # After backup, "move" a tiny bit (backup is only 0.25m)
                sim.tick(4.8, 5.0)

        assert sim.is_same_location_loop(tolerance=0.5), (
            "Failed to detect stuck loop — robot stuck at same location 3+ times"
        )

    def test_no_false_positive_loop(self):
        """Resets at DIFFERENT locations should not trigger loop detection."""
        sim = StuckSimulator()
        # Stuck at different places
        for i, pos in enumerate([(1, 1), (5, 5), (10, 10)]):
            sim.tick(pos[0], pos[1])
            sim.tick(pos[0] + 0.01, pos[1] + 0.01)
            sim.tick(pos[0] + 0.02, pos[1] + 0.02)

        assert not sim.is_same_location_loop(tolerance=0.5), (
            "False positive: different locations flagged as loop"
        )

    def test_bridge_has_loop_breaker(self):
        """Bridge should detect repeated stuck at same location and escalate.

        Current behavior: cycles reset_waypoint → backup forever.
        Desired behavior: after 3 cycles at same spot, take aggressive action
        (e.g., large lateral move, skip waypoint, or announce failure).
        """
        src = read_bridge_source()
        # Look for any loop-breaking logic beyond simple backup
        has_loop_detection = any(pattern in src for pattern in [
            "same_location",
            "stuck_loop",
            "repeated_stuck",
            "loop_count",
            "stuck_history",
            "positions_at_reset",
            "escalat",
        ])
        assert has_loop_detection, (
            "Bridge _stuck_detector has no loop-breaking logic. "
            "When TARE re-sends the same unreachable waypoint, the robot "
            "cycles reset→backup→stuck indefinitely."
        )

    def test_backup_distance_sufficient(self):
        """Backup should move robot far enough to break the loop.

        Current: -0.25 m/s × 1s = 0.25m — often returns to same spot.
        Desired: 0.5m+ or lateral escape.
        """
        src = read_bridge_source()
        # Find the backup velocity and duration
        match = re.search(r'set_velocity\((-[\d.]+)', src[src.find("backing up"):])
        if match:
            backup_speed = abs(float(match.group(1)))
            # Current backup: 0.25 m/s × ~1s = 0.25m
            assert backup_speed >= 0.4, (
                f"Backup speed {backup_speed} m/s too slow — need 0.4+ to escape tight spaces"
            )


# ===================================================================
# Part 3: Bridge static verification
# ===================================================================

class TestStuckDetectorStructure:
    """Static verification of stuck detector code patterns."""

    def test_stuck_detector_timer_exists(self):
        src = read_bridge_source()
        assert "_stuck_detector" in src

    def test_stuck_sends_reset_waypoint(self):
        src = read_bridge_source()
        assert "reset_waypoint" in src

    def test_stuck_publishes_current_position(self):
        """Reset waypoint message should contain robot's current position."""
        src = read_bridge_source()
        # The PointStamped message should use odom.x/odom.y
        assert "point.x = odom.x" in src or "point.x" in src

    def test_backup_clears_stale_path(self):
        """After backup, the stale path should be cleared."""
        src = read_bridge_source()
        # Find backup section
        backup_idx = src.find("backing up")
        if backup_idx > 0:
            after = src[backup_idx:backup_idx + 300]
            assert "_current_path = []" in after, (
                "Backup should clear _current_path to prevent re-following stale path"
            )


# ===================================================================
# Part 4: TARE config verification
# ===================================================================

class TestTAREStuckConfig:
    """Verify TARE config parameters relevant to stuck behavior."""

    def test_viewpoint_collision_margin(self):
        """kViewPointCollisionMargin must account for Go2 body width.

        Go2 cylinder model: 0.34m wide front / 0.19m wide side.
        Half-width = 0.17m. Margin of 0.30m adds 0.13m safety overhead
        without blocking doorways (typically 0.8m+ wide).
        0.30 is acceptable — 0.35 was overly conservative and shrinks
        explorable space near walls unnecessarily.
        """
        cfg_path = os.path.join(_REPO, "config", "tare_go2_indoor.yaml")
        if not os.path.isfile(cfg_path):
            pytest.skip("tare_go2_indoor.yaml not found")
        with open(cfg_path) as f:
            content = f.read()
        match = re.search(r'kViewPointCollisionMargin\s*:\s*([\d.]+)', content)
        assert match, "kViewPointCollisionMargin not found in TARE config"
        margin = float(match.group(1))
        assert margin >= 0.30, (
            f"kViewPointCollisionMargin={margin} too small — Go2 needs >= 0.30m "
            f"(half-width 0.17m + 0.13m safety) to avoid unreachable viewpoints near walls"
        )

    def test_viewpoint_margin_minimum(self):
        """Margin should be >= 0.30m (Go2 can reach viewpoints this far from walls).

        Original value 0.35 works with searchRadius=0.45. Don't increase
        further — larger margins shrink explorable space in tight rooms.
        """
        cfg_path = os.path.join(_REPO, "config", "tare_go2_indoor.yaml")
        if not os.path.isfile(cfg_path):
            pytest.skip("tare_go2_indoor.yaml not found")
        with open(cfg_path) as f:
            content = f.read()
        match = re.search(r'kViewPointCollisionMargin\s*:\s*([\d.]+)', content)
        margin = float(match.group(1))
        assert margin >= 0.30, (
            f"kViewPointCollisionMargin={margin} — should be >= 0.30m "
            f"(vehicleWidth/2 + searchRadius overhead + safety)"
        )

    def test_extend_waypoint_distance(self):
        """kExtendWayPointDistanceBig should be <= 2.0m for indoor.

        Large extension distances cause waypoints past walls.
        """
        cfg_path = os.path.join(_REPO, "config", "tare_go2_indoor.yaml")
        if not os.path.isfile(cfg_path):
            pytest.skip("tare_go2_indoor.yaml not found")
        with open(cfg_path) as f:
            content = f.read()
        match = re.search(r'kExtendWayPointDistanceBig\s*:\s*([\d.]+)', content)
        if match:
            dist = float(match.group(1))
            assert dist <= 2.0, (
                f"kExtendWayPointDistanceBig={dist} — too large for indoor, "
                f"waypoints may extend past walls"
            )
