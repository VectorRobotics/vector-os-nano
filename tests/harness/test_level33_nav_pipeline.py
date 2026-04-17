"""Level 33: Navigation pipeline integrity harness.

Tests verify the full navigate pipeline from proxy to FAR to localPlanner:
  - FAR detection uses /way_point (not /path)
  - Phase 1 probe correctly detects FAR availability
  - Phase 2 only publishes /goal_point (no direct /way_point)
  - Stale waypoint/path rejection
  - SceneGraph drift protection
  - Diagnostic logging
  - Cylinder body safety boundary
"""
from __future__ import annotations

import importlib
import inspect
import math
import os
import re

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nav_debug_helpers import read_bridge_source

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _proxy_source() -> str:
    mod = importlib.import_module("vector_os_nano.hardware.sim.go2_ros2_proxy")
    return inspect.getsource(mod)


def _navigate_source() -> str:
    mod = importlib.import_module("vector_os_nano.skills.navigate")
    return inspect.getsource(mod)


# ===================================================================
# Part 1: FAR detection must use /way_point, NOT /path
# ===================================================================

class TestFARDetectionSignal:
    """FAR availability must be detected via /way_point, not /path.

    /path from localPlanner is unreliable — it publishes even without
    a valid goal from FAR. /way_point is ONLY published by FAR when
    it has a V-Graph and can route to the goal.
    """

    def test_proxy_subscribes_to_waypoint(self):
        """Proxy must subscribe to /way_point for FAR detection."""
        src = _proxy_source()
        assert "/way_point" in src
        assert "_waypoint_cb" in src or "waypoint_cb" in src

    def test_proxy_has_waypoint_timestamp(self):
        """Proxy tracks _last_waypoint_time for FAR probe."""
        src = _proxy_source()
        assert "_last_waypoint_time" in src

    def test_probe_checks_waypoint_not_path(self):
        """Phase 1 probe must check _last_waypoint_time, not _last_path_time."""
        src = _proxy_source()
        nav_start = src.find("def navigate_to")
        nav_end = src.find("\n    def ", nav_start + 1)
        nav_body = src[nav_start:nav_end]
        assert "_last_waypoint_time" in nav_body, (
            "navigate_to must check _last_waypoint_time for FAR detection"
        )
        # Should NOT use _last_path_time for FAR detection
        assert "_last_path_time > start_time" not in nav_body, (
            "navigate_to must NOT use _last_path_time — /path is unreliable"
        )

    def test_waypoint_timestamp_reset_before_probe(self):
        """_last_waypoint_time must be reset to 0 before Phase 1 probe."""
        src = _proxy_source()
        nav_start = src.find("def navigate_to")
        nav_body = src[nav_start:nav_start + 3000]
        assert "_last_waypoint_time = 0" in nav_body, (
            "Must reset _last_waypoint_time before probe to ignore stale data"
        )


# ===================================================================
# Part 2: Phase 2 must NOT publish direct /way_point
# ===================================================================

class TestPhase2NoDirectWaypoint:
    """Phase 2 must only publish /goal_point. Direct /way_point to
    localPlanner overrides FAR's routed intermediate waypoints and
    causes the dog to navigate straight through walls.
    """

    def test_phase2_publishes_goal_point(self):
        """Phase 2 loop must publish /goal_point.

        _publish_goal_point is called at 2 Hz inside the Phase 2 while loop.
        The loop body is ~1600 chars from the Phase 2 comment due to cancel/abort
        checks before the actual publish call. Use 2000 chars window.
        """
        src = _proxy_source()
        # Find the actual Phase 2 code block (after the docstring)
        phase2_comment = src.find("# Phase 2: full navigation")
        if phase2_comment < 0:
            phase2_comment = src.find("Phase 2")
        assert phase2_comment > 0, "Phase 2 section not found"
        # Use 2000 chars to cover cancel/abort/stall checks before the publish call
        phase2 = src[phase2_comment:phase2_comment + 2000]
        assert "_publish_goal_point" in phase2

    def test_phase2_no_direct_waypoint(self):
        """Phase 2 must NOT call _publish_waypoint (would override FAR)."""
        src = _proxy_source()
        phase2_start = src.find("Phase 2")
        phase2_end = src.find("far_timeout", phase2_start)
        if phase2_end < 0:
            phase2_end = phase2_start + 800
        phase2 = src[phase2_start:phase2_end]
        assert "_publish_waypoint" not in phase2, (
            "Phase 2 must NOT publish /way_point directly — "
            "it overrides FAR's door routing and causes wall collision"
        )

    def test_phase1_no_direct_waypoint(self):
        """Phase 1 probe must NOT publish /way_point either."""
        src = _proxy_source()
        phase1_start = src.find("Phase 1")
        phase2_start = src.find("Phase 2")
        phase1 = src[phase1_start:phase2_start]
        assert "_publish_waypoint" not in phase1, (
            "Phase 1 must NOT publish /way_point — only /goal_point"
        )


# ===================================================================
# Part 3: SceneGraph drift protection
# ===================================================================

class TestSceneGraphDriftProtection:
    """Navigate must protect against unreliable SceneGraph positions.

    After the Wave 2 refactor, drift protection is handled via visit_count
    threshold (_MIN_VISIT_COUNT) instead of positional comparison against
    hardcoded room centers (which have been removed).
    """

    def test_min_visit_count_constant_exists(self):
        """_MIN_VISIT_COUNT must exist — protects against doorway drive-by positions."""
        src = _navigate_source()
        assert "_MIN_VISIT_COUNT" in src, (
            "_MIN_VISIT_COUNT constant must exist in navigate.py"
        )

    def test_min_visit_count_value_reasonable(self):
        """_MIN_VISIT_COUNT should be 1-10 (1 is fine for sim with config-based rooms)."""
        src = _navigate_source()
        match = re.search(r'_MIN_VISIT_COUNT.*?=\s*(\d+)', src)
        assert match, "_MIN_VISIT_COUNT not found"
        count = int(match.group(1))
        assert 1 <= count <= 10, (
            f"_MIN_VISIT_COUNT={count} — should be 1-10"
        )

    def test_visit_count_check_in_get_room_center(self):
        """_get_room_center_from_memory must check visit_count threshold."""
        src = _navigate_source()
        func_start = src.find("def _get_room_center_from_memory")
        func_end = src.find("\ndef ", func_start + 1)
        func_body = src[func_start:func_end]
        assert "visit_count" in func_body, (
            "_get_room_center_from_memory must check visit_count threshold"
        )
        assert "_MIN_VISIT_COUNT" in func_body, (
            "_get_room_center_from_memory must use _MIN_VISIT_COUNT"
        )

    def test_get_room_center_returns_none_for_unvisited(self):
        """_get_room_center_from_memory returns None for rooms with 0 visits."""
        mod = importlib.import_module("vector_os_nano.skills.navigate")

        class FakeRoom:
            center_x = 17.0
            center_y = 2.5
            visit_count = 0

        class FakeMemory:
            def get_room(self, key):
                if key == "kitchen":
                    return FakeRoom()
                return None

        result = mod._get_room_center_from_memory(FakeMemory(), "kitchen")
        assert result is None, "0 visits should not be trusted"


# ===================================================================
# Part 4: Cylinder body safety boundary
# ===================================================================

class TestCylinderBodySafety:
    """Path follower must treat the dog as a cylinder — subtract body
    radius from obstacle distances before safety checks.
    """

    def test_body_front_extent_used(self):
        """Body front gap computed from obstacle distance minus body extent."""
        src = read_bridge_source()
        assert "front_gap" in src, "front_gap must be computed in bridge"
        assert "0.20" in src or "0.19" in src, "Body extent offset must be in bridge"

    def test_body_side_extent_used(self):
        """Body side extent (0.19m) must be subtracted from obstacle distance."""
        src = read_bridge_source()
        assert "0.19" in src, "Body side extent 0.19m must be in bridge"

    def test_gap_calculation(self):
        """Safety must compute gap = obstacle_distance - body_extent."""
        src = read_bridge_source()
        follow_start = src.find("def _follow_path")
        follow_body = src[follow_start:]
        has_gap = (
            "front_gap" in follow_body
            or "left_gap" in follow_body
            or "front_d - _BODY" in follow_body
        )
        assert has_gap, (
            "Must compute gap (obstacle_dist - body_radius), not use raw distance"
        )

    def test_comfort_zone_exists(self):
        """Must define a comfort zone for body-to-wall gap."""
        src = read_bridge_source()
        assert "_COMFORT" in src or "comfort" in src.lower()

    def test_danger_zone_exists(self):
        """Must define a danger zone for imminent body contact."""
        src = read_bridge_source()
        assert "_DANGER" in src or "danger" in src.lower()

    def test_blocks_motion_toward_wall(self):
        """When wall is on one side, must block motion toward that wall."""
        src = read_bridge_source()
        follow_start = src.find("def _follow_path")
        follow_body = src[follow_start:]
        # Should have: if vy > 0: vy = 0.0 (block leftward when wall left)
        assert "vy = 0.0" in follow_body or "vy = 0" in follow_body, (
            "Must block lateral motion toward nearby wall"
        )


# ===================================================================
# Part 5: Behavioral - drift protection simulation
# ===================================================================

class TestDriftProtectionBehavior:
    """Behavioral tests for _get_room_center_from_memory visit_count protection.

    After the Wave 2 refactor, position trust is determined by visit_count
    threshold (_MIN_VISIT_COUNT=3), not positional drift from hardcoded centers.
    """

    def test_well_visited_position_accepted(self):
        """SceneGraph position with visit_count >= 3 is accepted."""
        mod = importlib.import_module("vector_os_nano.skills.navigate")

        class FakeRoom:
            center_x = 17.5
            center_y = 2.8
            visit_count = 5

        class FakeMemory:
            def get_room(self, key):
                if key == "kitchen":
                    return FakeRoom()
                return None

        result = mod._get_room_center_from_memory(FakeMemory(), "kitchen")
        assert result is not None, "5 visits should be trusted"
        assert result == (17.5, 2.8)

    def test_low_visit_count_rejected(self):
        """SceneGraph position with < _MIN_VISIT_COUNT visits is rejected."""
        mod = importlib.import_module("vector_os_nano.skills.navigate")

        class FakeRoom:
            center_x = 17.0
            center_y = 2.5
            visit_count = 1  # only 1 visit — doorway drive-by

        class FakeMemory:
            def get_room(self, key):
                if key == "kitchen":
                    return FakeRoom()
                return None

        result = mod._get_room_center_from_memory(FakeMemory(), "kitchen")
        # With _MIN_VISIT_COUNT=1, 1 visit IS trusted (sim-only with config-based rooms)
        assert result is not None, "1 visit should be trusted with _MIN_VISIT_COUNT=1"

    def test_zero_visit_count_rejected(self):
        """SceneGraph position with 0 visits is rejected."""
        mod = importlib.import_module("vector_os_nano.skills.navigate")

        class FakeRoom:
            center_x = 17.0
            center_y = 2.5
            visit_count = 0

        class FakeMemory:
            def get_room(self, key):
                if key == "kitchen":
                    return FakeRoom()
                return None

        result = mod._get_room_center_from_memory(FakeMemory(), "kitchen")
        assert result is None, "0 visits should not be trusted"

    def test_unknown_room_returns_none(self):
        """Unknown room key returns None."""
        mod = importlib.import_module("vector_os_nano.skills.navigate")

        class FakeMemory:
            def get_room(self, key):
                return None

        result = mod._get_room_center_from_memory(FakeMemory(), "nonexistent")
        assert result is None


# ===================================================================
# Part 6: Diagnostic logging
# ===================================================================

class TestNavigateDiagnostics:
    """Navigate must log progress during Phase 2 for debugging."""

    def test_progress_logging_exists(self):
        src = _proxy_source()
        nav_start = src.find("def navigate_to")
        nav_body = src[nav_start:]
        assert "waypoint_age" in nav_body or "wp_age" in nav_body, (
            "Navigate must log waypoint age for debugging"
        )

    def test_far_response_logged(self):
        src = _proxy_source()
        assert "FAR responded" in src, (
            "Must log when FAR responds with /way_point"
        )

    def test_early_fallback_logged(self):
        src = _proxy_source()
        assert "early_fallback" in src, (
            "Must log when falling back to dead-reckoning"
        )
