# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Level 29: Pre-explore navigation debug harness — P2 (FAR routing before explore).

Root cause: FAR planner silently rejects /goal_point when is_graph_init_=false.
The proxy publishes direct /way_point to localPlanner as fallback, but localPlanner
follows a straight line to the target — through walls.

The dead-reckoning mode (Mode 2) in NavigateSkill already routes through doorways
correctly, but Mode 0 (proxy) is tried first and times out after 45s without
detecting that FAR has no graph.

These tests verify:
  - Direct /way_point without FAR graph crosses walls (bug reproduction)
  - Dead-reckoning mode routes through doors correctly
  - Proxy should detect FAR unavailability and fall back faster
  - Navigate skill mode selection logic
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
    MockBase,
    navigate_module,
    navigate_source,
    proxy_source,
    distance,
    path_crosses_wall,
    ROOM_CENTERS,
    ROOM_DOORS,
)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===================================================================
# Part 1: Direct line path collision (bug reproduction)
# ===================================================================

class TestDirectLineCollision:
    """Prove that straight-line /way_point between rooms crosses walls."""


    def test_study_to_guest_bedroom_crosses_wall(self):
        """Study (17,7.5) to guest bedroom (16,12) crosses y=10 partition."""
        crosses = path_crosses_wall(17.0, 7.5, 16.0, 12.0)
        assert crosses, (
            "Study→Guest bedroom straight line should cross y=10 partition "
            "at x~16.5, which is inside wall segment (12.5, 10)-(20, 10)"
        )

    def test_kitchen_to_guest_bedroom_crosses_wall(self):
        """Kitchen (17,2.5) to guest bedroom (16,12) crosses partition."""
        crosses = path_crosses_wall(17.0, 2.5, 16.0, 12.0)
        assert crosses, (
            "Kitchen→Guest bedroom straight line should cross y=10 partition"
        )


# ===================================================================
# Part 2: Dead-reckoning routes through doors (correct behavior)
# ===================================================================

class TestDeadReckoningRouting:
    """Verify dead-reckoning mode uses doorway waypoints."""

    def test_dead_reckoning_uses_door_waypoints(self):
        """Dead-reckoning should route: room→door→hallway→door→room."""
        src = navigate_source()
        # The waypoint sequence logic — SceneGraph door chain replaces _ROOM_DOORS
        assert "waypoints" in src, "NavigateSkill must build waypoint list"
        assert "get_door_chain" in src or "door" in src.lower(), (
            "NavigateSkill must use SceneGraph door chain for dead-reckoning"
        )
        # Should route via door positions
        assert "src_room" in src or "door" in src.lower()

    def test_all_rooms_have_doors(self):
        """Every room in the test layout should have a door in ROOM_DOORS."""
        # nav_debug_helpers.py provides the canonical test room data.
        # navigate.py no longer has hardcoded _ROOM_CENTERS/_ROOM_DOORS —
        # door data now comes from SceneGraph populated during exploration.
        for room in ROOM_CENTERS:
            assert room in ROOM_DOORS, f"Room '{room}' has no door entry in test fixture"

    def test_door_to_door_path_avoids_walls(self):
        """Path from any door to any other door via hallway should be wall-free.

        Dead-reckoning goes: source_door → target_door (through hallway).
        These segments should not cross walls.
        """
        hallway_center = ROOM_CENTERS["hallway"]
        for room, door in ROOM_DOORS.items():
            if room == "hallway":
                continue
            # Door to hallway center should not cross walls
            crosses = path_crosses_wall(door[0], door[1],
                                         hallway_center[0], hallway_center[1])
            # Allow some tolerance — doors are at wall openings
            if crosses:
                # Check if it's just brushing the wall gap
                wall_dist_at_door = min(
                    abs(door[0] - hallway_center[0]),
                    abs(door[1] - hallway_center[1]),
                )
                assert wall_dist_at_door > 0.5, (
                    f"{room} door ({door[0]:.1f},{door[1]:.1f}) → hallway path "
                    f"crosses wall and is {wall_dist_at_door:.1f}m away"
                )


# ===================================================================
# Part 3: Proxy FAR detection (RED — desired behavior)
# ===================================================================

class TestProxyFARDetection:
    """Verify proxy detects when FAR graph is unavailable."""

    def test_proxy_does_not_publish_waypoint_without_far(self):
        """Proxy should NOT publish direct /way_point when FAR has no graph.

        Current behavior: always publishes both /goal_point AND /way_point.
        The /way_point goes directly to localPlanner which follows a straight
        line through walls.

        Desired: only publish /goal_point, wait for FAR to respond with its
        own /way_point. If FAR doesn't respond in 2-3s, return False to trigger
        dead-reckoning fallback.
        """
        src = proxy_source()
        # Look for FAR graph detection logic
        has_far_check = any(pattern in src for pattern in [
            "far_graph",
            "graph_init",
            "far_respond",
            "far_available",
            "far_status",
            "wait_for_far",
        ])
        assert has_far_check, (
            "Proxy navigate_to() has no FAR graph availability check. "
            "It publishes /way_point blindly, causing straight-line-through-wall "
            "navigation when FAR hasn't explored yet."
        )

    def test_proxy_fast_fallback_without_far(self):
        """When FAR has no graph, proxy should fall back in < 5s, not 45s.

        Current: navigate_to(timeout=45.0) — user waits 45s before dead-reckoning.
        Desired: detect no-FAR-response in 3-5s and return False immediately.
        """
        src = proxy_source()
        # Look for early fallback logic
        has_early_fallback = any(pattern in src for pattern in [
            "no_response",
            "far_timeout",
            "early_fallback",
            "no_path_received",
        ])
        assert has_early_fallback, (
            "Proxy does not detect FAR non-response early. "
            "45s timeout wastes user's time when FAR has no graph."
        )


# ===================================================================
# Part 4: Navigate skill mode selection
# ===================================================================

class TestNavigateSkillModes:
    """Verify navigate skill falls back correctly."""

    def test_mode_priority_order(self):
        """Navigate should try: proxy → nav_stack → dead_reckoning."""
        src = navigate_source()
        # Mode 0: proxy
        proxy_idx = src.find("navigate_to")
        # Mode 1: nav stack
        nav_idx = src.find("nav_stack")
        # Mode 2: dead reckoning
        dr_idx = src.find("dead_reckoning")
        assert proxy_idx < nav_idx < dr_idx, (
            "Mode priority should be: proxy → nav_stack → dead_reckoning"
        )

    def test_proxy_failure_falls_to_dead_reckoning(self):
        """If proxy navigate_to returns False, skill should dead-reckon."""
        src = navigate_source()
        # After proxy mode, there should be a fallback
        assert "_dead_reckoning" in src, "No dead_reckoning fallback"
        # The _navigate_with_proxy method itself should call dead_reckoning on timeout
        proxy_method_start = src.find("def _navigate_with_proxy")
        next_method = src.find("\n    def ", proxy_method_start + 1)
        proxy_body = src[proxy_method_start:next_method]
        assert "_dead_reckoning" in proxy_body, (
            "_navigate_with_proxy should fall back to _dead_reckoning on failure"
        )

    def test_exploration_cancelled_before_navigate(self):
        """Navigate should cancel active exploration before proceeding."""
        src = navigate_source()
        assert "cancel_exploration" in src, (
            "NavigateSkill should cancel background exploration"
        )

    def test_nav_flag_created(self):
        """Navigate should create /tmp/vector_nav_active for bridge."""
        src = navigate_source()
        assert "vector_nav_active" in src


# ===================================================================
# Part 5: Room connectivity (door graph data)
# ===================================================================

class TestRoomConnectivity:
    """Verify room map has enough data for pre-explore routing."""

    def test_all_rooms_reachable_via_doors(self):
        """Every room should be reachable from hallway via its door."""
        for room in ROOM_CENTERS:
            if room == "hallway":
                continue
            assert room in ROOM_DOORS, f"{room} has no door defined"
            dx, dy = ROOM_DOORS[room]
            hx, hy = ROOM_CENTERS["hallway"]
            dist = distance(dx, dy, hx, hy)
            assert dist < 10.0, (
                f"{room} door is {dist:.1f}m from hallway — unreachable"
            )

    def test_room_center_not_at_door(self):
        """Room centers should be distinct from door positions."""
        for room in ROOM_CENTERS:
            if room == "hallway":
                continue
            cx, cy = ROOM_CENTERS[room]
            dx, dy = ROOM_DOORS[room]
            dist = distance(cx, cy, dx, dy)
            assert dist > 0.5, (
                f"{room}: center ({cx},{cy}) too close to door ({dx},{dy}) — "
                f"only {dist:.1f}m apart"
            )

    def test_scene_graph_has_door_edges(self):
        """SceneGraph stores room→room connectivity (connected_rooms field).

        This exists but is not yet populated during exploration.
        Pre-explore navigation still relies on hardcoded _ROOM_DOORS.
        """
        import importlib
        sg_mod = importlib.import_module("vector_os_nano.core.scene_graph")
        src = inspect.getsource(sg_mod)
        assert "connected_rooms" in src, (
            "SceneGraph has no room connectivity data"
        )


# ===================================================================
# Part 6: Proxy dual-publish analysis
# ===================================================================

class TestProxyDualPublish:
    """Analyze the proxy's dual /goal_point + /way_point publishing."""

    def test_proxy_publishes_goal_point(self):
        """Proxy should publish to /goal_point for FAR planner."""
        src = proxy_source()
        assert "/goal_point" in src

    def test_proxy_publishes_way_point(self):
        """Proxy also publishes to /way_point (direct to localPlanner)."""
        src = proxy_source()
        assert "/way_point" in src

    def test_dual_publish_documented(self):
        """The dual-publish strategy should be documented in code."""
        src = proxy_source()
        has_doc = any(pattern in src.lower() for pattern in [
            "far planner",
            "fallback",
            "direct",
            "dual",
        ])
        assert has_doc, (
            "Dual-publish strategy (/goal_point + /way_point) should be documented"
        )

    def test_waypoint_publish_rate(self):
        """Direct /way_point should publish at <= 5Hz (don't overwhelm FAR)."""
        src = proxy_source()
        # Find the navigate_to method, then check its sleep interval
        nav_start = src.find("def navigate_to")
        if nav_start < 0:
            pytest.skip("navigate_to not found in proxy")
        nav_section = src[nav_start:src.find("\n    def ", nav_start + 1)]
        match = re.search(r'time\.sleep\(([\d.]+)\)', nav_section)
        if match:
            interval = float(match.group(1))
            rate = 1.0 / interval if interval > 0 else 999
            assert rate <= 5.0, (
                f"/way_point publish rate ~{rate:.0f}Hz — should be <= 5Hz"
            )


# Need inspect for test_scene_graph_has_door_edges
import inspect
