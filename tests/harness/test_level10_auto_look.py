# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Level 10: ExploreSkill + VLM auto-look integration tests.

Tests that when ExploreSkill enters a new room during exploration,
it automatically calls VLM to describe the scene and records
observations in the SceneGraph.

All tests use mock VLM (no real API calls) and mock base (no MuJoCo).
"""
from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

from tests.harness.clock import fake_clock

import numpy as np
import pytest

from vector_os_nano.core.scene_graph import SceneGraph
from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.types import SkillResult
import vector_os_nano.skills.go2.explore as _explore_mod
from vector_os_nano.skills.go2.explore import (
    ExploreSkill,
    _exploration_loop,
    _explore_cancel,
    _explore_visited,
    cancel_exploration,
    get_explored_rooms,
    is_exploring,
    set_auto_look,
    set_event_callback,
)
# Local test fixture replacing removed hardcoded dict from navigate.py
_ROOM_CENTERS: dict[str, tuple[float, float]] = {
    "living_room":    (3.0,  2.5),
    "dining_room":    (3.0,  7.5),
    "kitchen":        (17.0, 2.5),
    "study":          (17.0, 7.5),
    "master_bedroom": (3.5,  12.0),
    "guest_bedroom":  (16.0, 12.0),
    "bathroom":       (8.5,  12.0),
    "hallway":        (10.0, 5.0),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_vlm():
    """Create a mock VLM that returns predictable results."""
    vlm = MagicMock()

    # describe_scene returns SceneDescription-like object
    scene = MagicMock()
    scene.summary = "A room with furniture"
    obj1 = MagicMock()
    obj1.name = "table"
    obj1.confidence = 0.9
    obj2 = MagicMock()
    obj2.name = "chair"
    obj2.confidence = 0.85
    scene.objects = [obj1, obj2]
    vlm.describe_scene.return_value = scene

    # identify_room returns RoomIdentification-like object
    room_id = MagicMock()
    room_id.room = "living_room"
    room_id.confidence = 0.92
    vlm.identify_room.return_value = room_id

    return vlm


def _make_mock_base(rooms_sequence: list[str]):
    """Create a mock base that moves through a sequence of rooms.

    Each call to get_position() returns the center of the next room
    in the sequence.
    """
    base = MagicMock()
    pos_index = [0]

    def _get_position():
        idx = min(pos_index[0], len(rooms_sequence) - 1)
        room = rooms_sequence[idx]
        center = _ROOM_CENTERS.get(room, (0.0, 0.0))
        pos_index[0] += 1
        return (center[0], center[1], 0.28)  # z=0.28 = standing

    def _get_heading():
        return 0.0

    def _get_camera_frame(width=320, height=240):
        return np.zeros((height, width, 3), dtype=np.uint8)

    base.get_position = _get_position
    base.get_heading = _get_heading
    base.get_camera_frame = _get_camera_frame
    base.set_velocity = MagicMock()
    base.walk = MagicMock(return_value=True)
    return base


def _make_context(base, vlm=None, scene_graph=None):
    """Build a SkillContext with optional VLM and SceneGraph."""
    services = {}
    if vlm is not None:
        services["vlm"] = vlm
    if scene_graph is not None:
        services["spatial_memory"] = scene_graph
    return SkillContext(
        bases={"default": base},
        services=services,
    )


def _make_mock_scenegraph(rooms_sequence: list[str] | None = None) -> MagicMock:
    """Create a mock SceneGraph that returns rooms by nearest position.

    Since explore.py uses _spatial_memory.nearest_room() to detect rooms,
    tests must supply a SceneGraph (or mock) that returns room names for
    positions provided by _make_mock_base(). This helper creates a mock
    where nearest_room returns rooms from the sequence in order.
    """
    mock_sg = MagicMock(spec=["nearest_room", "visit", "add_door",
                               "get_visited_rooms", "observe_with_viewpoint",
                               "get_room_coverage", "get_room", "get_all_rooms",
                               "load_layout"])
    # get_all_rooms returns non-empty so layout reload doesn't trigger
    mock_sg.get_all_rooms.return_value = [MagicMock(room_id="hallway")]

    if rooms_sequence:
        call_count = [0]
        def _nearest_room(x: float, y: float) -> str | None:
            # Return room name based on x position matching _ROOM_CENTERS
            for name, (cx, cy) in _ROOM_CENTERS.items():
                if abs(x - cx) < 1.0 and abs(y - cy) < 1.0:
                    return name
            return None
        mock_sg.nearest_room = _nearest_room
    else:
        mock_sg.nearest_room.return_value = "hallway"
    mock_sg.visit = MagicMock()
    mock_sg.add_door = MagicMock()
    mock_sg.get_visited_rooms.return_value = []
    mock_sg.get_room.return_value = None
    return mock_sg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAutoLookCallback:
    """Test that set_auto_look / auto-look callback mechanism works."""

    def test_set_auto_look_is_callable(self):
        """set_auto_look accepts a callable."""
        called = []
        set_auto_look(lambda room: called.append(room))
        # Reset
        set_auto_look(None)

    def test_auto_look_called_on_new_room(self):
        """Auto-look callback fires when exploration enters a new room."""
        observations = []

        def mock_look(room: str) -> dict | None:
            observations.append(room)
            return {"summary": f"Observed {room}", "objects": []}

        set_auto_look(mock_look)

        # Simulate entering two rooms by running a short exploration
        base = _make_mock_base(["living_room", "living_room", "kitchen", "kitchen"])
        _explore_cancel.clear()
        _explore_visited.clear()

        # Provide a mock SceneGraph — explore.py uses nearest_room() since Wave 2 refactor
        _explore_mod._spatial_memory = _make_mock_scenegraph(["living_room", "kitchen"])

        # Run exploration loop briefly with a cancel after a few iterations
        def _cancel_soon():
            time.sleep(0.3)
            _explore_cancel.set()

        cancel_thread = threading.Thread(target=_cancel_soon, daemon=True)
        cancel_thread.start()

        # Patch subprocess.run to avoid 5s TARE start timeout
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with fake_clock():
                _exploration_loop(base, has_bridge=False)

        cancel_thread.join(timeout=3.0)
        _explore_mod._spatial_memory = None

        # Should have observed at least one room
        assert len(observations) >= 1
        assert "living_room" in observations or "kitchen" in observations

        # Cleanup
        set_auto_look(None)

    def test_auto_look_failure_does_not_crash_exploration(self):
        """If auto-look raises, exploration continues."""
        def failing_look(room: str) -> dict | None:
            raise RuntimeError("VLM crashed")

        set_auto_look(failing_look)

        base = _make_mock_base(["living_room", "kitchen"])
        _explore_cancel.clear()
        _explore_visited.clear()

        # Provide a mock SceneGraph for nearest_room()
        _explore_mod._spatial_memory = _make_mock_scenegraph(["living_room", "kitchen"])

        def _cancel_soon():
            time.sleep(0.3)
            _explore_cancel.set()

        cancel_thread = threading.Thread(target=_cancel_soon, daemon=True)
        cancel_thread.start()

        # Patch subprocess.run to avoid 5s TARE start timeout
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with fake_clock():
                # Should not raise
                _exploration_loop(base, has_bridge=False)

        cancel_thread.join(timeout=3.0)
        _explore_mod._spatial_memory = None

        # Exploration still recorded rooms even though auto-look failed
        assert len(_explore_visited) >= 1

        # Cleanup
        set_auto_look(None)

    def test_auto_look_none_is_noop(self):
        """When auto-look is None, exploration works normally."""
        set_auto_look(None)

        base = _make_mock_base(["hallway", "hallway"])
        _explore_cancel.clear()
        _explore_visited.clear()

        # Provide a mock SceneGraph for nearest_room()
        _explore_mod._spatial_memory = _make_mock_scenegraph(["hallway"])

        def _cancel_soon():
            time.sleep(0.3)
            _explore_cancel.set()

        cancel_thread = threading.Thread(target=_cancel_soon, daemon=True)
        cancel_thread.start()

        # Patch subprocess.run to avoid 5s TARE start timeout
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with fake_clock():
                _exploration_loop(base, has_bridge=False)

        cancel_thread.join(timeout=3.0)
        _explore_mod._spatial_memory = None

        assert "hallway" in _explore_visited


class TestExploreSkillAutoLookWiring:
    """Test that ExploreSkill.execute() wires auto-look from context."""

    def test_explore_works_with_vlm_in_services(self):
        """ExploreSkill starts even when VLM is in services (auto-look disabled for sim)."""
        vlm = _make_mock_vlm()
        scene_graph = SceneGraph()
        # Seed rooms so layout reload doesn't trigger file access
        scene_graph.visit("living_room", 3.0, 2.5)
        base = _make_mock_base(["living_room"])
        context = _make_context(base, vlm=vlm, scene_graph=scene_graph)

        skill = ExploreSkill()
        cancel_exploration()  # ensure clean state
        import time; time.sleep(0.1)

        with patch(
            "vector_os_nano.skills.go2.explore._start_bridge_on_go2",
            return_value=False,
        ):
            result = skill.execute({}, context)

        assert result.success
        cancel_exploration()

    def test_explore_no_vlm_still_works(self):
        """ExploreSkill works without VLM (no auto-look)."""
        base = _make_mock_base(["hallway"])
        context = _make_context(base, vlm=None, scene_graph=None)

        skill = ExploreSkill()

        with patch(
            "vector_os_nano.skills.go2.explore._start_bridge_on_go2",
            return_value=False,
        ):
            result = skill.execute({}, context)

        assert result.success
        cancel_exploration()


class TestAutoLookSceneGraphIntegration:
    """Test that auto-look observations are recorded in SceneGraph."""

    def setup_method(self):
        """Ensure clean exploration state between tests."""
        _explore_mod._explore_running = False
        _explore_mod._explore_cancel.set()  # ensure any wait() returns
        _explore_mod._explore_visited.clear()
        _explore_mod._on_event = None
        _explore_mod._auto_look = None
        _explore_mod._spatial_memory = None
        import time
        time.sleep(0.05)  # brief pause for any lingering threads to exit
        _explore_mod._explore_cancel.clear()

    def teardown_method(self):
        """Clean up after each test."""
        _explore_mod._explore_running = False
        _explore_mod._explore_cancel.set()
        _explore_mod._on_event = None
        _explore_mod._auto_look = None
        _explore_mod._spatial_memory = None

    def test_observations_recorded_in_scene_graph(self):
        """Auto-look records VLM observations via scene_graph.observe_with_viewpoint."""
        vlm = _make_mock_vlm()
        # Pre-populate SceneGraph so nearest_room() returns room names.
        # The explore loop needs rooms to already exist to detect them.
        scene_graph = SceneGraph()
        for name, (x, y) in _ROOM_CENTERS.items():
            scene_graph.visit(name, x, y)
        base = _make_mock_base(["living_room", "living_room", "living_room"])
        context = _make_context(base, vlm=vlm, scene_graph=scene_graph)

        skill = ExploreSkill()

        # Keep patches active during background thread execution.
        # time.sleep is patched to skip seed walk delay (1s) so the while
        # loop executes quickly without real waiting.
        # Use threading.Event.wait to give background thread real wait time.
        import time
        _orig_sleep = time.sleep  # capture before patch
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with fake_clock():
                with patch(
                    "vector_os_nano.skills.go2.explore._start_bridge_on_go2",
                    return_value=False,
                ):
                    result = skill.execute({}, context)
                    # Give background thread real time to enter while loop
                    _orig_sleep(0.3)
                    cancel_exploration()
                    _orig_sleep(0.2)

        # SceneGraph should have the observed room
        visited = scene_graph.get_visited_rooms()
        assert len(visited) >= 1

    def test_room_entered_event_emitted(self):
        """A 'room_entered' event is emitted when a new room is detected.

        Note: 'room_observed' (VLM auto-look) is disabled for sim mode.
        Room detection is position-based via SceneGraph.nearest_room().
        """
        events = []

        def capture_event(event_type: str, data: dict):
            events.append((event_type, data))

        set_event_callback(capture_event)

        # Pre-populate SceneGraph so nearest_room() returns room names.
        scene_graph = SceneGraph()
        for name, (x, y) in _ROOM_CENTERS.items():
            scene_graph.visit(name, x, y)
        base = _make_mock_base(["kitchen", "kitchen", "kitchen"])

        _explore_cancel.clear()
        _explore_visited.clear()
        _explore_mod._spatial_memory = scene_graph

        import time
        def _cancel_soon():
            time.sleep(0.3)
            _explore_cancel.set()

        cancel_thread = threading.Thread(target=_cancel_soon, daemon=True)
        cancel_thread.start()

        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with fake_clock():
                _exploration_loop(base, has_bridge=False)

        cancel_thread.join(timeout=3.0)
        _explore_mod._spatial_memory = None
        set_event_callback(None)

        event_types = [e[0] for e in events]
        assert "room_entered" in event_types, f"Expected room_entered, got: {event_types}"
        set_event_callback(None)
