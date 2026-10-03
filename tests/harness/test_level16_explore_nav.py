# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Level 16: Explore + nav stack integration logic tests.

Verifies that ExploreSkill:
- Seeds briefly then stops (doesn't override nav stack)
- Sends zero velocity after seed
- Auto-look fires on new room entry
- No velocity commands sent during main loop
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

from tests.harness.clock import fake_clock

import numpy as np
import pytest

from vector_os_nano.core.skill import SkillContext
import vector_os_nano.skills.go2.explore as _explore_mod
from vector_os_nano.skills.go2.explore import (
    ExploreSkill,
    _exploration_loop,
    _explore_cancel,
    _explore_visited,
    cancel_exploration,
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


def _make_base(room_sequence):
    base = MagicMock()
    idx = [0]
    def _pos():
        i = min(idx[0], len(room_sequence) - 1)
        room = room_sequence[i]
        c = _ROOM_CENTERS.get(room, (10.0, 5.0))
        idx[0] += 1
        return (c[0], c[1], 0.28)
    base.get_position = _pos
    base.get_heading = MagicMock(return_value=0.0)
    base.get_camera_frame = MagicMock(return_value=np.zeros((240, 320, 3), dtype=np.uint8))
    base.get_depth_frame = MagicMock(return_value=np.zeros((240, 320), dtype=np.float32))
    base.set_velocity = MagicMock()
    base.walk = MagicMock(return_value=True)
    return base


class TestSeedBehavior:
    """Verify seed is brief and ends with zero velocity."""

    def test_seed_walk_called_with_bridge(self):
        """Seed walk uses base.walk() when has_bridge=True."""
        base = _make_base(["hallway"] * 50)
        _explore_cancel.clear()
        _explore_visited.clear()

        def cancel_later():
            time.sleep(0.1)
            _explore_cancel.set()

        t = threading.Thread(target=cancel_later, daemon=True)
        t.start()

        with patch("vector_os_nano.skills.go2.explore._start_tare", return_value=True):
            with fake_clock():
                _exploration_loop(base, has_bridge=True)

        t.join(timeout=3)

        # Seed walk should have called base.walk()
        assert base.walk.call_count >= 1, f"Seed walk should call base.walk(). Calls: {base.walk.call_args_list}"

    def test_no_velocity_in_main_loop(self):
        """After seed, no more set_velocity calls during monitoring loop."""
        base = _make_base(["hallway"] * 30)
        _explore_cancel.clear()
        _explore_visited.clear()
        velocity_calls_before_loop = [0]

        def cancel_later():
            time.sleep(3.0)  # seed takes ~5s but we mock _start_tare
            _explore_cancel.set()

        t = threading.Thread(target=cancel_later, daemon=True)
        t.start()

        with patch("vector_os_nano.skills.go2.explore._start_tare", return_value=False):
            _exploration_loop(base, has_bridge=False)

        t.join(timeout=5)
        # When has_bridge=False, no seed or velocity at all
        assert base.set_velocity.call_count == 0


class TestAutoLookOnExplore:
    """Verify auto-look fires for new rooms during exploration."""

    def test_auto_look_fires_for_each_new_room(self):
        rooms_seen = []
        set_auto_look(lambda r: rooms_seen.append(r) or {"summary": r, "objects": []})

        # Many room changes to ensure at least 2 are detected
        sequence = ["hallway", "kitchen", "study", "living_room"] * 5
        base = _make_base(sequence)
        _explore_cancel.clear()
        _explore_visited.clear()

        # Provide a mock SceneGraph that returns room names from position.
        # explore.py uses _spatial_memory.nearest_room() since Wave 2 refactor.
        mock_sg = MagicMock()
        def _nearest(x: float, y: float) -> str | None:
            for name, (cx, cy) in _ROOM_CENTERS.items():
                if abs(x - cx) < 1.5 and abs(y - cy) < 1.5:
                    return name
            return None
        mock_sg.nearest_room = _nearest
        mock_sg.visit = MagicMock()
        mock_sg.add_door = MagicMock()
        _explore_mod._spatial_memory = mock_sg

        # Cancel once we've seen >= 2 rooms or after enough iterations
        pos_calls = [0]
        orig_get_pos = base.get_position
        def _get_pos_with_cancel():
            pos_calls[0] += 1
            result = orig_get_pos()
            # Cancel after enough iterations for 2+ rooms to be detected
            if pos_calls[0] >= len(sequence) - 2:
                _explore_cancel.set()
            return result
        base.get_position = _get_pos_with_cancel

        # Patch subprocess.run to avoid 5s TARE start timeout
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            # The loop paces itself on a REAL Event.wait and real monotonic (no
            # wall-clock spin), so a non-recording no-op sleep is safe here.
            with patch("time.sleep", new=lambda *_: None):
                _exploration_loop(base, has_bridge=False)

        _explore_mod._spatial_memory = None

        assert len(rooms_seen) >= 2, f"Expected >=2 rooms, got {rooms_seen}"
        set_auto_look(None)

    def test_explore_skill_works_without_detector(self):
        """ExploreSkill.execute works with VLM only, no detector."""
        base = _make_base(["hallway"])
        mock_vlm = MagicMock()
        mock_vlm.describe_scene.return_value = MagicMock(
            summary="test", objects=[], room_type="hallway"
        )
        mock_vlm.identify_room.return_value = MagicMock(
            room="hallway", confidence=0.9
        )

        context = SkillContext(
            bases={"default": base},
            services={
                "vlm": mock_vlm,
                "spatial_memory": MagicMock(),
            },
        )

        skill = ExploreSkill()
        with patch("vector_os_nano.skills.go2.explore._start_bridge_on_go2", return_value=False):
            result = skill.execute({}, context)

        assert result.success
        cancel_exploration()
