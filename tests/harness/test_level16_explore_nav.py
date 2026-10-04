# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Level 16: the TARE exploration monitor (explore backend 'tare').

Verifies that the monitor loop:
- seeds TARE with one brief walk (it does not override the nav stack);
- sends no velocity command while monitoring.
(The auto-look hook had no production setter and is deleted, WP08.)
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
    _exploration_loop,
    _explore_cancel,
    _explore_visited,
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
    """The seed is one brief walk; the monitor itself never commands velocity."""

    def _run(self, base, cancel_after: float) -> None:
        _explore_cancel.clear()
        _explore_visited.clear()

        def cancel_later():
            time.sleep(cancel_after)
            _explore_cancel.set()

        t = threading.Thread(target=cancel_later, daemon=True)
        t.start()
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):  # /start_exploration
            with fake_clock():
                _exploration_loop(base)
        t.join(timeout=3)

    def test_seed_walk_runs_once(self):
        base = _make_base(["hallway"] * 50)
        self._run(base, 0.1)
        assert base.walk.call_count == 1, base.walk.call_args_list

    def test_no_velocity_in_main_loop(self):
        base = _make_base(["hallway"] * 30)
        self._run(base, 0.5)
        assert base.set_velocity.call_count == 0
        assert not _explore_mod.is_exploring()
