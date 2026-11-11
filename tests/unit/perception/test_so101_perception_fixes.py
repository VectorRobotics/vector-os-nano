# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""SO-101 real-arm perception: one frameset per read, one EdgeTAM session owner,
a lost track times out, Moondream fails loud (no hardware needed)."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from vector_os_nano.core.types import CameraIntrinsics
from vector_os_nano.perception import pipeline as pl

_K = CameraIntrinsics(fx=100.0, fy=100.0, cx=8.0, cy=6.0, width=16, height=12)


class _Cam:
    def __init__(self) -> None:
        self.aligned = 0

    def get_aligned_frames(self):
        self.aligned += 1
        return np.zeros((12, 16, 3), np.uint8), np.full((12, 16), 500, np.uint16)

    def get_color_frame(self):
        raise AssertionError("color from its own frameset")

    def get_depth_frame(self):
        raise AssertionError("depth from its own frameset")

    def get_intrinsics(self):
        return _K


class _Tracker:
    def __init__(self) -> None:
        self.process_calls, self.stopped = 0, False

    def is_tracking(self):
        return not self.stopped

    def process_image(self, color):
        self.process_calls += 1
        return []

    def stop(self):
        self.stopped = True


def test_color_and_depth_come_from_one_frameset():
    """perception:realsense-split-framesets — color and depth were two wait_for_frames."""
    cam = _Cam()
    p = pl.PerceptionPipeline(camera=cam)
    assert p.get_point_cloud().shape[1] == 3 and cam.aligned == 1


def test_update_reads_the_background_loop_and_a_lost_track_times_out(monkeypatch):
    """perception:pipeline-concurrent-tracker — update() and the background loop both
    stepped one EdgeTAM session; a lost track's pose stayed forever."""
    tr = _Tracker()
    p = pl.PerceptionPipeline(camera=_Cam(), tracker=tr)
    p._last_tracked = ["latest"]  # type: ignore[list-item]
    hold = threading.Event()
    p._tracking_thread = threading.Thread(target=hold.wait, daemon=True)
    p._tracking_thread.start()  # "the background loop is alive"
    try:
        assert p.update() == ["latest"] and tr.process_calls == 0
    finally:
        hold.set()
        p._tracking_thread.join()
        p._tracking_thread = None

    monkeypatch.setattr(pl, "_TRACKING_TIMEOUT_S", 0.2)
    p.start_continuous_tracking()
    try:
        deadline = time.monotonic() + 3.0
        while not tr.stopped and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        p.stop_continuous_tracking()
    assert tr.stopped and p._last_tracked == [] and p._tracked_objects == []


def test_moondream_without_any_backend_fails_at_construction(monkeypatch):
    """perception:moondream-modes-broken — Station mode built a None client that crashed
    later (or detect returned [] silently)."""
    from vector_os_nano.perception import vlm

    monkeypatch.delenv("MOONDREAM_MODEL", raising=False)
    monkeypatch.delenv("MOONDREAM_API_KEY", raising=False)
    monkeypatch.setattr(vlm, "_STATION_ENDPOINT", "http://127.0.0.1:9/v1")
    with pytest.raises(RuntimeError, match="No Moondream backend"):
        vlm.VLMDetector()


def test_the_realsense_serial_is_config_not_code():
    from vector_os_nano.perception.realsense import RealSenseCamera

    assert RealSenseCamera()._serial is None
    assert RealSenseCamera(serial="123")._serial == "123"
