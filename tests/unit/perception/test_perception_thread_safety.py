# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""The shared detector and the EdgeTAM session serve one call at a time.

perception:edgetam-detector-thread-safety — explore's auto-observe, the grasp skill
and the detect capability call them from different threads; the lazy loads and the
EdgeTAM inference state are not re-entrant.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from vector_os_nano.perception import grounding_dino
from vector_os_nano.perception.camera_perception import CameraPerception


class _Overlap:
    """Records the most callers ever inside the guarded section at once."""

    def __init__(self) -> None:
        self.inside = 0
        self.peak = 0
        self._lock = threading.Lock()

    def enter(self) -> None:
        with self._lock:
            self.inside += 1
            self.peak = max(self.peak, self.inside)
        time.sleep(0.01)
        with self._lock:
            self.inside -= 1


def _hammer(fn) -> None:
    threads = [threading.Thread(target=fn) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_shared_detector_serializes_load_and_inference(monkeypatch) -> None:
    det = grounding_dino.GroundingDinoDetector()
    seen = _Overlap()
    monkeypatch.setattr(det, "_detect", lambda rgb, q: seen.enter() or [])
    _hammer(lambda: det.detect(np.zeros((4, 4, 3), np.uint8), "bottle"))
    assert seen.peak == 1


def test_shared_detector_singleton_is_built_once(monkeypatch) -> None:
    monkeypatch.setattr(grounding_dino, "_SHARED", None)
    built = []

    class _Slow(grounding_dino.GroundingDinoDetector):
        def __init__(self) -> None:
            built.append(1)
            time.sleep(0.01)
            super().__init__()

    monkeypatch.setattr(grounding_dino, "GroundingDinoDetector", _Slow)
    got = []
    _hammer(lambda: got.append(grounding_dino.get_shared_detector()))
    assert len(built) == 1 and len({id(g) for g in got}) == 1


def test_camera_perception_serializes_segmenter_sessions() -> None:
    seen = _Overlap()

    class _Tracker:
        def init_track(self, image, bboxes):
            seen.enter()
            return [{"mask": np.ones(image.shape[:2], np.uint8)}]

    p = CameraPerception(camera=None, tracker=_Tracker())
    img = np.zeros((6, 8, 3), np.uint8)
    _hammer(lambda: p.segment(img, (1, 1, 4, 4)))
    assert seen.peak == 1
