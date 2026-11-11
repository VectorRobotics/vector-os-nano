# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""CameraPerception: one snapshot, self-filtered depth, segmenter provenance."""
from __future__ import annotations

import logging

import numpy as np

from vector_os_nano.interfaces.devices import CameraFrame
from vector_os_nano.perception import camera_perception as cp


class _Camera:
    def __init__(self) -> None:
        self.calls = 0

    def capture(self, width: int = 32, height: int = 24) -> CameraFrame:
        self.calls += 1
        mask = np.zeros((height, width), bool)
        mask[:2] = True  # the arm's pixels
        K = np.array([[30.0, 0.0, width / 2], [0.0, 30.0, height / 2], [0.0, 0.0, 1.0]])
        return CameraFrame(rgb=np.zeros((height, width, 3), np.uint8), depth=np.ones((height, width), np.float32),
                           self_mask=mask, K=K, cam_pos=np.zeros(3), cam_mat=np.eye(3), sim_time=3.0,
                           camera="d435_rgb")


class _NoEdgeTAM:
    def init_track(self, image, bboxes):
        raise RuntimeError("weights absent")


def test_capture_is_one_frame_with_the_robot_zeroed():
    cam = _Camera()
    f = cp.CameraPerception(cam).capture()
    assert cam.calls == 1  # ONE camera instant, every field
    assert (f.depth[:2] == 0).all() and (f.depth[2:] == 1).all()
    assert (f.K[0, 2], f.K[1, 2]) == (16, 12) and f.sim_time == 3.0
    assert not hasattr(f, "seg")  # no segmentation ever reaches perception


def test_segmenter_provenance_and_a_loud_box_rect_fallback(caplog, monkeypatch):
    """perception:segmenter-provenance-missing + edgetam-provisioning-silent-fallback:
    the box-rect fallback was an INFO line and invisible to the result."""
    monkeypatch.setattr(cp, "_warned_box_rect", False)
    p = cp.CameraPerception(_Camera(), tracker=_NoEdgeTAM())
    rgb = np.zeros((24, 32, 3), np.uint8)
    with caplog.at_level(logging.WARNING):
        m = p.segment(rgb, (4, 4, 10, 12))
        p.segment(rgb, (4, 4, 10, 12))
    assert int(m.sum()) == 6 * 8 and p.last_segmenter == "box_rect"
    assert sum("box-rect fallback" in r.message for r in caplog.records) == 1  # once
    p.front_object_mask(rgb, None)
    assert p.last_segmenter == "hsv"
