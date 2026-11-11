# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Go2GraspPerception: one snapshot, self-filtered depth, segmenter provenance."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from vector_os_nano.interfaces.physics import RGBDFrame
from vector_os_nano.perception import go2_grasp_perception as g2


class _Base:
    def __init__(self) -> None:
        self.calls = []

    def get_rgbd_frame(self, w, h, kinds):
        self.calls.append((w, h, kinds))
        seg = np.full((h, w), -1, np.int32)
        seg[:2] = 7  # the arm's body
        K = np.array([[30.0, 0.0, w / 2], [0.0, 30.0, h / 2], [0.0, 0.0, 1.0]])
        return RGBDFrame(rgb=np.zeros((h, w, 3), np.uint8), depth=np.ones((h, w), np.float32),
                         seg=seg, K=K, cam_pos=np.zeros(3), cam_mat=np.eye(3), sim_time=3.0,
                         camera="d435_rgb")

    def self_mask(self, seg):
        return seg == 7


class _NoEdgeTAM:
    def init_track(self, image, bboxes):
        raise RuntimeError("weights absent")


def test_capture_is_one_frame_with_the_arm_zeroed():
    base = _Base()
    f = g2.Go2GraspPerception(base, width=32, height=24).capture()
    assert base.calls == [(32, 24, ("rgb", "depth", "seg"))]  # ONE render, every field
    assert (f.depth[:2] == 0).all() and (f.depth[2:] == 1).all()
    assert (f.K[0, 2], f.K[1, 2]) == (16, 12) and f.sim_time == 3.0  # the base's frame, depth replaced


class _EmptyEdgeTAM:
    def init_track(self, image, bboxes):
        return [{"mask": np.zeros(image.shape[:2], np.uint8)}]


def test_a_segmenter_that_cannot_run_is_an_error_never_a_box_rect():
    """perception:edgetam-silent-boxrect-fallback: a missing/failed EdgeTAM was
    replaced by a box-rectangle mask; now the caller sees the failure and its cause."""
    p = g2.Go2GraspPerception(_Base(), tracker=_NoEdgeTAM(), width=32, height=24)
    p.last_segmenter = "edgetam"  # from an earlier call: must not survive a failure
    with pytest.raises(RuntimeError, match="weights absent"):
        p.segment(np.zeros((24, 32, 3), np.uint8), (4, 4, 10, 12))
    assert p.last_segmenter is None


def test_segmenter_provenance_and_a_loud_box_rect_for_an_empty_mask(caplog, monkeypatch):
    """perception:segmenter-provenance-missing: which segmenter made the mask is recorded."""
    monkeypatch.setattr(g2, "_warned_box_rect", False)
    p = g2.Go2GraspPerception(_Base(), tracker=_EmptyEdgeTAM(), width=32, height=24)
    rgb = np.zeros((24, 32, 3), np.uint8)
    with caplog.at_level(logging.WARNING):
        m = p.segment(rgb, (4, 4, 10, 12))
        p.segment(rgb, (4, 4, 10, 12))
    assert int(m.sum()) == 6 * 8 and p.last_segmenter == "box_rect"
    assert sum("box-rect mask" in r.message for r in caplog.records) == 1  # once
    p.front_object_mask(rgb, None)
    assert p.last_segmenter == "hsv"
