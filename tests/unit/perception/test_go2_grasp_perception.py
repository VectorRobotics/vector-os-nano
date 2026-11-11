# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Go2GraspPerception: one snapshot, self-filtered depth, segmenter provenance."""
from __future__ import annotations

import logging

import numpy as np

from vector_os_nano.perception import go2_grasp_perception as g2
from vector_os_nano.perception.depth_projection import mujoco_intrinsics
from vector_os_nano.sim.runtime import RGBDFrame


class _Base:
    def __init__(self) -> None:
        self.calls = []

    def get_rgbd_frame(self, w, h, kinds):
        self.calls.append((w, h, kinds))
        seg = np.full((h, w), -1, np.int32)
        seg[:2] = 7  # the arm's geom
        return RGBDFrame(rgb=np.zeros((h, w, 3), np.uint8), depth=np.ones((h, w), np.float32),
                         seg=seg, K=mujoco_intrinsics(w, h), cam_pos=np.zeros(3),
                         cam_mat=np.eye(3).ravel(), sim_time=3.0, camera="d435_rgb")

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
    assert (f.K.width, f.K.height) == (32, 24)


def test_segmenter_provenance_and_a_loud_box_rect_fallback(caplog, monkeypatch):
    """perception:segmenter-provenance-missing + edgetam-provisioning-silent-fallback:
    the box-rect fallback was an INFO line and invisible to the result."""
    monkeypatch.setattr(g2, "_warned_box_rect", False)
    p = g2.Go2GraspPerception(_Base(), tracker=_NoEdgeTAM(), width=32, height=24)
    rgb = np.zeros((24, 32, 3), np.uint8)
    with caplog.at_level(logging.WARNING):
        m = p.segment(rgb, (4, 4, 10, 12))
        p.segment(rgb, (4, 4, 10, 12))
    assert int(m.sum()) == 6 * 8 and p.last_segmenter == "box_rect"
    assert sum("box-rect fallback" in r.message for r in caplog.records) == 1  # once
    p.front_object_mask(rgb, None)
    assert p.last_segmenter == "hsv"
