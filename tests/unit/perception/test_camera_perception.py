# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""CameraPerception: one snapshot, self-filtered depth, segmenter provenance."""
from __future__ import annotations

import logging

import numpy as np
import pytest

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


def test_named_capture_keeps_the_default_view_and_its_self_mask():
    from vector_os_nano.perception.object_localizer import capture_rgbd
    head, wrist = _Camera(), _Camera()
    perception = cp.CameraPerception(head, cameras={'head': head, 'wrist': wrist})
    frame = capture_rgbd(perception, camera='wrist')
    assert (head.calls, wrist.calls) == (0, 1)
    assert (frame[1][:2] == 0).all() and frame[2].width == 32
    perception.capture()
    assert (head.calls, wrist.calls) == (1, 1)
    with pytest.raises(ValueError, match='unknown camera.*head.*wrist'):
        perception.capture(camera='missing')
    assert (head.calls, wrist.calls) == (1, 1)


def test_named_view_cannot_reuse_another_cameras_box_even_at_the_same_pose():
    from vector_os_nano.skills.grasp_perception import _fresh
    rgb = np.zeros((24, 32, 3))
    frame = dict(size=[32, 24], cam_pos=[0, 0, 0], cam_mat=np.eye(3).tolist(), frame_id='head@3.0000')
    assert not _fresh(frame, rgb, [0, 0, 0], np.eye(3), 'wrist')
    assert _fresh(frame, rgb, [0, 0, 0], np.eye(3), 'head')
    del frame['frame_id']
    assert not _fresh(frame, rgb, [0, 0, 0], np.eye(3), 'head')


class _EmptyEdgeTAM:
    def init_track(self, image, bboxes):
        return [{"mask": np.zeros(image.shape[:2], np.uint8)}]


def test_a_segmenter_that_cannot_run_is_an_error_never_a_box_rect():
    """perception:edgetam-silent-boxrect-fallback: a missing/failed EdgeTAM was
    replaced by a box-rectangle mask; now the caller sees the failure and its cause."""
    p = cp.CameraPerception(_Camera(), tracker=_NoEdgeTAM())
    p.last_segmenter = "edgetam"  # from an earlier call: must not survive a failure
    with pytest.raises(RuntimeError, match="weights absent"):
        p.segment(np.zeros((24, 32, 3), np.uint8), (4, 4, 10, 12))
    assert p.last_segmenter is None


def test_segmenter_provenance_and_a_loud_box_rect_for_an_empty_mask(caplog, monkeypatch):
    """perception:segmenter-provenance-missing: which segmenter made the mask is recorded."""
    monkeypatch.setattr(cp, "_warned_box_rect", False)
    p = cp.CameraPerception(_Camera(), tracker=_EmptyEdgeTAM())
    rgb = np.zeros((24, 32, 3), np.uint8)
    with caplog.at_level(logging.WARNING):
        m = p.segment(rgb, (4, 4, 10, 12))
        p.segment(rgb, (4, 4, 10, 12))
    assert int(m.sum()) == 6 * 8 and p.last_segmenter == "box_rect"
    assert sum("box-rect mask" in r.message for r in caplog.records) == 1  # once
    p.front_object_mask(rgb, None)
    assert p.last_segmenter == "hsv"
