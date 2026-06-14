# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #8 R9 — color target detection (pure, offline, no GL).

detect_targets(rgb) finds the red/blue/green target boxes in a camera frame by
colour segmentation — deterministic, no new deps, robust on MuJoCo's basic
rendering for distinct colours (the DQ-10 panel's domain-gap concern is small
for saturated colours). Foundation for the R10 vision closed loop. Tested on
synthetic frames (no sim).
"""
from __future__ import annotations

import numpy as np

from vector_os_nano.perception.color_targets import detect_targets


def _frame(h=64, w=128):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestDetect:
    def test_empty_frame_no_detections(self):
        assert detect_targets(_frame()) == []

    def test_red_block_left_blue_block_right(self):
        f = _frame()
        f[20:44, 8:40] = (220, 20, 20)      # red, left
        f[20:44, 88:120] = (20, 20, 220)    # blue, right
        dets = {d["label"]: d for d in detect_targets(f)}
        assert "red" in dets and "blue" in dets
        # red is on the left → normalised x < 0; blue on the right → > 0
        assert dets["red"]["x_norm"] < 0
        assert dets["blue"]["x_norm"] > 0
        # areas are positive fractions
        assert 0 < dets["red"]["area_frac"] < 1

    def test_green_center(self):
        f = _frame()
        f[24:40, 56:72] = (20, 200, 20)
        dets = detect_targets(f)
        assert len(dets) == 1 and dets[0]["label"] == "green"
        assert abs(dets[0]["x_norm"]) < 0.2     # roughly centred

    def test_min_area_filters_speckle(self):
        f = _frame()
        f[0:2, 0:2] = (220, 20, 20)          # a 2x2 red speck
        # default min area should reject a tiny speck
        assert detect_targets(f) == []
        # a low threshold detects it
        assert any(d["label"] == "red" for d in detect_targets(f, min_area_frac=0.0))
