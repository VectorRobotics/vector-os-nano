# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""The detect capability routes between two detector backends, and every box carries
the provenance of its frame (perception:producer-box-no-provenance)."""
from __future__ import annotations

import numpy as np
import pytest

from vector_os_nano.interfaces.perception import Detection
from vector_os_nano.perception.detector_capability import DetectorCapability
from vector_os_nano.perception.rynnbrain import RynnBrainDetector, parse_reply
from vector_os_nano.interfaces.physics import RGBDFrame


class _Det:
    def __init__(self, label: str) -> None:
        self.label, self.calls = label, []

    def detect(self, rgb, query):
        self.calls.append(query)
        return [Detection(label=self.label, bbox=(1.0, 2.0, 30.0, 40.0), confidence=0.8)]


class _Snap:
    def capture(self):
        return RGBDFrame(rgb=np.zeros((48, 64, 3), np.uint8), depth=None, seg=None, K=None,
                         cam_pos=np.array([10.0, 3.0, 0.4]), cam_mat=np.eye(3).ravel(),
                         sim_time=12.5, camera="d435_rgb")


def test_backend_routes_to_the_named_model_and_boxes_carry_their_frame():
    gd, rb = _Det("a green bottle"), _Det("green bottle")
    cap = DetectorCapability(detector=gd, perception=_Snap(), rynnbrain=rb)
    res = cap.invoke({"query": "green bottle"}, None)
    assert res.success and gd.calls == ["green bottle"] and rb.calls == []
    assert res.output["backend"] == "grounding-dino"
    frame = res.output["detections"][0]["frame"]
    assert frame["size"] == [64, 48] and frame["sim_time"] == 12.5
    assert frame["cam_pos"] == [10.0, 3.0, 0.4] and frame["frame_id"] == "d435_rgb@12.5000"
    res = cap.invoke({"query": "绿色的瓶子", "backend": "rynnbrain"}, None)
    assert res.success and rb.calls == ["绿色的瓶子"] and res.output["backend"] == "rynnbrain"


def test_an_unknown_backend_fails_loud_with_the_valid_set():
    res = DetectorCapability(detector=_Det("x"), perception=_Snap()).invoke(
        {"query": "can", "backend": "yolo"}, None)
    assert not res.success and "grounding-dino" in res.error and "rynnbrain" in res.error


def test_rynnbrain_down_is_a_clear_failure_not_a_crash():
    pytest.importorskip("PIL", reason="pillow ([perception] extra): rynnbrain encodes the frame first")
    cap = DetectorCapability(perception=_Snap(), rynnbrain=RynnBrainDetector("http://127.0.0.1:9", timeout=0.5))
    res = cap.invoke({"query": "green bottle", "backend": "rynnbrain"}, None)
    assert not res.success and "not reachable" in res.error and "grounding-dino" in res.error


def test_rynnbrain_reply_is_scaled_from_thousandths_to_pixels():
    (d,) = parse_reply("<object> (100, 250), (300, 750) </object>", 640, 480, "green bottle")
    assert d.bbox == pytest.approx((64.0, 120.0, 192.0, 360.0)) and d.label == "green bottle"
    assert parse_reply("I cannot see it.", 640, 480, "x") == []


def test_native_detect_tool_passes_the_backend_and_names_it():
    pytest.importorskip("PIL", reason="pillow ([perception] extra): rynnbrain encodes the frame first")
    from vector_os_nano.vcli.native_tools import _NativeDetectTool

    gd, rb = _Det("a can"), _Det("can")
    tool = _NativeDetectTool(DetectorCapability(detector=gd, perception=_Snap(), rynnbrain=rb))
    out = tool.execute({"query": "red can", "backend": "rynnbrain"}, None)
    assert rb.calls == ["red can"] and "rynnbrain localized 1 object" in out.content
    assert "backend" in tool.input_schema["properties"]
    bad = DetectorCapability(perception=_Snap(), rynnbrain=RynnBrainDetector("http://127.0.0.1:9", timeout=0.5))
    out = _NativeDetectTool(bad).execute({"query": "red can", "backend": "rynnbrain"}, None)
    assert out.is_error and "not reachable" in out.content
