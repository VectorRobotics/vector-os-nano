# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""perception_grasp MODEL routing: NAMED->detector, colour->front, deictic->front.

Proves the new learned-model route ("route each instruction to the right
model+skill") without a sim or a real model: the perception object is mocked and
records which resolver fired. The detector route calls perception.detect (and NOT
front_object_mask); the colour/deictic routes call front_object_mask (and NOT
detect) — keeping the D47 colour path and the deictic path byte-behaviour-identical.
"""
from __future__ import annotations

import numpy as np
import pytest

from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.types import Detection
from vector_os_nano.core.world_model import WorldModel
from vector_os_nano.perception.depth_projection import mujoco_intrinsics
from vector_os_nano.skills.perception_grasp import (
    PerceptionGraspSkill,
    _names_object,
)
from vector_os_nano.perception.front_object import parse_color

_W, _H = 64, 48
_INTR = mujoco_intrinsics(_W, _H, vfov_deg=42.0)
# Camera facing +X from (9, 3, 0.8): depth=1 at centre pixel -> world z=0.8,
# satisfying _MIN_GRASP_Z (0.12 m) so the routing tests succeed as intended.
# MuJoCo xmat columns: right(+y), up(-z), -forward(-x) → cam_forward = +x.
_CAM_XPOS = np.array([9.0, 3.0, 0.8], dtype=np.float64)
_CAM_XMAT = np.array([
    0, 1, 0,
    0, 0, -1,
    -1, 0, 0,
], dtype=np.float64)


def _depth_mask():
    depth = np.zeros((_H, _W), dtype=np.float32)
    color = np.zeros((_H, _W, 3), dtype=np.uint8)
    mask = np.zeros((_H, _W), dtype=np.uint8)
    for v in range(21, 28):
        for u in range(29, 36):
            depth[v, u] = 1.0
            mask[v, u] = 1
    return depth, color, mask


class RoutingPerception:
    """Records whether detect (detector route) or front_object_mask (classical) fired."""

    def __init__(self, detections=None):
        self._depth, self._color, self._mask = _depth_mask()
        self._dets = detections if detections is not None else [
            Detection(label="can", bbox=(29.0, 21.0, 35.0, 27.0), confidence=0.9)
        ]
        self.calls: list[str] = []

    def capture(self):
        from vector_os_nano.sim.runtime import RGBDFrame
        return RGBDFrame(rgb=self._color, depth=self._depth, seg=None, K=_INTR,
                         cam_pos=_CAM_XPOS, cam_mat=_CAM_XMAT, sim_time=0.0, camera="d435_rgb")

    def detect_in(self, rgb, query):
        return self.detect(query)

    def get_color_frame(self):
        return self._color

    def get_depth_frame(self):
        return self._depth

    def get_intrinsics(self):
        return _INTR

    def get_camera_pose(self):
        return _CAM_XPOS, _CAM_XMAT

    def detect(self, query):
        self.calls.append(f"detect:{query}")
        return list(self._dets)

    def segment(self, image, bbox):
        self.calls.append("segment")
        return self._mask

    def front_object_mask(self, rgb=None, depth=None, *, color=None):
        self.calls.append(f"front:{color}" if color else "front")
        return self._mask


class _Arm:
    name = "piper"

    def ik_top_down(self, xyz, current_joints=None):
        return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    def move_joints(self, q, duration=1.0):
        return True

    def get_joint_positions(self):
        return [0.0] * 6

    def fk(self, q):
        return ([0.0, 0.0, 0.0], np.eye(3))


class _Gripper:
    def __init__(self):
        self._closed = False

    def open(self):
        self._closed = False
        return True

    def close(self):
        self._closed = True
        return True

    def is_holding(self):
        return self._closed


def _ctx(perc):
    return SkillContext(
        arm=_Arm(), gripper=_Gripper(), world_model=WorldModel(),
        perception=perc, config={"world": "apartment"},
    )


# --- the _names_object classifier (pure) ------------------------------------


@pytest.mark.parametrize("q", ["罐子", "瓶子", "the can", "red bottle", "拿起那个杯子", "banana"])
def test_named_object_detected(q):
    assert _names_object(q) is True


@pytest.mark.parametrize("q", ["前面的东西", "面前的", "抓红色的", "蓝色的", "东西", "object", ""])
def test_pure_colour_or_deictic_not_named(q):
    assert _names_object(q) is False


# --- routing behaviour ------------------------------------------------------


def test_named_query_routes_to_detector():
    """A NAMED object -> perception.detect (the learned model), NOT front_object."""
    perc = RoutingPerception()
    res = PerceptionGraspSkill().execute({"query": "罐子"}, _ctx(perc))
    assert res.success is True
    assert any(c.startswith("detect:") for c in perc.calls)
    assert not any(c.startswith("front") for c in perc.calls)


def test_named_with_colour_routes_to_hsv_not_detector():
    """CHANGE 1 (R40): '红色的罐子' has a noun AND a colour — colour wins, routes to
    front_object HSV resolver (NOT the detector). The HSV resolver reliably
    disambiguates by hue; grounding-dino is intermittent at 22 cm inter-can spacing.
    The verify LABEL still maps to the colour's scene key.
    """
    perc = RoutingPerception()
    res = PerceptionGraspSkill().execute({"query": "红色的罐子"}, _ctx(perc))
    assert res.success is True
    assert "front:red" in perc.calls, (
        f"expected front_object_mask(color='red') for colour+noun query, got: {perc.calls}")
    assert not any(c.startswith("detect:") for c in perc.calls), (
        f"grounding-dino detect was called despite a colour query: {perc.calls}")
    assert res.result_data["detection_label"] == "pickable_can_red"


def test_pure_colour_routes_to_front_object_with_color():
    """'抓红色的' (no noun) -> classical front_object(color='red'), NOT the detector."""
    perc = RoutingPerception()
    res = PerceptionGraspSkill().execute({"query": "抓红色的"}, _ctx(perc))
    assert res.success is True
    assert "front:red" in perc.calls
    assert not any(c.startswith("detect:") for c in perc.calls)
    assert res.result_data["detection_label"] == "pickable_can_red"


def test_deictic_routes_to_front_object():
    """'前面的东西' -> classical front_object(color=None), NOT the detector."""
    perc = RoutingPerception()
    res = PerceptionGraspSkill().execute({"query": "前面的东西"}, _ctx(perc))
    assert res.success is True
    assert "front" in perc.calls
    assert not any(c.startswith("detect:") for c in perc.calls)


# --- D48 caveat 1: PERCEPTUAL colour selection among detector boxes ----------


def test_named_colour_query_routes_to_hsv_and_sets_verify_label():
    """CHANGE 1 (R40): '红色的罐子' contains a noun AND a colour. parse_color finds
    'red' and _names_object finds '罐子', but colour presence takes precedence:
    use_front_resolver=True routes to front_object HSV, not the detector. The
    verify LABEL still maps to the colour's scene key (pickable_can_red)."""
    assert parse_color("红色的罐子") == "red"
    assert _names_object("红色的罐子") is True
    perc = RoutingPerception()
    res = PerceptionGraspSkill().execute({"query": "红色的罐子"}, _ctx(perc))
    assert res.success is True
    assert "front:red" in perc.calls, (
        f"expected HSV resolver for colour+noun query, got: {perc.calls}")
    assert not any(c.startswith("detect:") for c in perc.calls)
    assert res.result_data["detection_label"] == "pickable_can_red"


def test_select_detection_prefers_colour_match_over_max_confidence():
    """The crux: a colour-matching box wins even when a non-matching box scores higher.

    This is what makes colour selection PERCEPTUAL rather than centrality-staged:
    the green box has the highest confidence (scene authors green dead-centre) but
    'red' must still pick the lower-confidence RED box."""
    boxes = [
        Detection(label="green bottle", bbox=(10.0, 10.0, 20.0, 20.0), confidence=0.90),
        Detection(label="red can", bbox=(40.0, 10.0, 50.0, 20.0), confidence=0.55),
        Detection(label="blue bottle", bbox=(70.0, 10.0, 80.0, 20.0), confidence=0.50),
    ]
    chosen = PerceptionGraspSkill._select_detection(boxes, "red")
    assert chosen.label == "red can"
    chosen_blue = PerceptionGraspSkill._select_detection(boxes, "blue")
    assert chosen_blue.label == "blue bottle"
    chosen_green = PerceptionGraspSkill._select_detection(boxes, "green")
    assert chosen_green.label == "green bottle"


def test_select_detection_no_colour_is_plain_max_confidence():
    boxes = [
        Detection(label="bottle", bbox=(10.0, 10.0, 20.0, 20.0), confidence=0.40),
        Detection(label="can", bbox=(40.0, 10.0, 50.0, 20.0), confidence=0.80),
    ]
    assert PerceptionGraspSkill._select_detection(boxes, None).label == "can"


def test_select_detection_falls_back_when_colour_absent_from_labels():
    """If no box label names the colour (under-grounded adjective), fall back to
    plain max-confidence rather than failing."""
    boxes = [
        Detection(label="bottle", bbox=(10.0, 10.0, 20.0, 20.0), confidence=0.40),
        Detection(label="can", bbox=(40.0, 10.0, 50.0, 20.0), confidence=0.80),
    ]
    assert PerceptionGraspSkill._select_detection(boxes, "red").label == "can"
