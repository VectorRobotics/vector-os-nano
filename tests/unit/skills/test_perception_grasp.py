# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""RED-4 — PerceptionGraspSkill flow, fully mocked (no GPU / GL / model / sim).

Proves the honest perception-driven grasp end-to-end at the skill level:
  - the 3D grasp point is computed from the PERCEIVED depth + mask (via the pure
    grasp_point_from_rgbd), NOT from a ground-truth object pose — the world model
    is EMPTY and the arm exposes no get_object_positions, so any GT shortcut fails;
  - the proven PickTopDownSkill motion is REUSED via its params['target_xyz'] seam
    (no re-implemented motion);
  - each perception failure FAILS LOUD with a specific diagnosis, never a fabricated
    point or a silent GT fallback.
"""
from __future__ import annotations

import numpy as np
import pytest

from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.types import Detection
from vector_os_nano.core.world_model import WorldModel
from tests.harness.camera import pinhole
from vector_os_nano.perception.grasp_point import grasp_point_from_rgbd
from vector_os_nano.skills.perception_grasp import PerceptionGraspSkill

_W, _H = 64, 48
_INTR = pinhole(_W, _H, 42.0)
_K = np.array([[_INTR.fx, 0.0, _INTR.cx], [0.0, _INTR.fy, _INTR.cy], [0.0, 0.0, 1.0]])  # the frame's own K
# A camera pose that puts a 1 m-forward point at a physically plausible world spot:
# camera at (9.0, 3.0, 0.8), facing +X (forward). MuJoCo xmat convention:
#   col0 = cam right = world +y
#   col1 = cam up    = world -z (vertical)
#   col2 = cam -fwd  = world -x  (so cam_forward = +x)
# With depth=1.0 at center pixel this gives world z=0.8 (above the floor),
# satisfying the _MIN_GRASP_Z guard without any special-casing.
_CAM_XPOS = np.array([9.0, 3.0, 0.8], dtype=np.float64)
_CAM_XMAT = np.array([
    0, 1, 0,    # col0 = cam right = world +y
    0, 0, -1,   # col1 = cam up   = world -z
    -1, 0, 0,   # col2 = cam -fwd = world -x (cam forward = +x)
], dtype=np.float64)


def _depth_mask(depth_val: float = 1.0):
    depth = np.zeros((_H, _W), dtype=np.float32)
    color = np.zeros((_H, _W, 3), dtype=np.uint8)
    mask = np.zeros((_H, _W), dtype=np.uint8)
    for v in range(21, 28):
        for u in range(29, 36):
            depth[v, u] = depth_val
            mask[v, u] = 1
    return depth, color, mask


class FakePerception:
    """Mock backend with the exact surface PerceptionGraspSkill consumes."""

    def __init__(self, *, detections=None, mask=None, depth=None, color=None):
        d, c, m = _depth_mask()
        self._depth = depth if depth is not None else d
        self._color = color if color is not None else c
        self._mask = mask if mask is not None else m
        self._dets = detections if detections is not None else [
            Detection(label="banana", bbox=(29.0, 21.0, 35.0, 27.0), confidence=0.9)
        ]
        self.calls: list[str] = []

    def capture(self):
        from vector_os_nano.interfaces.physics import RGBDFrame
        return RGBDFrame(rgb=self._color, depth=self._depth, seg=None, K=_K,
                         cam_pos=_CAM_XPOS, cam_mat=_CAM_XMAT, sim_time=0.0, camera="d435_rgb")

    def detect_in(self, rgb, query):
        return self.detect(query)

    def get_color_frame(self):
        self.calls.append("color")
        return self._color

    def get_depth_frame(self):
        self.calls.append("depth")
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


class FakeArm:
    name = "piper"

    def __init__(self):
        self.ik_calls: list[tuple] = []
        self.move_calls = 0

    def ik_top_down(self, xyz, current_joints=None):
        self.ik_calls.append(tuple(float(v) for v in xyz))
        return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    def move_joints(self, q, duration=1.0):
        self.move_calls += 1
        return True

    def get_joint_positions(self):
        return [0.0] * 6

    def fk(self, q):
        return ([0.0, 0.0, 0.0], np.eye(3))


class FakeGripper:
    def __init__(self, holding_after_close=True):
        self._closed = False
        self._holding_after = holding_after_close

    def open(self):
        self._closed = False
        return True

    def close(self):
        self._closed = True
        return True

    def is_holding(self):
        return self._closed and self._holding_after


def _ctx(perception, *, arm=None, gripper=None, wm=None):
    return SkillContext(
        arm=arm if arm is not None else FakeArm(),
        gripper=gripper if gripper is not None else FakeGripper(),
        world_model=wm if wm is not None else WorldModel(),  # EMPTY but not None
        perception=perception,
        config={"world": "apartment"},
    )


def test_grasp_point_comes_from_perceived_depth_not_ground_truth():
    """The delegated target_xyz must equal grasp_point_from_rgbd(perceived inputs).

    World model is EMPTY and the arm has no get_object_positions — there is NO GT
    to read, so a passing run can only be using the perceived point.
    """
    perc = FakePerception()
    arm = FakeArm()
    skill = PerceptionGraspSkill()
    res = skill.execute({"query": "banana"}, _ctx(perc, arm=arm))

    assert res.success is True
    rd = res.result_data
    assert rd["perceived"] is True
    assert rd["detection_label"] == "banana"

    depth, color, mask = perc._depth, perc._color, perc._mask
    expected = grasp_point_from_rgbd(depth, color, mask, _INTR, _CAM_XPOS, _CAM_XMAT)
    assert expected is not None
    assert rd["grasp_world"] == pytest.approx([expected.x, expected.y, expected.z], abs=1e-6)

    # Motion was DELEGATED (ik solved) and aimed at the perceived x,y.
    assert arm.ik_calls, "PickTopDownSkill motion was not reused (no IK call)"
    assert arm.ik_calls[0][0] == pytest.approx(expected.x, abs=1e-6)
    assert arm.ik_calls[0][1] == pytest.approx(expected.y, abs=1e-6)


def test_pipeline_order_detect_then_segment():
    perc = FakePerception()
    PerceptionGraspSkill().execute({"query": "banana"}, _ctx(perc))
    di = next(i for i, c in enumerate(perc.calls) if c.startswith("detect"))
    si = perc.calls.index("segment")
    assert di < si, "must VLM-detect before EdgeTAM-segment"


def test_no_detections_fails_loud():
    perc = FakePerception(detections=[])
    res = PerceptionGraspSkill().execute({"query": "ghost"}, _ctx(perc))
    assert res.success is False
    assert res.result_data["diagnosis"] == "no_detections"


def test_empty_mask_fails_segmentation():
    perc = FakePerception(mask=np.zeros((_H, _W), dtype=np.uint8))
    res = PerceptionGraspSkill().execute({"query": "banana"}, _ctx(perc))
    assert res.success is False
    assert res.result_data["diagnosis"] == "segmentation_failed"


def test_all_zero_depth_fails_no_depth_points():
    perc = FakePerception(depth=np.zeros((_H, _W), dtype=np.float32))
    res = PerceptionGraspSkill().execute({"query": "banana"}, _ctx(perc))
    assert res.success is False
    assert res.result_data["diagnosis"] == "no_depth_points"


def test_no_perception_fails_loud():
    res = PerceptionGraspSkill().execute({"query": "banana"}, _ctx(None))
    assert res.success is False
    assert res.result_data["diagnosis"] in ("no_perception", "no_camera")


def test_ik_unreachable_surfaced():
    class DeadArm(FakeArm):
        def ik_top_down(self, xyz, current_joints=None):
            return None

    perc = FakePerception()
    res = PerceptionGraspSkill().execute({"query": "banana"}, _ctx(perc, arm=DeadArm()))
    assert res.success is False
    assert res.result_data["diagnosis"] == "ik_unreachable"


class FrontPerception(FakePerception):
    """Adds the deictic front-object resolver surface."""

    def front_object_mask(self, rgb=None, depth=None, *, color=None):
        self.calls.append(f"front:{color}" if color else "front")
        return self._mask


def test_deictic_query_uses_front_object_not_vlm():
    """'抓前面的东西' must resolve via the front-object mask, NOT a VLM name."""
    perc = FrontPerception()
    arm = FakeArm()
    res = PerceptionGraspSkill().execute({"query": "前面的东西"}, _ctx(perc, arm=arm))
    assert res.success is True
    assert res.result_data["perceived"] is True
    assert "front" in perc.calls
    assert not any(c.startswith("detect") for c in perc.calls)  # no VLM naming
    expected = grasp_point_from_rgbd(perc._depth, perc._color, perc._mask,
                                     _INTR, _CAM_XPOS, _CAM_XMAT)
    assert res.result_data["grasp_world"] == pytest.approx(
        [expected.x, expected.y, expected.z], abs=1e-6)


def test_deictic_nothing_in_front_fails_loud():
    perc = FrontPerception(mask=np.zeros((_H, _W), dtype=np.uint8))
    res = PerceptionGraspSkill().execute({"query": "前面的东西"}, _ctx(perc))
    assert res.success is False
    assert res.result_data["diagnosis"] == "no_detections"


def test_named_query_empty_vlm_falls_back_to_front():
    """A named query the VLM misses falls back to the front object (honest)."""
    perc = FrontPerception(detections=[])
    res = PerceptionGraspSkill().execute({"query": "banana"}, _ctx(perc))
    assert res.success is True
    assert "front" in perc.calls


def test_color_query_threads_color_and_maps_verify_label():
    """ATTRIBUTE grasp (D47): '抓红色的东西' resolves via front_object_mask(color='red')
    and the verify LABEL maps to the colour's scene name (pickable_can_red)."""
    perc = FrontPerception()
    arm = FakeArm()
    res = PerceptionGraspSkill().execute({"query": "抓红色的东西"}, _ctx(perc, arm=arm))
    assert res.success is True
    assert "front:red" in perc.calls               # colour threaded to the resolver
    assert not any(c.startswith("detect") for c in perc.calls)  # no VLM naming
    assert res.result_data["detection_label"] == "pickable_can_red"  # verify LABEL


def test_color_query_blue_maps_to_blue_bottle():
    perc = FrontPerception()
    res = PerceptionGraspSkill().execute({"query": "抓蓝色的"}, _ctx(perc, arm=FakeArm()))
    assert res.success is True
    assert "front:blue" in perc.calls
    assert res.result_data["detection_label"] == "pickable_bottle_blue"


def test_deictic_no_color_keeps_query_label():
    """A plain deictic query parses no colour: front_object_mask(color=None), label unchanged."""
    perc = FrontPerception()
    res = PerceptionGraspSkill().execute({"query": "前面的东西"}, _ctx(perc, arm=FakeArm()))
    assert res.success is True
    assert "front" in perc.calls and "front:red" not in perc.calls
    assert res.result_data["detection_label"] == "前面的东西"


class _FakeBase:
    """A go2 stand-in: walk(vx,..,duration) advances x by vx*duration*0.7 (gait
    under-shoot); get_position returns the live pose. Faces +x (toward the table)."""
    def __init__(self, x=10.0, y=3.0):
        self._x, self._y = x, y
        self.walks = 0
    def get_heading(self):
        return 0.0  # faces +x toward the table
    def get_position(self):
        return [self._x, self._y, 0.3]
    def walk(self, vx=0.0, vy=0.0, vyaw=0.0, duration=1.0):
        self.walks += 1
        self._x += vx * duration * 0.7  # under-shoot, like the real gait
        return True


def test_approach_object_converges_with_position_feedback():
    """The scripted forward-walk approach closes the gap to within reach via feedback
    despite gait under-shoot (D27: non-gated, the base walk primitive not FAR)."""
    from vector_os_nano.skills.perception_grasp import _approach_object, _GRASP_REACH_M
    base = _FakeBase(x=10.0, y=3.0)
    ok = _approach_object(base, (11.0, 3.0))  # object 1.0m ahead, out of ~0.34m reach
    assert ok
    assert base.walks >= 1                       # it actually walked
    assert (11.0 - base.get_position()[0]) <= _GRASP_REACH_M + 0.15  # within reach now


def test_approach_noop_when_already_in_reach():
    from vector_os_nano.skills.perception_grasp import _approach_object
    base = _FakeBase(x=10.97, y=3.0)  # already 0.03m from the object -> within reach_m=0.05
    ok = _approach_object(base, (11.0, 3.0))
    assert ok
    assert base.walks == 0           # no walk needed


def test_no_arm_fails_loud():
    perc = FakePerception()
    ctx = SkillContext(arm=None, gripper=FakeGripper(), world_model=WorldModel(),
                       perception=perc, config={"world": "apartment"})
    res = PerceptionGraspSkill().execute({"query": "banana"}, ctx)
    assert res.success is False
    assert res.result_data["diagnosis"] == "no_arm"


# --- R37 TASK A: TRUE producer->consumer box-flow composition -----------------
# The grasp CONSUMES a box passed in by an earlier detect step (via the
# ${detect.output.detections} / ${detect.output.boxes} binding) and back-projects
# it to a 3D grasp point — instead of running its OWN detect/front_object. The
# composition claim ("the grasp's target came from the routed detector") is proven
# by: segment fired, detect/front DID NOT, consumed_bbox=True, and the grasp point
# equals grasp_point_from_rgbd over the SAME perceived depth+mask.

# The producer DetectorCapability emits boxes as bbox lists and detections as dicts
# (Detection.to_dict). Match the FakePerception's mask box so segment yields the mask.
_PRODUCER_BOX = [29.0, 21.0, 35.0, 27.0]


def test_passed_detections_consumed_no_reperceive():
    """A passed detection box flows into the grasp; the grasp does NOT re-perceive."""
    perc = FrontPerception()
    arm = FakeArm()
    dets = [{"label": "green bottle", "bbox": _PRODUCER_BOX, "confidence": 0.91}]
    res = PerceptionGraspSkill().execute(
        {"query": "绿色的瓶子", "detections": dets}, _ctx(perc, arm=arm)
    )
    assert res.success is True
    rd = res.result_data
    # Composition evidence: the box was consumed, the re-perceive was suppressed.
    assert rd["consumed_bbox"] is True
    assert rd["reperceived"] is False
    # The bbox path fired (segment), the re-perceive path did NOT (no detect/front).
    assert "segment" in perc.calls
    assert not any(c.startswith("detect") for c in perc.calls)
    assert not any(c.startswith("front") for c in perc.calls)
    # The grasp point equals the perceived back-projection (same math, never GT).
    expected = grasp_point_from_rgbd(perc._depth, perc._color, perc._mask,
                                     _INTR, _CAM_XPOS, _CAM_XMAT)
    assert rd["grasp_world"] == pytest.approx([expected.x, expected.y, expected.z], abs=1e-6)


def test_passed_single_bbox_consumed():
    """A single `bbox` (e.g. ${detect.output.boxes.0}) is consumed in place of detect."""
    perc = FrontPerception()
    res = PerceptionGraspSkill().execute(
        {"query": "the bottle", "bbox": _PRODUCER_BOX}, _ctx(perc, arm=FakeArm())
    )
    assert res.success is True
    assert res.result_data["consumed_bbox"] is True
    assert "segment" in perc.calls
    assert not any(c.startswith("detect") for c in perc.calls)


def test_passed_bare_boxes_list_consumed():
    """A bare `boxes` list (${detect.output.boxes}) is consumed (first box)."""
    perc = FrontPerception()
    res = PerceptionGraspSkill().execute(
        {"query": "瓶子", "boxes": [_PRODUCER_BOX]}, _ctx(perc, arm=FakeArm())
    )
    assert res.success is True
    assert res.result_data["consumed_bbox"] is True
    assert not any(c.startswith("detect") for c in perc.calls)


def test_passed_detections_colour_selects_right_box():
    """Colour query selects the colour-matching box from passed detections (D49 rule).

    The green box is dead-centre / highest-confidence, but a 'red' query must pick the
    RED box by label — perceptual colour selection, not centrality. The verify LABEL
    then maps to the colour's scene key (pickable_can_red)."""
    perc = FrontPerception()
    dets = [
        {"label": "green bottle", "bbox": [10.0, 10.0, 14.0, 14.0], "confidence": 0.95},
        {"label": "red can", "bbox": _PRODUCER_BOX, "confidence": 0.80},
    ]
    res = PerceptionGraspSkill().execute(
        {"query": "拿起红色的罐子", "detections": dets}, _ctx(perc, arm=FakeArm())
    )
    assert res.success is True
    assert res.result_data["consumed_bbox"] is True
    # The red box (the lower-confidence one) was chosen by colour, and the verify
    # label is the red scene key.
    assert res.result_data["detection_label"] == "pickable_can_red"
    assert not any(c.startswith("detect") for c in perc.calls)


def test_no_passed_box_still_reperceives():
    """Back-compat: no box passed -> the skill re-perceives as before (re-perceive ON)."""
    perc = FakePerception()
    res = PerceptionGraspSkill().execute({"query": "banana"}, _ctx(perc))
    assert res.success is True
    assert res.result_data["consumed_bbox"] is False
    assert res.result_data["reperceived"] is True
    assert any(c.startswith("detect") for c in perc.calls)  # it DID re-perceive


def test_passed_box_no_depth_fails_loud_no_gt():
    """A consumed box over all-zero depth FAILS LOUD — never a GT substitute."""
    perc = FrontPerception(depth=np.zeros((_H, _W), dtype=np.float32))
    res = PerceptionGraspSkill().execute(
        {"query": "bottle", "bbox": _PRODUCER_BOX}, _ctx(perc, arm=FakeArm())
    )
    assert res.success is False
    assert res.result_data["diagnosis"] == "no_depth_points"
    assert res.result_data.get("consumed_bbox") is True


def test_passed_box_empty_mask_fails_loud():
    """A consumed box whose segment yields no mask FAILS LOUD (segmentation_failed)."""
    perc = FrontPerception(mask=np.zeros((_H, _W), dtype=np.uint8))
    res = PerceptionGraspSkill().execute(
        {"query": "bottle", "bbox": _PRODUCER_BOX}, _ctx(perc, arm=FakeArm())
    )
    assert res.success is False
    assert res.result_data["diagnosis"] == "segmentation_failed"


def test_malformed_passed_box_falls_back_to_reperceive():
    """A malformed box (wrong arity) is ignored; the skill re-perceives (fail-soft)."""
    perc = FakePerception()
    res = PerceptionGraspSkill().execute(
        {"query": "banana", "bbox": [1.0, 2.0]}, _ctx(perc)
    )
    assert res.success is True
    assert res.result_data["consumed_bbox"] is False
    assert any(c.startswith("detect") for c in perc.calls)


# ===== R40 CHANGE 1: colour+noun query routes to HSV front_object resolver =====

def test_colour_noun_query_routes_to_front_hsv_not_detector():
    """CHANGE 1 (R40): '绿色的瓶子' has both a colour AND a noun, but colour wins —
    the skill must use front_object_mask(color='green') NOT detect().

    The D47 HSV resolver had 100% selection among 3 close cans; grounding-dino is
    intermittent. A colour target disambiguates by hue, reliably.
    """
    perc = FrontPerception()
    arm = FakeArm()
    res = PerceptionGraspSkill().execute({"query": "绿色的瓶子"}, _ctx(perc, arm=arm))
    assert res.success is True
    # HSV resolver was used with the correct colour
    assert "front:green" in perc.calls, (
        f"expected front_object_mask(color='green') but got calls: {perc.calls}")
    # The learned detector was NOT called
    assert not any(c.startswith("detect") for c in perc.calls), (
        f"grounding-dino detect was called despite a colour query: {perc.calls}")
    # Verify label maps to the colour's scene key
    assert res.result_data["detection_label"] == "pickable_bottle_green"


def test_colour_noun_query_red_routes_to_hsv():
    """A red+noun query ('红色的罐子') also takes the HSV route (not just green)."""
    perc = FrontPerception()
    res = PerceptionGraspSkill().execute({"query": "红色的罐子"}, _ctx(perc, arm=FakeArm()))
    assert res.success is True
    assert "front:red" in perc.calls
    assert not any(c.startswith("detect") for c in perc.calls)
    assert res.result_data["detection_label"] == "pickable_can_red"


def test_no_colour_noun_query_still_uses_detector():
    """A named-no-colour query ('罐子', no colour) still uses the detector route.

    CHANGE 1 must not regress the noun-only -> detector path.
    """
    perc = FrontPerception()
    # FrontPerception has detections=["banana"] by default; "罐子" has no colour
    res = PerceptionGraspSkill().execute({"query": "罐子"}, _ctx(perc, arm=FakeArm()))
    # It may succeed or fail, but it MUST have called detect (not front_object_mask)
    assert any(c.startswith("detect") for c in perc.calls), (
        f"expected detect for noun-only query but got: {perc.calls}")
    assert "front:green" not in perc.calls and "front:red" not in perc.calls


# ===== R40 CHANGE 2: low-z back-projection guard =====

def test_low_z_grasp_point_fails_loud():
    """CHANGE 2 (R40): a back-projected grasp_point with z < _MIN_GRASP_Z must
    FAIL LOUD with diagnosis='low_z_backprojection' and return gp=None.

    Simulated by monkeypatching grasp_point_from_rgbd to return a floor-level z.
    """
    import unittest.mock as mock
    from vector_os_nano.core.types import Pose3D
    from vector_os_nano.skills.perception_grasp import _MIN_GRASP_Z

    floor_pose = Pose3D(x=10.5, y=3.0, z=0.039)  # the R39 t3 bad value
    perc = FrontPerception()
    arm = FakeArm()

    with mock.patch(
        "vector_os_nano.skills.perception_grasp.grasp_point_from_rgbd",
        return_value=floor_pose,
    ):
        res = PerceptionGraspSkill().execute({"query": "前面的东西"}, _ctx(perc, arm=arm))

    assert res.success is False
    rd = res.result_data
    assert rd["diagnosis"] == "low_z_backprojection", (
        f"expected low_z_backprojection, got {rd['diagnosis']}")
    assert rd.get("actual_z", 1.0) < _MIN_GRASP_Z


def test_normal_z_grasp_point_passes():
    """A back-projected grasp point with z≈0.32 (real can top) must NOT be rejected."""
    import unittest.mock as mock
    from vector_os_nano.core.types import Pose3D

    can_pose = Pose3D(x=10.5, y=3.0, z=0.32)  # typical can top height
    perc = FrontPerception()
    arm = FakeArm()

    with mock.patch(
        "vector_os_nano.skills.perception_grasp.grasp_point_from_rgbd",
        return_value=can_pose,
    ):
        res = PerceptionGraspSkill().execute({"query": "前面的东西"}, _ctx(perc, arm=arm))

    # Should succeed (IK returns a result from FakeArm)
    assert res.success is True
    assert res.result_data.get("grasp_world", [0, 0, 0])[2] == pytest.approx(0.32, abs=1e-6)


def test_low_z_passed_box_fails_loud():
    """CHANGE 2 (R40): the low-z guard also fires for the passed-box (consumed) path."""
    import unittest.mock as mock
    from vector_os_nano.core.types import Pose3D
    from vector_os_nano.skills.perception_grasp import _MIN_GRASP_Z

    floor_pose = Pose3D(x=10.5, y=3.0, z=0.039)
    perc = FrontPerception()
    arm = FakeArm()

    with mock.patch(
        "vector_os_nano.skills.perception_grasp.grasp_point_from_rgbd",
        return_value=floor_pose,
    ):
        res = PerceptionGraspSkill().execute(
            {"query": "绿色的瓶子", "bbox": _PRODUCER_BOX},
            _ctx(perc, arm=arm),
        )

    assert res.success is False
    rd = res.result_data
    assert rd["diagnosis"] == "low_z_backprojection", (
        f"expected low_z_backprojection at passed-box site, got {rd['diagnosis']}")
    assert rd.get("consumed_bbox") is True
    assert rd.get("actual_z", 1.0) < _MIN_GRASP_Z


def _frame_of(xpos, size=(_W, _H)):
    return {"size": list(size), "sim_time": 1.0, "cam_pos": [float(v) for v in xpos],
            "cam_mat": [float(v) for v in _CAM_XMAT]}


def test_a_box_from_the_current_frame_is_consumed():
    perc = FrontPerception()
    dets = [{"label": "green bottle", "bbox": _PRODUCER_BOX, "confidence": 0.9,
             "frame": _frame_of(_CAM_XPOS)}]
    res = PerceptionGraspSkill().execute({"query": "绿色的瓶子", "detections": dets}, _ctx(perc))
    assert res.success and not any(c.startswith("detect") for c in perc.calls)


def test_a_stale_producer_box_is_redetected_on_the_current_frame():
    """perception:producer-box-no-provenance — a box detected before the camera moved
    (a navigate, the arm stow) was segmented on a later frame where it no longer is."""
    perc = FrontPerception()
    moved = _CAM_XPOS + np.array([0.5, 0.0, 0.0])
    dets = [{"label": "green bottle", "bbox": [0.0, 0.0, 6.0, 6.0], "confidence": 0.9,
             "frame": _frame_of(moved)}]
    res = PerceptionGraspSkill().execute({"query": "绿色的瓶子", "detections": dets}, _ctx(perc))
    assert any(c.startswith("detect") for c in perc.calls), "the stale box must be re-detected"
    assert res.success and res.result_data["consumed_bbox"] is True
    perc = FrontPerception(detections=[])
    res = PerceptionGraspSkill().execute({"query": "绿色的瓶子", "detections": dets}, _ctx(perc))
    assert not res.success and res.result_data["diagnosis"] == "stale_box"
