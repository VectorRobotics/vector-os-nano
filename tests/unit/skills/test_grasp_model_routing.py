# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""perception_grasp target selection and MODEL routing (REF1: one referent rule).

In a world with a manifest the query must name ONE object of that world
(``WorldManifest.referents``) before any motion: the object is perceived by its own
hue (front_object_mask(color=...)) or, with no hue resolver, by the detector asked for
its English name, and labelled by its body name. A query naming only what is in front
takes the front-object route; anything else is refused (unknown_object /
ambiguous_object) with no arm, gripper, base or perception call. Without a manifest:
deictic -> front, only a colour -> the hue route, else the detector on the English
phrase. The perception object is mocked and records which resolver fired.
"""
from __future__ import annotations

import numpy as np
import pytest

from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.types import Detection
from tests.harness.camera import pinhole
from vector_os_nano.perception.front_object import parse_color
from vector_os_nano.skills.grasp import GraspSkill
from vector_os_nano.embodiments.config import load_embodiment_config
from vector_os_nano.vcli.cognitive import abort
from vector_os_nano.vcli.worlds.world_manifest import load_world

_W, _H = 64, 48
_INTR = pinhole(_W, _H, 42.0)
_K = np.array([[_INTR.fx, 0.0, _INTR.cx], [0.0, _INTR.fy, _INTR.cy], [0.0, 0.0, 1.0]])  # the frame's own K
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
        from vector_os_nano.interfaces.physics import RGBDFrame
        return RGBDFrame(rgb=self._color, depth=self._depth, seg=None, K=_K,
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
        self._paint(parse_color(query))
        self.calls.append(f"detect:{query}")
        return list(self._dets)

    def segment(self, image, bbox):
        self.calls.append("segment")
        return self._mask

    def _paint(self, colour):
        colors = {"red": [255, 0, 0], "green": [0, 255, 0], "blue": [0, 0, 255], "yellow": [255, 255, 0]}
        self._color[self._mask > 0] = colors.get(colour, [200, 200, 200])

    def front_object_mask(self, rgb=None, depth=None, *, color=None):
        self._paint(color)
        self.calls.append(f"front:{color}" if color else "front")
        return self._mask


class _NoFront(RoutingPerception):
    """A perception backend with no front-object (hue) resolver: the detector only."""

    front_object_mask = property(lambda self: (_ for _ in ()).throw(AttributeError("no front resolver")))


class _Arm:
    name = "piper"

    def __init__(self):
        self.commands: list[str] = []

    def ik_top_down(self, xyz, current_joints=None):
        return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    def move_joints(self, q, duration=1.0):
        self.commands.append("move_joints")
        return True

    def get_joint_positions(self):
        return [0.0] * 6

    def fk(self, q):
        return ([0.0, 0.0, 0.0], np.eye(3))


class _Gripper:
    def __init__(self):
        self._closed = False
        self.commands: list[str] = []

    def open(self):
        self.commands.append("open")
        self._closed = False
        return True

    def close(self):
        self.commands.append("close")
        self._closed = True
        return True

    def is_holding(self, name=None):
        return self._closed


class _Base:
    """A steerable base that records every motion command."""

    def __init__(self):
        self.commands: list[str] = []

    def walk(self, **kw):
        self.commands.append("walk")

    def get_position(self):
        return [9.0, 3.0, 0.3]

    def get_heading(self):
        return 0.0


def _ctx(perc, world="apartment", base=None):
    abort.clear_abort()
    cfg = load_embodiment_config("go2_piper")
    return SkillContext(
        arm=_Arm(), gripper=_Gripper(), base=base,
        perception=perc, services={"object_world": load_world(world) if world else None, "cancel": abort.current_token()}, config={"world": world, "skills": cfg.skills, "arm": {"postures": cfg.arm.postures}},
    )


def _motion(ctx) -> list[str]:
    return [*ctx.arm.commands, *ctx.gripper.commands, *(ctx.base.commands if ctx.base else [])]


# --- routing behaviour in the apartment (a world manifest) -------------------


@pytest.mark.parametrize("query, body, colour", [
    ("红色的罐子", "pickable_can_red", "red"),     # CHANGE 1 (R40): the hue resolver, never the detector
    ("抓红色的", "pickable_can_red", "red"),       # a colour marked by 的 refers: the only red object
    ("罐子", "pickable_can_red", "red"),           # the one can of this world (a unique category)
    ("the cup", "pickable_cup_yellow", "yellow"),
    ("绿色的瓶子", "pickable_bottle_green", "green"),
])
def test_a_named_object_is_perceived_by_its_own_hue_and_labelled_by_its_body(query, body, colour):
    perc = RoutingPerception()
    res = GraspSkill().execute({"query": query}, _ctx(perc))
    assert res.success is True, res.result_data
    assert perc.calls[0] == f"front:{colour}", perc.calls
    assert not any(c.startswith("detect:") for c in perc.calls), perc.calls
    assert res.result_data["detection_label"] == body


def test_without_a_hue_resolver_the_detector_is_asked_for_the_objects_own_name():
    """'罐子' names the red can: the detector is asked for 'red can' (the object's
    English name, colour included), never the bare noun, and the label is its body."""
    perc = _NoFront()
    res = GraspSkill().execute({"query": "罐子"}, _ctx(perc))
    assert res.success is True
    assert perc.calls[0] == "detect:red can", perc.calls
    assert res.result_data["detection_label"] == "pickable_can_red"


def test_deictic_routes_to_front_object():
    """'前面的东西' -> classical front_object(color=None), NOT the detector."""
    perc = RoutingPerception()
    res = GraspSkill().execute({"query": "前面的东西"}, _ctx(perc))
    assert res.success is True
    assert "front" in perc.calls
    assert not any(c.startswith("detect:") for c in perc.calls)


def test_named_colour_query_routes_to_hsv_and_sets_verify_label():
    """CHANGE 1 (R40): '红色的罐子' contains a noun AND a colour: the red can, whose hue
    the front resolver reads; the verify LABEL is its body (pickable_can_red)."""
    assert parse_color("红色的罐子") == "red"
    perc = RoutingPerception()
    res = GraspSkill().execute({"query": "红色的罐子"}, _ctx(perc))
    assert res.success is True
    assert "front:red" in perc.calls, (
        f"expected HSV resolver for colour+noun query, got: {perc.calls}")
    assert not any(c.startswith("detect:") for c in perc.calls)
    assert res.result_data["detection_label"] == "pickable_can_red"


# --- without a world manifest (no oracle world) -------------------------------


@pytest.mark.parametrize("query, first", [
    ("罐子", "detect:can"),              # the open-vocabulary detector on the English phrase
    ("banana", "detect:banana"),
    ("grab the red unicorn", "detect:red unicorn"),
    ("red", "front:red"),                # only a colour: the hue route
    ("红色的", "front:red"),
    ("the red one", "front:red"),
    ("前面的东西", "front"),              # only deictic: the front route
])
def test_without_a_manifest_each_query_takes_exactly_one_route(query, first):
    perc = RoutingPerception()
    res = GraspSkill().execute({"query": query}, _ctx(perc, world=None))
    assert res.success is True, res.result_data
    assert perc.calls[0] == first, perc.calls
    other = "front" if first.startswith("detect") else "detect"
    assert not any(c.startswith(other) for c in perc.calls), perc.calls


@pytest.mark.parametrize("query", ["红色的苹果", "白色的瓶子", "抓前面红色的苹果", "grab the red 苹果"])
def test_without_a_manifest_cjk_the_words_cannot_carry_is_refused(query):
    """No world's words account for 苹果 / 白: the detector would be handed a guess."""
    perc = RoutingPerception()
    ctx = _ctx(perc, world=None)
    res = GraspSkill().execute({"query": query}, ctx)
    assert not res.success and res.result_data["diagnosis"] == "unknown_object"
    assert perc.calls == [] and _motion(ctx) == []


def test_a_front_route_without_a_front_resolver_fails_and_never_detects():
    """No fallback from one route to another: a deictic query on a detector-only backend."""
    perc = _NoFront()
    res = GraspSkill().execute({"query": "前面的东西"}, _ctx(perc))
    assert not res.success and res.result_data["diagnosis"] == "perception_failed"
    assert perc.calls == []


# --- D48 caveat 1: PERCEPTUAL colour selection among detector boxes ----------








# --- a named grasp fails loud, never a substitute target (R2) ----------------


class _BrokenDetector(RoutingPerception):
    def detect(self, query):
        self._paint(parse_color(query))
        self.calls.append(f"detect:{query}")
        raise RuntimeError("grounding-dino weights missing")


class _BrokenDetectorNoFront(_NoFront, _BrokenDetector):
    pass


@pytest.mark.parametrize("perc, world", [(_BrokenDetector, None), (_BrokenDetectorNoFront, "apartment")])
def test_a_detector_error_fails_loud_never_the_front_blob(perc, world):
    """perception:named-grasp-falls-back-to-front-blob: the detector raised and the
    skill grasped the front-most vivid blob, labelled as the query."""
    perc = perc()
    res = GraspSkill().execute({"query": "罐子"}, _ctx(perc, world=world))
    assert not res.success and res.result_data["diagnosis"] == "perception_failed"
    assert "weights missing" in res.error_message
    assert not any(c.startswith("front") for c in perc.calls), perc.calls


def test_an_empty_detector_answer_fails_loud_never_the_front_blob():
    """skills-manip:silent-target-substitution: 'VLM empty -> front-object fallback'."""
    perc = RoutingPerception(detections=[])
    res = GraspSkill().execute({"query": "罐子"}, _ctx(perc, world=None))
    assert not res.success and res.result_data["diagnosis"] == "no_detections"
    assert not any(c.startswith("front") for c in perc.calls), perc.calls


# REF1 acceptance: the demonstrated holes (a colour route / the only object of that
# colour taken for a noun this world has not), the R2 rows, and the R2F2V scratch rows.
_REFUSED = [
    "yellow banana", "黄色的香蕉", "red bottle", "抓红色的瓶子", "blue can", "red unicorn", "红色的苹果",
    "red stool", "红色的凳子", "white cup",
    "白色的瓶子", "grab the black can", "抓那个苹果", "抓前面那个苹果", "前面的苹果", "抓红色的苹果",
    "grab the red unicorn", "抓前面红色的苹果", "grab that apple", "pick this banana", "grab the apple in front",
    "this banana", "黄色的香蕉皮鞋", "grab the red apple",
    "抓红苹果", "红色苹果", "grab the RED UNICORN", "red-unicorn", "the red unicorn please", "抓红色的unicorn",
    "grab the red 苹果", "拿那个红色的苹果", "Grab The Red Unicorn!", "抓起红色的独角兽", "green dragon",
    "the blue whale", "绿色的西瓜",
    # R2F2V-02: words beside the reference that are neither a grasp verb nor a spatial word
    "could you grab the red one", "grab the red one for me now", "the red thing on the left",
    "red",                       # a lone colour word is a noun ("the orange"), not a reference
]


@pytest.mark.parametrize("query", _REFUSED)
def test_a_phrase_the_world_does_not_resolve_is_refused_before_any_motion(query):
    perc = RoutingPerception()
    ctx = _ctx(perc, base=_Base())
    res = GraspSkill().execute({"query": query}, ctx)
    assert not res.success and res.result_data["diagnosis"] == "unknown_object", res.result_data
    assert perc.calls == [] and _motion(ctx) == [], (perc.calls, _motion(ctx))
    # the refusal names what this world has, so the model can retry by name
    assert res.result_data["known_objects"] == load_world("apartment").vocabulary()
    assert "red can" in res.error_message


@pytest.mark.parametrize("query, names", [
    ("瓶子", ["blue bottle", "green bottle"]),
    ("the bottle", ["blue bottle", "green bottle"]),
    ("the bottle in front", ["blue bottle", "green bottle"]),  # a spatial word never picks one
    ("box", ["orange box", "purple box"]),
])
def test_a_reference_that_fits_several_objects_is_refused_and_names_them(query, names):
    perc = RoutingPerception()
    ctx = _ctx(perc, base=_Base())
    res = GraspSkill().execute({"query": query}, ctx)
    assert not res.success and res.result_data["diagnosis"] == "ambiguous_object"
    assert res.result_data["candidates"] == names
    assert perc.calls == [] and _motion(ctx) == []


@pytest.mark.parametrize("query", [
    "grab that can", "抓前面那个罐子", "the can in front", "前面的罐子", "the nearest can", "pick up the red can",
])
def test_a_spatial_or_deictic_word_beside_a_reference_takes_that_object(query):
    """R2V-04 (a deictic word never hides the named object), the other side: the
    reference alone resolves, so the object is perceived by its hue, never the front blob."""
    perc = RoutingPerception()
    res = GraspSkill().execute({"query": query}, _ctx(perc))
    assert res.success, res.result_data
    assert perc.calls[0] == "front:red", perc.calls
    assert res.result_data["detection_label"] == "pickable_can_red"


@pytest.mark.parametrize("query", [
    "前面的东西", "抓前面的", "the thing in front", "grab that one", "this", "前面", "nearest object",
    "那个", "这个", "面前的", "东西", "object",
])
def test_a_pure_deictic_query_still_takes_the_front_object(query):
    """The other side of R2V-04: a query naming no object stays on the space resolver."""
    perc = RoutingPerception()
    res = GraspSkill().execute({"query": query}, _ctx(perc))
    assert res.success, res.error_message
    assert perc.calls[0] == "front" and not any(c.startswith("detect:") for c in perc.calls), perc.calls


@pytest.mark.parametrize("query", [
    "抓红色的", "红色的东西", "grab the red one", "the red object", "抓前面红色的", "grab the red can", "红色的罐子",
    "红的", "红色那个",
])
def test_a_colour_marked_as_a_reference_takes_the_one_object_of_that_colour(query):
    """The other side of R2F1V-01: a colour with a generic head or 的 (or a category of
    the same object) refers, to the only red object here."""
    perc = RoutingPerception()
    res = GraspSkill().execute({"query": query}, _ctx(perc))
    assert res.success, res.error_message
    assert "front:red" in perc.calls, perc.calls
    assert res.result_data["detection_label"] == "pickable_can_red"




