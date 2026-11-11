# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Pure offline unit tests for object_localizer.localize_objects_3d.

No MuJoCo, no GPU, no network — pure numpy synthetic geometry + stub
perception objects.  Tests are marked as unit (no sim marker).

World-point derivation reference
---------------------------------
The camera sits at (0, 0, 0.4) looking along world +x (MuJoCo xmat columns
right=-y, up=+z, -forward=-x). A mask centred on the principal point (cx, cy)
yields a camera-frame centroid at (0, 0, depth_val), which camera_to_world maps
to (depth_val, 0, 0.4) in the world frame.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from vector_os_nano.core.types import Detection
from vector_os_nano.perception.depth_projection import camera_to_world, mujoco_intrinsics
from vector_os_nano.perception.object_localizer import (
    Localized,
    _perception_is_usable,
    localize_objects_3d,
)
from vector_os_nano.sim.runtime import RGBDFrame

# ---------------------------------------------------------------------------
# Shared synthetic scene parameters
# ---------------------------------------------------------------------------

_W, _H = 64, 48
_INTR = mujoco_intrinsics(_W, _H, vfov_deg=42.0)
_CAM_XPOS = np.array([0.0, 0.0, 0.4])
_CAM_XMAT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64).reshape(9)


def _frame(rgb: np.ndarray, depth: np.ndarray) -> RGBDFrame:
    return RGBDFrame(rgb=rgb, depth=depth, seg=None, K=_INTR, cam_pos=_CAM_XPOS,
                     cam_mat=_CAM_XMAT, sim_time=1.0, camera="d435_rgb")

_DEPTH_VAL = 1.5  # metres — non-trivial, not 1.0 (avoids accidental pass)


def _make_depth_and_mask(depth_val: float = _DEPTH_VAL) -> tuple[np.ndarray, np.ndarray]:
    """Flat depth plane at depth_val, with a centred rectangular mask.

    Mask is a 7x7 block centred on (cx=32, cy=24) so the camera-frame
    centroid is (0, 0, depth_val) — exactly on the optical axis.
    """
    depth = np.zeros((_H, _W), dtype=np.float32)
    mask = np.zeros((_H, _W), dtype=np.uint8)
    us = range(29, 36)  # mean 32 = cx
    vs = range(21, 28)  # mean 24 = cy
    for v in vs:
        for u in us:
            depth[v, u] = depth_val
            mask[v, u] = 1
    return depth, mask


def _make_color() -> np.ndarray:
    return np.zeros((_H, _W, 3), dtype=np.uint8)


def _bbox_for_mask() -> tuple[float, float, float, float]:
    """Bounding box (x1, y1, x2, y2) for the synthetic centred mask."""
    return (29.0, 21.0, 35.0, 27.0)


# ---------------------------------------------------------------------------
# Stub perception (fake Go2GraspPerception)
# ---------------------------------------------------------------------------


class _StubPerception:
    """Minimal stub that mimics Go2GraspPerception's required API.

    Parameters
    ----------
    label:
        Label returned by detect().
    confidence:
        Score of the single returned Detection.
    depth_val:
        Depth plane value used for the synthetic depth frame.
    mask_override:
        If set, segment() returns this mask instead of the synthetic one.
    fail_segment:
        If True, segment() raises to exercise per-object error handling.
    """

    def __init__(
        self,
        label: str = "apple",
        confidence: float = 0.9,
        depth_val: float = _DEPTH_VAL,
        mask_override: np.ndarray | None = None,
        fail_segment: bool = False,
    ) -> None:
        self._label = label
        self._confidence = confidence
        self._depth_val = depth_val
        self._mask_override = mask_override
        self._fail_segment = fail_segment
        self._color = _make_color()
        self._depth, self._mask = _make_depth_and_mask(depth_val)

    def capture(self) -> RGBDFrame:
        return _frame(self._color, self._depth)

    def detect_in(self, rgb: np.ndarray, query: str) -> list[Detection]:
        return [
            Detection(
                label=self._label,
                bbox=_bbox_for_mask(),
                confidence=self._confidence,
            )
        ]

    def segment(self, image: np.ndarray, bbox: Any) -> np.ndarray | None:
        if self._fail_segment:
            raise RuntimeError("segment: simulated failure")
        if self._mask_override is not None:
            return self._mask_override
        return self._mask


class _StubPerceptionNoDetections(_StubPerception):
    """Returns empty detections list."""

    def detect_in(self, rgb: np.ndarray, query: str) -> list[Detection]:
        return []


class _StubPerceptionNullMask(_StubPerception):
    """segment() always returns None."""

    def segment(self, image: np.ndarray, bbox: Any) -> np.ndarray | None:
        return None


# ---------------------------------------------------------------------------
# Tests: _perception_is_usable guard
# ---------------------------------------------------------------------------


def test_guard_none_returns_false():
    assert _perception_is_usable(None) is False


def test_guard_missing_one_method_returns_false():
    """An object missing even one required method is not usable."""

    class _MissingSegment:
        def capture(self): ...
        def detect_in(self, rgb, q): ...
        # 'segment' deliberately absent

    assert _perception_is_usable(_MissingSegment()) is False


def test_guard_full_stub_returns_true():
    assert _perception_is_usable(_StubPerception()) is True


# ---------------------------------------------------------------------------
# Tests: localize_objects_3d returns [] for bad perception
# ---------------------------------------------------------------------------


def test_localize_returns_empty_when_perception_is_none():
    result = localize_objects_3d(None, ["apple"])
    assert result == []


def test_localize_returns_empty_when_perception_missing_method():
    class _NoDect:
        def capture(self): ...
        def segment(self, img, bbox): ...
        # no 'detect_in'

    result = localize_objects_3d(_NoDect(), ["apple"])
    assert result == []


def test_localize_empty_queries_returns_empty():
    result = localize_objects_3d(_StubPerception(), [])
    assert result == []


# ---------------------------------------------------------------------------
# Tests: happy-path localization
# ---------------------------------------------------------------------------


def test_localize_one_object_returns_correct_world_point():
    """Centred mask, identity cam pose, depth 1.5 m → world (0, 0, -1.5).

    The stub detect() returns one Detection; segment() returns the
    synthetic centred mask: the camera-frame centroid (0, 0, 1.5) lands 1.5 m
    ahead of the camera, at its height.
    """
    perc = _StubPerception(label="apple", depth_val=_DEPTH_VAL)
    results = localize_objects_3d(perc, ["apple"])

    assert len(results) == 1
    label, x, y, z = results[0][:4]
    assert label == "apple" and results[0].score == pytest.approx(0.9)

    # Cross-check with the proven camera_to_world transform.
    expected_wx, expected_wy, expected_wz = camera_to_world(
        0.0, 0.0, _DEPTH_VAL,
        cam_xpos=_CAM_XPOS, cam_xmat=_CAM_XMAT,
    )
    assert x == pytest.approx(expected_wx, abs=0.03)
    assert y == pytest.approx(expected_wy, abs=0.03)
    assert z == pytest.approx(expected_wz, abs=0.03)
    assert (x, z) == pytest.approx((_DEPTH_VAL, 0.4), abs=0.03)


def test_localize_multiple_different_queries():
    """Two distinct labels each return their own world point."""

    class _TwoLabelPerc:
        """detect() returns label matching query; same depth/mask for both."""

        def __init__(self) -> None:
            self._depth, self._mask = _make_depth_and_mask()
            self._color = _make_color()

        def capture(self): return _frame(self._color, self._depth)

        def detect_in(self, rgb, query: str) -> list[Detection]:
            return [Detection(label=query, bbox=_bbox_for_mask(), confidence=0.8)]

        def segment(self, image, bbox): return self._mask

    # Two queries on ONE spot: only one object is there, the better score keeps it.
    results = localize_objects_3d(_TwoLabelPerc(), ["cup", "bottle"])
    assert [r.label for r in results] == ["cup"]


def test_localize_keys_result_by_query_not_detector_label():
    """Regression (real-sim bug): grounding-dino returns labels like
    'a green bottle' for the query 'green bottle'.  The result MUST be keyed by
    the input QUERY so callers (look skill) can map their requested name ->
    position; keying by det.label stored every looked object at (0, 0).
    """
    perc = _StubPerception(label="a green bottle")  # detector label != query
    results = localize_objects_3d(perc, ["green bottle"])
    assert len(results) == 1
    assert results[0][0] == "green bottle"  # the QUERY, not "a green bottle"


def test_two_queries_on_one_spot_keep_the_better_score():
    """perception:localizer-no-plausibility-gate (dedupe) — 'can' and 'red can' became
    two scene-graph entries for one body."""

    class _Scored(_StubPerception):
        def detect_in(self, rgb, query):
            return [Detection(label=query, bbox=_bbox_for_mask(),
                              confidence={"can": 0.5, "metal can": 0.7}[query])]

    results = localize_objects_3d(_Scored(), ["can", "metal can"])
    assert [r.label for r in results] == ["metal can"]


# ---------------------------------------------------------------------------
# Tests: failure resilience
# ---------------------------------------------------------------------------


def test_localize_skips_object_when_no_detections():
    """No detections for the query → object absent from result, no crash."""
    perc = _StubPerceptionNoDetections()
    results = localize_objects_3d(perc, ["apple"])
    assert results == []


def test_localize_skips_object_when_segment_returns_none():
    """Null mask → grasp_point_from_rgbd gets no pixels → None → skip."""
    perc = _StubPerceptionNullMask()
    results = localize_objects_3d(perc, ["apple"])
    assert results == []


def test_localize_skips_object_when_segment_raises():
    """A failing segment call must not propagate — skip that object."""
    perc = _StubPerception(fail_segment=True)
    results = localize_objects_3d(perc, ["apple"])
    assert results == []


def test_localize_skips_object_when_mask_has_no_depth():
    """Mask over all-zero depth → grasp_point_from_rgbd returns None → skip."""
    zero_mask = np.ones((_H, _W), dtype=np.uint8)
    zero_depth_perc = _StubPerception(mask_override=zero_mask, depth_val=0.0)
    results = localize_objects_3d(zero_depth_perc, ["apple"])
    assert results == []


def test_localize_partial_failure_returns_successes():
    """One query fails, another succeeds — successes are not dropped."""

    class _SelectivePerc:
        def __init__(self) -> None:
            self._depth, self._mask = _make_depth_and_mask()
            self._color = _make_color()

        def capture(self): return _frame(self._color, self._depth)

        def detect_in(self, rgb, query: str) -> list[Detection]:
            return [Detection(label=query, bbox=_bbox_for_mask(), confidence=0.8)]

        def segment(self, image, bbox):
            # Only "good" gets a valid mask; "bad" returns None
            # We can't tell from bbox alone which query this was, but we can
            # use a call counter: first call OK (good), second call None (bad).
            self._calls = getattr(self, "_calls", 0) + 1
            return self._mask if self._calls == 1 else None

    results = localize_objects_3d(_SelectivePerc(), ["good", "bad"])
    assert len(results) == 1
    assert results[0][0] == "good"


# ---------------------------------------------------------------------------
# Tests: look skill wiring (unit — fake SkillContext + fake spatial_memory)
# ---------------------------------------------------------------------------


class _RecordingMemory:
    """Captures the kwargs passed to observe_with_viewpoint."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def observe_with_viewpoint(
        self,
        room: str,
        x: float,
        y: float,
        heading: float,
        objects: list[str],
        description: str = "",
        detected_objects: Any = None,
    ) -> None:
        self.calls.append(
            {
                "room": room,
                "objects": objects,
                "detected_objects": detected_objects,
            }
        )

    def visit(self, room: str, x: float, y: float) -> None:
        pass


class _FakeVLMResult:
    name: str
    description: str
    confidence: float

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = ""
        self.confidence = 0.9


class _FakeScene:
    def __init__(self, object_names: list[str]) -> None:
        self.objects = [_FakeVLMResult(n) for n in object_names]
        self.summary = "test scene"
        self.details = ""


class _FakeRoomID:
    room = "living_room"
    confidence = 0.9
    reasoning = ""


class _FakeVLM:
    def describe_scene(self, frame):
        return _FakeScene(["apple", "cup"])

    def identify_room(self, frame):
        return _FakeRoomID()


class _FakeBase:
    def get_camera_frame(self):
        return np.zeros((_H, _W, 3), dtype=np.uint8)

    def get_position(self):
        return (1.0, 2.0, 0.0)

    def get_heading(self):
        return 0.0


def _make_context(with_perception: bool = True) -> Any:
    """Build a minimal SkillContext for look skill wiring tests."""
    from vector_os_nano.core.skill import SkillContext

    perception = _StubPerception(label="apple") if with_perception else None
    mem = _RecordingMemory()
    ctx = SkillContext(
        base=_FakeBase(),
        services={"vlm": _FakeVLM(), "spatial_memory": mem},
        perception=perception,
    )
    return ctx, mem


def test_look_skill_passes_detected_objects_when_perception_present():
    """With perception available, observe_with_viewpoint gets detected_objects
    with non-None coordinates (not all zeros from names-only fallback).

    We patch localize_objects_3d to return a known world point so the test
    is deterministic and does not depend on the full depth pipeline.
    """
    import unittest.mock as mock
    from vector_os_nano.skills.go2.look import LookSkill

    ctx, mem = _make_context(with_perception=True)
    skill = LookSkill()

    # Patch localize_objects_3d to return a predictable non-zero world point.
    fake_loc = [Localized("apple", 1.1, 2.2, 3.3, 0.9, "edgetam")]
    with mock.patch(
        "vector_os_nano.skills.go2.look.localize_objects_3d",
        return_value=fake_loc,
    ) as mock_loc:
        result = skill.execute({}, ctx)

    assert result.success, result.error_message
    assert mock_loc.called, "localize_objects_3d should be called when perception is present"

    # Check the call that reached spatial memory.
    assert len(mem.calls) == 1
    call = mem.calls[0]
    detected = call["detected_objects"]
    assert detected is not None, "detected_objects must not be None when localization succeeded"
    # apple was found at (1.1, 2.2) XY
    apple_entry = next((t for t in detected if t[0] == "apple"), None)
    assert apple_entry is not None
    assert apple_entry[1] == pytest.approx(1.1, abs=1e-6)
    assert apple_entry[2] == pytest.approx(2.2, abs=1e-6)

    # Regression: 'cup' was reported by the VLM but NOT localized (the patched
    # localize_objects_3d returns only 'apple').  It must NOT appear in
    # detected_objects with a fake (0, 0) position — un-localized objects are
    # never stored at the origin (scene-graph pollution bug).
    cup_entry = next((t for t in detected if t[0] == "cup"), None)
    assert cup_entry is None, (
        "un-localized 'cup' must not be passed with a fake (0,0) position"
    )


def test_look_skill_stores_nothing_when_perception_absent():
    """docs-vs-code:zero-zero-pollution-all-fail — without a localizer the VLM's names
    used to be stored at (0, 0); now only the viewpoint is recorded, no object."""
    from vector_os_nano.skills.go2.look import LookSkill

    ctx, mem = _make_context(with_perception=False)
    skill = LookSkill()
    result = skill.execute({}, ctx)

    assert result.success, result.error_message
    assert len(mem.calls) == 1
    call = mem.calls[0]
    assert call["detected_objects"] == [] and call["objects"] == []
    assert sorted(result.result_data["unlocalized"]) == ["apple", "cup"]


def test_look_skill_includes_world_coords_in_result_data():
    """When localization succeeds, result_data objects include world_x/y/z."""
    import unittest.mock as mock
    from vector_os_nano.skills.go2.look import LookSkill

    ctx, _ = _make_context(with_perception=True)
    skill = LookSkill()

    fake_loc = [Localized("apple", 4.0, 5.0, 6.0, 0.9, "edgetam")]
    with mock.patch(
        "vector_os_nano.skills.go2.look.localize_objects_3d",
        return_value=fake_loc,
    ):
        result = skill.execute({}, ctx)

    assert result.success
    objects = result.result_data.get("objects", [])
    apple_obj = next((o for o in objects if o["name"] == "apple"), None)
    assert apple_obj is not None
    assert apple_obj["world_x"] == pytest.approx(4.0, abs=1e-6)
    assert apple_obj["world_y"] == pytest.approx(5.0, abs=1e-6)
    assert apple_obj["world_z"] == pytest.approx(6.0, abs=1e-6)


def test_look_skill_no_world_coords_when_localization_returns_empty():
    """When localize_objects_3d returns [], objects_data has no world_x/y/z keys."""
    import unittest.mock as mock
    from vector_os_nano.skills.go2.look import LookSkill

    ctx, _ = _make_context(with_perception=True)
    skill = LookSkill()

    with mock.patch(
        "vector_os_nano.skills.go2.look.localize_objects_3d",
        return_value=[],
    ):
        result = skill.execute({}, ctx)

    assert result.success
    objects = result.result_data.get("objects", [])
    for obj in objects:
        assert "world_x" not in obj
        assert "world_y" not in obj
        assert "world_z" not in obj


def _no_vlm_context(world: str | None) -> Any:
    from vector_os_nano.core.skill import SkillContext

    mem = _RecordingMemory()
    ctx = SkillContext(
        base=_FakeBase(), services={"spatial_memory": mem},
        perception=_StubPerception(label="green bottle"),
        config={"world": world} if world else {},
    )
    return ctx, mem


def test_look_without_a_vlm_localizes_the_world_vocabulary():
    """No VLM: look asks the detector for each of the world's own objects and
    records ONLY the ones it localized (never an unseen name)."""
    import unittest.mock as mock
    from vector_os_nano.skills.go2.look import LookSkill

    ctx, mem = _no_vlm_context("apartment")
    with mock.patch(
        "vector_os_nano.skills.go2.look.localize_objects_3d",
        return_value=[Localized("green bottle", 10.88, 3.0, 0.32, 0.6, "edgetam")],
    ) as loc:
        result = LookSkill().execute({}, ctx)
    assert result.success, result.error_message
    asked = loc.call_args.args[1]
    assert "green bottle" in asked and "red can" in asked  # the manifest vocabulary
    (call,) = mem.calls
    assert call["objects"] == []  # names are never stored, only localized objects
    assert call["detected_objects"] == [("green bottle", 10.88, 3.0, 0.32)]
    assert [o["name"] for o in result.result_data["objects"]] == ["green bottle"]


def test_look_without_a_vlm_or_a_world_still_fails_loud():
    from vector_os_nano.skills.go2.look import LookSkill

    ctx, mem = _no_vlm_context(None)
    result = LookSkill().execute({}, ctx)
    assert not result.success and result.diagnosis_code == "no_vlm"
    assert not mem.calls


# ---------------------------------------------------------------------------
# Plausibility gates (perception:localizer-no-plausibility-gate) and one frame
# (perception:localizer-box-from-later-frame / docs-vs-code:localizer-frame-skew)
# ---------------------------------------------------------------------------


class _Scene(_StubPerception):
    """A wall at 3 m filling the frame, optionally with an object patch in front."""

    def __init__(self, *, obj_depth: float | None, rgb_fill=(40, 170, 70), box=None,
                 score: float = 0.8) -> None:
        super().__init__(confidence=score)
        self._depth = np.full((_H, _W), 3.0, dtype=np.float32)
        self._color = np.full((_H, _W, 3), 90, dtype=np.uint8)
        self._box = box or _bbox_for_mask()
        x1, y1, x2, y2 = (int(v) for v in self._box)
        self._mask = np.zeros((_H, _W), dtype=np.uint8)
        self._mask[y1:y2 + 1, x1:x2 + 1] = 1
        if obj_depth is not None:
            self._depth[self._mask > 0] = obj_depth
        self._color[self._mask > 0] = rgb_fill
        self.frames_seen: list[int] = []

    def detect_in(self, rgb, query):
        self.frames_seen.append(id(rgb))
        return [Detection(label=query, bbox=self._box, confidence=self._confidence)]


def test_the_localizer_rejects_a_box_on_a_wall():
    """A confident box whose depth is flush with what surrounds it is a surface."""
    assert localize_objects_3d(_Scene(obj_depth=None), ["green bottle"]) == []


def test_an_object_standing_in_front_of_the_wall_is_localized():
    (r,) = localize_objects_3d(_Scene(obj_depth=1.2), ["green bottle"])
    assert r.x == pytest.approx(1.2, abs=0.03) and r.z == pytest.approx(0.4, abs=0.03)


def test_a_box_of_the_wrong_colour_is_rejected():
    """WP10: 'yellow cup' boxed on a green bottle was stored as a yellow cup."""
    assert localize_objects_3d(_Scene(obj_depth=1.2), ["yellow cup"]) == []
    assert localize_objects_3d(_Scene(obj_depth=1.2, rgb_fill=(240, 200, 40)), ["yellow cup"])


def test_low_scores_slivers_furniture_and_far_boxes_are_rejected():
    assert localize_objects_3d(_Scene(obj_depth=1.2, score=0.26), ["green bottle"]) == []
    sliver = (2.0, 2.0, 60.0, 4.0)
    assert localize_objects_3d(_Scene(obj_depth=1.2, box=sliver), ["green bottle"]) == []
    sofa = (4.0, 8.0, 60.0, 44.0)  # ~1.4 m wide at 1.2 m: no graspable object
    assert localize_objects_3d(_Scene(obj_depth=1.2, box=sofa), ["green bottle"]) == []
    assert localize_objects_3d(_Scene(obj_depth=5.5), ["green bottle"]) == []


def test_every_query_is_detected_on_the_one_captured_frame():
    scene = _Scene(obj_depth=1.2)
    localize_objects_3d(scene, ["green bottle", "blue bottle", "red can"])
    assert len(scene.frames_seen) == 3 and len(set(scene.frames_seen)) == 1
