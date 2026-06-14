# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #9 R1 (track A) — VlmSeekSkill: find & approach a SEMANTIC object.

Same perceive→decide→act state machine as VisionSeekSkill (shared _seek_loop),
but the perceive step grounds an object CLASS with a real VLM instead of colour
segmentation. Tested with an injected fake detector (no network) + a scripted
base, mirroring test_vision_seek. Honesty: a never-recognised target FAILS.
"""
from __future__ import annotations

from types import SimpleNamespace

from vector_os_nano.skills.vlm_seek import VlmSeekSkill


class _FakeDetector:
    """Returns scripted contract detections; ignores the frame, records query."""

    def __init__(self, script):
        self._script = list(script)
        self._i = 0
        self.queries = []

    def detect_targets(self, frame, query, min_area_frac=0.0025):  # noqa: ARG002
        self.queries.append(query)
        d = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return d


class _FakeBase:
    """Advances on forward walks until 'blocked' at x≈3.0, then holds — so the
    progress-stall (physical, collision-blocked) arrival fires, mirroring how a
    real VLM seek arrives (the VLM bbox area is too noisy to gate arrival)."""

    def __init__(self):
        self.actions = []
        self._x = 0.0

    def get_camera_frame(self, timeout=5.0):
        return "frame"

    def walk(self, vx, vy, vyaw, duration=0.3):
        self.actions.append(("forward" if vx > 0 else "turn", round(vyaw, 2)))
        if vx > 0 and self._x < 3.0:           # advance, then hit the wall/object
            self._x = min(3.0, self._x + 0.5)

    def stop(self):
        self.actions.append(("stop", 0.0))

    def get_position(self):
        return [self._x, 0.0, 0.77]


def _ctx(base):
    return SimpleNamespace(base=base, world_model=None, agent=None)


class TestVlmSeek:
    def test_finds_and_arrives(self):
        # chair centred & small (area never reaches the high VLM arrive bar) →
        # the robot walks forward, gets blocked at x≈3.0, and the progress-stall
        # fires arrival (the honest physical signal).
        det = _FakeDetector([[{"label": "chair", "x_norm": 0.0, "area_frac": 0.03}]])
        base = _FakeBase()
        res = VlmSeekSkill(detector=det).execute(
            {"label": "chair", "max_iters": 30}, _ctx(base))
        assert res.success is True
        assert res.result_data["reason"] == "arrived"
        assert res.result_data["seen"] is True
        assert res.result_data["transport"] == "vlm_vision"
        # the VLM was really queried with the requested class
        assert det.queries and det.queries[0] == "chair"

    def test_never_recognised_fails_honestly(self):
        det = _FakeDetector([[]])
        base = _FakeBase()
        res = VlmSeekSkill(detector=det).execute(
            {"label": "chair", "max_iters": 4}, _ctx(base))
        assert res.success is False
        assert res.result_data["seen"] is False
        assert "chair" in (res.error_message or "")

    def test_no_camera_base_fails(self):
        base = SimpleNamespace()        # no get_camera_frame
        res = VlmSeekSkill(detector=_FakeDetector([[]])).execute(
            {"label": "chair"}, _ctx(base))
        assert res.success is False
        assert res.result_data["diagnosis"] == "no_camera"
