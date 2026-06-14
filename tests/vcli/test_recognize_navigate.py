# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #9 R3 — RecognizeNavigateSkill: VLM recognise → lidar locate → navigate.

The reliable-arrival pivot (Case 21). Tested with an injected fake VLM + a fake
base exposing camera/lidar/pose/navigate_to (no sim, no network): the skill must
recognise (real VLM in prod), locate via lidar bearing+range, then hand the SENSED
location to navigate_to — never GT, never pixel servoing.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vector_os_nano.skills.recognize_navigate import RecognizeNavigateSkill


class _Lidar:
    def __init__(self, pts):
        self.points = np.array(pts, dtype=np.float32).reshape(-1, 4)


class _FakeBase:
    """Recognised chair is ahead; lidar has a hit at (2.0,0) → located. navigate_to
    records the goal and reports reached."""

    def __init__(self, lidar_pts):
        self._lidar = _Lidar(lidar_pts)
        self.nav_goal = None
        self.walks = 0

    def get_camera_frame(self, timeout=5.0):
        return "frame"

    def get_position(self):
        return [0.0, 0.0, 0.77]

    def get_heading(self):
        return 0.0

    def get_lidar_scan(self):
        return self._lidar

    def walk(self, vx, vy, vyaw, duration=0.4):
        self.walks += 1

    def stop(self):
        pass

    def navigate_to(self, x, y, tol=0.2):
        self.nav_goal = (x, y)
        return {"reached": True, "remaining": 0.1, "pos": [x, y, 0.77]}


class _Det:
    def __init__(self, seq):
        self._seq = list(seq)
        self._i = 0
        self.queries = []

    def detect_targets(self, frame, query, min_area_frac=0.0025):  # noqa: ARG002
        self.queries.append(query)
        d = self._seq[min(self._i, len(self._seq) - 1)]
        self._i += 1
        return d


def _ctx(base):
    return SimpleNamespace(base=base, world_model=None, agent=None)


class TestRecognizeNavigate:
    def test_recognise_locate_navigate(self):
        # chair recognised dead-ahead; lidar hit at (2,0) in that direction.
        base = _FakeBase([[2.0, 0.0, 0.3, 1.0]])
        det = _Det([[{"label": "chair", "x_norm": 0.0, "area_frac": 0.03}]])
        res = RecognizeNavigateSkill(detector=det).execute(
            {"label": "chair"}, _ctx(base))
        assert res.success is True
        assert res.result_data["transport"] == "recognize_navigate"
        # navigate_to was handed the SENSED location (≈ the lidar hit), not GT
        assert base.nav_goal is not None
        assert abs(base.nav_goal[0] - 2.0) < 1e-3 and abs(base.nav_goal[1]) < 1e-3
        assert det.queries[0] == "chair"        # VLM really queried

    def test_never_recognised_fails_honestly(self):
        base = _FakeBase([[2.0, 0.0, 0.3, 1.0]])
        det = _Det([[]])                         # VLM never grounds it
        res = RecognizeNavigateSkill(detector=det).execute(
            {"label": "chair", "max_iters": 4}, _ctx(base))
        assert res.success is False
        assert res.result_data["diagnosis"] == "not_found"
        assert base.nav_goal is None             # never navigated to a phantom

    def test_recognised_but_unranged_advances_then_fails(self):
        # recognised every frame but lidar has NO hit in the bearing → must advance
        # (walk) and, if still unranged within max_iters, fail "not_located".
        base = _FakeBase([[0.0, 3.0, 0.3, 1.0]])   # only a side hit, never in bearing
        det = _Det([[{"label": "chair", "x_norm": 0.0, "area_frac": 0.03}]])
        res = RecognizeNavigateSkill(detector=det).execute(
            {"label": "chair", "max_iters": 5}, _ctx(base))
        assert res.success is False
        assert res.result_data["diagnosis"] == "not_located"
        assert base.walks > 0                    # it tried to close in
        assert base.nav_goal is None

    def test_base_without_navigate_to_fails_loud(self):
        base = SimpleNamespace(get_camera_frame=lambda **k: "frame")
        res = RecognizeNavigateSkill(detector=_Det([[]])).execute(
            {"label": "chair"}, _ctx(base))
        assert res.success is False
        assert res.result_data["diagnosis"] == "no_navigate"
