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

# Import the MODULE so the rate-limit class is resolved LIVE (an earlier test,
# test_qwen_vl_wiring, importlib.reloads vlm_go2 → a cached class binding would
# not match the live class recognize_navigate catches).
from vector_os_nano.perception import vlm_go2 as _vlm_go2
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
    """Scripted fake detector. A sequence entry that is a _vlm_go2.VlmRateLimitError
    (instance or the class) is RAISED — mirroring production where the opt-in
    ``raise_on_rate_limit=True`` propagates a 429; any other entry is returned."""

    def __init__(self, seq):
        self._seq = list(seq)
        self._i = 0
        self.queries = []

    def detect_targets(self, frame, query, min_area_frac=0.0025,  # noqa: ARG002
                       raise_on_rate_limit=False):
        self.queries.append(query)
        d = self._seq[min(self._i, len(self._seq) - 1)]
        self._i += 1
        if d is _vlm_go2.VlmRateLimitError or isinstance(d, _vlm_go2.VlmRateLimitError):
            if raise_on_rate_limit:
                raise d if isinstance(d, _vlm_go2.VlmRateLimitError) else _vlm_go2.VlmRateLimitError("429")
            return []          # swallowed when caller did not opt in (legacy)
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
        # navigate_to was handed a STANDOFF point 0.7m in front of the SENSED
        # location (lidar hit 2.0 → goal 1.3), not GT and not the surface itself.
        assert base.nav_goal is not None
        assert abs(base.nav_goal[0] - 1.3) < 0.05 and abs(base.nav_goal[1]) < 1e-3
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


class TestRateLimit:
    """A throttled VLM (HTTP 429) must give up HONESTLY (vlm_unavailable) without
    blind-walking the robot or hammering the endpoint for the full max_iters —
    and must stay distinct from an honest 'object not in frame' miss."""

    def test_persistent_429_gives_up_without_walking(self, monkeypatch):
        import vector_os_nano.skills.recognize_navigate as rn
        monkeypatch.setattr(rn, "_RL_BACKOFF_S", 0.0)
        base = _FakeBase([[2.0, 0.0, 0.3, 1.0]])     # would locate if VLM worked
        det = _Det([_vlm_go2.VlmRateLimitError("429")])        # every call raises 429
        res = RecognizeNavigateSkill(detector=det).execute(
            {"label": "chair", "max_iters": 16}, _ctx(base))
        assert res.success is False
        assert res.result_data["diagnosis"] == "vlm_unavailable"
        assert res.result_data["rl_streak"] == rn._MAX_RL_STREAK
        assert base.walks == 0                        # CORE: zero blind walks
        assert base.nav_goal is None                  # never navigated to a phantom
        # capped: did NOT run all 16 iters — exactly _MAX_RL_STREAK VLM calls
        assert len(det.queries) == rn._MAX_RL_STREAK

    def test_429_then_success_rides_through_and_navigates(self, monkeypatch):
        import vector_os_nano.skills.recognize_navigate as rn
        monkeypatch.setattr(rn, "_RL_BACKOFF_S", 0.0)
        base = _FakeBase([[2.0, 0.0, 0.3, 1.0]])
        # raise twice, then ground the chair every frame (last entry repeats →
        # the second agreeing estimate confirms and navigation proceeds)
        det = _Det([_vlm_go2.VlmRateLimitError("429"), _vlm_go2.VlmRateLimitError("429"),
                    [{"label": "chair", "x_norm": 0.0, "area_frac": 0.03}]])
        res = RecognizeNavigateSkill(detector=det).execute(
            {"label": "chair", "max_iters": 16}, _ctx(base))
        assert res.success is True                     # rode through the 429s
        assert res.result_data["diagnosis"] == "ok"
        assert base.nav_goal is not None
        assert abs(base.nav_goal[0] - 1.3) < 0.05 and abs(base.nav_goal[1]) < 1e-3

    def test_single_429_between_good_frames_does_not_escalate(self, monkeypatch):
        import vector_os_nano.skills.recognize_navigate as rn
        monkeypatch.setattr(rn, "_RL_BACKOFF_S", 0.0)
        base = _FakeBase([[2.0, 0.0, 0.3, 1.0]])
        good = [{"label": "chair", "x_norm": 0.0, "area_frac": 0.03}]
        # alternating raise / good — streak resets on each good frame, so the
        # cap is never reached and the run completes normally (navigates).
        det = _Det([_vlm_go2.VlmRateLimitError("429"), good,
                    _vlm_go2.VlmRateLimitError("429"), good, good])
        res = RecognizeNavigateSkill(detector=det).execute(
            {"label": "chair", "max_iters": 16}, _ctx(base))
        assert res.result_data["diagnosis"] != "vlm_unavailable"
        assert res.success is True                     # good frames carried it


class TestDepthPath:
    """When the base exposes get_camera_observation (depth+pose), the skill must
    locate by DEPTH-AT-BBOX (semantic), not lidar — skipping intervening obstacles."""

    def test_prefers_depth_over_lidar(self):
        import math
        # cam looking +x, up +z (cam -z = +x)
        cam_mat = np.array([[0.0, 0.0, -1.0],
                            [-1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]], dtype=np.float64)
        depth = np.full((240, 320), 40.0, dtype=np.float32)
        depth[110:130, 150:170] = 3.0       # chair surface at centre, 3 m ahead

        class _ObsBase(_FakeBase):
            def get_camera_observation(self, timeout=5.0):
                return {"rgb": "frame", "depth": depth,
                        "cam_pos": np.array([0.0, 0.0, 1.0]),
                        "cam_mat": cam_mat.flatten(), "fovy": 70.0}

        # lidar has a CLOSER obstacle hit straight ahead (would mislead lidar path)
        base = _ObsBase([[1.5, 0.0, 0.3, 1.0]])
        det = _Det([[{"label": "chair", "x_norm": 0.0, "area_frac": 0.03, "y_norm": 0.0}]])
        res = RecognizeNavigateSkill(detector=det).execute({"label": "chair"}, _ctx(base))
        assert res.success is True
        assert res.result_data["located_by"] == "depth"
        # navigated to a standoff in front of the DEPTH point (3,0 → 2.3,0), NOT
        # the closer lidar obstacle (1.5,0) — proves depth was preferred.
        assert abs(base.nav_goal[0] - 2.3) < 0.2 and abs(base.nav_goal[1]) < 0.2
