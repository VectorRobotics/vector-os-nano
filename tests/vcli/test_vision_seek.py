# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #8 R10 — VisionSeekSkill: find & approach a target by RECOGNITION.

The skill loops camera→detect_targets→decide→actuate: not seen → scan (turn),
seen off-centre → turn toward it, seen & centred → walk forward, seen & large →
arrived. It uses the camera (not GT coordinates) to locate the target — the
last piece of req #5. The seek decision is a pure function (unit-tested here);
the execute loop is exercised with a fake base returning scripted detections.
"""
from __future__ import annotations

from types import SimpleNamespace

from vector_os_nano.skills.vision_seek import VisionSeekSkill, _seek_action


class TestSeekDecision:
    def test_not_seen_scans(self):
        act, vx, vyaw = _seek_action(None)
        assert act == "scan" and vx == 0.0 and vyaw != 0.0

    def test_seen_right_turns_right(self):
        # x_norm > 0 = target on the robot's right → turn right (vyaw < 0)
        act, vx, vyaw = _seek_action({"x_norm": 0.6, "area_frac": 0.02})
        assert act == "turn" and vyaw < 0 and vx == 0.0

    def test_seen_left_turns_left(self):
        act, vx, vyaw = _seek_action({"x_norm": -0.6, "area_frac": 0.02})
        assert act == "turn" and vyaw > 0

    def test_centred_walks_forward(self):
        act, vx, vyaw = _seek_action({"x_norm": 0.05, "area_frac": 0.03})
        assert act == "forward" and vx > 0

    def test_large_means_arrived(self):
        act, vx, vyaw = _seek_action({"x_norm": 0.0, "area_frac": 0.5})
        assert act == "arrived" and vx == 0.0 and vyaw == 0.0


class _FakeBase:
    """Scripted camera: returns frames whose detection sequence walks the robot
    from 'not seen' → 'seen right' → 'centred' → 'arrived'."""

    def __init__(self, script):
        self._script = list(script)   # list of detect_targets() returns
        self._i = 0
        self.actions = []

    def get_camera_frame(self, timeout=5.0):
        return "frame"               # opaque; detection is injected via _detect

    def detect(self):
        d = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return d

    def walk(self, vx, vy, vyaw, duration=0.3):
        self.actions.append(("forward" if vx > 0 else "turn", round(vyaw, 2)))

    def get_position(self):
        return [1.0, 0.0, 0.77]


def _ctx(base):
    return SimpleNamespace(base=base, world_model=None, agent=None)


class TestSeekLoop:
    def test_finds_and_arrives(self, monkeypatch):
        # not seen, not seen, seen-right, centred, arrived
        script = [
            [], [],
            [{"label": "red", "x_norm": 0.5, "area_frac": 0.03}],
            [{"label": "red", "x_norm": 0.05, "area_frac": 0.06}],
            [{"label": "red", "x_norm": 0.0, "area_frac": 0.5}],
        ]
        base = _FakeBase(script)
        # inject detection: the skill calls detect_targets(frame); route to fake
        import vector_os_nano.skills.vision_seek as vs
        monkeypatch.setattr(vs, "detect_targets", lambda frame, **k: base.detect())
        res = VisionSeekSkill().execute(
            {"label": "red", "max_iters": 10}, _ctx(base))
        assert res.success is True
        assert res.result_data["reason"] == "arrived"
        assert res.result_data["seen"] is True

    def test_never_seen_fails_honestly(self, monkeypatch):
        base = _FakeBase([[]])
        import vector_os_nano.skills.vision_seek as vs
        monkeypatch.setattr(vs, "detect_targets", lambda frame, **k: [])
        res = VisionSeekSkill().execute(
            {"label": "red", "max_iters": 4}, _ctx(base))
        assert res.success is False
        assert res.result_data["seen"] is False
        assert "not" in (res.error_message or "").lower()
