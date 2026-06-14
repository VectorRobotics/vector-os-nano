# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Batch 2 R7 — walk/turn motion-evidence contract (design review #6/#7).

WalkSkill used to return success=True for zero-distance, zero-speed,
wall-blocked and silently-dropped-lateral commands, echoing the COMMAND
distance as if measured. Motion skills now measure: result_data carries
start/moved_m (turn: turned_rad) /duration_s; a commanded motion that
produced almost none fails loud; distance<=0 is bad_params; and the skills
declare REAL verify templates (step_output evidence) so they graduate from
RobotWorld's transitional evidence-gate exemption.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from vector_os_nano.skills.go2.turn import TurnSkill
from vector_os_nano.skills.go2.walk import WalkSkill


class _MovingBase:
    """walk() teleports by the commanded integral — an honest mover."""

    def __init__(self):
        self.pos = [0.0, 0.0, 0.0]
        self.heading = 0.0

    def walk(self, vx, vy, vyaw, duration):
        self.pos[0] += vx * duration
        self.pos[1] += vy * duration
        self.heading += vyaw * duration
        return True

    def get_position(self):
        return list(self.pos)

    def get_heading(self):
        return self.heading


class _FrozenBase(_MovingBase):
    """Accepts commands, never moves (wall / dropped lateral)."""

    def walk(self, vx, vy, vyaw, duration):
        return True


def _ctx(base):
    return SimpleNamespace(base=base, world_model=None, agent=None)


class TestWalkEvidence:
    def test_zero_distance_is_bad_params(self):
        res = WalkSkill().execute({"distance": 0.0}, _ctx(_MovingBase()))
        assert res.success is False
        assert "distance" in (res.error_message or "")

    def test_moved_evidence_recorded(self):
        res = WalkSkill().execute(
            {"direction": "forward", "distance": 1.0}, _ctx(_MovingBase()))
        assert res.success is True
        rd = res.result_data
        assert rd["moved_m"] == 1.0
        assert rd["requested_m"] == 1.0
        assert rd["start"] == [0.0, 0.0, 0.0]
        assert rd["duration_s"] >= 0.0

    def test_no_motion_fails_loud(self):
        res = WalkSkill().execute(
            {"direction": "forward", "distance": 1.0}, _ctx(_FrozenBase()))
        assert res.success is False
        assert res.result_data.get("diagnosis") == "moved_short"
        assert res.result_data["moved_m"] == 0.0

    def test_dropped_lateral_now_fails(self):
        # habitat ignores vy — '往左走一米' used to be a 3.3s no-op PASS (#7).
        res = WalkSkill().execute(
            {"direction": "left", "distance": 1.0}, _ctx(_FrozenBase()))
        assert res.success is False

    def test_baseless_position_fails_open(self):
        # A base with no odometry cannot prove motion — keep legacy success
        # but mark the evidence as unmeasured (never fabricate moved_m).
        class _NoOdom:
            def walk(self, vx, vy, vyaw, duration):
                return True

        res = WalkSkill().execute(
            {"direction": "forward", "distance": 1.0}, _ctx(_NoOdom()))
        assert res.success is True
        assert res.result_data.get("moved_m") is None


class TestTurnEvidence:
    def test_turned_evidence_recorded(self):
        res = TurnSkill().execute(
            {"direction": "left", "angle": 90}, _ctx(_MovingBase()))
        assert res.success is True
        assert abs(res.result_data["turned_rad"] - math.pi / 2) < 0.05
        assert res.result_data["duration_s"] >= 0.0

    def test_frozen_turn_fails_loud(self):
        res = TurnSkill().execute(
            {"direction": "left", "angle": 90}, _ctx(_FrozenBase()))
        assert res.success is False
        assert res.result_data.get("diagnosis") == "turned_short"

    def test_zero_angle_is_bad_params(self):
        res = TurnSkill().execute({"angle": 0}, _ctx(_MovingBase()))
        assert res.success is False


class TestTemplatesAndExemptions:
    def test_motion_skills_declare_evidence_templates(self):
        assert "step_output('moved_m')" in WalkSkill.verify_template
        assert "step_output('turned_rad')" in TurnSkill.verify_template

    def test_walk_turn_graduated_from_exemption(self):
        from vector_os_nano.vcli.worlds.robot import RobotWorld

        exempt = RobotWorld().evidence_exempt_strategies()
        assert "walk" not in exempt
        assert "turn" not in exempt
        assert "explore" in exempt  # async skills stay until batch 3+
