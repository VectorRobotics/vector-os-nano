# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tests for Go2 quadruped skills."""
import math

import pytest
from unittest.mock import MagicMock
from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.world_model import WorldModel
from vector_os_nano.core.types import SkillResult


def _make_go2_context():
    """Create a SkillContext with a mock base (Go2)."""
    base = MagicMock()
    base.limits = (.5, .3, .5)
    yaw = [0.0]
    pos = [2.5, 2.5]

    def walk(vx, vy, vyaw, duration):  # a perfect gait: pose integrates the command
        c, s = math.cos(yaw[0]), math.sin(yaw[0])
        pos[0] += (vx * c - vy * s) * duration
        pos[1] += (vx * s + vy * c) * duration
        yaw[0] += vyaw * duration
        return True

    base.walk.side_effect = walk
    base.set_posture.return_value = True
    base.get_position.side_effect = lambda: (pos[0], pos[1], 0.27)
    base.get_heading.side_effect = lambda: yaw[0]
    return SkillContext(
        arm=None,
        gripper=None,
        perception=None,
        world_model=WorldModel(),
        calibration=None,
        base=base,
    )


class TestWalkSkill:
    def test_a_direction_with_zero_speed_limit_is_refused(self):
        from vector_os_nano.skills.go2.walk import WalkSkill
        ctx = _make_go2_context()
        ctx.base.limits = (.5, 0., .5)
        assert not WalkSkill().execute({"direction": "left"}, ctx).success
        ctx.base.walk.assert_not_called()

    @pytest.mark.parametrize("direction,axis", [("forward", 0), ("left", 1)])
    def test_duration_uses_the_actual_body_limit(self, direction, axis):
        from vector_os_nano.skills.go2.walk import WalkSkill
        ctx = _make_go2_context()
        ctx.base.limits = (.12, .08, .2)
        result = WalkSkill().execute({"direction": direction, "distance": 1., "speed": 2.}, ctx)
        args = ctx.base.walk.call_args.args
        assert result.success
        assert args[axis] == ctx.base.limits[axis]
        assert args[3] == pytest.approx(1. / ctx.base.limits[axis])

    def test_walk_forward(self):
        from vector_os_nano.skills.go2.walk import WalkSkill
        ctx = _make_go2_context()
        skill = WalkSkill()
        result = skill.execute({"direction": "forward", "distance": 1.0, "speed": 0.3}, ctx)
        assert isinstance(result, SkillResult)
        assert result.success
        ctx.base.walk.assert_called_once()
        args = ctx.base.walk.call_args
        assert args[0][0] > 0  # vx positive (forward)
        assert abs(args[0][1]) < 0.01  # vy ~0
        assert abs(args[0][2]) < 0.01  # vyaw ~0

    def test_walk_backward(self):
        from vector_os_nano.skills.go2.walk import WalkSkill
        ctx = _make_go2_context()
        skill = WalkSkill()
        result = skill.execute({"direction": "backward", "distance": 0.5}, ctx)
        assert result.success
        args = ctx.base.walk.call_args
        assert args[0][0] < 0  # vx negative

    def test_walk_no_base(self):
        from vector_os_nano.skills.go2.walk import WalkSkill
        ctx = _make_go2_context()
        ctx.base = None
        skill = WalkSkill()
        result = skill.execute({"direction": "forward"}, ctx)
        assert not result.success
        assert "no_base" in result.diagnosis_code

    def test_walk_skill_metadata(self):
        from vector_os_nano.skills.go2.walk import WalkSkill
        skill = WalkSkill()
        assert skill.name == "walk"
        assert "walk" in skill.__class__.__skill_aliases__
        assert "走" in skill.__class__.__skill_aliases__


class TestTurnSkill:
    def test_a_base_without_yaw_control_is_refused(self):
        from vector_os_nano.skills.go2.turn import TurnSkill
        ctx = _make_go2_context()
        ctx.base.limits = (.5, .3, 0.)
        assert not TurnSkill().execute({"angle": 90}, ctx).success
        ctx.base.walk.assert_not_called()

    def test_turn_uses_the_actual_body_limit(self):
        from vector_os_nano.skills.go2.turn import TurnSkill
        ctx = _make_go2_context()
        ctx.base.limits = (.12, .08, .2)
        assert TurnSkill().execute({"angle": 90}, ctx).success
        assert ctx.base.walk.call_args.args[2] == .2

    def test_turn_left(self):
        from vector_os_nano.skills.go2.turn import TurnSkill
        ctx = _make_go2_context()
        skill = TurnSkill()
        result = skill.execute({"direction": "left", "angle": 90}, ctx)
        assert result.success
        ctx.base.walk.assert_called_once()
        args = ctx.base.walk.call_args
        assert abs(args[0][0]) < 0.01  # vx ~0
        assert args[0][2] > 0  # vyaw positive (left turn)

    def test_turn_right(self):
        from vector_os_nano.skills.go2.turn import TurnSkill
        ctx = _make_go2_context()
        skill = TurnSkill()
        result = skill.execute({"direction": "right", "angle": 45}, ctx)
        assert result.success
        args = ctx.base.walk.call_args
        assert args[0][2] < 0  # vyaw negative (right turn)

    def test_turn_skill_metadata(self):
        from vector_os_nano.skills.go2.turn import TurnSkill
        skill = TurnSkill()
        assert skill.name == "turn"
        assert "转" in skill.__class__.__skill_aliases__


class TestStanceSkills:
    def test_stand(self):
        from vector_os_nano.skills.go2.stance import StandSkill
        ctx = _make_go2_context()
        skill = StandSkill()
        result = skill.execute({}, ctx)
        assert result.success
        ctx.base.set_posture.assert_called_once_with("stand")

    def test_sit(self):
        from vector_os_nano.skills.go2.stance import SitSkill
        ctx = _make_go2_context()
        skill = SitSkill()
        result = skill.execute({}, ctx)
        assert result.success
        ctx.base.set_posture.assert_called_once_with("sit")

    def test_lie_down(self):
        from vector_os_nano.skills.go2.stance import LieDownSkill
        ctx = _make_go2_context()
        skill = LieDownSkill()
        result = skill.execute({}, ctx)
        assert result.success
        ctx.base.set_posture.assert_called_once_with("lie_down")

    def test_a_posture_the_joints_did_not_reach_fails(self):
        from vector_os_nano.skills.go2.stance import SitSkill
        ctx = _make_go2_context()
        ctx.base.set_posture.return_value = False  # the base read its joints back off the posture
        result = SitSkill().execute({}, ctx)
        assert not result.success and result.diagnosis_code == "sit_failed"
        assert result.result_data["controller_reached_posture"] is False

    def test_a_body_without_the_posture_fails_naming_the_valid_ones(self):
        from vector_os_nano.skills.go2.stance import LieDownSkill
        ctx = _make_go2_context()
        ctx.base.set_posture.side_effect = KeyError("no posture 'lie_down'; postures: ['stand', 'sit']")
        result = LieDownSkill().execute({}, ctx)
        assert not result.success and "postures: ['stand', 'sit']" in result.error_message

    def test_stance_no_base(self):
        from vector_os_nano.skills.go2.stance import StandSkill
        ctx = _make_go2_context()
        ctx.base = None
        skill = StandSkill()
        result = skill.execute({}, ctx)
        assert not result.success


class TestSkillRegistration:
    def test_get_go2_skills(self):
        from vector_os_nano.skills.go2 import get_go2_skills
        skills = get_go2_skills()
        names = [s.name for s in skills]
        assert sorted(names) == sorted({
            "walk", "turn", "stand", "sit", "lie_down",
            "navigate", "navigate_to_object", "explore", "where_am_i", "stop",
            "look", "describe_scene", "patrol",
        })
