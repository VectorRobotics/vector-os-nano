# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tests for Go2 quadruped skills."""
import pytest
from unittest.mock import MagicMock
from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.world_model import WorldModel
from vector_os_nano.core.types import SkillResult


def _make_go2_context():
    """Create a SkillContext with a mock base (Go2)."""
    base = MagicMock()
    yaw = [0.0]

    def walk(vx, vy, vyaw, duration):  # a perfect gait: heading integrates the yaw rate
        yaw[0] += vyaw * duration
        return True

    base.walk.side_effect = walk
    base.stand.return_value = True
    base.sit.return_value = True
    base.lie_down.return_value = True
    base.get_position.return_value = (2.5, 2.5, 0.27)
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
        assert skill.__class__.__skill_direct__  # direct skill
        result = skill.execute({}, ctx)
        assert result.success
        ctx.base.stand.assert_called_once()

    def test_sit(self):
        from vector_os_nano.skills.go2.stance import SitSkill
        ctx = _make_go2_context()
        skill = SitSkill()
        assert skill.__class__.__skill_direct__
        result = skill.execute({}, ctx)
        assert result.success
        ctx.base.sit.assert_called_once()

    def test_lie_down(self):
        from vector_os_nano.skills.go2.stance import LieDownSkill
        ctx = _make_go2_context()
        skill = LieDownSkill()
        assert skill.__class__.__skill_direct__
        result = skill.execute({}, ctx)
        assert result.success
        ctx.base.lie_down.assert_called_once()

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
        assert len(skills) == 12
        names = {s.name for s in skills}
        assert names == {
            "walk", "turn", "stand", "sit", "lie_down",
            "navigate", "explore", "where_am_i", "stop",
            "look", "describe_scene", "patrol",
        }
