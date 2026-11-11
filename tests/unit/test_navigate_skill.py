# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tests for unified NavigateSkill -- hardware-agnostic navigation."""
import pytest
from unittest.mock import MagicMock
from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.world_model import WorldModel
from vector_os_nano.sim.nav.follow import NavResult


def _arrived(x: float = 10.0, y: float = 3.0) -> NavResult:
    return NavResult(reached=True, reason="arrived", target=[x, y], pos=[x, y, 0.3])


def _make_scene_graph():
    """Create a SceneGraph loaded with room_layout.yaml for navigate tests."""
    import os
    from vector_os_nano.core.scene_graph import SceneGraph
    sg = SceneGraph()
    layout = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config", "worlds", "apartment.yaml",
    )
    if os.path.isfile(layout):
        sg.load_layout(layout)
    # Mark all rooms as visited (visit_count >= 1) so navigate trusts positions
    for room_id in list(sg._rooms):
        r = sg._rooms[room_id]
        sg._rooms[room_id] = type(r)(
            room_id=r.room_id, center_x=r.center_x, center_y=r.center_y,
            area=r.area, visit_count=1, last_visited=1.0,
            representative_description=r.representative_description,
            connected_rooms=r.connected_rooms,
        )
    return sg


def _make_context():
    base = MagicMock()
    base.walk.return_value = True
    base.get_position.return_value = [10.0, 3.0, 0.27]
    base.get_heading.return_value = 0.0
    base.navigate_to = MagicMock(return_value=_arrived())
    sg = _make_scene_graph()
    services = {"spatial_memory": sg}
    return SkillContext(
        bases={"go2": base},
        world_model=WorldModel(),
        services=services,
    )


class TestNavigateSkillMetadata:
    def test_skill_name(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        skill = NavigateSkill()
        assert skill.name == "navigate"

    def test_aliases(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        skill = NavigateSkill()
        assert "去" in skill.__class__.__skill_aliases__
        assert "navigate" in skill.__class__.__skill_aliases__

    def test_has_required_metadata(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        skill = NavigateSkill()
        assert hasattr(skill, "description")
        assert hasattr(skill, "parameters")
        assert "room" in skill.parameters
        assert hasattr(skill, "failure_modes")
        assert "no_base" in skill.failure_modes
        assert "unknown_room" in skill.failure_modes
        assert "navigation_failed" in skill.failure_modes


class TestNavigateWithNavStack:
    def test_proxy_navigate_to_used_when_available(self):
        """Mode 0: base.navigate_to() takes priority when present (MuJoCoGo2 / MuJoCoG1)."""
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context()
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        assert result.success
        # Proxy mode: base.navigate_to called, NOT nav service
        ctx.base.navigate_to.assert_called_once()

    def test_proxy_receives_kitchen_coordinates(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context()
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        args = ctx.base.navigate_to.call_args
        # Kitchen center is (17.0, 2.5)
        assert 15 < args[0][0] < 19
        assert 1 < args[0][1] < 4

    def test_nav_unknown_room_not_sent_to_nav_client(self):
        """Unknown room should fail before reaching nav client."""
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context()
        skill = NavigateSkill()
        result = skill.execute({"room": "mars_base"}, ctx)
        assert not result.success
        assert "unknown_room" in result.diagnosis_code
        ctx.base.navigate_to.assert_not_called()

    def test_a_base_without_a_planner_fails_loud(self):
        """skills-nav:dead-doorchain-proxy-branch: no navigate_to = no door-chain walk."""
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context()
        del ctx.base.navigate_to
        result = NavigateSkill().execute({"room": "kitchen"}, ctx)
        assert not result.success and result.diagnosis_code == "navigation_failed"


class TestNavigateRooms:
    def test_named_room(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context()
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        assert result.success

    def test_unknown_room(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context()
        skill = NavigateSkill()
        result = skill.execute({"room": "nonexistent_room"}, ctx)
        assert not result.success
        assert "unknown_room" in result.diagnosis_code

    def test_chinese_room_name(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context()
        skill = NavigateSkill()
        result = skill.execute({"room": "厨房"}, ctx)
        assert result.success

    def test_no_base(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = SkillContext(world_model=WorldModel())
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        assert not result.success
        assert "no_base" in result.diagnosis_code

    def test_already_at_destination(self):
        """Robot already near target room center -- should succeed immediately."""
        from vector_os_nano.skills.navigate import NavigateSkill
        # Position (3.0, 2.5) is the living_room center
        base = MagicMock()
        base.walk.return_value = True
        base.get_position.return_value = [3.0, 2.5, 0.27]
        base.get_heading.return_value = 0.0
        base.navigate_to = MagicMock(return_value=_arrived(3.0, 2.5))
        sg = _make_scene_graph()
        ctx = SkillContext(bases={"go2": base}, world_model=WorldModel(),
                          services={"spatial_memory": sg})
        skill = NavigateSkill()
        result = skill.execute({"room": "living_room"}, ctx)
        assert result.success

    def test_result_data(self):
        """Success includes the room and the position in result_data."""
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context()
        skill = NavigateSkill()
        result = skill.execute({"room": "kitchen"}, ctx)
        assert result.success
        assert result.result_data is not None
        assert "room" in result.result_data
        assert result.result_data["room"] == "kitchen"

    def test_alias_english_shorthand(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context()
        skill = NavigateSkill()
        # "bedroom" is an alias for master_bedroom
        result = skill.execute({"room": "bedroom"}, ctx)
        assert result.success
        assert result.result_data["room"] == "master_bedroom"

    def test_empty_room_param(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context()
        skill = NavigateSkill()
        result = skill.execute({"room": ""}, ctx)
        assert not result.success
        assert "unknown_room" in result.diagnosis_code

    def test_missing_room_param(self):
        from vector_os_nano.skills.navigate import NavigateSkill
        ctx = _make_context()
        skill = NavigateSkill()
        result = skill.execute({}, ctx)
        assert not result.success
        assert "unknown_room" in result.diagnosis_code


class TestNavigateFromGo2Package:
    def test_go2_skills_includes_navigate(self):
        from vector_os_nano.skills.go2 import get_go2_skills
        skills = get_go2_skills()
        names = {s.name for s in skills}
        assert "navigate" in names

    def test_go2_navigate_is_unified(self):
        """The NavigateSkill from go2 package should be the unified one."""
        from vector_os_nano.skills.go2 import NavigateSkill as Go2Nav
        from vector_os_nano.skills.navigate import NavigateSkill as UnifiedNav
        assert Go2Nav is UnifiedNav

    def test_go2_navigate_module_removed(self):
        """go2/navigate.py shim has been removed -- import must fail."""
        import importlib
        import pytest
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("vector_os_nano.skills.go2.navigate")
