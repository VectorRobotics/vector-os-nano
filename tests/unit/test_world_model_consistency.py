"""Unit tests for WorldModel.apply_skill_effects() — pick mode=hold and mode=drop.

TDD RED phase: these tests are written before the implementation is changed.
They verify the new branching behaviour specified in T3.
"""
from __future__ import annotations

import time

import pytest


@pytest.fixture
def wm():
    from vector_os_nano.core.world_model import WorldModel
    return WorldModel()


@pytest.fixture
def success_result():
    from vector_os_nano.core.types import SkillResult
    return SkillResult(success=True)


def _make_obj(label: str, obj_id: str | None = None):
    from vector_os_nano.core.world_model import ObjectState
    return ObjectState(
        object_id=obj_id or f"obj_{label}",
        label=label,
        x=0.2,
        y=0.0,
        z=0.03,
        confidence=0.9,
        state="on_table",
        last_seen=time.time(),
    )


class TestPickDropMode:
    """mode='drop' (default) — object is removed, gripper becomes open."""

    def test_pick_drop_removes_only_picked_object(self, wm, success_result):
        """pick(drop, banana) must NOT remove mug or bottle."""
        banana = _make_obj("banana", "obj_banana")
        mug = _make_obj("mug", "obj_mug")
        bottle = _make_obj("bottle", "obj_bottle")
        wm.add_object(banana)
        wm.add_object(mug)
        wm.add_object(bottle)

        wm.apply_skill_effects("pick", {"mode": "drop", "object_label": "banana"}, success_result)

        assert wm.get_object("obj_banana") is None, "banana must be removed after pick-drop"
        assert wm.get_object("obj_mug") is not None, "mug must survive pick-drop of banana"
        assert wm.get_object("obj_bottle") is not None, "bottle must survive pick-drop of banana"

    def test_pick_drop_clears_held_object(self, wm, success_result):
        """After drop, robot.held_object is None and gripper is open."""
        wm.add_object(_make_obj("cup", "obj_cup"))
        wm.update_robot_state(held_object="obj_cup", gripper_state="holding")

        wm.apply_skill_effects("pick", {"mode": "drop", "object_label": "cup"}, success_result)

        robot = wm.get_robot()
        assert robot.held_object is None
        assert robot.gripper_state == "open"

    def test_apply_skill_effects_pick_drop(self, wm, success_result):
        """pick-drop via object_id removes the object and clears held_object."""
        wm.add_object(_make_obj("spoon", "obj_spoon"))
        wm.add_object(_make_obj("fork", "obj_fork"))

        wm.apply_skill_effects("pick", {"mode": "drop", "object_id": "obj_spoon"}, success_result)

        assert wm.get_object("obj_spoon") is None
        assert wm.get_object("obj_fork") is not None
        assert wm.get_robot().held_object is None
        assert wm.get_robot().gripper_state == "open"

    def test_multi_pick_preserves_remaining(self, wm, success_result):
        """Sequential drops: after picking banana, mug is still present for the next pick."""
        banana = _make_obj("banana", "obj_banana")
        mug = _make_obj("mug", "obj_mug")
        wm.add_object(banana)
        wm.add_object(mug)

        # First pick — banana
        wm.apply_skill_effects("pick", {"mode": "drop", "object_label": "banana"}, success_result)
        assert wm.get_object("obj_mug") is not None, "mug must survive first pick"

        # Second pick — mug
        wm.apply_skill_effects("pick", {"mode": "drop", "object_label": "mug"}, success_result)
        assert wm.get_object("obj_banana") is None
        assert wm.get_object("obj_mug") is None


class TestPickHoldMode:
    """mode='hold' — object is marked grasped, robot.held_object is set."""

    def test_pick_hold_marks_object_grasped(self, wm, success_result):
        """pick(hold, banana) sets banana.state='grasped' and robot.held_object."""
        banana = _make_obj("banana", "obj_banana")
        wm.add_object(banana)

        wm.apply_skill_effects("pick", {"mode": "hold", "object_label": "banana"}, success_result)

        obj = wm.get_object("obj_banana")
        assert obj is not None, "banana must still exist after pick-hold"
        assert obj.state == "grasped"

        robot = wm.get_robot()
        assert robot.held_object == "obj_banana"
        assert robot.gripper_state == "holding"

    def test_apply_skill_effects_pick_hold(self, wm, success_result):
        """pick-hold via object_id sets held_object and gripper state."""
        wm.add_object(_make_obj("cup", "obj_cup"))

        wm.apply_skill_effects("pick", {"mode": "hold", "object_id": "obj_cup"}, success_result)

        robot = wm.get_robot()
        assert robot.held_object == "obj_cup"
        assert robot.gripper_state == "holding"

    def test_pick_hold_preserves_other_objects(self, wm, success_result):
        """pick-hold on one object does not touch the others."""
        wm.add_object(_make_obj("apple", "obj_apple"))
        wm.add_object(_make_obj("pear", "obj_pear"))

        wm.apply_skill_effects("pick", {"mode": "hold", "object_label": "apple"}, success_result)

        assert wm.get_object("obj_pear") is not None
        assert wm.get_object("obj_apple") is not None
        assert wm.get_object("obj_apple").state == "grasped"

    def test_pick_hold_object_remains_in_world(self, wm, success_result):
        """The grasped object is NOT removed from _objects during hold."""
        wm.add_object(_make_obj("mug", "obj_mug"))

        wm.apply_skill_effects("pick", {"mode": "hold", "object_id": "obj_mug"}, success_result)

        assert wm.get_object("obj_mug") is not None, "mug must remain in world model during hold"


# ---------------------------------------------------------------------------
# DetectSkill merge logic tests (T6)
# ---------------------------------------------------------------------------

from unittest.mock import Mock

from vector_os_nano.skills.detect import DetectSkill
from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.types import Detection, TrackedObject, Pose3D


def _make_detect_context(world: "WorldModel", detections, tracked_objects=None):
    """Build a SkillContext wired to mock perception for detect tests."""
    mock_perception = Mock()
    mock_perception.detect = Mock(return_value=detections)
    if tracked_objects is not None:
        mock_perception.track = Mock(return_value=tracked_objects)
    else:
        mock_perception.track = Mock(return_value=[])

    mock_cal = Mock()
    mock_cal.camera_to_base = Mock(side_effect=lambda p: p)  # identity transform

    return SkillContext(
        arm=None,
        gripper=None,
        perception=mock_perception,
        world_model=world,
        calibration=mock_cal,
        config={},
    )


class TestDetectSkillMergeLogic:
    """DetectSkill must reuse existing world model IDs instead of creating new ones."""

    def test_detect_merges_existing_objects(self):
        """Add banana_0 to world model; detect banana → ID stays banana_0, not banana_1."""
        from vector_os_nano.core.world_model import WorldModel, ObjectState

        world = WorldModel()
        world.add_object(ObjectState(
            object_id="banana_0",
            label="banana",
            x=0.1, y=0.1, z=0.0,
            confidence=0.8,
            state="on_table",
        ))

        detections = [Detection(label="banana", bbox=(100, 100, 200, 200), confidence=0.9)]
        ctx = _make_detect_context(world, detections)

        skill = DetectSkill()
        skill.execute({"query": "banana"}, ctx)

        assert world.get_object("banana_0") is not None, "banana_0 must be preserved"
        assert world.get_object("banana_1") is None, "banana_1 must NOT be created on merge"

    def test_detect_creates_new_for_unknown(self):
        """Empty world model → detect creates object with _0 suffix."""
        from vector_os_nano.core.world_model import WorldModel

        world = WorldModel()
        detections = [Detection(label="apple", bbox=(50, 50, 150, 150), confidence=0.85)]
        ctx = _make_detect_context(world, detections)

        skill = DetectSkill()
        skill.execute({"query": "apple"}, ctx)

        assert world.get_object("apple_0") is not None, "apple_0 must be created for new object"

    def test_detect_updates_position_on_merge(self):
        """banana_0 at (0.1, 0.1, 0.0) → detect with pose (0.2, 0.2, 0.3) → position updated."""
        from vector_os_nano.core.world_model import WorldModel, ObjectState

        world = WorldModel()
        world.add_object(ObjectState(
            object_id="banana_0",
            label="banana",
            x=0.1, y=0.1, z=0.0,
            confidence=0.8,
            state="on_table",
        ))

        detections = [Detection(label="banana", bbox=(100, 100, 200, 200), confidence=0.9)]
        tracked = [
            TrackedObject(
                track_id=1,
                label="banana",
                bbox_2d=(100, 100, 200, 200),
                pose=Pose3D(x=0.2, y=0.2, z=0.3),
                confidence=0.9,
            )
        ]
        ctx = _make_detect_context(world, detections, tracked)

        skill = DetectSkill()
        skill.execute({"query": "banana"}, ctx)

        obj = world.get_object("banana_0")
        assert obj is not None
        # Position should be updated to the tracked pose values
        assert abs(obj.x - 0.2) < 1e-6, f"x should be 0.2, got {obj.x}"
        assert abs(obj.y - 0.2) < 1e-6, f"y should be 0.2, got {obj.y}"
