# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""DetectSkill keeps the world model consistent: it reuses object ids on a re-detect."""
from __future__ import annotations

from unittest.mock import Mock

from vector_os_nano.core.skill import SkillContext
from vector_os_nano.core.types import Detection, Pose3D, TrackedObject
from vector_os_nano.core.world_model import WorldModel
from vector_os_nano.skills.detect import DetectSkill


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
        tracked = [TrackedObject(track_id=1, label="apple", bbox_2d=(50, 50, 150, 150),
                                 pose=Pose3D(x=0.2, y=0.0, z=0.05), confidence=0.85)]
        ctx = _make_detect_context(world, detections, tracked)

        skill = DetectSkill()
        skill.execute({"query": "apple"}, ctx)

        assert world.get_object("apple_0") is not None, "apple_0 must be created for new object"
        # a 2D-only detection (no tracked pose) never enters the world model
        DetectSkill().execute({"query": "pear"}, _make_detect_context(
            world, [Detection(label="pear", bbox=(1, 1, 9, 9), confidence=0.8)]))
        assert world.get_objects_by_label("pear") == []

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
