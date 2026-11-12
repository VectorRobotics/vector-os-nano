# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Unit tests for vector_os_nano.core.world_model (the session object store R4 replaces)."""
from __future__ import annotations

import json
import time

import pytest


@pytest.fixture
def wm():
    from vector_os_nano.core.world_model import WorldModel
    return WorldModel()


@pytest.fixture
def cup_obj():
    from vector_os_nano.core.world_model import ObjectState
    return ObjectState(
        object_id="obj_001",
        label="cup",
        x=0.2,
        y=0.05,
        z=0.03,
        confidence=0.95,
        state="on_table",
        last_seen=time.time(),
    )


@pytest.fixture
def bottle_obj():
    from vector_os_nano.core.world_model import ObjectState
    return ObjectState(
        object_id="obj_002",
        label="bottle",
        x=0.15,
        y=-0.08,
        z=0.05,
        confidence=0.85,
        state="on_table",
        last_seen=time.time(),
    )


class TestAddGetObject:
    def test_add_and_get_by_id(self, wm, cup_obj):
        wm.add_object(cup_obj)
        retrieved = wm.get_object("obj_001")
        assert retrieved is not None
        assert retrieved.object_id == "obj_001"
        assert retrieved.label == "cup"

    def test_get_nonexistent_returns_none(self, wm):
        assert wm.get_object("nonexistent") is None

    def test_add_overwrites_existing(self, wm, cup_obj):
        from vector_os_nano.core.world_model import ObjectState
        wm.add_object(cup_obj)
        updated = ObjectState(
            object_id="obj_001",
            label="cup",
            x=0.3, y=0.1, z=0.05,
            confidence=0.99,
            state="grasped",
            last_seen=time.time(),
        )
        wm.add_object(updated)
        retrieved = wm.get_object("obj_001")
        assert retrieved.state == "grasped"
        assert retrieved.x == pytest.approx(0.3)


class TestRemoveObject:
    def test_remove_existing(self, wm, cup_obj):
        wm.add_object(cup_obj)
        wm.remove_object("obj_001")
        assert wm.get_object("obj_001") is None

    def test_remove_nonexistent_does_not_raise(self, wm):
        wm.remove_object("does_not_exist")  # Should not raise

    def test_remove_leaves_others_intact(self, wm, cup_obj, bottle_obj):
        wm.add_object(cup_obj)
        wm.add_object(bottle_obj)
        wm.remove_object("obj_001")
        assert wm.get_object("obj_002") is not None


class TestGetObjects:
    def test_get_objects_empty(self, wm):
        assert wm.get_objects() == []

    def test_get_objects_returns_all(self, wm, cup_obj, bottle_obj):
        wm.add_object(cup_obj)
        wm.add_object(bottle_obj)
        objs = wm.get_objects()
        assert len(objs) == 2

    def test_get_objects_by_label_cup(self, wm, cup_obj, bottle_obj):
        wm.add_object(cup_obj)
        wm.add_object(bottle_obj)
        cups = wm.get_objects_by_label("cup")
        assert len(cups) == 1
        assert cups[0].label == "cup"

    def test_get_objects_by_label_missing(self, wm, cup_obj):
        wm.add_object(cup_obj)
        result = wm.get_objects_by_label("phone")
        assert result == []

    def test_get_objects_by_label_multiple(self, wm):
        from vector_os_nano.core.world_model import ObjectState
        for i in range(3):
            obj = ObjectState(
                object_id=f"cup_{i}", label="cup",
                x=0.1 * i, y=0.0, z=0.03,
                last_seen=time.time(),
            )
            wm.add_object(obj)
        cups = wm.get_objects_by_label("cup")
        assert len(cups) == 3


def test_an_empty_label_matches_nothing(wm, cup_obj, bottle_obj):
    """core:worldmodel-empty-label-removes-arbitrary: '' was a substring of every
    label, so a pick with no label removed an arbitrary object."""
    wm.add_object(cup_obj)
    wm.add_object(bottle_obj)
    assert wm.get_objects_by_label("") == [] and wm.get_objects_by_label("  ") == []


class TestRobotState:
    def test_default_robot_state(self, wm):
        robot = wm.get_robot()
        assert robot.gripper_state == "open"
        assert robot.held_object is None
        assert robot.is_moving is False

    def test_update_gripper_state(self, wm):
        wm.update_robot_state(gripper_state="closed")
        assert wm.get_robot().gripper_state == "closed"

    def test_update_held_object(self, wm):
        wm.update_robot_state(held_object="obj_001", gripper_state="holding")
        robot = wm.get_robot()
        assert robot.held_object == "obj_001"
        assert robot.gripper_state == "holding"

    def test_update_is_moving(self, wm):
        wm.update_robot_state(is_moving=True)
        assert wm.get_robot().is_moving is True

    def test_update_joint_positions(self, wm):
        joints = (0.1, -1.2, 0.5, 0.8, 0.3)
        wm.update_robot_state(joint_positions=joints)
        assert wm.get_robot().joint_positions == joints

    def test_update_preserves_unset_fields(self, wm):
        """Updating one field should not reset others to defaults."""
        wm.update_robot_state(gripper_state="closed", held_object="obj_001")
        wm.update_robot_state(is_moving=True)
        robot = wm.get_robot()
        assert robot.gripper_state == "closed"
        assert robot.held_object == "obj_001"
        assert robot.is_moving is True


class TestSerialization:
    def test_to_dict_is_json_serializable(self, wm, cup_obj, bottle_obj):
        wm.add_object(cup_obj)
        wm.add_object(bottle_obj)
        d = wm.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_to_dict_has_objects_key(self, wm, cup_obj):
        wm.add_object(cup_obj)
        d = wm.to_dict()
        assert "objects" in d
        assert len(d["objects"]) == 1

    def test_to_dict_has_robot_key(self, wm):
        d = wm.to_dict()
        assert "robot" in d
        assert "gripper_state" in d["robot"]

    def test_to_dict_roundtrip_object_count(self, wm, cup_obj, bottle_obj):
        wm.add_object(cup_obj)
        wm.add_object(bottle_obj)
        d = wm.to_dict()
        assert len(d["objects"]) == 2
