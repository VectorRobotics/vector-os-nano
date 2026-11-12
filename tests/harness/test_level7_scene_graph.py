# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Level 7 — SceneGraph: three-layer spatial scene representation.

Tests the SysNav-inspired hierarchical scene graph at
``vector_os_nano.core.scene_graph``.  Every test is pure Python — no
MuJoCo, no real API calls.

Layers under test
-----------------
    RoomNode  (frozen dataclass)
    ViewpointNode  (frozen dataclass with computed coverage_area)
    ObjectNode  (frozen dataclass with to_dict())
    SceneGraph  (three-layer store + SpatialMemory backward-compat API)

Reference: SysNav (arXiv 2603.06914v1) — hierarchical scene representation
where rooms contain discrete viewpoints and each viewpoint anchors a set of
VLM-detected objects.

Test classes
------------
    TestSceneGraphDataModel    L7-0  node dataclasses and basic graph ops
    TestViewpointLogic         L7-1  viewpoint threshold and coverage maths
    TestObjectMerge            L7-2  object creation and deduplication
    TestBackwardCompat         L7-4  SpatialMemory backward-compatible API
    TestPersistence            L7-5  YAML save/load round-trip
    TestObserveWithViewpoint   L7-6  full viewpoint-aware observation flow
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Ensure repo root is importable
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from vector_os_nano.core.scene_graph import (  # noqa: E402
    ObjectNode,
    RoomNode,
    SceneGraph,
    ViewpointNode,
    _ROOM_AREA_DEFAULT,
    _VIEWPOINT_FOV_DEG,
    _VIEWPOINT_MIN_DISTANCE,
    _VIEWPOINT_RANGE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sg() -> SceneGraph:
    """Return a fresh SceneGraph with no persist path."""
    return SceneGraph()


def _room(room_id: str = "kitchen", **kwargs: Any) -> RoomNode:
    return RoomNode(room_id=room_id, **kwargs)


def _vp(
    viewpoint_id: str = "vp_abc",
    room_id: str = "kitchen",
    x: float = 0.0,
    y: float = 0.0,
    **kwargs: Any,
) -> ViewpointNode:
    return ViewpointNode(
        viewpoint_id=viewpoint_id, room_id=room_id, x=x, y=y, **kwargs
    )


def _obj(
    object_id: str = "obj_001",
    category: str = "chair",
    room_id: str = "kitchen",
    **kwargs: Any,
) -> ObjectNode:
    return ObjectNode(object_id=object_id, category=category, room_id=room_id, **kwargs)


# ===========================================================================
# L7-0: Node dataclasses and basic graph operations
# ===========================================================================


class TestSceneGraphDataModel:
    """L7-0: Node dataclasses and basic graph operations."""

    def test_room_node_frozen(self) -> None:
        """RoomNode is immutable — attribute assignment raises FrozenInstanceError."""
        room = _room()
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            room.room_id = "bedroom"  # type: ignore[misc]

    def test_viewpoint_node_frozen(self) -> None:
        """ViewpointNode is immutable."""
        vp = _vp()
        with pytest.raises(Exception):
            vp.x = 99.0  # type: ignore[misc]

    def test_object_node_frozen(self) -> None:
        """ObjectNode is immutable."""
        obj = _obj()
        with pytest.raises(Exception):
            obj.category = "sofa"  # type: ignore[misc]

    def test_viewpoint_node_coverage_area(self) -> None:
        """ViewpointNode.coverage_area uses the FOV cone formula."""
        vp = _vp()
        half_angle = math.radians(_VIEWPOINT_FOV_DEG / 2)
        expected = 0.5 * _VIEWPOINT_RANGE**2 * math.sin(2 * half_angle)
        assert abs(vp.coverage_area - expected) < 1e-9

    def test_viewpoint_coverage_area_positive(self) -> None:
        """coverage_area is always a positive number."""
        vp = _vp()
        assert vp.coverage_area > 0.0

    def test_object_node_to_dict(self) -> None:
        """ObjectNode serializes to a dict with the correct keys and values."""
        obj = ObjectNode(
            object_id="obj_test",
            category="sofa",
            description="a red sofa",
            confidence=0.9,
            room_id="living_room",
            x=1.0,
            y=2.0,
            z=0.0,
            viewpoint_ids=("vp_1", "vp_2"),
        )
        d = obj.to_dict()
        assert d["object_id"] == "obj_test"
        assert d["category"] == "sofa"
        assert d["confidence"] == 0.9
        assert d["room_id"] == "living_room"
        assert d["x"] == 1.0
        assert d["viewpoint_ids"] == ["vp_1", "vp_2"]
        assert isinstance(d["attributes"], dict)

    def test_room_node_to_dict_keys(self) -> None:
        """RoomNode.to_dict includes all expected keys."""
        room = _room(center_x=3.0, center_y=4.0, visit_count=2)
        d = room.to_dict()
        for key in (
            "room_id", "center_x", "center_y", "area",
            "visit_count", "last_visited", "representative_description",
            "connected_rooms",
        ):
            assert key in d, f"missing key: {key}"

    def test_add_room_and_get(self) -> None:
        """Add a room and retrieve it by ID."""
        sg = _sg()
        room = _room("living_room", center_x=1.5, center_y=2.5)
        sg.add_room(room)
        result = sg.get_room("living_room")
        assert result is not None
        assert result.room_id == "living_room"
        assert result.center_x == 1.5

    def test_get_room_missing_returns_none(self) -> None:
        """get_room on an unknown ID returns None."""
        sg = _sg()
        assert sg.get_room("nowhere") is None

    def test_get_all_rooms(self) -> None:
        """get_all_rooms returns every room that was added."""
        sg = _sg()
        sg.add_room(_room("kitchen"))
        sg.add_room(_room("bedroom"))
        sg.add_room(_room("bathroom"))
        ids = {r.room_id for r in sg.get_all_rooms()}
        assert ids == {"kitchen", "bedroom", "bathroom"}

    def test_add_viewpoint_updates_room_description(self) -> None:
        """Adding a viewpoint with scene_summary updates room's representative_description."""
        sg = _sg()
        sg.add_room(_room("kitchen"))
        vp = _vp(
            viewpoint_id="vp_k1",
            room_id="kitchen",
            scene_summary="A bright kitchen with stainless appliances.",
        )
        sg.add_viewpoint(vp)
        room = sg.get_room("kitchen")
        assert room is not None
        assert room.representative_description == "A bright kitchen with stainless appliances."

    def test_add_viewpoint_no_summary_keeps_existing_description(self) -> None:
        """Adding a viewpoint with empty summary does not overwrite room description."""
        sg = _sg()
        sg.add_room(RoomNode(room_id="office", representative_description="existing desc"))
        vp = _vp(viewpoint_id="vp_o1", room_id="office", scene_summary="")
        sg.add_viewpoint(vp)
        room = sg.get_room("office")
        assert room is not None
        assert room.representative_description == "existing desc"


# ===========================================================================
# L7-1: Viewpoint creation threshold and coverage
# ===========================================================================


class TestViewpointLogic:
    """L7-1: Viewpoint creation threshold and coverage."""

    def test_should_add_first_viewpoint(self) -> None:
        """First viewpoint in a room always passes the distance check."""
        sg = _sg()
        sg.add_room(_room("kitchen"))
        assert sg.should_add_viewpoint("kitchen", 0.0, 0.0) is True

    def test_should_not_add_close_viewpoint(self) -> None:
        """Viewpoint within _VIEWPOINT_MIN_DISTANCE of an existing one is rejected."""
        sg = _sg()
        sg.add_room(_room("kitchen"))
        sg.add_viewpoint(_vp(viewpoint_id="vp_1", room_id="kitchen", x=0.0, y=0.0))
        # 0.5 m away — less than 1.5 m threshold
        assert sg.should_add_viewpoint("kitchen", 0.5, 0.0) is False

    def test_should_add_far_viewpoint(self) -> None:
        """Viewpoint > _VIEWPOINT_MIN_DISTANCE from all existing ones is accepted."""
        sg = _sg()
        sg.add_room(_room("kitchen"))
        sg.add_viewpoint(_vp(viewpoint_id="vp_1", room_id="kitchen", x=0.0, y=0.0))
        # 2.0 m away — above 1.5 m threshold
        assert sg.should_add_viewpoint("kitchen", 2.0, 0.0) is True

    def test_should_add_viewpoint_different_room_not_constrained(self) -> None:
        """Viewpoints in different rooms do not block each other."""
        sg = _sg()
        sg.add_room(_room("kitchen"))
        sg.add_room(_room("bedroom"))
        sg.add_viewpoint(_vp(viewpoint_id="vp_k", room_id="kitchen", x=0.0, y=0.0))
        # bedroom has no viewpoints at (0.0, 0.0) — should be accepted
        assert sg.should_add_viewpoint("bedroom", 0.0, 0.0) is True

    def test_room_coverage_zero_without_viewpoints(self) -> None:
        """Coverage is 0.0 when no viewpoints have been recorded."""
        sg = _sg()
        sg.add_room(_room("kitchen"))
        assert sg.get_room_coverage("kitchen") == 0.0

    def test_room_coverage_increases_with_viewpoints(self) -> None:
        """Coverage grows as viewpoints are added."""
        sg = _sg()
        sg.add_room(RoomNode(room_id="kitchen", area=_ROOM_AREA_DEFAULT))
        sg.add_viewpoint(_vp(viewpoint_id="vp_1", room_id="kitchen", x=0.0, y=0.0))
        cov1 = sg.get_room_coverage("kitchen")
        # Add second viewpoint far enough away
        sg.add_viewpoint(_vp(viewpoint_id="vp_2", room_id="kitchen", x=3.0, y=0.0))
        cov2 = sg.get_room_coverage("kitchen")
        assert cov2 > cov1

    def test_room_coverage_capped_at_one(self) -> None:
        """Coverage never exceeds 1.0 regardless of how many viewpoints are added."""
        sg = _sg()
        # Tiny room — viewpoints will exceed area immediately
        sg.add_room(RoomNode(room_id="closet", area=0.1))
        for i in range(10):
            sg.add_viewpoint(
                _vp(viewpoint_id=f"vp_{i}", room_id="closet", x=float(i * 5), y=0.0)
            )
        assert sg.get_room_coverage("closet") <= 1.0

    def test_room_coverage_unknown_room_uses_default_area(self) -> None:
        """Coverage for a viewpoint in an unregistered room uses default area."""
        sg = _sg()
        # Add a viewpoint without registering the room
        sg._viewpoints["vp_x"] = ViewpointNode(
            viewpoint_id="vp_x", room_id="ghost_room", x=0.0, y=0.0
        )
        cov = sg.get_room_coverage("ghost_room")
        assert 0.0 < cov <= 1.0


# ===========================================================================
# L7-2: Object creation and deduplication
# ===========================================================================


class TestObjectMerge:
    """L7-2: Object creation and deduplication."""

    def test_merge_creates_new_object(self) -> None:
        """merge_object creates a new ObjectNode when none exists."""
        sg = _sg()
        sg.add_room(_room("kitchen"))
        obj = sg.merge_object(
            category="fridge",
            room_id="kitchen",
            viewpoint_id="vp_1",
            description="large silver fridge",
            confidence=0.9,
        )
        assert obj.category == "fridge"
        assert obj.room_id == "kitchen"
        assert "vp_1" in obj.viewpoint_ids
        assert len(sg.find_objects_in_room("kitchen")) == 1

    def test_merge_same_category_same_room_deduplicates(self) -> None:
        """Same category in same room merges into one object, not two."""
        sg = _sg()
        sg.add_room(_room("kitchen"))
        sg.merge_object(category="chair", room_id="kitchen", viewpoint_id="vp_1")
        sg.merge_object(category="chair", room_id="kitchen", viewpoint_id="vp_2")
        objects = sg.find_objects_in_room("kitchen")
        assert len(objects) == 1

    def test_merge_same_category_different_room_no_merge(self) -> None:
        """Same category in different rooms creates separate objects."""
        sg = _sg()
        sg.add_room(_room("kitchen"))
        sg.add_room(_room("bedroom"))
        sg.merge_object(category="chair", room_id="kitchen", viewpoint_id="vp_1")
        sg.merge_object(category="chair", room_id="bedroom", viewpoint_id="vp_2")
        assert len(sg.find_objects_in_room("kitchen")) == 1
        assert len(sg.find_objects_in_room("bedroom")) == 1

    def test_merge_updates_viewpoint_ids(self) -> None:
        """Merged object accumulates viewpoint_ids from all observations."""
        sg = _sg()
        sg.add_room(_room("kitchen"))
        sg.merge_object(category="table", room_id="kitchen", viewpoint_id="vp_A")
        sg.merge_object(category="table", room_id="kitchen", viewpoint_id="vp_B")
        sg.merge_object(category="table", room_id="kitchen", viewpoint_id="vp_C")
        objs = sg.find_objects_in_room("kitchen")
        assert len(objs) == 1
        vp_ids = set(objs[0].viewpoint_ids)
        assert {"vp_A", "vp_B", "vp_C"} == vp_ids

    def test_merge_keeps_higher_confidence(self) -> None:
        """merge_object retains the higher confidence value from either observation."""
        sg = _sg()
        sg.add_room(_room("living_room"))
        sg.merge_object(
            category="sofa", room_id="living_room",
            viewpoint_id="vp_1", confidence=0.6,
        )
        sg.merge_object(
            category="sofa", room_id="living_room",
            viewpoint_id="vp_2", confidence=0.95,
        )
        objs = sg.find_objects_in_room("living_room")
        assert objs[0].confidence == pytest.approx(0.95)

    def test_merge_updates_description_when_confidence_higher(self) -> None:
        """merge_object replaces description only when incoming confidence is higher."""
        sg = _sg()
        sg.add_room(_room("office"))
        sg.merge_object(
            category="desk", room_id="office",
            viewpoint_id="vp_1", description="brown desk", confidence=0.5,
        )
        sg.merge_object(
            category="desk", room_id="office",
            viewpoint_id="vp_2", description="white standing desk", confidence=0.85,
        )
        objs = sg.find_objects_in_room("office")
        assert objs[0].description == "white standing desk"

    def test_find_objects_by_category_exact(self) -> None:
        """find_objects_by_category returns objects with matching category."""
        sg = _sg()
        sg.add_room(_room("bedroom"))
        sg.merge_object(category="bed", room_id="bedroom", viewpoint_id="vp_1")
        sg.merge_object(category="wardrobe", room_id="bedroom", viewpoint_id="vp_1")
        results = sg.find_objects_by_category("bed")
        assert len(results) == 1
        assert results[0].category == "bed"

    def test_find_objects_by_category_substring(self) -> None:
        """find_objects_by_category('chair') matches 'office_chair'."""
        sg = _sg()
        sg.add_room(_room("office"))
        sg.merge_object(category="office_chair", room_id="office", viewpoint_id="vp_1")
        sg.merge_object(category="desk", room_id="office", viewpoint_id="vp_1")
        results = sg.find_objects_by_category("chair")
        assert len(results) == 1
        assert results[0].category == "office_chair"

    def test_find_objects_by_category_case_insensitive(self) -> None:
        """find_objects_by_category is case-insensitive."""
        sg = _sg()
        sg.add_room(_room("kitchen"))
        sg.merge_object(category="Fridge", room_id="kitchen", viewpoint_id="vp_1")
        results = sg.find_objects_by_category("fridge")
        assert len(results) == 1


# ===========================================================================
# L7-4: SpatialMemory backward compatibility
# ===========================================================================


class TestBackwardCompat:
    """L7-4: SpatialMemory backward compatibility."""

    def test_visit_creates_room(self) -> None:
        """visit() creates a RoomNode when the room does not exist."""
        sg = _sg()
        sg.visit("hallway", 5.0, 3.0)
        room = sg.get_room("hallway")
        assert room is not None
        assert room.center_x == 5.0
        assert room.center_y == 3.0

    def test_visit_increments_count(self) -> None:
        """Calling visit() multiple times increments visit_count."""
        sg = _sg()
        sg.visit("kitchen", 0.0, 0.0)
        sg.visit("kitchen", 0.0, 0.0)
        sg.visit("kitchen", 0.0, 0.0)
        room = sg.get_room("kitchen")
        assert room is not None
        assert room.visit_count == 3

    def test_observe_creates_objects(self) -> None:
        """observe() creates ObjectNodes for each item in the objects list."""
        sg = _sg()
        sg.visit("living_room", 0.0, 0.0)
        sg.observe("living_room", ["tv", "sofa", "coffee_table"], "A living room with TV.")
        objs = sg.find_objects_in_room("living_room")
        categories = {o.category for o in objs}
        assert categories == {"tv", "sofa", "coffee_table"}

    def test_observe_without_prior_visit_still_works(self) -> None:
        """observe() on an unvisited room auto-creates the room."""
        sg = _sg()
        sg.observe("bathroom", ["sink", "mirror"])
        assert sg.get_room("bathroom") is not None
        assert len(sg.find_objects_in_room("bathroom")) == 2

    def test_get_visited_rooms(self) -> None:
        """get_visited_rooms returns rooms with visit_count > 0."""
        sg = _sg()
        sg.visit("kitchen", 0.0, 0.0)
        sg.visit("bedroom", 1.0, 1.0)
        # Add a room via add_room without visiting
        sg.add_room(RoomNode(room_id="garage", visit_count=0))
        visited = sg.get_visited_rooms()
        assert "kitchen" in visited
        assert "bedroom" in visited
        assert "garage" not in visited

    def test_get_room_summary_format(self) -> None:
        """get_room_summary returns a non-empty string including visited room info."""
        sg = _sg()
        sg.visit("kitchen", 0.0, 0.0)
        sg.observe("kitchen", ["fridge", "counter"], "Bright kitchen.")
        summary = sg.get_room_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "kitchen" in summary

    def test_get_room_summary_empty(self) -> None:
        """get_room_summary on an empty graph returns a sensible fallback string."""
        sg = _sg()
        summary = sg.get_room_summary()
        assert "No rooms" in summary or len(summary) > 0

# ===========================================================================
# L7-5: YAML save/load round-trip
# ===========================================================================


class TestPersistence:
    """L7-5: YAML save/load round-trip."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Save graph, create new instance, load, verify all data preserved."""
        save_file = str(tmp_path / "scene.yaml")
        sg = SceneGraph(persist_path=save_file)

        # Populate all three layers
        sg.add_room(RoomNode(
            room_id="kitchen",
            center_x=1.0, center_y=2.0,
            area=20.0, visit_count=3,
            representative_description="Modern kitchen",
        ))
        sg.add_viewpoint(ViewpointNode(
            viewpoint_id="vp_k1", room_id="kitchen",
            x=1.0, y=2.0, heading=0.5,
            scene_summary="Stainless fridge visible",
        ))
        sg.merge_object(
            category="fridge", room_id="kitchen",
            viewpoint_id="vp_k1", description="large fridge",
            confidence=0.92, x=1.2, y=2.3,
        )
        sg.save()

        # Load into a fresh instance
        sg2 = SceneGraph(persist_path=save_file)
        sg2.load()

        # Rooms — add_viewpoint with a scene_summary updates representative_description
        room = sg2.get_room("kitchen")
        assert room is not None
        assert room.visit_count == 3
        assert room.representative_description == "Stainless fridge visible"
        assert room.area == pytest.approx(20.0)

        # Viewpoints
        vps = sg2.get_viewpoints_in_room("kitchen")
        assert len(vps) == 1
        assert vps[0].scene_summary == "Stainless fridge visible"
        assert vps[0].heading == pytest.approx(0.5)

        # Objects
        objs = sg2.find_objects_by_category("fridge")
        assert len(objs) == 1
        assert objs[0].description == "large fridge"
        assert objs[0].confidence == pytest.approx(0.92)

    def test_load_missing_file_noop(self, tmp_path: Path) -> None:
        """Loading from a non-existent file leaves the graph empty."""
        sg = SceneGraph(persist_path=str(tmp_path / "does_not_exist.yaml"))
        sg.load()
        assert sg.get_all_rooms() == []

    def test_save_no_path_noop(self) -> None:
        """save() with no persist_path is a no-op (no file created, no exception)."""
        sg = SceneGraph()  # no persist_path
        sg.add_room(_room("kitchen"))
        sg.save()  # should not raise

    def test_round_trip_preserves_object_viewpoint_ids(self, tmp_path: Path) -> None:
        """viewpoint_ids tuple survives YAML serialization as a list, then tuple."""
        save_file = str(tmp_path / "vp_ids.yaml")
        sg = SceneGraph(persist_path=save_file)
        sg.add_room(_room("office"))
        sg.merge_object(category="desk", room_id="office", viewpoint_id="vp_1")
        sg.merge_object(category="desk", room_id="office", viewpoint_id="vp_2")
        sg.save()

        sg2 = SceneGraph(persist_path=save_file)
        sg2.load()
        objs = sg2.find_objects_by_category("desk")
        assert len(objs) == 1
        vp_ids = set(objs[0].viewpoint_ids)
        assert {"vp_1", "vp_2"} == vp_ids


# ===========================================================================
# L7-6: Full viewpoint-aware observation flow
# ===========================================================================


class TestObserveWithViewpoint:
    """L7-6: Full viewpoint-aware observation flow."""

    def test_observe_with_viewpoint_creates_all(self) -> None:
        """observe_with_viewpoint creates viewpoint + objects; looking is not
        arriving, so it never visits the room (skills-nav:look-vlm-room-phantoms)."""
        sg = _sg()
        sg.add_room(_room("bedroom"))
        vp = sg.observe_with_viewpoint(
            room="bedroom",
            x=2.0, y=3.0, heading=1.57,
            objects=["bed", "nightstand"],
            description="Cozy bedroom.",
        )
        assert vp is not None
        assert vp.room_id == "bedroom"
        room = sg.get_room("bedroom")
        assert room is not None and room.visit_count == 0
        objs = sg.find_objects_in_room("bedroom")
        cats = {o.category for o in objs}
        assert cats == {"bed", "nightstand"}

    def test_observe_with_viewpoint_returns_viewpoint_node(self) -> None:
        """Return value is a ViewpointNode with correct position and heading."""
        sg = _sg()
        sg.add_room(_room("hallway"))
        vp = sg.observe_with_viewpoint(
            room="hallway", x=4.0, y=0.0, heading=3.14, objects=[]
        )
        assert vp is not None
        assert vp.x == pytest.approx(4.0)
        assert vp.y == pytest.approx(0.0)
        assert vp.heading == pytest.approx(3.14)

    def test_observe_with_viewpoint_skips_close(self) -> None:
        """Too close to an existing viewpoint → no new VP returned, but objects added."""
        sg = _sg()
        sg.add_room(_room("kitchen"))
        # First observation at (0, 0)
        sg.observe_with_viewpoint(
            room="kitchen", x=0.0, y=0.0, heading=0.0,
            objects=["fridge"],
        )
        vp_count_before = len(sg.get_viewpoints_in_room("kitchen"))

        # Second observation 0.5m away — should be skipped
        result = sg.observe_with_viewpoint(
            room="kitchen", x=0.5, y=0.0, heading=0.0,
            objects=["counter"],
        )
        assert result is None
        assert len(sg.get_viewpoints_in_room("kitchen")) == vp_count_before

        # Object from the skipped observation is still merged in
        objs = sg.find_objects_in_room("kitchen")
        cats = {o.category for o in objs}
        assert "counter" in cats

    def test_observe_with_viewpoint_far_second_accepted(self) -> None:
        """Second viewpoint far enough from the first is accepted."""
        sg = _sg()
        sg.add_room(_room("living_room"))
        sg.observe_with_viewpoint(
            room="living_room", x=0.0, y=0.0, heading=0.0, objects=[]
        )
        vp2 = sg.observe_with_viewpoint(
            room="living_room", x=3.0, y=0.0, heading=0.0, objects=[]
        )
        assert vp2 is not None
        assert len(sg.get_viewpoints_in_room("living_room")) == 2

    def test_stats_reflects_all_layers(self) -> None:
        """stats() accurately counts rooms, viewpoints, objects, and visited rooms."""
        sg = _sg()
        sg.add_room(_room("kitchen"))
        sg.add_room(_room("bedroom"))
        sg.observe_with_viewpoint(
            room="kitchen", x=0.0, y=0.0, heading=0.0,
            objects=["fridge", "stove"],
        )
        sg.observe_with_viewpoint(
            room="bedroom", x=5.0, y=0.0, heading=0.0,
            objects=["bed"],
        )
        s = sg.stats()
        assert s["rooms"] == 2
        assert s["viewpoints"] == 2
        assert s["objects"] == 3   # fridge, stove, bed
        assert s["visited_rooms"] == 0  # observing a room is not visiting it

    def test_observe_with_viewpoint_updates_room_description(self) -> None:
        """description passed to observe_with_viewpoint becomes room description."""
        sg = _sg()
        sg.add_room(_room("study"))
        sg.observe_with_viewpoint(
            room="study", x=0.0, y=0.0, heading=0.0,
            objects=[], description="Quiet study with bookshelves.",
        )
        room = sg.get_room("study")
        assert room is not None
        assert room.representative_description == "Quiet study with bookshelves."
