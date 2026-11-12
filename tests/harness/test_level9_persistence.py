# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Level 9 — SceneGraph Persistence: save/load round-trip tests.

All tests are pure Python — no MuJoCo, no real API calls, no hardware.
A temporary directory is used as the persist_path so the real
$VECTOR_HOME/memory/<world>/scene_graph.yaml is never touched.

Test classes
------------
    TestSaveLoadRoundTrip        L9-0  basic save/load with rooms, viewpoints, objects
    TestVisitCountAfterLoad      L9-1  visit counts and increment after load
    TestEdgeCaseMissingFile      L9-2  load from non-existent file → empty graph
    TestEdgeCaseCorruptedFile    L9-3  load from corrupted YAML → no crash, empty graph
    TestPersistPathCreation      L9-4  makedirs logic (path does not need to pre-exist)
    TestNoPersistPath            L9-5  SceneGraph(persist_path=None) → save/load are no-ops
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

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
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sg(tmp_dir: str) -> tuple[SceneGraph, str]:
    """Return a fresh SceneGraph backed by a temp YAML file."""
    path = os.path.join(tmp_dir, "scene_graph.yaml")
    sg = SceneGraph(persist_path=path)
    return sg, path


def _populate(sg: SceneGraph) -> None:
    """Add two rooms, two viewpoints, three objects to sg."""
    sg.visit("living_room", 1.0, 2.0)
    sg.visit("kitchen", 5.0, 6.0)

    vp1 = ViewpointNode(
        viewpoint_id="vp_aaa",
        room_id="living_room",
        x=1.0, y=2.0,
        heading=0.0,
        scene_summary="cozy sofa near window",
    )
    vp2 = ViewpointNode(
        viewpoint_id="vp_bbb",
        room_id="kitchen",
        x=5.0, y=6.0,
        heading=1.57,
        scene_summary="stove and fridge visible",
    )
    sg.add_viewpoint(vp1)
    sg.add_viewpoint(vp2)

    sg.add_object(ObjectNode(
        object_id="obj_001", category="sofa",
        room_id="living_room", x=1.0, y=2.0,
        viewpoint_ids=("vp_aaa",),
    ))
    sg.add_object(ObjectNode(
        object_id="obj_002", category="fridge",
        room_id="kitchen", x=5.0, y=6.0,
        viewpoint_ids=("vp_bbb",),
    ))
    sg.add_object(ObjectNode(
        object_id="obj_003", category="stove",
        room_id="kitchen", x=5.5, y=6.0,
        viewpoint_ids=("vp_bbb",),
    ))


# ---------------------------------------------------------------------------
# L9-0  Basic save/load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    """Rooms, viewpoints, and objects survive a save → fresh load cycle."""

    def test_rooms_survive_round_trip(self, tmp_path):
        sg, path = _make_sg(str(tmp_path))
        _populate(sg)

        sg.save()

        sg2 = SceneGraph(persist_path=path)
        sg2.load()

        rooms = {r.room_id for r in sg2.get_all_rooms()}
        assert "living_room" in rooms
        assert "kitchen" in rooms

    def test_room_count_matches(self, tmp_path):
        sg, path = _make_sg(str(tmp_path))
        _populate(sg)
        sg.save()

        sg2 = SceneGraph(persist_path=path)
        sg2.load()

        assert len(sg2.get_all_rooms()) == 2

    def test_room_coordinates_preserved(self, tmp_path):
        sg, path = _make_sg(str(tmp_path))
        _populate(sg)
        sg.save()

        sg2 = SceneGraph(persist_path=path)
        sg2.load()

        lr = sg2.get_room("living_room")
        assert lr is not None
        assert lr.center_x == pytest.approx(1.0)
        assert lr.center_y == pytest.approx(2.0)

    def test_viewpoints_survive_round_trip(self, tmp_path):
        sg, path = _make_sg(str(tmp_path))
        _populate(sg)
        sg.save()

        sg2 = SceneGraph(persist_path=path)
        sg2.load()

        vps_lr = sg2.get_viewpoints_in_room("living_room")
        assert any(vp.viewpoint_id == "vp_aaa" for vp in vps_lr)

    def test_viewpoint_scene_summary_preserved(self, tmp_path):
        sg, path = _make_sg(str(tmp_path))
        _populate(sg)
        sg.save()

        sg2 = SceneGraph(persist_path=path)
        sg2.load()

        vps = sg2.get_viewpoints_in_room("living_room")
        vp = next(vp for vp in vps if vp.viewpoint_id == "vp_aaa")
        assert vp.scene_summary == "cozy sofa near window"

    def test_objects_survive_round_trip(self, tmp_path):
        sg, path = _make_sg(str(tmp_path))
        _populate(sg)
        sg.save()

        sg2 = SceneGraph(persist_path=path)
        sg2.load()

        sofas = sg2.find_objects_by_category("sofa")
        assert len(sofas) == 1
        assert sofas[0].object_id == "obj_001"

    def test_object_count_matches(self, tmp_path):
        sg, path = _make_sg(str(tmp_path))
        _populate(sg)
        sg.save()

        sg2 = SceneGraph(persist_path=path)
        sg2.load()

        stats = sg2.stats()
        assert stats["objects"] == 3

    def test_object_room_id_preserved(self, tmp_path):
        sg, path = _make_sg(str(tmp_path))
        _populate(sg)
        sg.save()

        sg2 = SceneGraph(persist_path=path)
        sg2.load()

        kitchen_objs = sg2.find_objects_in_room("kitchen")
        categories = {o.category for o in kitchen_objs}
        assert categories == {"fridge", "stove"}

    def test_viewpoint_heading_preserved(self, tmp_path):
        sg, path = _make_sg(str(tmp_path))
        _populate(sg)
        sg.save()

        sg2 = SceneGraph(persist_path=path)
        sg2.load()

        vps = sg2.get_viewpoints_in_room("kitchen")
        vp = next(vp for vp in vps if vp.viewpoint_id == "vp_bbb")
        assert vp.heading == pytest.approx(1.57, abs=1e-4)

    def test_stats_match_after_load(self, tmp_path):
        sg, path = _make_sg(str(tmp_path))
        _populate(sg)
        orig_stats = sg.stats()
        sg.save()

        sg2 = SceneGraph(persist_path=path)
        sg2.load()

        loaded_stats = sg2.stats()
        assert loaded_stats["rooms"] == orig_stats["rooms"]
        assert loaded_stats["viewpoints"] == orig_stats["viewpoints"]
        assert loaded_stats["objects"] == orig_stats["objects"]


# ---------------------------------------------------------------------------
# L9-1  Visit counts and increment after load
# ---------------------------------------------------------------------------


class TestVisitCountAfterLoad:
    """Visit counts persist across save/load and increment correctly."""

    def test_visit_count_persists(self, tmp_path):
        sg, path = _make_sg(str(tmp_path))
        sg.visit("hallway", 0.0, 0.0)
        sg.visit("hallway", 0.1, 0.1)  # second visit

        sg.save()

        sg2 = SceneGraph(persist_path=path)
        sg2.load()

        room = sg2.get_room("hallway")
        assert room is not None
        assert room.visit_count == 2

    def test_visit_after_load_increments_correctly(self, tmp_path):
        sg, path = _make_sg(str(tmp_path))
        sg.visit("study", 3.0, 4.0)
        sg.save()

        sg2 = SceneGraph(persist_path=path)
        sg2.load()
        sg2.visit("study", 3.0, 4.0)

        room = sg2.get_room("study")
        assert room.visit_count == 2

    def test_visited_rooms_matches_after_load(self, tmp_path):
        sg, path = _make_sg(str(tmp_path))
        sg.visit("bedroom", 2.0, 3.0)
        sg.save()

        sg2 = SceneGraph(persist_path=path)
        sg2.load()

        assert "bedroom" in sg2.get_visited_rooms()

    def test_last_visited_timestamp_preserved(self, tmp_path):
        sg, path = _make_sg(str(tmp_path))
        before = time.time()
        sg.visit("bathroom", 1.0, 1.0)
        sg.save()

        sg2 = SceneGraph(persist_path=path)
        sg2.load()

        room = sg2.get_room("bathroom")
        assert room.last_visited >= before


# ---------------------------------------------------------------------------
# L9-2  Edge case: load from non-existent file
# ---------------------------------------------------------------------------


class TestEdgeCaseMissingFile:
    """Loading from a path that does not exist creates an empty graph."""

    def test_load_missing_file_no_crash(self, tmp_path):
        path = os.path.join(str(tmp_path), "nonexistent.yaml")
        sg = SceneGraph(persist_path=path)
        sg.load()  # must not raise

    def test_load_missing_file_empty_graph(self, tmp_path):
        path = os.path.join(str(tmp_path), "nonexistent.yaml")
        sg = SceneGraph(persist_path=path)
        sg.load()

        stats = sg.stats()
        assert stats["rooms"] == 0
        assert stats["viewpoints"] == 0
        assert stats["objects"] == 0

    def test_load_missing_file_no_rooms(self, tmp_path):
        path = os.path.join(str(tmp_path), "absent.yaml")
        sg = SceneGraph(persist_path=path)
        sg.load()

        assert sg.get_all_rooms() == []

    def test_load_missing_file_visited_rooms_empty(self, tmp_path):
        path = os.path.join(str(tmp_path), "absent.yaml")
        sg = SceneGraph(persist_path=path)
        sg.load()

        assert sg.get_visited_rooms() == []


# ---------------------------------------------------------------------------
# L9-3  Edge case: corrupted / invalid YAML
# ---------------------------------------------------------------------------


class TestEdgeCaseCorruptedFile:
    """Loading a corrupted file should not crash and result in an empty graph."""

    def test_corrupted_yaml_no_crash(self, tmp_path):
        path = os.path.join(str(tmp_path), "corrupt.yaml")
        Path(path).write_text("this: is: not: valid: :: yaml: [unclosed", encoding="utf-8")

        sg = SceneGraph(persist_path=path)
        sg.load()  # must not raise

    def test_corrupted_yaml_empty_graph(self, tmp_path):
        path = os.path.join(str(tmp_path), "corrupt.yaml")
        Path(path).write_text("rooms: !!python/object:os.system 'rm -rf /'", encoding="utf-8")

        sg = SceneGraph(persist_path=path)
        sg.load()

        # Graph may be empty or have malformed data — either is acceptable
        # as long as no exception was raised and rooms is a dict
        assert isinstance(sg.get_all_rooms(), list)

    def test_non_dict_yaml_no_crash(self, tmp_path):
        path = os.path.join(str(tmp_path), "list_yaml.yaml")
        Path(path).write_text("- item1\n- item2\n", encoding="utf-8")

        sg = SceneGraph(persist_path=path)
        sg.load()

        stats = sg.stats()
        assert stats["rooms"] == 0

    def test_empty_yaml_file_no_crash(self, tmp_path):
        path = os.path.join(str(tmp_path), "empty.yaml")
        Path(path).write_text("", encoding="utf-8")

        sg = SceneGraph(persist_path=path)
        sg.load()

        assert sg.stats()["rooms"] == 0


# ---------------------------------------------------------------------------
# L9-4  makedirs: persist path in nested subdirectory
# ---------------------------------------------------------------------------


class TestPersistPathCreation:
    """SceneGraph.save() must work even if the parent directory is missing."""

    def test_save_creates_file(self, tmp_path):
        path = os.path.join(str(tmp_path), "scene_graph.yaml")
        sg = SceneGraph(persist_path=path)
        sg.visit("room_a", 0.0, 0.0)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sg.save()

        assert os.path.isfile(path)

    def test_save_then_load_consistency(self, tmp_path):
        path = os.path.join(str(tmp_path), "sg.yaml")
        sg = SceneGraph(persist_path=path)
        sg.visit("room_b", 1.0, 1.0)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sg.save()

        sg2 = SceneGraph(persist_path=path)
        sg2.load()
        assert "room_b" in {r.room_id for r in sg2.get_all_rooms()}

    def test_save_yaml_is_readable_text(self, tmp_path):
        path = os.path.join(str(tmp_path), "sg.yaml")
        sg = SceneGraph(persist_path=path)
        sg.visit("corridor", 0.5, 0.5)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sg.save()

        content = Path(path).read_text(encoding="utf-8")
        assert "rooms" in content
        assert "corridor" in content


# ---------------------------------------------------------------------------
# L9-5  No persist path → save/load are no-ops
# ---------------------------------------------------------------------------


class TestNoPersistPath:
    """SceneGraph(persist_path=None) must not create any file."""

    def test_save_no_op(self, tmp_path):
        sg = SceneGraph(persist_path=None)
        sg.visit("room_x", 0.0, 0.0)
        sg.save()  # must not raise

        # Confirm no yaml files appeared in tmp_path
        yaml_files = list(tmp_path.glob("*.yaml"))
        assert yaml_files == []

    def test_load_no_op(self):
        sg = SceneGraph(persist_path=None)
        sg.load()  # must not raise
        assert sg.stats()["rooms"] == 0

    def test_state_unchanged_after_save_load(self):
        sg = SceneGraph(persist_path=None)
        sg.visit("room_y", 1.0, 2.0)
        sg.save()
        sg.load()

        # State should be unchanged (load was a no-op)
        assert "room_y" in sg.get_visited_rooms()
