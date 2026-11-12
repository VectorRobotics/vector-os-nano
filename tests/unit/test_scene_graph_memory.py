# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""SceneGraph as trustworthy spatial memory (WP11): instances, None sentinel, fixed
rooms, atomic world-keyed persistence. Each test names the ledger bug it pins."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from vector_os_nano.core.scene_graph import SceneGraph
from vector_os_nano.vcli.worlds.world_manifest import load_world

_APT = load_world("apartment")


def test_numpy_scalars_round_trip(tmp_path) -> None:
    """core:scenegraph-numpy-persist-loss — np.float64 wrote python tags; load()
    then started empty and the next save overwrote the file."""
    p = tmp_path / "sg.yaml"
    sg = SceneGraph(persist_path=str(p))
    sg.visit("kitchen", np.float64(17.0), np.float32(1.7))
    sg.add_door("kitchen", "hallway", np.float64(13.5), np.float64(3.0))
    sg.merge_object("green bottle", "kitchen", "vp1", confidence=np.float64(0.7),
                    x=np.float64(10.88), y=np.float64(3.0), z=np.float32(0.32))
    sg.save()
    assert "!!python" not in p.read_text()
    yaml.safe_load(p.read_text())  # plain YAML
    back = SceneGraph(persist_path=str(p))
    back.load()
    (o,) = back.find_objects_by_category("green bottle")
    assert (o.x, o.y) == pytest.approx((10.88, 3.0)) and o.z == pytest.approx(0.32, abs=1e-6)
    assert back.get_room("kitchen").center_x == 17.0
    assert back.get_door("kitchen", "hallway") == pytest.approx((13.5, 3.0))


def test_two_same_category_objects_a_metre_apart_stay_two() -> None:
    """core:merge-object-collapses-instances — merge keyed on category+room only."""
    sg = SceneGraph()
    sg.merge_object("chair", "dining_room", "vp1", x=2.0, y=7.0)
    sg.merge_object("chair", "dining_room", "vp1", x=3.0, y=7.0)
    assert len(sg.find_objects_by_category("chair")) == 2
    sg.merge_object("chair", "dining_room", "vp2", x=3.05, y=7.02)  # re-seen: same one
    chairs = sg.find_objects_by_category("chair")
    assert len(chairs) == 2
    near = min(chairs, key=lambda o: abs(o.x - 3.05))
    assert near.n_obs == 2 and (near.x, near.y) == (3.05, 7.02)


def test_unknown_position_is_none_and_moves_as_a_whole() -> None:
    """core:merge-object-zero-sentinel + core:scenegraph-drops-z — x=0/y=0 kept stale
    coordinates and mixed old x with new y; z was never stored."""
    sg = SceneGraph()
    o = sg.merge_object("cup", "kitchen", "vp1")
    assert o.x is None and o.y is None and not o.localized
    o = sg.merge_object("cup", "kitchen", "vp2", x=16.0, y=0.0, z=0.9)  # y=0 is a real y
    assert (o.x, o.y, o.z) == (16.0, 0.0, 0.9)
    o2 = sg.merge_object("cup", "kitchen", "vp3")  # an unlocalized sighting never moves it
    assert o2.object_id == o.object_id and (o2.x, o2.y, o2.z) == (16.0, 0.0, 0.9)
    o3 = sg.merge_object("cup", "kitchen", "vp4", x=16.1, y=0.05)
    assert (o3.x, o3.y, o3.z) == (16.1, 0.05, None) and o3.last_seen >= o.last_seen


def test_layout_rooms_are_fixed_and_rooms_are_boxes() -> None:
    """core:room-center-drift — every visit averaged the robot pose into the centre."""
    sg = SceneGraph()
    assert sg.load_layout(str(_APT.path)) == len(_APT.rooms)
    for _ in range(20):
        sg.visit("kitchen", 14.2, 4.9)  # a corner of the kitchen
    k = sg.get_room("kitchen")
    assert (k.center_x, k.center_y) == _APT.rooms["kitchen"] and k.fixed
    # membership, not the nearest centre: (13.9, 4.0) is hallway floor although the
    # kitchen point (17, 1.7) is not far
    assert sg.nearest_room(13.9, 4.0) == "hallway"
    assert sg.nearest_room(10.88, 3.0) == "hallway"
    o = sg.merge_object("red can", "kitchen", "vp", x=10.9, y=3.22)
    assert o.room_id == "hallway"  # the object's own room, not the observer's guess


def test_door_keys_survive_odd_room_ids_and_load_replaces(tmp_path) -> None:
    """core:scenegraph-persist-keys — 'a|b' keys broke on '|' in room ids; load merged
    into the current state instead of replacing it."""
    p = tmp_path / "sg.yaml"
    sg = SceneGraph(persist_path=str(p))
    sg.add_door("room|a", "room-b", 1.0, 2.0)
    sg.save()
    back = SceneGraph(persist_path=str(p))
    back.merge_object("stale thing", "x", "vp", x=1.0, y=1.0)
    back.load()
    assert back.get_door("room|a", "room-b") == (1.0, 2.0)
    assert back.find_objects_by_category("stale thing") == []


def test_unreadable_file_is_kept_aside_not_overwritten(tmp_path, caplog) -> None:
    p = tmp_path / "sg.yaml"
    p.write_text("objects: !!python/object:numpy.float64 [oops")
    sg = SceneGraph(persist_path=str(p))
    sg.load()
    assert sg.stats()["objects"] == 0
    assert list(tmp_path.glob("sg.yaml.bad-*")), "the bad file must be kept"
    assert "starting empty" in caplog.text


def test_save_is_atomic_and_autosaves_after_a_mutation(tmp_path) -> None:
    """docs-vs-code:scene-graph-persist-fragile — saved only on REPL exit, non-atomic."""
    p = tmp_path / "sg.yaml"
    sg = SceneGraph(persist_path=str(p))
    sg.merge_object("green bottle", "hallway", "vp", x=10.88, y=3.0)
    deadline = time.monotonic() + 3.0
    while not p.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert p.exists(), "a mutation must reach disk without an explicit save()"
    sg.merge_object("blue bottle", "hallway", "vp", x=10.9, y=2.78)  # inside the debounce
    deadline = time.monotonic() + 3.0
    while "blue bottle" not in p.read_text() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert "blue bottle" in p.read_text(), "the last change of a burst is saved too"
    assert [f.name for f in tmp_path.iterdir()] == ["sg.yaml"], "no temp file left behind"


def test_persisted_format_is_unchanged(tmp_path) -> None:
    """golden/scene_graph.yaml was written by the SceneGraph of release/1.0 before the
    SG1 split (fixed and discovered rooms, doors, viewpoints, localized, half-localized
    and unlocalized objects, zh text, attributes); loading it and saving it back must
    give the same bytes, so a file on disk survives an upgrade and a downgrade."""
    golden = Path(__file__).with_name("golden") / "scene_graph.yaml"
    p = tmp_path / "scene_graph.yaml"
    p.write_bytes(golden.read_bytes())
    sg = SceneGraph(persist_path=str(p))
    sg.load()
    assert sg.stats() == {"rooms": 3, "viewpoints": 2, "objects": 4, "visited_rooms": 3}
    sg.save()
    assert p.read_bytes() == golden.read_bytes()


def test_clear_is_in_place_and_keeps_the_world_layout() -> None:
    sg = SceneGraph()
    sg.load_layout(str(_APT.path))
    sg.merge_object("green bottle", "hallway", "vp", x=10.88, y=3.0)
    holder = sg
    sg.clear()
    assert holder is sg and sg.stats()["objects"] == 0
    assert sg.get_room("kitchen").fixed and sg.get_door("kitchen", "hallway")


def test_zh_and_en_names_find_the_same_instance() -> None:
    """docs-vs-code:object-name-match-substring — substring category match; zh names
    never matched an en category."""
    sg = SceneGraph(resolve=_APT.resolve)
    o = sg.merge_object("green bottle", "hallway", "vp", x=10.88, y=3.0)
    sg.merge_object("blue bottle", "hallway", "vp", x=10.9, y=2.78)
    sg.merge_object("candle", "hallway", "vp", x=1.0, y=1.0)
    for q in ("绿色的瓶子", "green bottle", "绿瓶", "pickable_bottle_green"):
        assert [x.object_id for x in sg.find_objects_by_category(q)] == [o.object_id], q
    assert len(sg.find_objects_by_category("bottle")) == 2
    assert sg.find_objects_by_category("can") == []  # never 'candle'
    # an observation named in zh merges into the en-named instance
    assert sg.merge_object("绿色的瓶子", "hallway", "vp2", x=10.89, y=3.01).object_id == o.object_id


def test_update_object_records_a_pick_event() -> None:
    sg = SceneGraph()
    o = sg.merge_object("green bottle", "hallway", "vp", x=10.88, y=3.0, z=0.32)
    held = sg.update_object(o.object_id, state="held", x=10.5, y=3.0, z=0.6)
    assert held.attributes["state"] == "held" and (held.x, held.z) == (10.5, 0.6)
    assert sg.update_object("nope", state="held") is None
