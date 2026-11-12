# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""World manifests (config/worlds/*.yaml): name resolution, and the data checked
against the MJCF it describes (floor boxes, surfaces, pickables, grasp welds)."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from vector_os_nano.vcli.worlds.world_manifest import (
    WorldManifestError,
    load_world,
    parse_world,
)

_ROOM_XML = Path(__file__).resolve().parents[3] / "vector_os_nano" / "hardware" / "sim" / "go2_room.xml"


@pytest.fixture(scope="module")
def apt():
    return load_world("apartment")


@pytest.mark.parametrize("text, body", [
    ("pickable_bottle_green", "pickable_bottle_green"),
    ("green bottle", "pickable_bottle_green"),
    ("the green bottle please", "pickable_bottle_green"),
    ("绿色的瓶子", "pickable_bottle_green"),
    ("把绿色瓶子拿过来", "pickable_bottle_green"),
    ("绿瓶", "pickable_bottle_green"),          # colour + category, no alias
    ("红色的", "pickable_can_red"),              # the only red object
    ("bring me the red can", "pickable_can_red"),
    ("黄色的杯子", "pickable_cup_yellow"),
    ("purple box", "pickable_box_purple"),
    ("瓶子", None),                              # two bottles: ambiguous
    ("box", None),                               # two boxes
    ("green can", None),                         # no such object
    ("can you see anything", None),              # a category word alone never picks one
    ("罐子", None),
    ("a red stool", None),                       # a colour + scenery names furniture
    ("红色的凳子", None),
    ("红色的东西", "pickable_can_red"),
    ("grab the red can, not the green bottle", None),  # two objects named: ambiguous
    ("把红色罐子旁边的绿色瓶子拿来", None),
    ("unicorn", None),
    ("", None),
])
def test_resolve(apt, text, body) -> None:
    assert apt.resolve(text) == body


def test_tabletop_world(apt) -> None:
    tt = load_world("tabletop")
    assert tt.resolve("香蕉") == "banana" and tt.resolve("pick up the duck") == "duck"
    assert tt.surfaces["tray"].box == (0.20, 0.0, 0.50, 0.25)
    assert tt.user is None and not tt.rooms


def test_unknown_world_and_bad_refs_fail_loud(tmp_path) -> None:
    with pytest.raises(WorldManifestError, match="valid: .*apartment"):
        load_world("no_such_world")
    with pytest.raises(WorldManifestError, match="surface='nowhere'"):
        parse_world({"surfaces": {}, "pickables": {"x": {
            "category": "c", "colour": "red", "aliases": ["x"], "surface": "nowhere"}}}, "t", tmp_path)
    with pytest.raises(WorldManifestError, match="needs body, box, height"):
        parse_world({"surfaces": {"s": {"box": [0, 0, 1, 1]}}}, "t", tmp_path)


def _floor_boxes() -> dict[str, tuple[float, ...]]:
    out = {}
    for g in ET.parse(_ROOM_XML).getroot().iter("geom"):
        name = g.get("name", "")
        if name.startswith("f_"):
            (x, y, _), (sx, sy, _) = ([float(v) for v in g.get(k).split()] for k in ("pos", "size"))
            out[name] = (x - sx, y - sy, x + sx, y + sy)
    return out


def test_room_boxes_are_the_floor_geoms(apt) -> None:
    floors = _floor_boxes()
    exact = {"living_room": "f_living", "dining_room": "f_dining", "kitchen": "f_kitchen",
             "study": "f_study", "master_bedroom": "f_master", "guest_bedroom": "f_guest"}
    for room, geom in exact.items():
        assert apt.boxes[room] == pytest.approx(floors[geom]), room
    b, f = apt.boxes["bathroom"], floors["f_bath"]  # the bathroom adds the merged laundry
    assert b[0] <= f[0] and b[1] <= f[1] and b[2] >= f[2] and b[3] >= f[3]
    for room, (x, y) in apt.rooms.items():  # every seed lies in its own room's box
        x0, y0, x1, y1 = apt.boxes[room]
        assert x0 <= x <= x1 and y0 <= y <= y1, room
    ux, uy = apt.user
    x0, y0, x1, y1 = apt.boxes["hallway"]
    assert x0 <= ux <= x1 and y0 <= uy <= y1


def test_manifest_matches_the_compiled_scene(apt) -> None:
    mj = pytest.importorskip("mujoco")
    import numpy as np

    from vector_os_nano.embodiments import registry
    from vector_os_nano.hardware.sim.scene_builder import room_scene

    m = mj.MjModel.from_xml_path(str(room_scene(registry.load("go2_piper"), apt).xml_path))
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    free = {
        mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)
        if m.body_jntadr[b] >= 0 and m.jnt_type[m.body_jntadr[b]] == mj.mjtJoint.mjJNT_FREE
    }
    assert free - {"base_link"} == set(apt.pickables)  # every free object is listed, and only those
    welds = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_EQUALITY, i) for i in range(m.neq)
             if m.eq_type[i] == mj.mjtEq.mjEQ_WELD}
    assert welds == {f"grasp_{p}" for p in apt.pickables}
    for s in apt.surfaces.values():  # height + footprint = the body's collision box top
        bid = m.body(s.body).id
        g = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_type[g] == mj.mjtGeom.mjGEOM_BOX
                 and (m.geom_contype[g] or m.geom_conaffinity[g]))
        c, h = d.geom_xpos[g], m.geom_size[g]
        assert c[2] + h[2] == pytest.approx(s.height, abs=1e-6), s.name
        assert np.allclose(s.box, (c[0] - h[0], c[1] - h[1], c[0] + h[0], c[1] + h[1]), atol=0.006), s.name
    for p in apt.pickables.values():  # each starts inside its surface's footprint
        x, y = d.body(p.name).xpos[:2]
        x0, y0, x1, y1 = apt.surfaces[p.surface].box
        assert x0 <= x <= x1 and y0 <= y <= y1, p.name
