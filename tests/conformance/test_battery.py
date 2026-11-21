# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The oracle-parity battery (scenes/battery.xml + the `piper` body).

Ground-truth quantities are read the way the world side reads them: canonical shape
AABBs over the backend's state (clearance), the backend's contacts and its
segmentation render, the same on every backend. The arm's own pick runs through the
generic drivers in test_drivers.py.
"""
from __future__ import annotations

import math

import pytest

from tests.conformance.rig import clearance
from vector_os_nano.embodiments import registry
from vector_os_nano.sim.scene.compose import compose
from vector_os_nano.sim.scene.reader import GEOM_MESH, geoms
from vector_os_nano.schemas.world import load_world

pytestmark = pytest.mark.sim


def _settled(load, backend, scene):
    w = load(backend, scene)
    w.step(500)
    return w


def test_resting_object_touches_its_support(backend, load, battery_scene):
    sc = battery_scene
    w = _settled(load, backend, sc)
    for name in ("box", "cylinder"):
        obj = sc.body(name)
        assert sc.body("table") in w.contacts(obj), name
        assert abs(clearance(sc, w.state()[0], obj)) < 3e-3, name


def test_object_held_5mm_above_is_not_in_contact(backend, load, battery_scene):
    sc = battery_scene
    w = _settled(load, backend, sc)
    box = sc.body("box")
    q = w.state()[0]
    q[sc.free_q(box).start + 2] += 0.005
    w.reset(q)  # fresh: no contact survives from the resting state
    assert sc.body("table") not in w.contacts(box)
    assert clearance(sc, w.state()[0], box) == pytest.approx(0.005, abs=5e-4)


@pytest.mark.parametrize("name", ["box", "cylinder"])
@pytest.mark.parametrize("lift, touching", [(0.2e-3, True), (0.5e-3, True), (0.9e-3, True), (1.5e-3, False)])
def test_object_held_under_1mm_above_is_in_contact(backend, load, battery_scene, name, lift, touching):
    """Charter 3.1: a gap <= 1 mm is a contact on every backend, whatever the engine's
    own margins (MuJoCo: 0; Bullet: a breaking threshold under 1 mm for small shapes).
    The object is set upright at a MEASURED clearance."""
    sc = battery_scene
    w = _settled(load, backend, sc)
    obj = sc.body(name)
    q = w.state()[0]
    s = sc.free_q(obj)
    q[s.start + 3:s.start + 7] = [1, 0, 0, 0]
    q[s.start + 2] -= clearance(sc, q, obj) - lift
    w.reset(q)
    assert clearance(sc, w.state()[0], obj) == pytest.approx(lift, abs=1e-6)
    assert (sc.body("table") in w.contacts(obj)) is touching, f"{name} at {lift * 1000:.1f} mm"


def test_own_shapes_and_never_colliding_pairs_are_no_contact(backend, load, battery_scene):
    """MuJoCo's filter on every backend: a body never touches itself, nor a body it never
    collides with (the Piper's adjacent links interpenetrate; the table sits on the world)."""
    sc = battery_scene
    w = _settled(load, backend, sc)
    m = sc.model
    for b in range(1, m.nbody):
        got = w.contacts(b)
        assert b not in got, sc.body_name(b)
        parent = int(m.body_parentid[b])
        assert parent == 0 or parent not in got, (sc.body_name(b), sc.body_name(parent))
    assert 0 not in w.contacts(sc.body("table"))


@pytest.mark.parametrize("lift, touching", [(0.3e-3, True), (1.5e-3, False)])
def test_a_box_resting_on_an_edge_keeps_the_contact_sentence(backend, load, battery_scene, lift, touching):
    """S2cV-02, pinned: Bullet rounds box edges by its collision margin, so edge-down it
    reads a contact only up to ~0.55 mm (corner-down ~0.25 mm), not 1 mm: conservative
    (fewer contacts, a stricter grip gate). Within 0.3 mm both engines touch, past 1.5 mm
    neither does."""
    sc = battery_scene
    w = _settled(load, backend, sc)
    box = sc.body("box")
    q = w.state()[0]
    s = sc.free_q(box)
    q[s.start + 3:s.start + 7] = [math.cos(math.pi / 8), math.sin(math.pi / 8), 0.0, 0.0]  # 45 deg about x
    q[s.start + 2] -= clearance(sc, q, box) - lift
    w.reset(q)
    assert clearance(sc, w.state()[0], box) == pytest.approx(lift, abs=1e-6)
    assert (sc.body("table") in w.contacts(box)) is touching, f"edge-down at {lift * 1000:.1f} mm"


def test_no_gripper_finger_or_pickable_shape_on_pybullet_is_a_mesh(battery_scene):
    """S2cV-01, pinned: Bullet's distance query over-reports a convex hull's gap (a
    contact at ~2.0 mm one mesh, ~2.8 mm mesh on mesh, against the 1 mm sentence), so a
    grip gate over mesh fingers or a mesh object would be looser on PyBullet. Every
    finger and every pickable of the bodies this release runs on PyBullet (the `piper`
    on the battery table and on its workbench) collides through primitives only."""
    cfg = registry.load("piper")
    world = load_world(cfg.world)
    workbench = compose(world.scene, cfg, spawn=world.spawn, pickables=tuple(world.pickables))
    for sc, objects in ((battery_scene, ("box", "cylinder")), (workbench, tuple(world.pickables))):
        bodies = {sc.body(b) for b in (*cfg.arm.gripper.finger_bodies, *objects)}
        shapes = [g for g in geoms(sc.model) if g.body in bodies and g.collide]
        assert {g.body for g in shapes} == bodies  # each one collides
        assert not [sc.body_name(g.body) for g in shapes if g.type == GEOM_MESH]


def test_segmentation_pixel_counts_agree_across_backends(both, load, battery_scene):
    sc = battery_scene
    counts = {}
    for backend in both:
        w = _settled(load, backend, sc)
        seg = w.render("overhead", 320, 240, ("seg",)).seg
        counts[backend] = {name: int((seg == sc.body(name)).sum()) for name in ("box", "cylinder", "table")}
    for name in ("box", "cylinder", "table"):
        a, b = counts["mujoco"][name], counts["pybullet"][name]
        assert min(a, b) > 50 and abs(a - b) <= 0.15 * max(a, b), counts
