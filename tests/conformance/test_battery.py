# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The oracle-parity battery and the contact-gated pick (scenes/battery.xml + the Piper).

Ground-truth quantities are read the way the world side reads them: canonical shape
AABBs over the backend's state (clearance), the backend's contacts, its attachments
and its segmentation render. Only the PhysicsWorld API drives the arm, and the grasp
is the world side's own decision (sim/grasp.py), the same on every backend.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tests.conformance.rig import clearance, ik
from vector_os_nano.interfaces.physics import GraspSpec
from vector_os_nano.sim import grasp as g
from vector_os_nano.sim.scene import canonical

pytestmark = pytest.mark.sim

DOWN = np.array([0.0, 0.0, -1.0])
SEED = [0.0, 1.87, -0.98, 0.0, 0.77, 0.0]  # a top-down configuration over the table
OPEN = 0.035                                # per-finger travel of joint7


class Arm:
    """The scripted Piper: IK on the canonical model, set-points through set_targets."""

    def __init__(self, world, scene) -> None:
        self.w, self.sc = world, scene
        self.joints = [scene.joint(f"joint{i}") for i in range(1, 7)]
        self.jaw = scene.joint("joint7")
        self.site = int(scene.model.site("ee_site").id)
        self.fingers = {scene.body("link7"), scene.body("link8")}
        self.hand = scene.body("link6")
        self.spec = GraspSpec(weld_body=self.hand, finger_bodies=tuple(sorted(self.fingers)),
                                jaw_joint=self.jaw, closed=0.0, min_opening=0.005)

    def move(self, pos: np.ndarray, jaw: float, steps: int = 750) -> None:
        q = self.w.state()[0]
        q[[self.sc.joint_q(j) for j in self.joints]] = SEED
        q = ik(self.sc, q, self.site, self.joints, np.asarray(pos, dtype=float), DOWN)
        self.w.set_targets(self.joints + [self.jaw], [*q[[self.sc.joint_q(j) for j in self.joints]], jaw])
        self.w.step(steps)

    def servo_jaw(self, jaw: float, steps: int = 500) -> float:
        self.w.set_targets([self.jaw], [jaw])
        self.w.step(steps)
        return float(self.w.state()[0][self.sc.joint_q(self.jaw)])

    def gripped(self, obj: int) -> bool:
        """Every finger body touches the object."""
        return self.fingers <= self.w.contacts(obj)

    def grip(self) -> tuple[float, int]:
        """Approach the box with open jaws, close them on it: (jaw q where they stop, box)."""
        box = self.sc.body("box")
        at = self.w.state()[0][self.sc.free_q(box)][:3]
        self.move(at + [0, 0, 0.10], OPEN)
        self.move(at, OPEN)
        assert g.grasp(self.w, self.spec) is None and not self.w.attached()  # jaws open: no finger touches
        return self.servo_jaw(0.0), box


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


def test_contact_gated_pick_attach_and_lift(backend, load, battery_scene):
    sc = battery_scene
    w = _settled(load, backend, sc)
    arm = Arm(w, sc)
    _jaw, box = arm.grip()
    at = w.state()[0][sc.free_q(box)][:3]
    assert arm.gripped(box), f"finger contacts {[w.contacts(f) for f in arm.fingers]}"
    assert g.grasp(w, arm.spec) == box
    assert w.attached() == {box: arm.hand}
    # attached but still resting: the attach log says held, the AABB rule says not lifted
    assert abs(clearance(sc, w.state()[0], box)) < 3e-3
    assert sc.body("table") in w.contacts(box)
    arm.move(at + [0, 0, 0.05], 0.0)
    q = w.state()[0]
    assert clearance(sc, q, box) == pytest.approx(0.05, abs=5e-3)
    assert sc.body("table") not in w.contacts(box)
    arm.move(at + [0, 0, 0.10], 0.0)
    assert clearance(sc, w.state()[0], box) >= 0.05 + 0.03  # it follows the hand up


def test_fingers_3mm_away_do_not_grip(backend, load, battery_scene):
    sc = battery_scene
    w = _settled(load, backend, sc)
    arm = Arm(w, sc)
    touching, box = arm.grip()  # where the jaws stop on the box
    assert arm.gripped(box)
    arm.servo_jaw(touching + 0.003)
    assert not arm.gripped(box)
    assert not (arm.fingers & w.contacts(box))
    assert g.grasp(w, arm.spec) is None and w.attached() == {}


def test_grasp_needs_the_jaws_held_open_and_a_declared_pair(backend, load, battery_scene):
    """Both fingers on the box is not enough: jaws squeezed under min_opening, or a
    weld body with no grasp pair for the box, attach nothing."""
    sc = battery_scene
    w = _settled(load, backend, sc)
    arm = Arm(w, sc)
    jaw, box = arm.grip()
    assert arm.gripped(box)
    squeezed = dataclasses.replace(arm.spec, min_opening=jaw + 1e-3)
    assert g.grasp(w, squeezed) is None
    wrist = dataclasses.replace(arm.spec, weld_body=sc.body("link5"))  # no (link5, box) pair
    assert g.grasp(w, wrist) is None
    assert w.attached() == {}
    assert g.grasp(w, arm.spec) == box
    assert g.release(w, sc.body("link5")) == [] and w.attached() == {box: arm.hand}  # not its parent
    assert g.release(w, arm.hand) == [] and w.attached() == {}


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
