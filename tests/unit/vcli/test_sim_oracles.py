# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""GT oracles (vcli/worlds/sim_oracles) on the COMPILED go2_piper apartment scene.

Every predicate gets a positive and an adversarial negative case. The rig is the
real MJCF + SimRuntime (caller-stepped); the robot is pinned kinematically each
tick so only the objects run physics, and grasps are posed (bottle between the
Piper fingers, weld on) rather than driven, so each case isolates one oracle input.
"""
from __future__ import annotations

import dataclasses
import math
import time

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from vector_os_nano.core.scene_graph import ObjectNode, SceneGraph  # noqa: E402
from vector_os_nano.embodiments.dof_layout import DofLayout  # noqa: E402
from vector_os_nano.hardware.sim.mujoco_go2 import _build_room_scene_xml, _grasp_pairs  # noqa: E402
from vector_os_nano.sim.backends.mujoco.world import MuJoCoWorld  # noqa: E402
from vector_os_nano.sim.ground_truth import GroundTruth  # noqa: E402
from vector_os_nano.sim.runtime import SimRuntime  # noqa: E402
from vector_os_nano.sim.scene import compiled_scene  # noqa: E402
from vector_os_nano.sim.scene.compiled import inactive_welds  # noqa: E402
from vector_os_nano.vcli.worlds.sim_oracles import rig_namespace  # noqa: E402
from vector_os_nano.vcli.worlds.world_manifest import load_world  # noqa: E402

_GRASP_SEED = [0.0, 1.57, -1.57, 0.0, 1.22, 0.0]  # top-down, fingers in free air
_STAND = [0.0, 0.9, -1.8] * 4
GREEN = "pickable_bottle_green"


class Rig:
    """The compiled scene + a kinematic pin for the robot's root, legs, arm and jaws."""

    def __init__(self) -> None:
        world = MuJoCoWorld(compiled_scene(
            _build_room_scene_xml(with_arm=True), robot_root="base_link", grasp_pairs=_grasp_pairs(True)))
        m, d = world.model, world.data
        self.m, self.d = m, d
        self.L = DofLayout(m, "base_link", 12)
        jq = lambda n: int(m.jnt_qposadr[m.joint(n).id])  # noqa: E731
        self.arm_adr = [jq(f"piper_joint{i}") for i in range(1, 7)]
        self.j7, self.j8 = jq("piper_joint7"), jq("piper_joint8")
        self.base = [10.0, 3.0, 0.32, 0.0]  # x, y, z, yaw
        self.arm = [0.0] * 6
        self.jaw = 0.035
        self.rt = SimRuntime(world, threaded=False, paced=False, name="oracle_test")
        self.rt.call(self._pin)
        self.rt.step(300, controller=self._pin)  # objects settle into resting contact
        self.qpos0 = d.qpos.copy()
        self.gt = GroundTruth(self.rt, robot_root_bid=self.L.root_bid)
        self.turn = (-math.inf, -math.inf)
        self.sg: SceneGraph | None = None
        self.ns = rig_namespace(self.rt, "go2_piper", belief=lambda: self.sg,
                                turn_start=lambda: self.turn)

    def _pin(self, w) -> None:
        m, d = w.model, w.data
        a = self.L.root_qpos_adr
        x, y, z, yaw = self.base
        d.qpos[a:a + 3] = [x, y, z]
        d.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        d.qpos[self.L.joint_qpos_start:self.L.joint_qpos_start + 12] = _STAND
        d.qpos[self.arm_adr] = self.arm
        d.qpos[self.j7], d.qpos[self.j8] = self.jaw, -self.jaw
        d.qvel[self.L.root_dof_adr:self.L.root_dof_adr + 26] = 0.0
        mj.mj_forward(m, d)

    def reset(self) -> None:
        """Back to the settled scene, no weld on, a fresh turn starting now."""
        self.base, self.arm, self.jaw = [10.0, 3.0, 0.32, 0.0], [0.0] * 6, 0.035
        self.sg = None

        def fn(w):
            m, d = w.model, w.data
            d.qpos[:] = self.qpos0
            d.qvel[:] = 0.0
            d.eq_active[:] = [int(m.eq_type[i] != mj.mjtEq.mjEQ_WELD) for i in range(m.neq)]
            self._pin(w)

        self.rt.call(fn)
        self.turn = (self.gt.sim_time, time.time())

    def weld(self, name: str, on: bool = True) -> None:
        i = next(i for i in range(self.m.neq) if self.m.eq_type[i] == mj.mjtEq.mjEQ_WELD
                 and mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_BODY, int(self.m.eq_obj2id[i])) == name)
        self.rt.call(lambda w: w.data.eq_active.__setitem__(i, int(on)))

    def move(self, name: str, xyz) -> None:
        a = int(self.m.jnt_qposadr[self.m.body(name).jntadr[0]])

        def fn(w):
            w.data.qpos[a:a + 3] = xyz
            w.data.qpos[a + 3:a + 7] = [1.0, 0.0, 0.0, 0.0]
            self._pin(w)

        self.rt.call(fn)

    def grasp_pose(self, jaw: float) -> np.ndarray:
        """Arm at the top-down seed; returns the point between the two finger pads."""
        self.arm, self.jaw = list(_GRASP_SEED), 0.035
        self.rt.call(self._pin)
        pads = []
        for body in ("piper_link7", "piper_link8"):
            bid = self.m.body(body).id
            gs = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == bid
                  and (self.m.geom_contype[g] or self.m.geom_conaffinity[g])]
            pads.append(np.mean([self.d.geom_xpos[g] for g in gs], axis=0))
        self.jaw = jaw
        return (pads[0] + pads[1]) / 2.0


@pytest.fixture(scope="module")
def rig() -> Rig:
    r = Rig()
    yield r
    r.rt.close()


@pytest.fixture()
def ns(rig: Rig):
    rig.reset()
    return rig.ns


def _held_green(rig: Rig, jaw: float = 0.02) -> None:
    """The green bottle between the closed fingers, in the air, welded."""
    rig.move(GREEN, rig.grasp_pose(jaw))
    rig.weld(GREEN)


# --------------------------------------------------------------------- objects

def test_robot_bodies_are_never_scene_objects(rig: Rig) -> None:
    names = set(rig.gt.object_positions())
    assert names == set(load_world("apartment").pickables)
    assert "base_link" not in names and not any(n.startswith("piper") for n in names)


def test_every_pickable_rests_on_its_manifest_surface(ns, rig: Rig) -> None:
    world = load_world("apartment")
    for name, p in world.pickables.items():
        assert ns["object_on"](name, p.surface), name
        assert not ns["holding_object"](name), name
        assert ns["released"](name), name
    assert not ns["object_on"](GREEN, "dining_table")  # the wrong surface
    assert not ns["object_on"](GREEN, "no_such_table")


# --------------------------------------------------------------------- gripper

def test_resting_on_the_table_is_not_holding(ns) -> None:
    assert ns["object_on"]("green bottle", "pick_table")
    assert not ns["holding_object"]("green bottle")
    assert not ns["holding_object"]()
    assert not ns["lifted"]("green bottle")


def test_welded_in_place_without_lift_is_not_holding(ns, rig: Rig) -> None:
    rig.weld(GREEN)
    assert not ns["holding_object"]("green bottle")
    assert not ns["lifted"]("green bottle")
    assert not ns["object_on"]("green bottle", "pick_table")  # welded = not resting
    assert not ns["released"]("green bottle")
    assert not ns["released"]()


def test_lifted_by_weld_and_finger_contacts_is_holding(ns, rig: Rig) -> None:
    _held_green(rig)
    assert ns["holding_object"]("green bottle")
    assert ns["holding_object"]("绿色的瓶子")
    assert ns["holding_object"](GREEN)
    assert ns["holding_object"]()
    assert ns["lifted"]("green bottle", 0.03)
    assert ns["kept_holding"]("green bottle")
    assert not ns["holding_object"]("blue bottle")  # target-aware
    assert not ns["object_on"]("green bottle", "pick_table")


def test_lift_without_finger_contact_is_not_holding(ns, rig: Rig) -> None:
    _held_green(rig, jaw=0.035)  # jaws open: only the wrist touches the bottle
    assert ns["lifted"]("green bottle")
    assert not ns["holding_object"]("green bottle")


def test_finger_contact_without_weld_is_not_holding(ns, rig: Rig) -> None:
    rig.move(GREEN, rig.grasp_pose(0.02))
    assert not ns["holding_object"]("green bottle")
    assert not ns["lifted"]("green bottle")


def test_held_but_resting_on_its_support_is_not_lifted(ns, rig: Rig) -> None:
    rig.weld(GREEN)
    rig.move(GREEN, [10.88, 3.0, 0.325])  # 5 mm proud of the table: under the 3 cm lift
    assert not ns["lifted"]("green bottle")
    assert not ns["holding_object"]("green bottle")


def test_kept_holding_is_false_after_a_drop_this_turn(ns, rig: Rig) -> None:
    rig.turn = (rig.gt.sim_time - 1.0, time.time())
    _held_green(rig)
    rig.weld(GREEN, False)  # the scripted drop
    rig.gt.weld_events()  # (a read publishes the change)
    rig.weld(GREEN, True)  # ... and a re-grab
    assert ns["holding_object"]("green bottle")
    assert not ns["kept_holding"]("green bottle")
    rig.turn = (rig.gt.sim_time + 1.0, time.time())  # a later turn did not see the drop
    assert ns["kept_holding"]("green bottle")


def test_object_at(ns) -> None:
    assert ns["object_at"]("green bottle", 10.88, 3.0)
    assert not ns["object_at"]("green bottle", 10.0, 3.0)
    assert not ns["object_at"]("green bottle", 10.0, 3.0, 99)  # clamped to 0.5 m


# --------------------------------------------------------------------- arm

def test_arm_at_home_on_the_piper_at_zero(ns, rig: Rig) -> None:
    assert rig.arm == [0.0] * 6
    assert ns["arm_at_home"]()
    rig.arm = [0.0, 0.3, 0.0, 0.0, 0.0, 0.0]
    rig.rt.call(rig._pin)
    assert not ns["arm_at_home"]()


# --------------------------------------------------------------------- base

def test_base_oracles(ns, rig: Rig) -> None:
    assert ns["at_position"](10.0, 3.0)
    assert not ns["at_position"](12.0, 3.0)
    assert not ns["at_position"](13.0, 3.0, 99)  # clamped to 2 m
    assert ns["facing"](0.0) and not ns["facing"](math.pi)
    assert ns["in_room"]("hallway") and not ns["in_room"]("kitchen")
    assert not ns["in_room"]("no_such_room")
    assert ns["near_object"]("green bottle", 1.0)
    assert not ns["near_object"]("purple box")
    assert not ns["near_object"]("green bottle", 0.5)
    assert not ns["near_object"]("user")  # 2 m away
    assert not ns["near_object"]("user", 99)  # clamped to 1.5 m
    rig.base = [8.3, 3.0, 0.32, 0.0]
    rig.rt.call(rig._pin)
    assert ns["near_object"]("user")
    assert ns["get_position"]()[:2] == pytest.approx((8.3, 3.0))


def test_in_view_is_segmentation_of_the_robot_camera(ns, rig: Rig) -> None:
    assert ns["in_view"]("green bottle")
    assert not ns["in_view"]("purple box")  # behind the robot, on the hallway console
    rig.base = [10.0, 3.0, 0.32, math.pi]  # turned around
    rig.rt.call(rig._pin)
    assert not ns["in_view"]("green bottle")


# --------------------------------------------------------------------- perception claim

def _instance(category: str, x: float, y: float, seen: float) -> ObjectNode:
    return ObjectNode(object_id=f"obj_{category}_{x}", category=category, x=x, y=y, first_seen=seen)


def test_localized_judges_the_scene_graph_against_gt(ns, rig: Rig) -> None:
    rig.turn = (rig.gt.sim_time, time.time())
    rig.sg = SceneGraph()
    assert not ns["localized"]("green bottle")  # nothing on the map
    now = time.time() + 1.0
    rig.sg.add_object(_instance("blue bottle", 10.88, 3.0, now))  # wrong object at its spot
    rig.sg.add_object(_instance("bottle", 10.88, 3.0, now))  # ambiguous
    rig.sg.add_object(_instance("green bottle", 11.9, 3.0, now))  # 1 m off
    assert not ns["localized"]("green bottle")
    assert not ns["localized"]("green bottle", 99)  # tol clamped to 0.2 m
    rig.sg.add_object(_instance("green bottle", 10.90, 2.78, now))  # where the BLUE one is
    assert not ns["localized"]("green bottle", 0.2)
    rig.sg.add_object(_instance("green bottle", None, None, now))  # seen, never localized
    assert not ns["localized"]("green bottle")
    rig.sg.add_object(_instance("green bottle", 10.9, 3.05, now))
    assert ns["localized"]("green bottle")  # an unlocalized instance never hides it
    assert ns["localized"]("绿色的瓶子")
    rig.turn = (rig.gt.sim_time, now + 10.0)  # seen before this turn started: stale
    assert not ns["localized"]("green bottle")
    old = next(o for o in rig.sg.find_objects_by_category("") if (o.x, o.y) == (10.9, 3.05))
    # re-observed during this turn: its last_seen (not first_seen) is what counts
    rig.sg.add_object(dataclasses.replace(old, last_seen=now + 11.0))
    assert ns["localized"]("green bottle")
    rig.sg = None
    assert not ns["localized"]("green bottle")


def test_unknown_object_fails_closed(ns) -> None:
    for fn in ("holding_object", "lifted", "kept_holding", "released", "in_view", "localized"):
        assert ns[fn]("unicorn") is False, fn
    assert ns["object_on"]("unicorn", "pick_table") is False
    assert ns["near_object"]("unicorn") is False


# --------------------------------------------------------------------- so101 tabletop

def test_so101_tabletop_oracles() -> None:
    from vector_os_nano.hardware.sim.mujoco_arm import _DEFAULT_SCENE_XML
    from vector_os_nano.embodiments import registry

    world = MuJoCoWorld(compiled_scene(
        _DEFAULT_SCENE_XML, robot_root="base_link", grasp_pairs=inactive_welds(_DEFAULT_SCENE_XML)))
    m, d = world.model, world.data
    stance = registry.load("so101").stance
    for j, v in stance.items():
        d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
    rt = SimRuntime(world, threaded=False, paced=False, name="so101_test")
    try:
        rt.call(lambda w: mj.mj_forward(w.model, w.data))
        ns = rig_namespace(rt, "so101")
        assert "at_position" not in ns  # fixed base: no base oracles
        assert ns["arm_at_home"]()
        rt.step(500)  # the objects settle onto the table (the unpowered arm sags)
        assert not ns["arm_at_home"]()
        assert ns["object_on"]("banana", "table") and not ns["holding_object"]("banana")
        i = next(i for i in range(m.neq)
                 if mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.eq_obj2id[i])) == "banana")
        rt.call(lambda w: w.data.eq_active.__setitem__(i, 1))
        assert not ns["holding_object"]("banana")  # welded, not lifted
        a = int(m.jnt_qposadr[m.body("banana").jntadr[0]])
        rt.call(lambda w: (w.data.qpos.__setitem__(a + 2, w.data.qpos[a + 2] + 0.10), mj.mj_forward(w.model, w.data)))
        assert ns["holding_object"]("banana")  # a weld-only gripper: weld + lift
        assert ns["holding_object"]("香蕉")
    finally:
        rt.close()
