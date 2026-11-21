# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The generic drivers on every backend: the SAME Arm + Gripper + Camera (sim/drivers),
built from a manifest alone, drive two different arms through ONE SimRuntime(world):
the `piper` (6-DoF, parallel jaws) on the battery table and the `so101` (5-DoF, one
rotating jaw) on its tabletop world.

What they report is checked against the world side: GroundTruth for held / lifted /
released, the backend's contacts for the grip gate, its segmentation for the self mask.
"""
from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import yaml

import vector_os_nano
from vector_os_nano.embodiments import registry
from vector_os_nano.embodiments.config import parse_embodiment_config
from vector_os_nano.interfaces import ArmProtocol, CameraProtocol, GripperProtocol
from vector_os_nano.sim.drivers.arm import Arm
from vector_os_nano.sim.drivers.camera import Camera
from vector_os_nano.sim.drivers.gripper import Gripper
from vector_os_nano.sim.ground_truth import GroundTruth
from vector_os_nano.sim.runtime import SimRuntime
from vector_os_nano.sim.scene import canonical
from vector_os_nano.sim.scene.compose import compose
from vector_os_nano.schemas.world import parse_world

pytestmark = pytest.mark.sim

LIFT_M = 0.03  # the lifted oracle's threshold


class Case(NamedTuple):
    """What a test needs to know about one body in its scene (test data, not product)."""

    scene: str                  # the conftest scene fixture
    picks: tuple[str, str]      # two objects of different shape
    pose: list[float]           # a reachable joint pose to move to and hold
    far: list[float]            # a far pose (the stop test interrupts the move to it)
    gap_3mm: float              # 3 mm between a finger and the object, in jaw units
    opens: frozenset[str]       # the fingers the jaw joint moves away from the object
    rise: float                 # how far the squeezing jaws may raise a resting object (m)
    table: str                  # the support body ...
    table_point: tuple[float, float, float]  # ... and a free point on its top


CASES = {
    "piper": Case("battery_scene", ("box", "cylinder"), [0.3, 1.6, -1.2, 0.2, 0.6, -0.4],
                  [0.8, 1.6, -1.0, 0.3, 0.5, 0.5], 0.003, frozenset({"link7", "link8"}), 3e-3,
                  "table", (0.40, 0.0, 0.10)),
    # One jaw moves (the gripper joint, rad, 64 mm from its hinge to its pad) and presses
    # the object against the fixed one; its pad sweeps up as it closes (a few mm of lift).
    "so101": Case("tabletop_scene", ("screwdriver", "bottle"), [0.3, -0.3, 0.2, 0.1, -0.1],
                  [0.8, 0.5, 0.5, 0.3, 0.5], 0.003 / 0.064, frozenset({"jaw"}), 8e-3,
                  "table", (0.35, -0.15, 0.02)),
}


class Body:
    """A manifest's drivers on one world, caller-stepped."""

    def __init__(self, robot: str, world) -> None:
        self.case = CASES[robot]
        self.cfg = cfg = registry.load(robot)
        world.step(500)  # the objects settle on the table
        self.rt = SimRuntime(world, substeps=5, threaded=False, paced=False, name="conf")
        self.arm, self.gripper = Arm(cfg.arm, self.rt), Gripper(cfg.arm.gripper, self.rt)
        self.camera = Camera(cfg.camera, self.rt)
        self.gt = GroundTruth(self.rt)
        sc = self.sc = world.scene
        self.hand = sc.body(cfg.arm.gripper.weld_body)
        self.fingers = {sc.body(b) for b in cfg.arm.gripper.finger_bodies}
        self.jaw = sc.joint(cfg.arm.gripper.joint)

    def reach(self, name: str, dz: float = 0.0, duration: float = 2.0) -> bool:
        x, y, z = self.gt.object_positions()[name]
        q = self.arm.ik_top_down((x, y, z + dz))
        assert q is not None, f"top-down IK has no solution {dz * 100:.0f} cm over the {name}"
        return self.arm.move_joints(q, duration=duration)

    def around(self, name: str) -> int:
        """Open jaws lowered around *name*: nothing touches it, nothing is gripped."""
        assert self.gripper.open()
        assert self.reach(name, 0.05, 3.0) and self.reach(name, 0.0, 2.0)
        body = self.sc.body(name)
        assert not (self.fingers & self.rt.call(lambda w: w.contacts(body)))
        return body

    def servo_jaw(self, q: float) -> float:
        """World side, the jaw alone (no grasp decision): where it stops."""
        jaw = [self.jaw]
        self.rt.call(lambda w: w.set_targets(jaw, [q]))
        self.rt.wait_sim(1.0)
        return float(self.rt.snapshot().q[self.sc.joint_q(self.jaw)])

    def touching(self, body: int) -> set[int]:
        return self.fingers & self.rt.call(lambda w: w.contacts(body))


@pytest.fixture(params=sorted(CASES))
def body(request, backend, load) -> Body:
    scene = request.getfixturevalue(CASES[request.param].scene)
    b = Body(request.param, load(backend, scene))
    yield b
    b.rt.close()


def test_the_drivers_are_the_protocols_and_keep_the_world_private(body):
    assert isinstance(body.arm, ArmProtocol) and isinstance(body.gripper, GripperProtocol)
    assert isinstance(body.camera, CameraProtocol)
    for dev in (body.arm, body.gripper, body.camera):
        assert not any(hasattr(dev, a) for a in ("runtime", "world", "gt", "get_object_positions"))
    assert body.arm.joint_names == list(body.cfg.arm.joints) and body.arm.dof == len(body.cfg.arm.joints)
    with pytest.raises(ValueError, match=f"expected {body.arm.dof}"):
        body.arm.move_joints([0.0, 0.0])


def test_move_to_a_pose_and_hold_it(body):
    target = body.case.pose
    assert body.arm.move_joints(target, duration=2.0)
    body.rt.wait_sim(1.0)  # held, not just reached
    err = np.abs(np.array(body.arm.get_joint_positions()) - target).max()
    assert err < body.cfg.arm.track_tol, err


def test_home_is_the_manifest_pose(body):
    assert body.arm.move_joints(body.case.pose, duration=2.0)
    assert body.arm.home(duration=2.0)
    err = np.abs(np.array(body.arm.get_joint_positions()) - body.cfg.arm.home).max()
    assert err < body.cfg.arm.track_tol, err


def test_a_move_that_cannot_arrive_reports_false(body):
    """move_joints checks arrival: a target past the joint limits is never reached."""
    beyond = [hi + 0.5 for _lo, hi in body.arm._kin.range]
    assert body.arm.move_joints(beyond, duration=1.0) is False


def test_ik_and_fk_never_touch_the_live_state(body):
    before = body.rt.snapshot().q.copy()
    pos, rot = body.arm.fk(list(body.case.far))
    assert len(pos) == 3 and np.array(rot).shape == (3, 3)
    q = body.arm.ik(tuple(pos))
    assert q is not None and np.linalg.norm(np.array(body.arm.fk(q)[0]) - pos) < 3e-3
    q = body.arm.ik(tuple(pos), current_joints=[v + .1 for v in body.case.far], target_rotation=rot)
    assert q is not None
    reached, orientation = body.arm.fk(q)
    assert np.linalg.norm(np.array(reached) - pos) < 3e-3
    assert np.linalg.norm(np.array(orientation) - rot) < .03
    assert body.arm.ik((5.0, 5.0, 5.0)) is None and body.arm.ik_top_down((5.0, 5.0, 5.0)) is None
    assert np.array_equal(body.rt.snapshot().q, before)


@pytest.mark.parametrize("which", [0, 1])
def test_a_contact_gated_pick_is_held_lifted_and_released(body, which):
    name = body.case.picks[which]
    obj = body.around(name)
    assert body.gripper.close() and body.gripper.is_holding(name)
    assert body.fingers <= body.touching(obj)
    assert body.gt.attached_to(name) == body.cfg.arm.gripper.weld_body
    assert body.gt.clearance(name) < body.case.rise  # attached but still resting: not lifted yet
    assert body.reach(name, 0.06)  # straight up
    assert body.gt.attached_to(name) == body.cfg.arm.gripper.weld_body and body.gt.clearance(name) >= LIFT_M
    assert body.gripper.open() and not body.gripper.is_holding()
    assert body.gt.attachments() == {}


def test_closing_on_nothing_attaches_nothing(body):
    assert body.gripper.open()
    body.rt.wait_sim(1.0)
    assert body.gripper.get_position() == pytest.approx(1.0, abs=0.05)
    assert body.gripper.close() and not body.gripper.is_holding()
    assert body.gripper.get_position() < 0.1 and body.gt.attachments() == {}


def test_contact_only_grip_has_no_constraint_and_notices_lost_contact(body):
    name = body.case.picks[0]
    obj = body.around(name)
    grip = Gripper(dataclasses.replace(body.cfg.arm.gripper, assist=False), body.rt)
    assert grip.close() and grip.is_holding(name)
    assert body.fingers <= body.touching(obj)
    assert body.rt.attachments() == {} and not body.rt.attach_log
    body.servo_jaw(body.cfg.arm.gripper.open)
    assert not grip.is_holding(name) and body.rt.attachments() == {}


def test_contact_only_scene_does_not_generate_grasp_welds():
    from tests.conformance.conftest import SCENES
    import mujoco as mj

    cfg = registry.load('piper')
    cfg = dataclasses.replace(cfg, arm=dataclasses.replace(
        cfg.arm, gripper=dataclasses.replace(cfg.arm.gripper, assist=False)))
    scene = compose(SCENES / 'battery.xml', cfg, spawn=(0.,0.,.2,0.), pickables=('box','cylinder'))
    assert scene.grasp_pairs == ()
    assert not np.any(scene.model.eq_type == int(mj.mjtEq.mjEQ_WELD))


def test_fingers_3mm_away_do_not_grip(body):
    box = body.around(body.case.picks[0])
    touching = body.servo_jaw(body.cfg.arm.gripper.closed)  # where the jaws stop on the box
    assert body.fingers <= body.touching(box)
    body.servo_jaw(touching + body.case.gap_3mm)
    assert not body.touching(box) & {body.sc.body(f) for f in body.case.opens}
    assert body.rt.grasp(body.gripper._grasp) is None and body.gt.attachments() == {}


def test_grasp_needs_the_jaws_held_open_and_a_declared_pair(body):
    """Every finger on the object is not enough: jaws squeezed under min_opening, or a
    weld body with no grasp pair for the object, attach nothing."""
    box = body.around(body.case.picks[0])
    jaw = body.servo_jaw(body.cfg.arm.gripper.closed)
    assert body.fingers <= body.touching(box)
    spec = body.gripper._grasp
    assert body.rt.grasp(dataclasses.replace(spec, min_opening=jaw - spec.closed + 1e-3)) is None
    root = body.sc.robot_root  # no (robot root, object) pair
    assert body.rt.grasp(dataclasses.replace(spec, weld_body=root)) is None and body.rt.attachments() == {}
    assert body.rt.grasp(spec) == box
    assert body.rt.release(root) == [] and body.rt.attachments() == {box: body.hand}
    assert body.rt.release(body.hand) == [] and body.rt.attachments() == {}


def test_stop_preempts_a_move(body):
    out: dict = {}
    t = threading.Thread(target=lambda: out.update(ok=body.arm.move_joints(body.case.far, 3.0),
                                                   t=body.rt.sim_time), daemon=True)
    t0 = body.rt.sim_time
    t.start()
    while body.rt.sim_time < t0 + 0.5 and t.is_alive():
        time.sleep(0.005)
    body.arm.stop()
    t_stop = body.rt.sim_time
    t.join(timeout=30.0)
    assert out["ok"] is False and out["t"] - t_stop <= 0.05
    joints = list(body.arm._joints)
    held = body.rt.call(lambda w: w.targets(joints))
    body.rt.wait_sim(0.3)
    assert np.array_equal(body.rt.call(lambda w: w.targets(joints)), held)  # nothing written after the hold


def test_a_move_that_dies_mid_way_still_counts_what_it_commanded(body, monkeypatch):
    """Charter 3.5: an error with no commanded motion is a REFUSED call. A move that
    raises after writing set-points did command motion, so the counter must say so
    (it was only added after the interpolation loop)."""
    from vector_os_nano.sim.runtime import ACTOR_EPOCH

    real, calls = body.rt.wait_sim, []

    def dying(dt, **kw):
        calls.append(dt)
        if len(calls) > 3:
            raise RuntimeError("the stepper died")
        return real(dt, **kw)

    monkeypatch.setattr(body.rt, "wait_sim", dying)
    before = body.arm.ctrl_motion()
    with ACTOR_EPOCH.tool(), pytest.raises(RuntimeError, match="stepper died"):
        body.arm.move_joints(body.case.far, 3.0)
    assert body.arm.ctrl_motion() > before


def test_a_capture_is_metric_and_masks_the_arm_and_nothing_else(body):
    sc, cam = body.sc, body.cfg.camera
    f = body.camera.capture()  # caller-stepped: the segmentation below is the same instant
    seg = body.rt.render(cam.name, cam.width, cam.height, ("seg",)).seg
    assert not hasattr(f, "seg") and f.rgb.shape == (cam.height, cam.width, 3)
    fy = (cam.height / 2) / np.tan(np.radians(cam.fovy) / 2)  # the intrinsics are the camera's
    assert np.allclose(f.K, [[fy, 0, cam.width / 2], [0, fy, cam.height / 2], [0, 0, 1]])
    robot = sorted(sc.subtree(sc.robot_root))
    assert f.self_mask.any() and np.array_equal(f.self_mask, np.isin(seg, robot))
    for name in (*body.case.picks, body.case.table):  # visible, and never the robot's
        assert (seg == sc.body(name)).any() and not f.self_mask[seg == sc.body(name)].any()
    # metric depth at a known surface: the table top through one pixel's centre
    K, R, c = f.K, f.cam_mat, f.cam_pos
    top = np.array(body.case.table_point)
    p = R.T @ (top - c)
    col, row = int(K[0, 0] * p[0] / -p[2] + K[0, 2]), int(K[1, 2] - K[1, 1] * p[1] / -p[2])
    assert seg[row, col] == sc.body(body.case.table)
    ray = R @ np.array([(col + 0.5 - K[0, 2]) / K[0, 0], -(row + 0.5 - K[1, 2]) / K[1, 1], -1.0])
    expected = (top[2] - c[2]) / ray[2]  # the ray's camera-z component is -1: t is the z-depth
    assert f.depth[row, col] == pytest.approx(expected, abs=2e-3)


def test_a_variant_arm_is_data_alone(tmp_path, backend, load):
    """A second SO-101 from data only: another home (its robot.yaml), a narrower pan
    range (its model file) and a mount 8 cm higher (its world's spawn). The same
    drivers move it; no Python names it."""
    pkg = Path(vector_os_nano.__file__).parent
    stock = pkg / "hardware" / "sim" / "mjcf" / "so101" / "so101.xml"
    model = tmp_path / "so101_variant.xml"
    model.write_text(stock.read_text()
                     .replace('meshdir="meshes"', f'meshdir="{stock.parent / "meshes"}"')
                     .replace('"shoulder_pan" axis="0 0 1" range="-1.92 1.92"', '"shoulder_pan" axis="0 0 1" range="-1 1"')
                     .replace('"shoulder_pan"  kp="100" ctrlrange="-1.92 1.92"', '"shoulder_pan"  kp="100" ctrlrange="-1 1"'))
    raw = yaml.safe_load((pkg / "embodiments" / "so101" / "robot.yaml").read_text())
    raw.update(id="so101_variant", model={"path": str(model), "root_body": "base_link"})
    raw["arm"]["home"] = [0.5, -0.8, 0.4, 0.9, 0.0]
    cfg = parse_embodiment_config(raw, "so101_variant/robot.yaml")
    world_yaml = pkg.parent / "config" / "worlds" / "tabletop.yaml"
    wraw = yaml.safe_load(world_yaml.read_text())
    wraw["spawn"]["z"] = 0.10  # a riser under the base
    world = parse_world(wraw, "tabletop", world_yaml)
    scene = compose(world.scene, cfg, spawn=world.spawn, pickables=tuple(world.pickables))
    assert scene.model.jnt_range[scene.joint("shoulder_pan")].tolist() == [-1.0, 1.0]

    rt = SimRuntime(load(backend, scene), substeps=5, threaded=False, paced=False, name="variant")
    try:
        arm = Arm(cfg.arm, rt)
        assert arm.home(duration=2.0)
        assert np.abs(np.array(arm.get_joint_positions()) - cfg.arm.home).max() < cfg.arm.track_tol
        mount = canonical.body_poses(scene, rt.snapshot().q)[0][scene.robot_root]
        assert mount[2] == pytest.approx(0.10, abs=1e-6)
        assert arm.move_joints([1.5, -0.8, 0.4, 0.9, 0.0], duration=1.0) is False  # past its pan range
        assert arm.get_joint_positions()[0] < 1.0 + cfg.arm.track_tol
    finally:
        rt.close()
