# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The generic drivers on every backend: the SAME Arm + Gripper + Camera (sim/drivers),
built from the `piper` manifest, drive the battery table through ONE SimRuntime(world).

What they report is checked against the world side: GroundTruth for held / lifted /
released, the backend's contacts for the grip gate, its segmentation for the self mask.
"""
from __future__ import annotations

import dataclasses
import threading
import time

import numpy as np
import pytest

from vector_os_nano.embodiments import registry
from vector_os_nano.interfaces import ArmProtocol, CameraProtocol, GripperProtocol
from vector_os_nano.sim import grasp as g
from vector_os_nano.sim.drivers.arm import Arm
from vector_os_nano.sim.drivers.camera import Camera
from vector_os_nano.sim.drivers.gripper import Gripper
from vector_os_nano.sim.ground_truth import GroundTruth
from vector_os_nano.sim.runtime import SimRuntime

pytestmark = pytest.mark.sim

LIFT_M = 0.03  # the lifted oracle's threshold


class Body:
    """The `piper` manifest's drivers on one battery world, caller-stepped."""

    def __init__(self, world) -> None:
        self.cfg = cfg = registry.load("piper")
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


@pytest.fixture
def body(backend, load, battery_scene) -> Body:
    b = Body(load(backend, battery_scene))
    yield b
    b.rt.close()


def test_the_drivers_are_the_protocols_and_keep_the_world_private(body):
    assert isinstance(body.arm, ArmProtocol) and isinstance(body.gripper, GripperProtocol)
    assert isinstance(body.camera, CameraProtocol)
    for dev in (body.arm, body.gripper, body.camera):
        assert not any(hasattr(dev, a) for a in ("runtime", "world", "gt", "get_object_positions"))


def test_move_to_a_pose_and_hold_it(body):
    target = [0.3, 1.6, -1.2, 0.2, 0.6, -0.4]
    assert body.arm.move_joints(target, duration=2.0)
    body.rt.wait_sim(1.0)  # held, not just reached
    err = np.abs(np.array(body.arm.get_joint_positions()) - target).max()
    assert err < body.cfg.arm.track_tol, err


@pytest.mark.parametrize("name", ["box", "cylinder"])
def test_a_contact_gated_pick_is_held_lifted_and_released(body, name):
    obj = body.around(name)
    assert body.gripper.close() and body.gripper.is_holding(name)
    assert body.fingers <= body.touching(obj)
    assert body.gt.attached_to(name) == "link6"
    assert body.gt.clearance(name) < 3e-3  # attached but still resting: not lifted yet
    lift = body.arm.get_joint_positions()
    lift[1] = max(0.0, lift[1] - 0.5)
    assert body.arm.move_joints(lift, duration=2.0)
    assert body.gt.attached_to(name) == "link6" and body.gt.clearance(name) >= LIFT_M
    assert body.gripper.open() and not body.gripper.is_holding()
    assert body.gt.attachments() == {}


def test_fingers_3mm_away_do_not_grip(body):
    box = body.around("box")
    touching = body.servo_jaw(body.cfg.arm.gripper.closed)  # where the jaws stop on the box
    assert body.fingers <= body.touching(box)
    body.servo_jaw(touching + 0.003)
    assert not body.touching(box)
    assert body.rt.grasp(body.gripper._grasp) is None and body.gt.attachments() == {}


def test_grasp_needs_the_jaws_held_open_and_a_declared_pair(body):
    """Both fingers on the box is not enough: jaws squeezed under min_opening, or a weld
    body with no grasp pair for the box, attach nothing."""
    box = body.around("box")
    jaw = body.servo_jaw(0.0)
    assert body.fingers <= body.touching(box)
    spec = body.gripper._grasp
    assert body.rt.grasp(dataclasses.replace(spec, min_opening=jaw + 1e-3)) is None
    wrist = dataclasses.replace(spec, weld_body=body.sc.body("link5"))  # no (link5, box) pair
    assert body.rt.grasp(wrist) is None and body.rt.attachments() == {}
    assert body.rt.grasp(spec) == box
    assert body.rt.release(body.sc.body("link5")) == [] and body.rt.attachments() == {box: body.hand}
    assert body.rt.release(body.hand) == [] and body.rt.attachments() == {}


def test_stop_preempts_a_move(body):
    out: dict = {}
    t = threading.Thread(target=lambda: out.update(ok=body.arm.move_joints([0.8, 1.6, -1.0, 0.3, 0.5, 0.5], 3.0),
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


def test_a_capture_is_metric_and_masks_the_arm_and_nothing_else(body):
    sc, cam = body.sc, body.cfg.camera
    f = body.camera.capture()  # caller-stepped: the segmentation below is the same instant
    seg = body.rt.render(cam.name, cam.width, cam.height, ("seg",)).seg
    assert not hasattr(f, "seg") and f.rgb.shape == (cam.height, cam.width, 3)
    robot = sorted(sc.subtree(sc.robot_root))
    assert f.self_mask.any() and np.array_equal(f.self_mask, np.isin(seg, robot))
    for name in ("box", "cylinder", "table"):  # visible, and never the robot's
        assert (seg == sc.body(name)).any() and not f.self_mask[seg == sc.body(name)].any()
    # metric depth at a known surface: the table top (z 0.10) through one pixel's centre
    K, R, c = f.K, f.cam_mat, f.cam_pos
    p = R.T @ (np.array([0.40, 0.0, 0.10]) - c)  # a table point between the objects
    col, row = int(K[0, 0] * p[0] / -p[2] + K[0, 2]), int(K[1, 2] - K[1, 1] * p[1] / -p[2])
    assert seg[row, col] == sc.body("table")
    ray = R @ np.array([(col + 0.5 - K[0, 2]) / K[0, 0], -(row + 0.5 - K[1, 2]) / K[1, 1], -1.0])
    expected = (0.10 - c[2]) / ray[2]  # the ray's camera-z component is -1: t is the z-depth
    assert f.depth[row, col] == pytest.approx(expected, abs=2e-3)
