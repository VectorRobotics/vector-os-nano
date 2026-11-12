# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The generic Arm / Gripper / Camera (sim/drivers) on go2_piper's runtime.

Runs a single MuJoCoGo2+Piper sim for the whole module (module-scope fixture). All
tests are read-only or perform small arm motions; object positions in the shared scene
are NOT touched.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from vector_os_nano.embodiments import registry  # noqa: E402
from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2  # noqa: E402
from vector_os_nano.interfaces import ArmProtocol, CameraProtocol, GripperProtocol  # noqa: E402
from vector_os_nano.sim.drivers.arm import Arm  # noqa: E402
from vector_os_nano.sim.drivers.camera import Camera  # noqa: E402
from vector_os_nano.sim.drivers.gripper import Gripper  # noqa: E402


@pytest.fixture(scope="module")
def sim():
    """Module-scope Go2+Piper sim, headless (the arm is selected by with_arm)."""
    cfg = registry.load("go2_piper")
    go2 = MuJoCoGo2(gui=False, room=True, backend="mpc", with_arm=True)
    try:
        go2.connect()
        go2.stand()  # settled: IK and a later fk must see the same standing body
        rt = go2.runtime
        yield {"go2": go2, "rt": rt, "cfg": cfg,
               "piper": Arm(cfg.arm, rt, wait_mount=go2.wait_settled),
               "gripper": Gripper(cfg.arm.gripper, rt), "camera": Camera(cfg.camera, rt)}
    finally:
        go2.disconnect()


def test_the_drivers_conform_and_keep_the_world_private(sim):
    assert isinstance(sim["piper"], ArmProtocol) and isinstance(sim["gripper"], GripperProtocol)
    assert isinstance(sim["camera"], CameraProtocol)
    for dev in ("piper", "gripper", "camera"):
        assert not any(hasattr(sim[dev], a) for a in ("runtime", "world", "gt", "get_object_positions"))


def test_dof_and_joint_names(sim):
    assert sim["piper"].dof == 6
    assert sim["piper"].joint_names == [f"piper_joint{i}" for i in range(1, 7)]


def test_get_joint_positions_returns_six_floats(sim):
    qs = sim["piper"].get_joint_positions()
    assert len(qs) == 6 and all(isinstance(q, float) for q in qs)


def test_kinematics_run_on_a_private_copy(sim):
    """IK runs on its own MjData of the canonical model, never the data the stepper steps."""
    kin = sim["piper"]._kin
    assert kin._d is not sim["go2"]._mj.data and kin._m is sim["rt"].world.scene.model


# -- FK -------------------------------------------------------------------------------


def test_fk_at_zeros_returns_pose(sim):
    pos, rot = sim["piper"].fk([0.0] * 6)
    # the dog at (10, 3) stands; the Piper base on its back, the arm upright: EE above it
    assert 9.9 < pos[0] < 10.5 and 2.9 < pos[1] < 3.1 and 0.45 < pos[2] < 0.75
    for row in np.array(rot):
        assert abs(np.linalg.norm(row) - 1.0) < 1e-5


def test_fk_different_configs_differ(sim):
    pos0, _ = sim["piper"].fk([0.0] * 6)
    pos1, _ = sim["piper"].fk([0.5, 1.0, -0.5, 0.0, 0.0, 0.0])
    assert np.linalg.norm(np.array(pos0) - np.array(pos1)) > 0.1


# -- IK -------------------------------------------------------------------------------


def test_ik_top_down_reachable_below_base(sim):
    piper = sim["piper"]
    target = (10.40, 3.00, 0.30)  # 40 cm ahead, below the Piper base
    q = piper.ik_top_down(target)
    assert q is not None, "IK must converge for an easy target"
    pos, rot = piper.fk(q)
    assert np.linalg.norm(np.array(pos) - np.array(target)) < 5e-3
    assert np.array(rot)[2, 2] < -0.95  # finger axis (tool z) points down


def test_ik_top_down_unreachable_returns_none(sim):
    assert sim["piper"].ik_top_down((11.9, 3.0, 0.30)) is None  # 2 m ahead


def test_ik_without_orientation_is_more_permissive(sim):
    assert sim["piper"].ik((10.35, 3.00, 0.40)) is not None


def test_ik_seeds_include_the_live_arm_and_respect_the_limits(sim, monkeypatch):
    piper = sim["piper"]
    seen: list[list[float]] = []
    monkeypatch.setattr(piper._kin, "ik_top_down", lambda q, t, seeds: seen.extend(seeds))
    piper.ik_top_down((10.5, 3.0, 0.3), current_joints=[0.2] * 6)
    assert seen[0] == [0.2] * 6 and seen[1] == pytest.approx(piper.get_joint_positions(), abs=1e-3)
    assert seen[2:] == [list(s) for s in sim["cfg"].arm.ik_seeds]
    lo, hi = piper._kin.range[:, 0], piper._kin.range[:, 1]
    for s in sim["cfg"].arm.ik_seeds:
        assert np.all(np.asarray(s) >= lo) and np.all(np.asarray(s) <= hi), s


# -- motion ---------------------------------------------------------------------------


def test_move_joints_wrong_length_raises(sim):
    with pytest.raises(ValueError):
        sim["piper"].move_joints([0.0] * 5, duration=0.1)


def test_move_joints_writes_the_set_points(sim):
    piper, rt = sim["piper"], sim["rt"]
    target = [0.1, 0.2, -0.3, 0.0, 0.0, 0.0]
    piper.move_joints(target, duration=0.1)  # only the set-points matter here
    joints = list(piper._joints)
    np.testing.assert_allclose(rt.call(lambda w: w.targets(joints)), target, atol=1e-12)


def test_stop_holds_current_position(sim):
    """stop() writes the joint positions of its own step boundary as the set-points:
    read at that same boundary (the stepper parked), they are equal bit for bit."""
    piper, rt = sim["piper"], sim["rt"]
    piper.move_joints([0.3, 0.6, -0.5, 0.0, 0.2, 0.0], duration=0.2)  # still settling
    joints, qadr = list(piper._joints), list(piper._qadr)
    with rt.paused():
        t = rt.sim_time
        piper.stop()
        ctrl, q, t_read = rt.call(lambda w: (w.targets(joints), w.state()[0][qadr], w.time()))
    assert t_read == t  # no tick between stop() and the read
    assert np.array_equal(ctrl, q)


# -- gripper --------------------------------------------------------------------------


def test_gripper_open_close(sim):
    g = sim["gripper"]
    g.open()
    time.sleep(0.8)
    assert g.get_position() > 0.8
    g.close()
    time.sleep(0.8)
    assert g.get_position() < 0.1
    g.open()
    time.sleep(0.5)


def test_gripper_is_holding_empty(sim):
    """With nothing between the jaws, is_holding is False open and closed."""
    g = sim["gripper"]
    g.open()
    assert g.is_holding() is False
    g.close()
    assert g.is_holding() is False
    g.open()


# -- camera: the self mask is the robot's own bodies; no segmentation crosses ---------


def test_capture_masks_the_robot_by_body_id_and_carries_no_segmentation(sim):
    rt, cam = sim["rt"], sim["camera"]
    scene = rt.world.scene
    with rt.paused():  # one instant for the frame and the world-side segmentation
        f = cam.capture()
        seg = rt.render("d435_rgb", 320, 240, ("seg",)).seg
    assert not hasattr(f, "seg") and f.rgb.shape == (240, 320, 3) and f.depth.shape == (240, 320)
    robot = sorted(scene.subtree(scene.robot_root))
    assert np.array_equal(f.self_mask, np.isin(seg, robot))
    # the gripper's fingers are robot bodies, whatever their names; a pickable is not
    assert scene.body("piper_link7") in robot and scene.body("base_link") in robot
    assert scene.body("pickable_bottle_green") not in robot
    valid = f.depth[f.depth > 0]
    assert valid.size and valid.min() >= 0.1 and valid.max() <= 10.0  # the manifest's depth range
