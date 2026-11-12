# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The generic Arm + Gripper on go2_piper, the dog seated at the pick table (one sim per module).

* a grasp is a contact-gated weld: an object beside the jaws (inside the old 6 cm
  "magnet" radius) is never welded; one between them is, and lifts;
* is_holding(name) is target-aware and a release is verified;
* move_joints reports tracking: blocked by the table -> False;
* stop() / an abort preempt an in-flight move within one set-point period;
* the robot's own bodies are never scene objects.
"""
from __future__ import annotations

import math
import threading

import numpy as np
import pytest

pytest.importorskip("mujoco")

from vector_os_nano.embodiments import registry
from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
from vector_os_nano.sim.drivers.arm import Arm
from vector_os_nano.sim.drivers.gripper import Gripper
from vector_os_nano.sim.ground_truth import GroundTruth
from vector_os_nano.vcli.cognitive import abort

_SEAT = (10.46, 3.0)  # dog body 42 cm behind the green bottle, head-on
_GREEN = "pickable_bottle_green"


@pytest.fixture(scope="module")
def rig():
    cfg = registry.load("go2_piper")
    go2 = MuJoCoGo2(gui=False, room=True, backend="auto", with_arm=True)
    try:
        go2.connect()
        go2.stand()
        arm = Arm(cfg.arm, go2.runtime, abort=abort.is_abort_requested, wait_mount=go2.wait_settled)
        grip = Gripper(cfg.arm.gripper, go2.runtime)
        rq = go2._mj.layout.root_qpos_adr
        go2.runtime.call(lambda w: w.data.qpos.__setitem__(slice(rq, rq + 2), _SEAT))
        go2.reset_pose()
        go2.wait_sim(1.5)
        yield go2, arm, grip
    finally:
        go2.disconnect()


def _restore_objects(w) -> None:
    """Every pickable back to its compiled rest pose, at rest."""
    import mujoco as mj

    _m, d = w.model, w.data

    for j in range(_m.njnt):
        if _m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE and mj.mj_id2name(
                _m, mj.mjtObj.mjOBJ_BODY, _m.jnt_bodyid[j]).startswith("pickable_"):
            a, v = _m.jnt_qposadr[j], _m.jnt_dofadr[j]
            d.qpos[a:a + 7] = _m.qpos0[a:a + 7]
            d.qvel[v:v + 6] = 0.0


@pytest.fixture()
def seated(rig):
    """Each test starts with the jaws open, nothing welded, the arm at home, objects at rest."""
    go2, arm, grip = rig
    abort.clear_abort()
    assert grip.open()
    arm.home(duration=1.5)
    go2.runtime.call(_restore_objects)
    go2.wait_sim(0.3)
    yield rig
    abort.clear_abort()


def _top_down_to(arm, xyz, duration=3.0) -> bool:
    q = arm.ik_top_down(xyz)
    assert q is not None, f"IK unreachable {xyz}"
    return arm.move_joints(q, duration=duration)


def _objects(go2) -> dict[str, list[float]]:
    """Where the scene's objects are: ground truth, read by the test, never by a driver."""
    return GroundTruth(go2.runtime).object_positions()


def test_the_robot_is_not_a_scene_object(seated) -> None:
    go2, _arm, _grip = seated
    objs = _objects(go2)
    assert _GREEN in objs and "base_link" not in objs
    assert not any(n.startswith(("piper_", "base_")) for n in objs)


def test_an_object_beside_the_jaws_is_never_welded(seated) -> None:
    """EE 5 cm from the bottle centre (inside the old 6 cm magnet), jaws beside it."""
    go2, arm, grip = seated
    g = _objects(go2)[_GREEN]
    beside = (g[0] - 0.05, g[1], g[2])
    assert _top_down_to(arm, (beside[0], beside[1], beside[2] + 0.02), 4.0)
    assert _top_down_to(arm, beside, 2.0)
    ee, _ = arm.fk(arm.get_joint_positions())
    assert math.dist(ee, g) < 0.06  # the magnet would have welded this
    assert grip.close()
    assert grip.is_holding() is False
    assert go2.runtime.attachments() == {}


def test_an_object_between_the_jaws_is_gripped_lifted_and_released(seated) -> None:
    go2, arm, grip = seated
    g = _objects(go2)[_GREEN]
    assert _top_down_to(arm, (g[0], g[1], g[2] + 0.02), 4.0)
    assert _top_down_to(arm, tuple(g), 2.0)
    assert grip.close()
    assert grip.is_holding() and grip.is_holding(_GREEN)
    assert not grip.is_holding("pickable_can_red")  # target-aware
    sc = go2.runtime.world.scene
    assert go2.runtime.attachments() == {sc.body(_GREEN): sc.body("piper_link6")}
    lift = arm.get_joint_positions()
    lift[1] = max(0.0, lift[1] - 0.5)
    assert arm.move_joints(lift, duration=2.0)
    assert _objects(go2)[_GREEN][2] > g[2] + 0.03
    assert grip.open()
    assert grip.is_holding() is False and go2.runtime.attachments() == {}


def test_a_failed_release_is_reported_and_keeps_the_hold(seated, monkeypatch) -> None:
    go2, arm, grip = seated
    g = _objects(go2)[_GREEN]
    assert _top_down_to(arm, (g[0], g[1], g[2] + 0.02), 4.0) and _top_down_to(arm, tuple(g), 2.0)
    assert grip.close() and grip.is_holding(_GREEN)

    def _refuse(_fn):
        raise RuntimeError("sim: runtime closed")

    monkeypatch.setattr(go2.runtime, "call", _refuse)
    assert grip.open() is False  # reported, and the held state is not dropped
    monkeypatch.undo()
    assert grip.is_holding(_GREEN)  # still welded, still known
    assert grip.open() and not grip.is_holding()


def test_a_move_blocked_by_the_table_returns_false(seated) -> None:
    _go2, arm, _grip = seated
    gap = (10.85, 3.11)  # table top between the green bottle and the red can
    assert _top_down_to(arm, (*gap, 0.36), 4.0), "free-air move must track"
    q_in = arm.ik_top_down((*gap, 0.22))  # 6 cm below the table top
    assert q_in is not None
    assert arm.move_joints(q_in, duration=2.0) is False
    err = max(abs(a - b) for a, b in zip(arm.get_joint_positions(), q_in))
    assert err > arm._spec.track_tol


def _move_in_thread(arm, rt, q, duration):
    out: dict = {}

    def run() -> None:
        out["ok"] = arm.move_joints(q, duration=duration)
        out["t"] = rt.sim_time

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, out


def test_stop_preempts_a_move_within_50_sim_ms(seated) -> None:
    go2, arm, _grip = seated
    rt = go2.runtime
    t, out = _move_in_thread(arm, rt, [0.8, 1.2, -1.0, 0.3, 0.5, 0.5], 3.0)
    go2.wait_sim(0.5)
    t_stop = rt.sim_time
    q_stop = np.array(arm.get_joint_positions())
    arm.stop()
    t.join(timeout=10.0)
    assert out["ok"] is False
    assert out["t"] - t_stop <= 0.05, f"move returned {out['t'] - t_stop:.3f} sim-s after stop()"
    joints = list(arm._joints)
    ctrl = rt.call(lambda w: w.targets(joints))
    go2.wait_sim(0.3)
    np.testing.assert_allclose(rt.call(lambda w: w.targets(joints)), ctrl)  # nothing written after the hold
    assert np.max(np.abs(np.array(arm.get_joint_positions()) - q_stop)) < 0.1


def test_an_abort_preempts_a_move(seated) -> None:
    go2, arm, _grip = seated
    t, out = _move_in_thread(arm, go2.runtime, [0.8, 1.2, -1.0, 0.3, 0.5, 0.5], 3.0)
    go2.wait_sim(0.3)
    t_abort = go2.runtime.sim_time
    abort.request_abort()
    t.join(timeout=10.0)
    assert out["ok"] is False and out["t"] - t_abort <= 0.05


def test_ik_right_after_a_walk_solves_for_the_stopped_body(seated) -> None:
    """A walk returns while its stop still brakes the trunk (MPC trot to zero, then the
    stance ramp: ~3 cm lower, ~1.5 deg less pitch, ~1.4 sim-s). IK solved in that
    transient planned for a body that was gone when the arm arrived: E3a's approach
    move dragged the fingers into the trunk, 0/3 (INT1)."""
    go2, arm, _grip = seated
    if not go2._use_mpc:
        pytest.skip("the stop brake is the MPC gait's")
    try:
        go2.walk(0.0, 0.0, 0.4, 0.6)  # turn in place a little, then stop
        x, y, _ = go2.get_position()
        h = go2.get_heading()
        target = (x + 0.38 * math.cos(h), y + 0.38 * math.sin(h), 0.34)
        q = arm.ik_top_down(target)
        assert q is not None
        go2.wait_sim(2.0)  # whatever is left of the brake
        pos, _ = arm.fk(q)
        assert math.dist(pos, target) < 0.01, pos
    finally:  # back to the seat for the module's other tests
        rq = go2._mj.layout.root_qpos_adr
        go2.runtime.call(lambda w: w.data.qpos.__setitem__(slice(rq, rq + 7), [*_SEAT, 0.3, 1.0, 0.0, 0.0, 0.0]))
        go2.reset_pose()
        go2.wait_sim(1.5)
