# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Piper arm + gripper physics, the dog seated at the pick table (one sim per module).

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

from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
from vector_os_nano.hardware.sim.mujoco_piper import MuJoCoPiper
from vector_os_nano.hardware.sim.mujoco_piper_gripper import MuJoCoPiperGripper
from vector_os_nano.vcli.cognitive import abort

_SEAT = (10.46, 3.0)  # dog body 42 cm behind the green bottle, head-on
_GREEN = "pickable_bottle_green"


@pytest.fixture(scope="module")
def rig():
    go2 = MuJoCoGo2(gui=False, room=True, backend="auto", with_arm=True)
    arm = grip = None
    try:
        go2.connect()
        go2.stand()
        arm, grip = MuJoCoPiper(go2), MuJoCoPiperGripper(go2)
        arm.connect()
        grip.connect()
        rq = go2._mj.layout.root_qpos_adr
        go2.runtime.call(lambda _m, d: d.qpos.__setitem__(slice(rq, rq + 2), _SEAT))
        go2.reset_pose()
        go2.wait_sim(1.5)
        yield go2, arm, grip
    finally:
        for dev in (grip, arm):
            if dev is not None:
                dev.disconnect()
        go2.disconnect()


def _restore_objects(_m, d) -> None:
    """Every pickable back to its compiled rest pose, at rest."""
    import mujoco as mj

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


def test_the_robot_is_not_a_scene_object(seated) -> None:
    _go2, arm, _grip = seated
    objs = arm.get_object_positions()
    assert _GREEN in objs and "base_link" not in objs
    assert not any(n.startswith(("piper_", "base_")) for n in objs)


def test_an_object_beside_the_jaws_is_never_welded(seated) -> None:
    """EE 5 cm from the bottle centre (inside the old 6 cm magnet), jaws beside it."""
    _go2, arm, grip = seated
    g = arm.get_object_positions()[_GREEN]
    beside = (g[0] - 0.05, g[1], g[2])
    assert _top_down_to(arm, (beside[0], beside[1], beside[2] + 0.02), 4.0)
    assert _top_down_to(arm, beside, 2.0)
    ee, _ = arm.fk(arm.get_joint_positions())
    assert math.dist(ee, g) < 0.06  # the magnet would have welded this
    assert grip.close()
    assert grip.is_holding() is False
    assert not any(grip.weld_is_active().values())


def test_an_object_between_the_jaws_is_gripped_lifted_and_released(seated) -> None:
    _go2, arm, grip = seated
    g = arm.get_object_positions()[_GREEN]
    assert _top_down_to(arm, (g[0], g[1], g[2] + 0.02), 4.0)
    assert _top_down_to(arm, tuple(g), 2.0)
    assert grip.close()
    assert grip.is_holding() and grip.is_holding(_GREEN)
    assert not grip.is_holding("pickable_can_red")  # target-aware
    assert grip.weld_is_active()[_GREEN]
    lift = arm.get_joint_positions()
    lift[1] = max(0.0, lift[1] - 0.5)
    assert arm.move_joints(lift, duration=2.0)
    assert arm.get_object_positions()[_GREEN][2] > g[2] + 0.03
    assert grip.open()
    assert grip.is_holding() is False and not any(grip.weld_is_active().values())


def test_a_move_blocked_by_the_table_returns_false(seated) -> None:
    _go2, arm, _grip = seated
    gap = (10.85, 3.11)  # table top between the green bottle and the red can
    assert _top_down_to(arm, (*gap, 0.36), 4.0), "free-air move must track"
    q_in = arm.ik_top_down((*gap, 0.22))  # 6 cm below the table top
    assert q_in is not None
    assert arm.move_joints(q_in, duration=2.0) is False
    err = max(abs(a - b) for a, b in zip(arm.get_joint_positions(), q_in))
    assert err > arm._spec.track_tol


def _move_in_thread(arm, q, duration):
    out: dict = {}

    def run() -> None:
        out["ok"] = arm.move_joints(q, duration=duration)
        out["t"] = arm._go2.runtime.sim_time

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, out


def test_stop_preempts_a_move_within_50_sim_ms(seated) -> None:
    go2, arm, _grip = seated
    rt = go2.runtime
    t, out = _move_in_thread(arm, [0.8, 1.2, -1.0, 0.3, 0.5, 0.5], 3.0)
    go2.wait_sim(0.5)
    t_stop = rt.sim_time
    q_stop = np.array(arm.get_joint_positions())
    arm.stop()
    t.join(timeout=10.0)
    assert out["ok"] is False
    assert out["t"] - t_stop <= 0.05, f"move returned {out['t'] - t_stop:.3f} sim-s after stop()"
    ids = arm._arm_actuator_ids
    ctrl = rt.call(lambda _m, d: d.ctrl[ids].copy())
    go2.wait_sim(0.3)
    np.testing.assert_allclose(rt.call(lambda _m, d: d.ctrl[ids].copy()), ctrl)  # nothing written after the hold
    assert np.max(np.abs(np.array(arm.get_joint_positions()) - q_stop)) < 0.1


def test_an_abort_preempts_a_move(seated) -> None:
    go2, arm, _grip = seated
    t, out = _move_in_thread(arm, [0.8, 1.2, -1.0, 0.3, 0.5, 0.5], 3.0)
    go2.wait_sim(0.3)
    t_abort = go2.runtime.sim_time
    abort.request_abort()
    t.join(timeout=10.0)
    assert out["ok"] is False and out["t"] - t_abort <= 0.05
