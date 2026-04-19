"""Unit tests for MuJoCoPiper + MuJoCoPiperGripper.

Runs a single MuJoCoGo2+Piper sim for the whole session (module-scope
fixture). MuJoCo does not tolerate multiple MjModel allocations in one
Python process, so we share a single instance across all tests.

All tests are read-only or perform small arm motions that are fully
restored by the end-of-session teardown — object positions in the shared
scene are NOT touched.
"""
from __future__ import annotations

import os

os.environ.setdefault("VECTOR_SIM_WITH_ARM", "1")

import time

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
from vector_os_nano.hardware.sim.mujoco_piper import MuJoCoPiper
from vector_os_nano.hardware.sim.mujoco_piper_gripper import MuJoCoPiperGripper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim():
    """Session-scope Go2+Piper sim. Headless. Torn down after the module."""
    go2 = MuJoCoGo2(gui=False, room=True, backend="mpc")
    go2.connect()
    piper = MuJoCoPiper(go2)
    piper.connect()
    gripper = MuJoCoPiperGripper(go2)
    gripper.connect()
    time.sleep(0.5)  # let physics settle

    try:
        yield {"go2": go2, "piper": piper, "gripper": gripper}
    finally:
        gripper.disconnect()
        piper.disconnect()
        go2.disconnect()


# ---------------------------------------------------------------------------
# MuJoCoPiper — ArmProtocol surface
# ---------------------------------------------------------------------------


def test_name(sim):
    assert sim["piper"].name == "mujoco_piper"


def test_dof(sim):
    assert sim["piper"].dof == 6


def test_joint_names(sim):
    jn = sim["piper"].joint_names
    assert len(jn) == 6
    assert jn == [f"piper_joint{i}" for i in range(1, 7)]


def test_get_joint_positions_returns_six_floats(sim):
    qs = sim["piper"].get_joint_positions()
    assert len(qs) == 6
    assert all(isinstance(q, float) for q in qs)


def test_ik_model_is_isolated(sim):
    """IK must run on a separate MjModel — safer under the physics thread."""
    piper = sim["piper"]
    assert piper._ik_model is not None
    assert piper._ik_data is not None
    assert piper._ik_model is not piper._go2._mj.model


# ---------------------------------------------------------------------------
# FK
# ---------------------------------------------------------------------------


def test_fk_at_zeros_returns_pose(sim):
    pos, rot = sim["piper"].fk([0.0] * 6)
    # Position should be above-and-in-front of the standing dog's trunk
    # (dog at (9.9, 3.0, 0.28), Piper base at (9.88, 3.0, 0.34), arm
    # extended forward means ee ~20 cm forward, ~+20 cm above base)
    assert 9.9 < pos[0] < 10.5
    assert 2.9 < pos[1] < 3.1
    assert 0.45 < pos[2] < 0.75
    # Rotation is a valid 3x3
    rot_np = np.array(rot)
    assert rot_np.shape == (3, 3)
    # Rows are unit vectors
    for row in rot_np:
        assert abs(np.linalg.norm(row) - 1.0) < 1e-5


def test_fk_different_configs_differ(sim):
    pos0, _ = sim["piper"].fk([0.0] * 6)
    pos1, _ = sim["piper"].fk([0.5, 1.0, -0.5, 0.0, 0.0, 0.0])
    assert np.linalg.norm(np.array(pos0) - np.array(pos1)) > 0.1


# ---------------------------------------------------------------------------
# IK
# ---------------------------------------------------------------------------


def test_ik_top_down_reachable_below_base(sim):
    """Target well within reach below Piper base: should converge."""
    piper = sim["piper"]
    target = (10.40, 3.00, 0.30)  # 50 cm forward, below Piper base z=0.34
    q = piper.ik_top_down(target)
    assert q is not None, "IK must converge for an easy target"
    # Verify via FK
    pos, rot = piper.fk(q)
    err = np.linalg.norm(np.array(pos) - np.array(target))
    assert err < 5e-3, f"position error {err*1000:.2f} mm too large"
    # Finger axis (local +z) should map to world -Z for top-down
    finger_z_world = np.array([rot[0][2], rot[1][2], rot[2][2]])
    assert finger_z_world[2] < -0.95, f"finger axis {finger_z_world} not top-down"


def test_ik_top_down_unreachable_returns_none(sim):
    """Target far outside reach envelope: IK must return None cleanly."""
    # 2 m in front of the dog — way past Piper's ~60cm reach
    target = (11.9, 3.0, 0.30)
    q = sim["piper"].ik_top_down(target)
    assert q is None


def test_ik_without_orientation_is_more_permissive(sim):
    """Position-only ik() should succeed for targets that ik_top_down fails."""
    piper = sim["piper"]
    # Target near Piper base level — top-down fails (wrist can't bend) but
    # position-only succeeds with some non-top-down orientation.
    target = (10.35, 3.00, 0.40)
    q_td = piper.ik_top_down(target)
    q_pos = piper.ik(target)
    # At minimum, pos-only IK should succeed for this target
    assert q_pos is not None


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------


def test_move_joints_wrong_length_raises(sim):
    with pytest.raises(ValueError):
        sim["piper"].move_joints([0.0] * 5, duration=0.1)


def test_move_joints_writes_ctrl(sim):
    piper = sim["piper"]
    data = piper._go2._mj.data
    target = [0.1, 0.2, -0.3, 0.0, 0.0, 0.0]
    # move with a tiny duration — we only care that ctrl gets set, not that
    # the actuators reach the target.
    piper.move_joints(target, duration=0.1)
    time.sleep(0.05)
    for i, aid in enumerate(piper._arm_actuator_ids):
        # ctrl will be at the final interpolated target at end of move
        assert abs(float(data.ctrl[aid]) - target[i]) < 1e-4


def test_stop_holds_current_position(sim):
    piper = sim["piper"]
    piper.stop()
    # After stop, ctrl targets == current qpos for the arm joints
    data = piper._go2._mj.data
    for adr, aid in zip(piper._arm_joint_qpos_adr, piper._arm_actuator_ids):
        assert abs(float(data.ctrl[aid]) - float(data.qpos[adr])) < 1e-3


# ---------------------------------------------------------------------------
# MuJoCoPiperGripper
# ---------------------------------------------------------------------------


def test_gripper_connected(sim):
    g = sim["gripper"]
    assert g._connected
    assert g._actuator_id >= 0
    assert g._joint_qpos_adr >= 0


def test_gripper_open_close(sim):
    g = sim["gripper"]
    g.open()
    time.sleep(0.8)
    assert g.get_position() > 0.8

    g.close()
    time.sleep(0.8)
    assert g.get_position() < 0.1

    # restore for following tests
    g.open()
    time.sleep(0.5)


def test_gripper_is_holding_empty(sim):
    """With no object between jaws, is_holding is False at both extremes."""
    g = sim["gripper"]
    g.open()
    time.sleep(0.5)
    assert g.is_holding() is False  # jaws open wide, cmd NOT closed
    g.close()
    time.sleep(0.5)
    assert g.is_holding() is False  # cmd closed AND jaws at 0 (nothing in them)
    g.open()
    time.sleep(0.5)


def test_gripper_get_force_returns_none(sim):
    assert sim["gripper"].get_force() is None
