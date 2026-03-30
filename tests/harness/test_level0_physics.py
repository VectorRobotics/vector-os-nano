"""Level 0 — MuJoCo physics verification.

Tests at this level use mujoco directly (no MuJoCoGo2 wrapper, no convex_mpc).
They verify the model file is sane and physics produces expected results.

Actuator ordering in go2.xml (ctrl[0..11]):
    0  FR_hip    1  FR_thigh    2  FR_calf
    3  FL_hip    4  FL_thigh    5  FL_calf
    6  RR_hip    7  RR_thigh    8  RR_calf
    9  RL_hip   10  RL_thigh   11  RL_calf

qpos layout:
    [0:3]  base position (x, y, z)
    [3:7]  base quaternion (w, x, y, z)
    [7:19] 12 joint angles (same ordering as ctrl)
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MJCF_DIR = (
    Path(__file__).resolve().parents[2]
    / "vector_os_nano" / "hardware" / "sim" / "mjcf" / "go2"
)

_SCENE_XML = """<mujoco>
  <include file="go2.xml"/>
  <worldbody>
    <light name="main" pos="0 0 5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="50 50 0.1" rgba="0.8 0.8 0.8 1"/>
  </worldbody>
</mujoco>"""


def _load_model():
    """Load go2 model with a flat ground plane. Returns (model, data)."""
    import mujoco
    import tempfile

    scene_path = _MJCF_DIR / "_test_scene.xml"
    scene_path.write_text(_SCENE_XML)
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        return model, data
    finally:
        scene_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.level0
def test_model_loads():
    """Model has 12 actuators and base_link body exists."""
    import mujoco

    model, _data = _load_model()

    assert model.nu == 12, f"Expected 12 actuators, got {model.nu}"

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    assert base_id >= 0, "base_link body not found in model"


@pytest.mark.level0
def test_joint_responds_to_torque():
    """Apply 10 Nm to FL_thigh actuator (ctrl[4]) for 100 steps; joint moves > 0.01 rad."""
    import mujoco

    model, data = _load_model()

    # Record initial FL_thigh joint position (qpos[7:19], index 4 = FL_thigh)
    mujoco.mj_forward(model, data)
    q_initial = float(data.qpos[7 + 4])  # FL_thigh

    # Apply torque only to FL_thigh; leave all others at zero
    data.ctrl[:] = 0.0
    data.ctrl[4] = 10.0  # FL_thigh

    for _ in range(100):
        mujoco.mj_step(model, data)

    q_final = float(data.qpos[7 + 4])
    displacement = abs(q_final - q_initial)

    assert not math.isnan(q_final), "FL_thigh joint position is NaN after stepping"
    assert displacement > 0.01, (
        f"FL_thigh joint did not respond to torque: "
        f"initial={q_initial:.4f} rad, final={q_final:.4f} rad, "
        f"displacement={displacement:.4f} rad (expected > 0.01)"
    )


@pytest.mark.level0
def test_gravity_pulls_robot_down():
    """With zero ctrl, the robot falls under gravity — z decreases."""
    import mujoco

    model, data = _load_model()
    mujoco.mj_forward(model, data)

    z_initial = float(data.qpos[2])

    data.ctrl[:] = 0.0
    for _ in range(500):
        mujoco.mj_step(model, data)

    z_final = float(data.qpos[2])

    assert not math.isnan(z_final), "Base z position is NaN after free fall"
    assert z_final < z_initial, (
        f"Robot did not fall under gravity: z_initial={z_initial:.4f}, z_final={z_final:.4f}"
    )


@pytest.mark.level0
def test_no_nan_in_qpos_qvel_after_torque():
    """Full state (qpos, qvel) is NaN-free after 200 steps of joint torques."""
    import mujoco

    model, data = _load_model()
    mujoco.mj_forward(model, data)

    # Apply moderate torques across all joints
    data.ctrl[:] = 5.0

    for _ in range(200):
        mujoco.mj_step(model, data)

    assert not np.any(np.isnan(data.qpos)), "NaN detected in qpos"
    assert not np.any(np.isnan(data.qvel)), "NaN detected in qvel"


@pytest.mark.level0
def test_actuator_count_and_limits():
    """All 12 actuators have non-zero torque limits in the MJCF."""
    import mujoco

    model, _data = _load_model()

    assert model.nu == 12, f"Expected 12 actuators, got {model.nu}"
    for i in range(model.nu):
        lo, hi = model.actuator_ctrlrange[i]
        assert lo < 0.0, f"Actuator {i} lower limit should be negative, got {lo}"
        assert hi > 0.0, f"Actuator {i} upper limit should be positive, got {hi}"
