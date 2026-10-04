# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Unit tests for the embodiment config schema + loader (Stage 1).

Offline + fast: pure YAML parsing, NO sim / MuJoCo / torch. Asserts that the
go2 + g1 manifests faithfully carry the constants the drivers hardcode today,
so Stage 2 can wire the generic driver to READ them with confidence.
"""
from __future__ import annotations

import math

import pytest

from vector_os_nano.embodiments import EmbodimentConfig, load_embodiment_config
from vector_os_nano.embodiments.config import (
    EmbodimentConfigError,
    parse_embodiment_config,
)


# ---------------------------------------------------------------------------
# go2 manifest — must match mujoco_go2.py constants
# ---------------------------------------------------------------------------


def test_go2_loads_and_is_frozen() -> None:
    cfg = load_embodiment_config("go2")
    assert isinstance(cfg, EmbodimentConfig)
    assert cfg.id == "go2"
    # frozen dataclass — assignment must raise (Rule 6 immutability).
    with pytest.raises(Exception):
        cfg.id = "nope"  # type: ignore[misc]


def test_go2_spawn_and_root_body() -> None:
    cfg = load_embodiment_config("go2")
    # connect(): qpos[0]=10.0, qpos[1]=3.0, qpos[2]=0.35
    assert cfg.spawn.xy == (10.0, 3.0)
    assert cfg.spawn.base_height == pytest.approx(0.35)
    # _Go2Model.base_bid = mj_name2id(..., "base_link")
    assert cfg.model.root_body == "base_link"


def test_go2_stance_matches_stand_joints() -> None:
    cfg = load_embodiment_config("go2")
    # _STAND_JOINTS = [0.0, 0.9, -1.8] * 4 (hip, thigh, calf per leg).
    assert len(cfg.stance) == 12
    for leg in ("FL", "FR", "RL", "RR"):
        assert cfg.stance[f"{leg}_hip"] == pytest.approx(0.0)
        assert cfg.stance[f"{leg}_thigh"] == pytest.approx(0.9)
        assert cfg.stance[f"{leg}_calf"] == pytest.approx(-1.8)


def test_go2_sensors() -> None:
    cfg = load_embodiment_config("go2")
    by_role = {s.role: s for s in cfg.sensors}
    # d435_rgb / d435_depth cameras on base_link.
    assert by_role["camera"].name == "d435_rgb"
    assert by_role["camera"].mount_body == "base_link"
    assert by_role["depth"].name == "d435_depth"
    # Lidar offset 0.3 m forward, 0.2 m up (_LIDAR_OFFSET_X / _LIDAR_OFFSET_Z).
    lidar = by_role["lidar"]
    assert lidar.pos == (0.3, 0.0, 0.2)
    assert lidar.params["tilt_deg"] == pytest.approx(-20.0)
    assert lidar.params["update_interval_steps"] == 200
    assert lidar.params["range_max"] == pytest.approx(12.0)


def test_go2_capabilities() -> None:
    cfg = load_embodiment_config("go2")
    cap = cfg.capabilities
    assert cap.has_base is True
    assert cap.has_arm is False
    assert cap.has_gripper is False
    assert cap.camera is True
    assert cap.lidar is True
    # supports_holonomic -> True is carried as a base motion trait in policy.spec.
    assert cfg.policy.spec["holonomic"] is True


def test_go2_policy() -> None:
    cfg = load_embodiment_config("go2")
    assert cfg.policy.ref == "convex_mpc"
    # _MPC_Z_DES = 0.27 and velocity limits _VX_MAX/_VY_MAX/_VYAW_MAX.
    assert cfg.policy.spec["mpc_z_des"] == pytest.approx(0.27)
    assert cfg.policy.spec["vx_max"] == pytest.approx(0.8)
    assert cfg.policy.spec["vy_max"] == pytest.approx(0.4)
    assert cfg.policy.spec["vyaw_max"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# g1 manifest — must match mujoco_g1.py constants
# ---------------------------------------------------------------------------


def test_g1_spawn_and_root_body() -> None:
    cfg = load_embodiment_config("g1")
    # _G1_SPAWN_X=10.0, _G1_SPAWN_Y=3.0, _G1_PELVIS_Z=0.793
    assert cfg.spawn.xy == (10.0, 3.0)
    assert cfg.spawn.base_height == pytest.approx(0.793)
    # _G1Offsets keys on model.body("g1_pelvis").
    assert cfg.model.root_body == "g1_pelvis"


def test_g1_stance_matches_default_angles() -> None:
    cfg = load_embodiment_config("g1")
    # _DEFAULT_ANGLES = [-0.1, 0, 0, 0.3, -0.2, 0] * 2 in g1_12dof joint order.
    assert len(cfg.stance) == 12
    expected = {
        "left_hip_pitch_joint": -0.1,
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.3,
        "left_ankle_pitch_joint": -0.2,
        "left_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": -0.1,
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.3,
        "right_ankle_pitch_joint": -0.2,
        "right_ankle_roll_joint": 0.0,
    }
    for name, angle in expected.items():
        assert cfg.stance[name] == pytest.approx(angle), name


def test_g1_sensors() -> None:
    cfg = load_embodiment_config("g1")
    by_role = {s.role: s for s in cfg.sensors}
    # Head RGB camera 'g1_head_rgb' on g1_pelvis, pos (0.04, 0, 0.42).
    cam = by_role["camera"]
    assert cam.name == "g1_head_rgb"
    assert cam.mount_body == "g1_pelvis"
    assert cam.pos == (0.04, 0.0, 0.42)
    # Lidar offset _LIDAR_OFFSET_X=0.0, _LIDAR_OFFSET_Z=0.72.
    assert by_role["lidar"].pos == (0.0, 0.0, 0.72)


def test_g1_policy_and_capabilities() -> None:
    cfg = load_embodiment_config("g1")
    assert cfg.policy.ref == "assets/g1_gait/motion.pt"
    # 47-obs / 12-action / 50 Hz (decimation 10 @ 0.002 s) / gait 0.8 s.
    assert cfg.policy.spec["num_obs"] == 47
    assert cfg.policy.spec["num_actions"] == 12
    assert cfg.policy.spec["decimation"] == 10
    assert cfg.policy.spec["sim_dt"] == pytest.approx(0.002)
    assert cfg.policy.spec["gait_period"] == pytest.approx(0.8)
    assert cfg.nav["body_radius"] == pytest.approx(0.30)
    assert cfg.policy.spec["holonomic"] is False
    cap = cfg.capabilities
    assert cap.has_base is True
    assert cap.has_arm is False
    assert cap.camera is True
    assert cap.lidar is True


# ---------------------------------------------------------------------------
# Loader fail-loud behavior (Rule 8)
# ---------------------------------------------------------------------------


def test_missing_embodiment_raises_clear_error() -> None:
    with pytest.raises(EmbodimentConfigError) as exc:
        load_embodiment_config("does_not_exist")
    assert "does_not_exist" in str(exc.value)
    assert "robot.yaml" in str(exc.value)


def test_missing_required_field_raises_clear_error() -> None:
    # A manifest missing 'spawn' must fail loud, naming the field.
    raw = {
        "id": "broken",
        "display_name": "Broken",
        "model": {"path": "x.xml", "root_body": "base"},
        # no 'spawn'
        "stance": {"j": 0.0},
        "policy": {"ref": "p"},
        "capabilities": {},
    }
    with pytest.raises(EmbodimentConfigError) as exc:
        parse_embodiment_config(raw, ctx="broken/robot.yaml")
    assert "spawn" in str(exc.value)


def test_both_embodiments_distinct() -> None:
    go2 = load_embodiment_config("go2")
    g1 = load_embodiment_config("g1")
    assert go2.id != g1.id
    assert go2.model.root_body != g1.model.root_body
    # g1 spawn height clearly higher than go2 (biped vs quadruped).
    assert g1.spawn.base_height > go2.spawn.base_height
    assert not math.isnan(g1.spawn.base_height)
