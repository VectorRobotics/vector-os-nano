# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Unit tests for the embodiment config schema + loader.

Offline + fast: pure YAML parsing, NO sim / MuJoCo / torch. The go2 manifests carry
every constant the generic LeggedBase and its policies read (the ``base:`` block and
the lidar); the g1 manifest the constants its driver hardcodes until S4b.
"""
from __future__ import annotations

import math

import pytest

from vector_os_nano.schemas.robot import EmbodimentConfig, load_embodiment_config
from vector_os_nano.embodiments.registry import validate
from vector_os_nano.schemas.robot import (
    EmbodimentConfigError,
    parse_embodiment_config,
)


# ---------------------------------------------------------------------------
# go2 manifest — the base block the generic LeggedBase reads
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
    # the entry hall; a legged base stands there at its base.height (no drop height)
    assert not hasattr(cfg, "spawn")
    assert cfg.model.root_body == "base_link"


def test_robot_cannot_override_its_world_spawn():
    raw = _legged()
    raw['spawn'] = {'xy': [0, 0], 'heading': 0}
    with pytest.raises(EmbodimentConfigError, match="unknown key 'spawn'"):
        parse_embodiment_config(raw)


@pytest.mark.parametrize('schema', [None, True, 0, 2])
def test_robot_schema_version_is_required(schema):
    raw = _legged()
    if schema is None:
        del raw['schema']
    else:
        raw['schema'] = schema
    with pytest.raises(EmbodimentConfigError, match='schema') as exc:
        parse_embodiment_config(raw, ctx='variant/robot.yaml')
    assert 'variant/robot.yaml' in str(exc.value)
    if schema == 2:
        assert 'upgrade vector_os_nano' in str(exc.value)


def test_capabilities_follow_components_and_cannot_be_authored():
    raw = _legged()
    raw['capabilities'] = {'has_base': False}
    with pytest.raises(EmbodimentConfigError, match="unknown key 'capabilities'"):
        parse_embodiment_config(raw)
    del raw['capabilities'], raw['base'], raw['sensors']
    cfg = parse_embodiment_config(raw)
    assert not any(vars(cfg.capabilities).values())


def test_duplicate_robot_keys_fail_before_loading(tmp_path, monkeypatch):
    from vector_os_nano.schemas import robot

    folder = tmp_path / 'duplicate'
    folder.mkdir()
    (folder / 'robot.yaml').write_text('schema: 1\nschema: 1\n')
    monkeypatch.setattr(robot, '_EMBODIMENTS_DIR', tmp_path)
    with pytest.raises(EmbodimentConfigError, match='duplicate key schema'):
        robot.load_embodiment_config('duplicate')


def test_go2_base_block() -> None:
    b = load_embodiment_config("go2").base
    legs = [f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR") for part in ("hip", "thigh", "calf")]
    assert b.joints == tuple(legs)  # the order the policy sees: qpos order
    assert [b.stance[j] for j in legs] == [0.0, 0.9, -1.8] * 4
    assert [b.postures["sit"][j] for j in legs] == [0.0, 1.5, -2.5] * 4
    assert [b.postures["lie_down"][j] for j in legs] == [0.0, 2.0, -2.7] * 4
    assert (b.kp, b.kd, b.torque_scale) == (120.0, 3.5, 0.9)
    assert (b.height, b.holonomic, b.limits, b.fall_z) == (0.279, True, (0.8, 0.4, 4.0), 0.15)
    assert (b.nav.body_radius, b.nav.speed, b.nav.vyaw_max, b.nav.explorer) == (0.28, 0.4, 0.8, "tare")


def test_go2_sensors() -> None:
    cfg = load_embodiment_config("go2")
    by_role = {s.role: s for s in cfg.sensors}
    # The d435_rgb camera of go2.xml, rendered at 640x480, depth over the D435 range.
    cam = cfg.camera
    assert by_role["camera"] is cam and (cam.name, cam.width, cam.height) == ("d435_rgb", 640, 480)
    assert cam.depth == (0.1, 10.0) and cam.pos is None  # in the model, not placed by the compiler
    lidar = cfg.lidar
    assert by_role["lidar"] is lidar and (lidar.body, lidar.pos) == ("base_link", (0.3, 0.0, 0.2))
    assert lidar.elevations_deg == tuple(float(e) for e in range(-8, 53, 2))
    assert (lidar.tilt_deg, lidar.n_azimuth, lidar.range, lidar.period_s) == (-20.0, 360, 12.0, 0.2)


def test_go2_capabilities() -> None:
    cap = load_embodiment_config("go2").capabilities
    assert (cap.has_base, cap.has_arm, cap.has_gripper, cap.camera, cap.lidar) == (True, False, False, True, True)


def test_go2_policy() -> None:
    p = load_embodiment_config("go2").base.policy
    assert (p.ref, p.timestep, p.tick_hz) == ("convex_mpc", 0.001, 200.0)
    assert p.params == {"file": "model/go2_description.urdf", "gait_hz": 3, "duty": 0.6, "z_des": 0.27, "brake_s": 0.35, "brake_ramp_s": 1.0}
    assert p.on("mujoco") == ("convex_mpc", p.params)
    assert p.on("pybullet") == ("sinusoidal", {"freq_hz": 2.0, "thigh_amp": 0.25, "calf_amp": 0.25, "hip_amp": 0.10})


def _legged(**base) -> dict:
    """A minimal legged manifest whose base block is the go2's, overridden by *base*."""
    import yaml

    from vector_os_nano.schemas.robot import _EMBODIMENTS_DIR

    raw = yaml.safe_load((_EMBODIMENTS_DIR / "go2" / "robot.yaml").read_text())
    raw["base"].update(base)
    return raw


@pytest.mark.parametrize("change, error", [
    ({"stance": {"FL_hip_joint": 0.0}}, r"base\.stance: no angle for \['FL_thigh_joint'"),
    ({"stance_typo": 1}, r"base: unknown key 'stance_typo'"),
    ({"postures": {"stand": {}}}, r"'stand' is the stance"),
    ({"limits": {"vx": 1.0, "vy": 1.0}}, r"base\.limits: missing required field 'vyaw'"),
    ({"joints": ["FL_hip_joint", "FL_hip_joint"]}, r"each base joint once"),
    ({"policy": {"ref": "convex_mpc", "timestep": 0.001, "tick_hz": 300}}, r"tick_hz=300.0 must divide"),
    ({"policy": {"ref": "convex_mpc", "timestep": 0.001, "tick_hz": 200, "by_backend": {"bullet": {"ref": "s"}}}},
     r"unknown backend 'bullet'"),
    ({"policy": {"ref": "convex_mpc", "timestep": 0.001, "tick_hz": 200, "by_backend": {"pybullet": {}}}},
     r"by_backend\.pybullet must be a mapping with a 'ref'"),
    ({"nav": {"body_radius": 0.3, "speed": 0.4, "vyaw_max": 1.0, "fall_z": 0.1}}, r"nav: unknown key 'fall_z'"),
])
def test_a_bad_base_block_fails_naming_the_key(change: dict, error: str) -> None:
    with pytest.raises(EmbodimentConfigError, match=error):
        validate(parse_embodiment_config(_legged(**change), ctx="quad/robot.yaml"), ctx="quad/robot.yaml")


@pytest.mark.parametrize("edit, error", [
    (lambda p: p.update(ref="rl_magic"),
     r"^quad/robot\.yaml\.base\.policy\.ref: no control policy 'rl_magic'; policies: \['convex_mpc', 'sinusoidal', 'swerve', 'torchscript'\]$"),
    (lambda p: p.update(gait_hzz=3),
     r"^quad/robot\.yaml\.base\.policy: keys \[.*'gait_hzz'.*\]; convex_mpc takes exactly \['brake_ramp_s'"),
    (lambda p: p.pop("duty"), r"^quad/robot\.yaml\.base\.policy: keys .*; convex_mpc takes exactly .*'duty'"),
    (lambda p: p.update(gait_hz="fast"), r"^quad/robot\.yaml\.base\.policy: gait_hz: 'fast' is not a number$"),
    (lambda p: p["by_backend"]["pybullet"].update(ref="nope"),
     r"^quad/robot\.yaml\.base\.policy\.by_backend\.pybullet\.ref: no control policy 'nope'; policies: \["),
    (lambda p: p["by_backend"]["pybullet"].update(freq_hzz=2.0),
     r"^quad/robot\.yaml\.base\.policy\.by_backend\.pybullet: keys .*; sinusoidal takes exactly"),
])
def test_a_bad_policy_row_fails_at_load_naming_the_file_and_key(edit, error: str) -> None:
    """The policy ref (and every by_backend ref) is a row of sim.policies.POLICIES and its
    keys are exactly the row's, checked at load, not when the sim builds the body."""
    raw = _legged()
    edit(raw["base"]["policy"])
    with pytest.raises(EmbodimentConfigError, match=error):
        validate(parse_embodiment_config(raw, ctx="quad/robot.yaml"), ctx="quad/robot.yaml")


@pytest.mark.parametrize("lidar, error", [
    ({"elevations_deg": [-2, 2]}, r"must include 0"),
    ({"mount_body": "base_link"}, r"unknown key 'mount_body'"),
    ({"role": "sonar"}, r"unknown role 'sonar'"),
])
def test_a_bad_lidar_entry_fails_naming_the_key(lidar: dict, error: str) -> None:
    raw = _legged()
    raw["sensors"][1] = {**raw["sensors"][1], **lidar}
    with pytest.raises(EmbodimentConfigError, match=error):
        validate(parse_embodiment_config(raw, ctx="quad/robot.yaml"), ctx="quad/robot.yaml")


# ---------------------------------------------------------------------------
# g1 manifest — must match mujoco_g1.py constants
# ---------------------------------------------------------------------------


def test_g1_spawn_and_root_body() -> None:
    cfg = load_embodiment_config("g1")
    # _G1_SPAWN_X=10.0, _G1_SPAWN_Y=3.0, _G1_PELVIS_Z=0.793
    assert not hasattr(cfg, "spawn")
    assert cfg.base.height == pytest.approx(0.78)
    # _G1Offsets keys on model.body("g1_pelvis").
    assert cfg.model.root_body == "pelvis"


def test_g1_stance_matches_default_angles() -> None:
    cfg = load_embodiment_config("g1")
    # _DEFAULT_ANGLES = [-0.1, 0, 0, 0.3, -0.2, 0] * 2 in g1_12dof joint order.
    assert len(cfg.base.stance) == 12
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
        assert cfg.base.stance[name] == pytest.approx(angle), name


def test_g1_sensors() -> None:
    cfg = load_embodiment_config("g1")
    by_role = {s.role: s for s in cfg.sensors}
    # Head RGB camera 'g1_head_rgb' (mujoco_g1's scene build places it), colour only.
    cam = cfg.camera
    assert by_role["camera"] is cam and (cam.name, cam.width, cam.height) == ("head_rgb", 640, 480)
    assert cam.depth == (0.1, 10.0)
    # Lidar offset _LIDAR_OFFSET_X=0.0, _LIDAR_OFFSET_Z=0.72.
    assert by_role["lidar"].pos == (0.0, 0.0, 0.72)


def test_g1_policy_and_capabilities() -> None:
    cfg = load_embodiment_config("g1")
    assert cfg.base.policy.params["file"] == "motion.pt"  # relative to the asset folder
    # 47-obs / 12-action / 50 Hz (decimation 10 @ 0.002 s) / gait 0.8 s.
    from vector_os_nano.sim.policies import TERMS
    assert sum(TERMS[t["term"]](12) for t in cfg.base.policy.params["obs"]) == 47
    assert len(cfg.base.joints) == 12
    assert cfg.base.policy.params["decimation"] == 10
    assert cfg.base.policy.timestep == pytest.approx(0.002)
    assert cfg.base.policy.params["obs"][-1]["period"] == pytest.approx(0.8)
    assert cfg.base.nav.body_radius == pytest.approx(0.30)
    assert cfg.base.holonomic is False
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


@pytest.mark.parametrize("missing", ["model"])
def test_missing_required_field_raises_clear_error(missing: str) -> None:
    # A body with a base (capabilities default to one) needs every one of these.
    raw = _legged()
    del raw[missing]
    with pytest.raises(EmbodimentConfigError) as exc:
        parse_embodiment_config(raw, ctx="broken/robot.yaml")
    assert f"'{missing}'" in str(exc.value) and "broken/robot.yaml" in str(exc.value)


def test_a_fixed_arm_needs_no_spawn_stance_or_policy() -> None:
    """A body without a base stands at its world's spawn: those blocks are a base's."""
    cfg = load_embodiment_config("piper")
    assert not hasattr(cfg, "spawn") and cfg.base is None
    assert cfg.arm.dof == 6 and cfg.camera.pos is not None


@pytest.mark.parametrize("where, raw", [
    ("broken/robot.yaml", {"schema": 1, "display_name": "A", "model": {"file": "x", "root_body": "b"},
                           "grasp": {}}),
    ("broken/robot.yaml.arm", {"arm": {"joints": ["j"], "ee": {"site": "s"}, "home": [0.0], "dof": 1}}),
    ("broken/robot.yaml.arm.ee", {"arm": {"joints": ["j"], "ee": {"site": "s", "body": "b"}, "home": [0.0]}}),
    ("broken/robot.yaml.arm.gripper", {"arm": {"joints": ["j"], "ee": {"site": "s"}, "home": [0.0], "gripper": {
        "joint": "g", "open": 1.0, "closed": 0.0, "finger_bodies": ["f"], "weld_body": "w",
        "min_opening": 0.0, "actuator": "g"}}}),
    ("broken/robot.yaml.sensors[0]", {"sensors": [{"role": "camera", "name": "c", "width": 2, "height": 2,
                                                   "params": {}}]}),
])
def test_an_unknown_key_fails_naming_the_file_the_key_and_the_valid_set(where: str, raw: dict) -> None:
    """Strict manifests: a misspelt or retired key is an error, never silently ignored."""
    full = {"schema": 1, "display_name": "A", "model": {"file": "x", "root_body": "b"},
            **raw}
    with pytest.raises(EmbodimentConfigError) as exc:
        parse_embodiment_config(full, ctx="broken/robot.yaml")
    msg = str(exc.value)
    assert msg.startswith(where + ": unknown key") and "valid: [" in msg, msg


def test_a_camera_placed_without_a_position_fails_loud() -> None:
    raw = {"schema": 1, "display_name": "A", "model": {"file": "x", "root_body": "b"},
           "sensors": [{"role": "camera", "name": "c", "width": 2, "height": 2, "body": "b"}]}
    with pytest.raises(EmbodimentConfigError, match=r"sensors\[0\]\.body places a camera"):
        parse_embodiment_config(raw, ctx="a/robot.yaml")


def test_both_embodiments_distinct() -> None:
    go2 = load_embodiment_config("go2")
    g1 = load_embodiment_config("g1")
    assert go2.id != g1.id
    assert go2.model.root_body != g1.model.root_body
    # g1 stands clearly higher than go2 (biped vs quadruped).
    assert g1.base.height > go2.base.height
    assert not math.isnan(g1.base.height)


# ---------------------------------------------------------------------------
# Arm block (go2_piper): typed, every morphology name + the gripper contact gate
# ---------------------------------------------------------------------------


def _with_arm(arm: dict) -> dict:
    return {
        "schema": 1, "display_name": "A", "model": {"file": "x.xml", "root_body": "b"},
        "arm": arm,
    }


def test_go2_piper_arm_block_is_typed() -> None:
    arm = load_embodiment_config("go2_piper").arm
    assert arm.dof == 6 and arm.joints == tuple(f"piper_joint{i}" for i in range(1, 7))
    assert arm.home == (0.0,) * 6 and arm.track_tol == 0.08
    assert arm.ee_site == "piper_ee_site" and len(arm.ik_seeds) == 3
    g = arm.gripper
    assert (g.joint, g.open, g.closed, g.min_opening) == ("piper_joint7", 0.035, 0.0, 0.005)
    assert g.finger_bodies == ("piper_link7", "piper_link8") and g.weld_body == "piper_link6"


@pytest.mark.parametrize("bad, field", [
    ({"joints": ["j1"]}, "home"),
    ({"home": [0.0]}, "home"),
    ({"ik_seeds": [[0.0, 0.0, 0.0]]}, "ik_seeds[0]"),
    ({"gripper": {"joint": "g", "open": 1.0, "closed": 0.0, "finger_bodies": [], "weld_body": "w",
                  "min_opening": 0.0}}, "gripper.finger_bodies"),
])
def test_arm_block_dof_mismatch_fails_loud(bad: dict, field: str) -> None:
    arm = {"ee": {"site": "s"}, "joints": ["j1", "j2"], "home": [0.0, 0.0], **bad}
    with pytest.raises(EmbodimentConfigError) as exc:
        parse_embodiment_config(_with_arm(arm), ctx="a/robot.yaml")
    assert f"arm.{field}" in str(exc.value)
