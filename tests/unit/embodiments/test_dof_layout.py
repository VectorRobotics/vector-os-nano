# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Behavior-preservation tests for the generic DofLayout helper (Stage 2).

These are the CRITICAL Rule-11 guardrail: they prove that reading spawn/stance
from ``embodiments/<id>/robot.yaml`` and introspecting the qpos/qvel slice
addresses from the compiled model produces BYTE-IDENTICAL values to the
constants the mujoco_go2 / mujoco_g1 drivers used to hardcode.

A regression here means the refactor changed robot behavior — which is forbidden
(the mandate is behavior-preserving). The model-compiling tests are headless
(MUJOCO_GL=egl) and compile a model directly from the scene XML (no driver, no
physics, no GL window) — fast, and excluded from the heavy sim-instance suites.

Run (offline subset):
    PATH=/usr/bin:$PATH MUJOCO_GL=egl .venv/bin/python -m pytest \
        tests/unit/embodiments/test_dof_layout.py -q
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vector_os_nano.embodiments.config import load_embodiment_config
from vector_os_nano.embodiments.dof_layout import (
    DofLayout,
    _resolve_stance_angle,
    build_robot_geom_set,
)

# Driver constants (the values behavior must stay identical to).
from vector_os_nano.hardware.sim.mujoco_go2 import (
    _STAND_JOINTS,
    _build_room_scene_xml,
)
from vector_os_nano.hardware.sim.mujoco_g1 import (
    _DEFAULT_ANGLES,
    _G1_PELVIS_Z,
    _G1_SPAWN_X,
    _G1_SPAWN_Y,
    _G1Offsets,
    _NUM_ACTIONS,
    _build_g1_room_scene_xml,
)


# ---------------------------------------------------------------------------
# Model fixtures — compile each scene once (headless, no physics, no driver).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def go2_model():
    import mujoco

    scene = _build_room_scene_xml()
    return mujoco.MjModel.from_xml_path(str(scene))


@pytest.fixture(scope="module")
def g1_model():
    import mujoco

    scene = _build_g1_room_scene_xml()
    return mujoco.MjModel.from_xml_path(str(scene))


# ---------------------------------------------------------------------------
# (a) Address introspection: go2 -> (0, 7); g1 -> the _G1Offsets pair.
# ---------------------------------------------------------------------------


def test_go2_layout_addresses_are_self_consistent(go2_model) -> None:
    layout = DofLayout(go2_model, "base_link", 12)
    # S3b unified scene-build onto MjSpec.attach: the room is the base spec and
    # the robot is ATTACHED LAST (so the room's pickable_* bodies keep unprefixed
    # names), which lands go2's base freejoint at qpos[21], NOT qpos[0] as the
    # legacy <include> build did. DofLayout introspects this — that is exactly why
    # the driver must read these addresses, never hardcode 0/7. The CONTRACT
    # (the relationships the driver relies on) is what's behavior-preserving:
    assert layout.quat_start == layout.root_qpos_adr + 3
    assert layout.joint_qpos_start == layout.root_qpos_adr + 7
    assert layout.angvel_start == layout.root_dof_adr + 3
    assert layout.joint_dof_start == layout.root_dof_adr + 6
    # The base freejoint sits at qpos[0] under <include>, qpos[21] under attach;
    # either way it is a single 7-dof freejoint introspected from the model.
    assert layout.root_qpos_adr >= 0
    assert layout.robot_geom_ids  # non-empty self-geom set


def test_g1_layout_matches_g1offsets(g1_model) -> None:
    layout = DofLayout(g1_model, "g1_pelvis", _NUM_ACTIONS)
    off = _G1Offsets(g1_model)
    # Every field _G1Offsets exposed must equal the generic layout's.
    assert layout.root_qpos_adr == off.pelvis_qpos_adr
    assert layout.root_dof_adr == off.pelvis_dof_adr
    assert layout.joint_qpos_start == off.leg_qpos_start
    assert layout.joint_dof_start == off.leg_dof_start
    assert layout.angvel_start == off.angvel_start
    assert layout.quat_start == off.quat_start
    assert layout.root_bid == off.pelvis_bid
    assert layout.robot_geom_ids == off.robot_geom_ids


# ---------------------------------------------------------------------------
# (b) Stance vector built from config == the old hardcoded array element-wise.
# ---------------------------------------------------------------------------


def test_go2_stance_vector_equals_stand_joints(go2_model) -> None:
    layout = DofLayout(go2_model, "base_link", 12)
    cfg = load_embodiment_config("go2")
    stance_vec = layout.build_stance_vector(go2_model, cfg.stance)
    expected = np.asarray(_STAND_JOINTS, dtype=np.float32)
    assert stance_vec.shape == expected.shape
    np.testing.assert_array_equal(stance_vec, expected)


def test_g1_stance_vector_equals_default_angles(g1_model) -> None:
    layout = DofLayout(g1_model, "g1_pelvis", _NUM_ACTIONS)
    cfg = load_embodiment_config("g1")
    stance_vec = layout.build_stance_vector(g1_model, cfg.stance)
    expected = np.asarray(_DEFAULT_ANGLES, dtype=np.float32)
    assert stance_vec.shape == expected.shape
    # _DEFAULT_ANGLES is the qpos-slice order (left leg 6, right leg 6); the
    # vector must match exactly, position-for-position.
    np.testing.assert_array_equal(stance_vec, expected)


# ---------------------------------------------------------------------------
# (c) Spawn read from config == the old spawn constants.
# ---------------------------------------------------------------------------


def test_go2_spawn_from_config_matches_constants() -> None:
    cfg = load_embodiment_config("go2")
    # connect(): qpos[0]=10.0, qpos[1]=3.0, qpos[2]=0.35
    assert cfg.spawn.xy[0] == 10.0
    assert cfg.spawn.xy[1] == 3.0
    assert cfg.spawn.base_height == pytest.approx(0.35)


def test_g1_spawn_from_config_matches_constants() -> None:
    cfg = load_embodiment_config("g1")
    assert cfg.spawn.xy[0] == _G1_SPAWN_X
    assert cfg.spawn.xy[1] == _G1_SPAWN_Y
    assert cfg.spawn.base_height == pytest.approx(_G1_PELVIS_Z)


# ---------------------------------------------------------------------------
# Pure-logic: the suffix-tolerant stance resolver + leg-order independence.
# (No model needed — guards the prefix-stripping logic in isolation.)
# ---------------------------------------------------------------------------


def test_resolve_stance_exact_key() -> None:
    stance = {"FL_hip": 0.0, "FL_thigh": 0.9, "FL_calf": -1.8}
    assert _resolve_stance_angle("FL_thigh", stance) == 0.9


def test_resolve_stance_prefix_stripped() -> None:
    # Model joint carries the "g1_" attach prefix; manifest key does not.
    stance = {"left_hip_pitch_joint": -0.1, "left_knee_joint": 0.3}
    assert _resolve_stance_angle("g1_left_knee_joint", stance) == 0.3


def test_resolve_stance_no_match_fails_loud() -> None:
    with pytest.raises(ValueError) as exc:
        _resolve_stance_angle("unknown_joint", {"FL_hip": 0.0})
    assert "unknown_joint" in str(exc.value)
    assert "FL_hip" in str(exc.value)


def test_go2_actuator_vs_qpos_order_does_not_corrupt_stance(go2_model) -> None:
    """Go2 actuators are FR,FL,RR,RL but legs in qpos are FL,FR,RL,RR.

    The stance vector is built in qpos-slice order (not actuator order), so even
    though every leg shares the same triple [0, 0.9, -1.8] (masking the order),
    we assert the leg-joint NAME order the layout extracts is the qpos order —
    the property that keeps non-uniform stances byte-identical too.
    """
    layout = DofLayout(go2_model, "base_link", 12)
    names = layout._leg_joint_names_in_qpos_order(go2_model)
    # qpos declaration order from go2.xml body tree: FL, FR, RL, RR.
    expected_prefixes = ["FL", "FL", "FL", "FR", "FR", "FR",
                         "RL", "RL", "RL", "RR", "RR", "RR"]
    got_prefixes = [n.split("_")[0] for n in names]
    assert got_prefixes == expected_prefixes, names


def test_build_robot_geom_set_subtree(go2_model) -> None:
    layout = DofLayout(go2_model, "base_link", 12)
    direct = build_robot_geom_set(go2_model, layout.root_bid)
    assert direct == layout.robot_geom_ids
    assert len(direct) > 0
