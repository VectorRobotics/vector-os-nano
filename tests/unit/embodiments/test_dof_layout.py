# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Canonical addresses replace per-driver DofLayout offsets.

The room's free objects precede the robot: a root-at-zero assumption must fail.
Compile-only checks preserve the former address, stance and self-mask guards.
"""
from __future__ import annotations

import numpy as np
import pytest

from vector_os_nano.embodiments import registry
from vector_os_nano.hardware.sim.scene_builder import room_scene
from vector_os_nano.sim.scene import reader
from vector_os_nano.schemas.world import load_world


@pytest.fixture(params=["go2", "g1"])
def body(request, g1_assets):
    cfg = registry.load(request.param)
    return cfg, room_scene(cfg, load_world(cfg.world))


def test_free_and_joint_addresses_are_canonical(body):
    cfg, scene = body
    fq, fd = scene.free_q(scene.robot_root), scene.free_qd(scene.robot_root)
    assert fq.start > 0 and fq.stop - fq.start == 7
    assert fd.start > 0 and fd.stop - fd.start == 6
    joints = [scene.joint(n) for n in cfg.base.joints]
    assert [scene.joint_q(j) for j in joints] == list(range(fq.stop, fq.stop + 12))
    assert [scene.joint_qd(j) for j in joints] == list(range(fd.stop, fd.stop + 12))
    assert [scene.model.joint(j).name for j in joints] == list(cfg.base.joints)


def test_stance_preserves_joint_order_and_precision(body):
    cfg, scene = body
    stance = np.array([cfg.base.stance[n] for n in cfg.base.joints], dtype=np.float64)
    expected = [0.0, 0.9, -1.8] * 4 if cfg.id == "go2" else [-0.1, 0, 0, 0.3, -0.2, 0] * 2
    np.testing.assert_array_equal(stance, expected)
    # Resolve by joint name even when actuator order differs from qpos order.
    limits = reader.motor_limits(scene, [scene.joint(n) for n in cfg.base.joints])
    assert limits.shape == stance.shape and (limits > 0).all()


def test_root_without_a_free_joint_is_refused(body):
    _cfg, scene = body
    for name in ("world", "pick_table"):
        with pytest.raises(KeyError, match="has no free joint"):
            scene.free_q(scene.body(name))
        with pytest.raises(KeyError, match="has no free joint"):
            scene.free_qd(scene.body(name))


def test_self_mask_contains_only_robot_descendants(body):
    _cfg, scene = body
    robot = scene.subtree(scene.robot_root)
    assert scene.robot_root in robot and scene.body("pick_table") not in robot
    geoms = [g for g in reader.geoms(scene.model) if g.body in robot]
    assert geoms and any(g.collide for g in geoms)
