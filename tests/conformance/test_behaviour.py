# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Behavioural bounds (charter section 8, S2): dynamics are held to outcomes, not traces.

The Piper runs on its model's own position servos. The legged bounds (stance height,
walk, turn, postures, stop, fall) run the generic LeggedBase: test_legged.py.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.sim


def test_mimic_jaws_recover_position_error(backend, load, robot_scene):
    scene = robot_scene('piper')
    w = load(backend, scene)
    arm = [scene.joint(f'joint{i}') for i in range(1, 7)]
    jaw, follower = (scene.joint(f'joint{i}') for i in (7, 8))
    qa, qb = scene.joint_q(jaw), scene.joint_q(follower)
    pose = [0., 1.57, -1.3485, 0., 0., 0.]
    q = scene.model.qpos0.copy()
    q[[scene.joint_q(j) for j in arm]] = pose
    q[qa], q[qb] = .025, -.015  # reset disturbance, not an actor position write
    w.reset(q)
    w.set_targets(arm + [jaw], pose + [.03])
    w.step(int(1. / scene.timestep))
    actual = w.state()[0]
    assert abs(actual[qa] + actual[qb]) < .0005
    assert abs(actual[qa] - .03) < .001


@pytest.mark.parametrize("pose", [[0.0, 1.57, -1.3485, 0.0, 0.0, 0.0], [0.5, 1.57, -1.3485, 0.0, 0.6, 0.0]])
def test_piper_holds_a_commanded_pose(backend, load, robot_scene, pose):
    """Gravity compensation and armature are the model's (frictionloss has its own rig:
    test_contract's drag hinge)."""
    scene = robot_scene("piper")
    w = load(backend, scene)
    arm = [scene.joint(f"joint{i}") for i in range(1, 7)]
    adr = [scene.joint_q(j) for j in arm]
    home = [0.0, 1.57, -1.3485, 0.0, 0.0, 0.0]
    q = scene.model.qpos0.copy()
    q[adr] = home
    w.reset(q)
    w.set_targets(arm + [scene.joint("joint7")], pose + [0.0])
    worst = 0.0
    for _ in range(4):
        w.step(int(0.5 / scene.timestep))
        worst = np.abs(w.state()[0][adr] - pose).max()
    assert worst < 0.05, f"held {worst:.3f} rad off the commanded pose"
