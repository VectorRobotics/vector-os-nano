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
