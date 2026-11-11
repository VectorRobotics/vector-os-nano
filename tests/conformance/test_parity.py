# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Kinematic and inertial parity: the engine's own bodies against the canonical model,
for the five 1.0 bodies (charter section 8, S2)."""
from __future__ import annotations

import numpy as np
import pytest

from tests.conformance.rig import qmat, quat_angle
from vector_os_nano.sim.scene import canonical

pytestmark = pytest.mark.sim

BODIES = ["so101", "piper", "go2", "go2_piper", "g1"]
# How each backend holds the model's inertia: Bullet has no joint armature, so the
# PyBullet backend adds a hinge's armature to its link inertia about the joint axis.
ARMATURE_IN_LINK_INERTIA = {"mujoco": False, "pybullet": True}


def _random_q(scene, rng: np.random.Generator) -> np.ndarray:
    m = scene.model
    q = m.qpos0.copy()
    for j in range(m.njnt):
        a = int(m.jnt_qposadr[j])
        if m.jnt_type[j] == 0:  # free: a random base pose around where it was
            quat = rng.normal(size=4)
            q[a:a + 7] = np.concatenate([q[a:a + 3] + rng.uniform(-1, 1, 3), quat / np.linalg.norm(quat)])
        else:
            lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-3.0, 3.0)
            q[a] = rng.uniform(lo, hi)
    return q


@pytest.mark.parametrize("body", BODIES)
def test_link_frames_match_canonical_fk(backend, load, robot_scene, body):
    scene = robot_scene(body)
    world = load(backend, scene)
    moving = np.flatnonzero(scene.model.body_weldid != 0)  # every body that is not welded to the world
    assert set(scene.subtree(scene.robot_root)) - {0} <= set(moving) | {scene.robot_root}
    rng = np.random.default_rng(7)
    for _ in range(10):
        q = _random_q(scene, rng)
        world.reset(q)
        pos, quat = world._link_frames()
        assert np.isfinite(pos[moving]).all(), "the engine does not move every movable body"
        cpos, cquat = canonical.body_poses(scene, q)
        ok = np.isfinite(pos[:, 0])
        assert np.abs(pos[ok] - cpos[ok]).max() < 1e-5
        assert quat_angle(quat[ok], cquat[ok]).max() < 1e-5


@pytest.mark.parametrize("body", BODIES)
def test_inertia_matches_the_model(backend, load, robot_scene, body):
    scene = robot_scene(body)
    m = scene.model
    world = load(backend, scene)
    mass, com, tensor = world._inertials()
    moving = np.flatnonzero(m.body_weldid != 0)
    assert np.isfinite(mass[moving]).all()
    for b in moving:
        R = qmat(m.body_iquat[b])
        expect = R @ np.diag(m.body_inertia[b]) @ R.T
        j = int(m.body_jntadr[b])
        if ARMATURE_IN_LINK_INERTIA[backend] and m.body_jntnum[b] and m.jnt_type[j] == 3:  # hinge
            n = m.jnt_axis[j] / np.linalg.norm(m.jnt_axis[j])
            expect = expect + m.dof_armature[m.jnt_dofadr[j]] * np.outer(n, n)
        scale = max(float(np.abs(expect).max()), 1e-12)
        assert mass[b] == pytest.approx(m.body_mass[b], rel=1e-6, abs=1e-12), scene.body_name(b)
        assert np.abs(com[b] - m.body_ipos[b]).max() <= 1e-6 * max(float(np.abs(m.body_ipos[b]).max()), 1e-6)
        assert np.abs(tensor[b] - expect).max() <= 1e-6 * scale, scene.body_name(b)
