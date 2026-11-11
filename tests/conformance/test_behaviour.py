# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Behavioural bounds (charter section 8, S2): dynamics are held to outcomes, not traces.

The go2 runs today's controller exactly (mujoco_go2: KP, KD, torque limits, a 200 Hz
torque held for 5 physics steps, the sinusoidal gait targets); the Piper runs on its
model's own position servos.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tests.conformance.rig import yaw
from vector_os_nano.hardware.sim.mujoco_go2 import (
    _CTRL_DECIM, _KD, _KP, _STAND_JOINTS, _TAU_LIMITS, _compute_gait_targets,
)

pytestmark = pytest.mark.sim

LEGS = [f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR") for part in ("hip", "thigh", "calf")]
STAND = np.array(_STAND_JOINTS)


class Go2:
    """Today's go2 PD loop on the PhysicsWorld API."""

    def __init__(self, world, scene, xy: tuple[float, float]) -> None:
        self.w = world
        self.legs = [scene.joint(n) for n in LEGS]
        self.qa = [scene.joint_q(j) for j in self.legs]
        self.da = [scene.joint_qd(j) for j in self.legs]
        self.base = scene.free_q(scene.robot_root)
        q = scene.model.qpos0.copy()
        q[self.base] = [xy[0], xy[1], 0.35, 1, 0, 0, 0]  # the manifest spawn height
        q[self.qa] = STAND
        world.reset(q)

    def tick(self, q_des: np.ndarray) -> np.ndarray:
        q, qd = self.w.state()
        self.w.set_torque(self.legs, np.clip(_KP * (q_des - q[self.qa]) - _KD * qd[self.da], -_TAU_LIMITS, _TAU_LIMITS))
        self.w.step(_CTRL_DECIM)  # held for 5 physics steps
        return self.w.state()[0][self.base]


def test_go2_pd_stance_height_agrees(both, load, robot_scene):
    scene = robot_scene("go2")
    height = {}
    for backend in both:
        dog = Go2(load(backend, scene), scene, (10.0, 3.0))  # the manifest spawn in the room
        for _ in range(300):
            pose = dog.tick(STAND)
        before = pose[2]
        for _ in range(100):
            pose = dog.tick(STAND)
        assert abs(pose[2] - before) < 1e-3, f"{backend}: not settled"
        assert abs(yaw(pose[3:]) - 0.0) < 0.05 and pose[2] > 0.25, backend
        height[backend] = pose[2]
    assert abs(height["mujoco"] - height["pybullet"]) < 5e-3, height


@pytest.mark.parametrize("pose", [[0.0, 1.57, -1.3485, 0.0, 0.0, 0.0], [0.5, 1.57, -1.3485, 0.0, 0.6, 0.0]])
def test_piper_holds_a_commanded_pose(backend, load, robot_scene, pose):
    """Gravity compensation, armature and the frictionloss deadband are the model's."""
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


def _walk(load, scene, vx: float, vyaw: float, seconds: float) -> tuple[float, float, float]:
    dog = Go2(load("pybullet", scene), scene, (0.0, 0.0))
    for _ in range(200):  # stand 1 s
        start = dog.tick(STAND)
    zmin, turned, last = math.inf, 0.0, yaw(start[3:])
    for _ in range(int(seconds * 200)):
        pose = dog.tick(_compute_gait_targets(dog.w.time(), vx, 0.0, vyaw))
        zmin = min(zmin, pose[2])
        turned += (yaw(pose[3:]) - last + math.pi) % (2 * math.pi) - math.pi
        last = yaw(pose[3:])
    return float(pose[0] - start[0]) / seconds, zmin, turned


def test_go2_walks_on_pybullet(pybullet_backend, load, flat_go2_scene):
    speed, zmin, _ = _walk(load, flat_go2_scene, 0.4, 0.0, 5.0)
    assert speed >= 0.2, f"{speed:.3f} m/s"
    assert zmin > 0.2, f"the base dipped to {zmin:.3f} m"


def test_go2_turns_on_pybullet(pybullet_backend, load, flat_go2_scene):
    _, zmin, turned = _walk(load, flat_go2_scene, 0.0, 1.0, 4.0)
    assert turned > 1.0, f"turned {math.degrees(turned):.0f} deg"
    assert zmin > 0.2
