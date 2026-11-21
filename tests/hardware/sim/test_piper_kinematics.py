# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Arm kinematics on the canonical model (sim/scene/kinematics.py), on the go2_piper
scene, against a MOVED base, with no physics running.

The arm rides the dog, so FK/IK read the mount from the live state, never the dog at
spawn (the PiperROS2Proxy wrote the base into a pickable's freejoint and solved every
grasp for a phantom dog at spawn). The "live" state is a q vector the test writes.
"""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from vector_os_nano.embodiments import registry  # noqa: E402
from vector_os_nano.sim.scene.compose import compose  # noqa: E402
from vector_os_nano.sim.scene import kinematics  # noqa: E402
from vector_os_nano.schemas.world import load_world  # noqa: E402

_BASE_Z = 0.30  # standing trunk height


class _Rig:
    def __init__(self) -> None:
        cfg = registry.load("go2_piper")
        self.scene = compose(cfg, load_world(cfg.world))
        self.spec = cfg.arm
        self.kin = kinematics.ArmKinematics(self.scene, self.spec.joints, self.spec.ee_site)
        self.q = self.scene.model.qpos0.copy()

    def place(self, x: float, y: float, yaw: float, arm=None) -> np.ndarray:
        s = self.scene.free_q(self.scene.robot_root)
        self.q[s] = [x, y, _BASE_Z, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        if arm is not None:
            for name, v in zip(self.spec.joints, arm):
                self.q[self.scene.joint_q(self.scene.joint(name))] = v
        return self.q

    def seeds(self, extra=()):
        return [*extra, self.kin.arm_q(self.q), *self.spec.ik_seeds]


@pytest.fixture(scope="module")
def rig() -> _Rig:
    return _Rig()


def _pose(x: float, y: float, yaw: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = kinematics._rot_z(yaw)
    T[:3, 3] = [x, y, _BASE_Z]
    return T


def test_fk_follows_a_moved_base(rig) -> None:
    """Base at spawn vs (12.8, 4, yaw pi/2): the EE moves rigidly with the base."""
    q = [0.3, 1.2, -1.0, 0.2, 0.6, 0.4]
    p_a, r_a = rig.kin.fk(rig.place(10.0, 3.0, 0.0), q)
    p_b, r_b = rig.kin.fk(rig.place(12.8, 4.0, math.pi / 2), q)
    T = _pose(12.8, 4.0, math.pi / 2) @ np.linalg.inv(_pose(10.0, 3.0, 0.0))
    np.testing.assert_allclose(p_b, T[:3, :3] @ p_a + T[:3, 3], atol=1e-6)
    np.testing.assert_allclose(r_b, T[:3, :3] @ np.array(r_a), atol=1e-6)
    assert math.dist(p_a, p_b) > 2.0


def test_ik_solves_against_the_moved_base(rig) -> None:
    q = rig.place(12.8, 4.0, math.pi / 2)
    target = (12.8, 4.0 + 0.45, 0.25)  # 45 cm ahead of the turned dog
    sol = rig.kin.ik_top_down(q, target, rig.seeds())
    assert sol is not None
    pos, rot = rig.kin.fk(q, sol)
    assert math.dist(pos, target) < 3e-3
    assert np.array(rot)[2, 2] < -0.999  # tool z = world -Z


def test_precise_ik_resolves_motion_smaller_than_the_legacy_tolerance(rig):
    kin = kinematics.ArmKinematics(rig.scene, rig.spec.joints, rig.spec.ee_site, position_tolerance=.0001)
    seed = [0.3, 1.2, -1., .2, .6, .4]
    state = rig.place(10., 3., 0., arm=seed)
    pos, rotation = kin.fk(state, seed)
    target = [pos[0], pos[1], pos[2]+.0005]
    solution = kin.ik(state, target, seed, target_rotation=rotation)
    assert solution is not None
    assert math.dist(kin.fk(state, solution)[0], target) < .0001


@pytest.fixture(scope="module")
def feasible(rig) -> list[tuple[float, float, float]]:
    """250 body-frame (forward, left, z) targets — 0.15-0.65 m ahead, bearing +-0.6 rad,
    z 0-0.5 — that a random multi-start proves reachable top-down (dog at spawn)."""
    q = rig.place(10.0, 3.0, 0.0, arm=[0.0] * 6)
    rng = np.random.default_rng(7)
    lo, hi = rig.kin.range[:, 0], rig.kin.range[:, 1]
    out: list[tuple[float, float, float]] = []
    while len(out) < 250:
        r, b, z = rng.uniform(0.15, 0.65), rng.uniform(-0.6, 0.6), rng.uniform(0.0, 0.5)
        f, left = r * math.cos(b), r * math.sin(b)
        if rig.kin.ik_top_down(q, (10.0 + f, 3.0 + left, z), [list(rng.uniform(lo, hi)) for _ in range(6)]):
            out.append((f, left, z))
    return out


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, math.pi])
def test_top_down_reach_is_heading_independent(rig, feasible, yaw) -> None:
    """The canned seeds solve nearly every feasible target at every heading.

    Measured: 250/250 at 0, pi/2 and pi; the old world-X jaw with the shipped
    seeds solved 185/250 at heading 0 (250 at pi/2 and pi).
    """
    q = rig.place(10.0, 3.0, yaw, arm=[0.0] * 6)
    c, s = math.cos(yaw), math.sin(yaw)
    solved = sum(
        rig.kin.ik_top_down(q, (10.0 + c * f - s * left, 3.0 + s * f + c * left, z), rig.seeds()) is not None
        for f, left, z in feasible
    )
    assert solved >= 240, f"yaw={yaw:.2f}: {solved}/250"


def test_a_180_degree_flip_is_not_convergence(rig) -> None:
    """sin(theta) of a pi rotation is 0: the old metric accepted a gripper pointing UP."""
    q = rig.place(10.0, 3.0, 0.0)
    q0 = [0.0, 1.57, -1.57, 0.0, 1.22, 0.0]
    pos, rot = rig.kin.fk(q, q0)
    flipped = np.array(rot) @ np.diag([1.0, -1.0, -1.0])  # rotated pi about the tool x
    with rig.kin._lock:
        rig.kin._sync(q)
        sol = rig.kin._iterate(tuple(pos), flipped, q0)
    if sol is not None:
        _p, r = rig.kin.fk(q, sol)
        assert kinematics._rot_angle(flipped @ np.array(r).T) < kinematics._IK_ROT_TOL


@pytest.mark.parametrize('rotation', [np.zeros((3,3)), np.diag([1,1,-1]), np.eye(2), np.full((3,3), np.nan)])
def test_oriented_ik_rejects_invalid_rotations(rig, rotation):
    with pytest.raises(ValueError, match='proper rotation'):
        rig.kin.ik(rig.q, [0,0,0], [0]*6, target_rotation=rotation)


def test_the_mount_is_every_robot_joint_outside_the_arm(rig) -> None:
    """The dog's free joint and legs come from the live state; the gripper and the scene's
    objects never do (they are not what the arm rides on)."""
    sc, kin = rig.scene, rig.kin
    root = sc.free_q(sc.robot_root)
    legs = [sc.joint_q(sc.joint(f"{leg}_{part}_joint")) for leg in ("FL", "FR", "RL", "RR")
            for part in ("hip", "thigh", "calf")]
    assert sorted(kin._mount) == sorted([*range(root.start, root.stop), *legs])


def test_fk_waits_for_the_kinematics_lock(rig) -> None:
    """fk (the verify oracle's thread) and IK (the skill's) share one scratch MjData."""
    done = threading.Event()
    with rig.kin._lock:
        t = threading.Thread(target=lambda: (rig.kin.fk(rig.q, [0.0] * 6), done.set()), daemon=True)
        t.start()
        assert not done.wait(0.2)
    assert done.wait(5.0)


# -- legacy kinematics, within MuJoCo 4 floating-point roundoff ---------------------
# golden/piper_kinematics.json holds MuJoCoPiper's ik_top_down / ik / fk on these 16
# cases, recorded from that driver on release/1.0 just before S3a deleted it (the live
# comparison passed on 84fcde0). JSON floats round-trip exactly.

_BASES = [(10.0, 3.0, 0.0), (10.46, 3.0, 0.0), (12.8, 4.0, math.pi / 2), (9.5, 2.0, -2.5)]
_TARGETS = [(0.45, 0.0, 0.30), (0.40, 0.08, 0.22), (0.38, -0.05, 0.34), (2.0, 0.0, 0.3)]
_ARM = [0.1, 0.9, -0.8, 0.05, 0.3, -0.2]


def test_the_kinematics_keep_the_old_driver_numbers(rig) -> None:
    golden = json.loads((Path(__file__).with_name("golden") / "piper_kinematics.json").read_text())
    cases = [((x, y, yaw), (x + math.cos(yaw) * f - math.sin(yaw) * left,
                            y + math.sin(yaw) * f + math.cos(yaw) * left, z))
             for x, y, yaw in _BASES for f, left, z in _TARGETS]
    assert len(golden) == len(cases) == 16
    for (base, target), old in zip(cases, golden):
        q = rig.place(*base, arm=_ARM).copy()
        pos, rot = rig.kin.fk(q, _ARM)
        # MuJoCo 4 changes floating-point roundoff; preserve twelve decimal places.
        for got, want in ((rig.kin.ik_top_down(q, target, rig.seeds()), old['td']),
                          (rig.kin.ik(q, target, rig.kin.arm_q(q)), old['ik']),
                          (pos, old['fk'][0]), (rot, old['fk'][1])):
            if want is None:
                assert got is None, (base, target)
            else:
                np.testing.assert_allclose(got, want, rtol=0, atol=1e-12, err_msg=str((base, target)))
    assert sum(o["td"] is not None for o in golden) == 12  # solutions, not only misses
