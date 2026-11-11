# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Arm kinematics on the canonical model (sim/scene/kinematics.py), on the go2_piper
scene, against a MOVED base, with no physics running.

The arm rides the dog, so FK/IK read the mount from the live state, never the dog at
spawn (the PiperROS2Proxy wrote the base into a pickable's freejoint and solved every
grasp for a phantom dog at spawn). The "live" state is a q vector the test writes.
"""
from __future__ import annotations

import math
import threading

import numpy as np
import pytest

pytest.importorskip("mujoco")

from vector_os_nano.embodiments import registry  # noqa: E402
from vector_os_nano.hardware.sim.mujoco_go2 import _build_room_scene_xml  # noqa: E402
from vector_os_nano.sim.scene import compiled_scene, kinematics  # noqa: E402

_BASE_Z = 0.30  # standing trunk height


class _Rig:
    def __init__(self) -> None:
        self.scene = compiled_scene(_build_room_scene_xml(with_arm=True), robot_root="base_link")
        self.spec = registry.load("go2_piper").arm
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


# -- today's numbers: the kinematics replace MuJoCoPiper's bit for bit ---------------

_BASES = [(10.0, 3.0, 0.0), (10.46, 3.0, 0.0), (12.8, 4.0, math.pi / 2), (9.5, 2.0, -2.5)]
_TARGETS = [(0.45, 0.0, 0.30), (0.40, 0.08, 0.22), (0.38, -0.05, 0.34), (2.0, 0.0, 0.3)]
_ARM = [0.1, 0.9, -0.8, 0.05, 0.3, -0.2]


def _cases():
    for x, y, yaw in _BASES:
        c, s = math.cos(yaw), math.sin(yaw)
        for f, left, z in _TARGETS:
            yield (x, y, yaw), (x + c * f - s * left, y + s * f + c * left, z)


def test_the_kinematics_keep_the_old_driver_numbers(rig) -> None:
    from types import SimpleNamespace

    from vector_os_nano.embodiments.dof_layout import DofLayout
    from vector_os_nano.hardware.sim.mujoco_piper import MuJoCoPiper

    q_live = rig.scene.model.qpos0.copy()
    go2 = SimpleNamespace(
        _connected=True, _scene_xml_path=str(rig.scene.xml_path), wait_settled=lambda: True,
        _mj=SimpleNamespace(model=rig.scene.model, layout=DofLayout(rig.scene.model, "base_link", 12)),
        runtime=SimpleNamespace(snapshot=lambda: SimpleNamespace(q=q_live.copy())),
    )
    old = MuJoCoPiper(go2, rig.spec)
    old.connect()
    solved = 0
    for base, target in _cases():
        q_live[:] = rig.place(*base, arm=_ARM)
        new_td = rig.kin.ik_top_down(q_live, target, rig.seeds())
        assert old.ik_top_down(target) == new_td, (base, target)
        assert old.ik(target) == rig.kin.ik(q_live, target, rig.kin.arm_q(q_live)), (base, target)
        assert old.fk(_ARM) == rig.kin.fk(q_live, _ARM)
        solved += new_td is not None
    assert solved >= 8  # the comparison covers solutions, not only misses
