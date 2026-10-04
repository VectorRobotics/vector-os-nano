# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""MuJoCoPiper kinematics against a MOVED base, with no physics running.

The arm rides the dog, so FK/IK must read the base pose from the live state —
never assume the dog at spawn (the PiperROS2Proxy wrote the base into qpos[0:7],
a pickable's freejoint, and solved every grasp for a phantom dog at spawn). The
"live" state here is a stub runtime whose snapshot the test sets, so every
assertion is about the kinematics alone.
"""
from __future__ import annotations

import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from vector_os_nano.embodiments import registry
from vector_os_nano.embodiments.dof_layout import DofLayout
from vector_os_nano.hardware.sim import mujoco_piper as mp
from vector_os_nano.hardware.sim.mujoco_go2 import _build_room_scene_xml
from vector_os_nano.hardware.sim.mujoco_piper import MuJoCoPiper

_BASE_Z = 0.30  # standing trunk height


class _Runtime:
    def __init__(self, model) -> None:
        self.model = model
        self.qpos = model.qpos0.copy()

    def snapshot(self):
        return SimpleNamespace(qpos=self.qpos.copy())


class _Go2:
    """What MuJoCoPiper reads from its parent: model, layout, runtime, scene file."""

    _connected = True

    def __init__(self, scene: str) -> None:
        model = mujoco.MjModel.from_xml_path(scene)
        self._mj = SimpleNamespace(model=model, layout=DofLayout(model, "base_link", 12))
        self.runtime = _Runtime(model)
        self._scene_xml_path = scene

    def place(self, x: float, y: float, yaw: float, arm=None) -> None:
        L = self._mj.layout
        q = self.runtime.qpos
        q[L.root_qpos_adr:L.root_qpos_adr + 7] = [x, y, _BASE_Z, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        if arm is not None:
            arm_q0 = L.joint_qpos_start + L.num_actuated
            q[arm_q0:arm_q0 + 6] = arm


@pytest.fixture(scope="module")
def rig():
    go2 = _Go2(str(_build_room_scene_xml(with_arm=True)))
    piper = MuJoCoPiper(go2, registry.load("go2_piper").arm)
    piper.connect()
    yield go2, piper
    piper.disconnect()


def _pose(x: float, y: float, yaw: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = mp._rot_z(yaw)
    T[:3, 3] = [x, y, _BASE_Z]
    return T


def test_fk_follows_a_moved_base(rig) -> None:
    """Base at spawn vs (12.8, 4, yaw pi/2): the EE moves rigidly with the base."""
    go2, piper = rig
    q = [0.3, 1.2, -1.0, 0.2, 0.6, 0.4]
    go2.place(10.0, 3.0, 0.0)
    p_a, r_a = piper.fk(q)
    go2.place(12.8, 4.0, math.pi / 2)
    p_b, r_b = piper.fk(q)
    T = _pose(12.8, 4.0, math.pi / 2) @ np.linalg.inv(_pose(10.0, 3.0, 0.0))
    np.testing.assert_allclose(p_b, T[:3, :3] @ p_a + T[:3, 3], atol=1e-6)
    np.testing.assert_allclose(r_b, T[:3, :3] @ np.array(r_a), atol=1e-6)
    assert math.dist(p_a, p_b) > 2.0


def test_ik_solves_against_the_moved_base(rig) -> None:
    go2, piper = rig
    go2.place(12.8, 4.0, math.pi / 2)
    target = (12.8, 4.0 + 0.45, 0.25)  # 45 cm ahead of the turned dog
    q = piper.ik_top_down(target)
    assert q is not None
    pos, rot = piper.fk(q)
    assert math.dist(pos, target) < 3e-3
    assert np.array(rot)[2, 2] < -0.999  # tool z = world -Z


@pytest.fixture(scope="module")
def feasible(rig) -> list[tuple[float, float, float]]:
    """250 body-frame (forward, left, z) targets — 0.15-0.65 m ahead, bearing +-0.6 rad,
    z 0-0.5 — that a random multi-start proves reachable top-down (dog at spawn)."""
    go2, piper = rig
    go2.place(10.0, 3.0, 0.0, arm=[0.0] * 6)
    rng = np.random.default_rng(7)
    out: list[tuple[float, float, float]] = []
    with piper._ik_lock:
        piper._sync_ik_base_from_live()
        lo, hi = piper._ik_range[:, 0], piper._ik_range[:, 1]
        while len(out) < 250:
            r, b, z = rng.uniform(0.15, 0.65), rng.uniform(-0.6, 0.6), rng.uniform(0.0, 0.5)
            f, left = r * math.cos(b), r * math.sin(b)
            if any(piper._ik_iterate((10.0 + f, 3.0 + left, z), mp._rot_z(y) @ mp._R_TOP_DOWN,
                                     list(rng.uniform(lo, hi))) is not None
                   for y in (0.0, math.pi) for _ in range(6)):
                out.append((f, left, z))
    return out


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, math.pi])
def test_top_down_reach_is_heading_independent(rig, feasible, yaw) -> None:
    """The canned seeds solve nearly every feasible target at every heading.

    Measured: 250/250 at 0, pi/2 and pi; the old world-X jaw with the shipped
    seeds solved 185/250 at heading 0 (250 at pi/2 and pi).
    """
    go2, piper = rig
    go2.place(10.0, 3.0, yaw, arm=[0.0] * 6)
    c, s = math.cos(yaw), math.sin(yaw)
    solved = sum(
        piper.ik_top_down((10.0 + c * f - s * left, 3.0 + s * f + c * left, z)) is not None
        for f, left, z in feasible
    )
    assert solved >= 240, f"yaw={yaw:.2f}: {solved}/250"


def test_a_180_degree_flip_is_not_convergence(rig) -> None:
    """sin(theta) of a pi rotation is 0: the old metric accepted a gripper pointing UP."""
    go2, piper = rig
    go2.place(10.0, 3.0, 0.0)
    q0 = [0.0, 1.57, -1.57, 0.0, 1.22, 0.0]
    pos, rot = piper.fk(q0)
    flipped = np.array(rot) @ np.diag([1.0, -1.0, -1.0])  # rotated pi about the tool x
    with piper._ik_lock:
        piper._sync_ik_base_from_live()
        sol = piper._ik_iterate(tuple(pos), flipped, q0)
    if sol is not None:
        _p, r = piper.fk(sol)
        assert mp._rot_angle(flipped @ np.array(r).T) < mp._IK_ROT_TOL


def test_ik_seeds_include_the_live_arm_and_respect_the_limits(rig, monkeypatch) -> None:
    go2, piper = rig
    live = [0.1, 0.9, -0.8, 0.05, 0.3, -0.2]
    go2.place(10.0, 3.0, 0.0, arm=live)
    seen: list[list[float]] = []
    monkeypatch.setattr(piper, "_ik_iterate", lambda t, r, s: seen.append(list(s)))
    assert piper.ik_top_down((10.5, 3.0, 0.3)) is None
    assert seen[0] == pytest.approx(live)  # the explicit live seed, not IK residue
    assert len(seen) == 2 * (1 + len(piper._spec.ik_seeds))  # heading 0: world X and +pi
    lo, hi = piper._ik_range[:, 0], piper._ik_range[:, 1]
    for s in piper._spec.ik_seeds:
        assert np.all(np.asarray(s) >= lo) and np.all(np.asarray(s) <= hi), s


def test_fk_waits_for_the_ik_lock(rig) -> None:
    """fk (the verify oracle's thread) and IK (the skill's) share one scratch MjData."""
    _go2, piper = rig
    done = threading.Event()
    with piper._ik_lock:
        t = threading.Thread(target=lambda: (piper.fk([0.0] * 6), done.set()), daemon=True)
        t.start()
        assert not done.wait(0.2)
    assert done.wait(5.0)
