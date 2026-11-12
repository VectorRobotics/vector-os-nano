# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The ONE legged base on every engine (charter sections 3.3, 3.4, 7): the SAME
LeggedBase, built from a manifest by the session's own ``legged_body``, on flat ground.

Rows: MuJoCo with the go2's convex MPC and with the sinusoidal gait, PyBullet with the
sinusoidal gait (its by_backend row; convex_mpc there is refused by ``supports``), and a
second quadruped that exists only as a test manifest. Caller-stepped (lockstep), so every
number is the same on every run.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.conformance.rig import yaw
from vector_os_nano.embodiments import registry
from vector_os_nano.embodiments.config import parse_embodiment_config
from vector_os_nano.interfaces import BaseProtocol
from vector_os_nano.sim import backends
from vector_os_nano.sim.scene.compose import compose
from vector_os_nano.sim.session import legged_body

pytestmark = pytest.mark.sim

SCENES = Path(__file__).parent / "scenes"
QUAD2 = Path(__file__).parent / "robots" / "quad2.yaml"  # a second quadruped: data only

ROWS = [("mujoco", "go2", "convex_mpc"), ("mujoco", "go2", "sinusoidal"), ("pybullet", "go2", "sinusoidal"),
        ("mujoco", "quad2", "convex_mpc")]


def _manifest(body: str):
    if body == "quad2":
        return parse_embodiment_config(yaml.safe_load(QUAD2.read_text()), ctx="quad2.yaml")
    return registry.load(body)


def _with_policy(cfg, backend: str, ref: str):
    """*cfg* running the policy row *ref* on *backend* (the go2's sinusoidal row is its
    PyBullet one)."""
    policy = cfg.base.policy
    if policy.on(backend)[0] == ref:
        return cfg
    row = next(dict(v) for v in policy.by_backend.values() if v["ref"] == ref)
    policy = dataclasses.replace(policy, by_backend={**policy.by_backend, backend: row})
    return dataclasses.replace(cfg, base=dataclasses.replace(cfg.base, policy=policy))


class Body:
    """A manifest's legged body standing on flat ground on one engine."""

    def __init__(self, backend: str, cfg) -> None:
        x, y = cfg.spawn.xy
        scene = compose(SCENES / "flat.xml", cfg, spawn=(x, y, 0.0, cfg.spawn.heading),
                        timestep=cfg.base.policy.timestep)
        self.cfg = cfg
        self.rt, self.base, self.lidar = legged_body(cfg, scene, backend=backend)
        self.rt.start()
        assert not self.rt.threaded

    def pose(self) -> np.ndarray:
        sc = self.rt.world.scene
        return self.rt.snapshot().q[sc.free_q(sc.robot_root)]

    def upright(self, pose: np.ndarray | None = None) -> bool:
        """The body's z axis within 30 deg of up and its root above 72 % of its stand
        height (the S2 bound for the go2: 0.2 m)."""
        p = self.pose() if pose is None else pose
        w, x, y, z = p[3:]
        tilt_ok = 1.0 - 2.0 * (x * x + y * y) > math.cos(math.radians(30))
        return bool(tilt_ok and p[2] > 0.72 * self.cfg.base.height)

    def run(self, seconds: float, every: float = 0.05):
        """Step *seconds*, yielding the root pose every *every* sim seconds."""
        for _ in range(round(seconds / every)):
            assert self.base.wait_sim(every)
            yield self.pose()

    def close(self) -> None:
        self.base.disconnect()
        self.rt.close()


@pytest.fixture(params=ROWS, ids=["-".join(r) for r in ROWS])
def body(request, monkeypatch):
    backend, name, ref = request.param
    monkeypatch.setenv("VECTOR_SIM_LOCKSTEP", "1")
    if backend == "pybullet":
        pytest.importorskip("pybullet", reason="the pybullet extra is not installed")
    cfg = _with_policy(_manifest(name), backend, ref)
    assert backends.get(backend).supports(cfg, None) is None
    b = Body(backend, cfg)
    assert b.base.gait_backend == ref
    yield b
    b.close()


def test_the_base_is_the_protocol_member_by_member(body):
    base = body.base
    assert isinstance(base, BaseProtocol)
    for name in BaseProtocol.__protocol_attrs__:
        assert hasattr(base, name), name
    assert base.limits == body.cfg.base.limits and base.holonomic is body.cfg.base.holonomic
    assert not any(hasattr(base, a) for a in ("runtime", "world", "gt", "model", "data"))


def test_it_stands_at_the_manifest_height(body):
    assert body.base.set_posture("stand")
    z = [p[2] for p in body.run(1.0)]
    assert abs(z[-1] - body.cfg.base.height) < 5e-3, z[-1]
    assert max(z[-10:]) - min(z[-10:]) < 1e-3 and body.upright()


def test_it_stands_at_the_same_height_on_both_engines(both, monkeypatch):
    """The go2's stance on its PyBullet row and on MuJoCo (with the same row): within 5 mm."""
    monkeypatch.setenv("VECTOR_SIM_LOCKSTEP", "1")
    height = {}
    for backend in both:
        b = Body(backend, _with_policy(registry.load("go2"), backend, "sinusoidal"))
        try:
            assert b.base.set_posture("stand")
            height[backend] = list(b.run(1.0))[-1][2]
        finally:
            b.close()
    assert abs(height["mujoco"] - height["pybullet"]) < 5e-3, height


def test_it_walks_forward(body):
    assert body.base.set_posture("stand")
    start = body.pose().copy()
    body.base.set_velocity(0.4, 0.0, 0.0)
    poses = list(body.run(5.0))
    body.base.stop()
    end = poses[-1]
    heading = yaw(start[3:])
    ahead = (end[0] - start[0]) * math.cos(heading) + (end[1] - start[1]) * math.sin(heading)
    assert ahead / 5.0 >= 0.2, f"{ahead / 5.0:.3f} m/s"
    assert all(body.upright(p) for p in poses), min(p[2] for p in poses)
    drift = abs(math.remainder(yaw(end[3:]) - heading, 2 * math.pi))
    assert drift < math.radians(30), f"heading drifted {math.degrees(drift):.0f} deg"


def test_it_turns(body):
    assert body.base.set_posture("stand")
    turned, last = 0.0, yaw(body.pose()[3:])
    body.base.set_velocity(0.0, 0.0, 1.0)
    for p in body.run(4.0):
        turned += math.remainder(yaw(p[3:]) - last, 2 * math.pi)
        last = yaw(p[3:])
        assert body.upright(p), p
    assert turned > 1.0, f"turned {math.degrees(turned):.0f} deg"


def test_a_posture_holds_until_a_velocity_command(body):
    stand = body.cfg.base.height
    assert body.base.set_posture("sit")
    low = [p[2] for p in body.run(3.0)]  # 3 s after the posture reported: still sat
    assert max(low) < stand - 0.03, (min(low), max(low))
    body.base.stop()  # a stop is not a velocity command
    assert max(p[2] for p in body.run(1.0)) < stand - 0.03
    body.base.set_velocity(0.3, 0.0, 0.0)  # it gets up and walks
    list(body.run(2.0))
    body.base.stop()
    list(body.run(2.0))
    assert body.pose()[2] > stand - 0.03 and body.upright()


def test_an_unknown_posture_names_the_valid_ones(body):
    with pytest.raises(KeyError, match="'stand'"):
        body.base.set_posture("handstand")


def test_a_posture_the_joints_cannot_reach_is_reported(monkeypatch):
    """Adversarial: hips asked past their 60 deg limit end there, and set_posture says so."""
    monkeypatch.setenv("VECTOR_SIM_LOCKSTEP", "1")
    cfg = registry.load("go2")
    splits = {j: (1.4 if j.endswith("hip_joint") else a) for j, a in cfg.base.stance.items()}
    cfg = dataclasses.replace(cfg, base=dataclasses.replace(cfg.base, postures={**cfg.base.postures, "splits": splits}))
    b = Body("mujoco", cfg)
    try:
        assert b.base.set_posture("stand") is True
        assert b.base.set_posture("splits") is False
    finally:
        b.close()


def test_a_stop_from_a_walk_at_the_limit_brings_it_to_rest(body):
    assert body.base.set_posture("stand")
    body.base.set_velocity(body.cfg.base.limits[0], 0.0, 0.0)
    walking = list(body.run(2.0))
    body.base.stop()
    sc = body.rt.world.scene
    speed = []
    for _ in body.run(2.0):
        v = body.rt.snapshot().qd[sc.free_qd(sc.robot_root)]
        speed.append(math.hypot(v[0], v[1]))
    assert max(speed[-10:]) < 0.05, speed[-10:]
    assert all(p[2] > body.cfg.base.fall_z for p in walking) and body.upright()


def test_a_tipped_body_reports_the_fall(body):
    """Started on its back, legs in the air: the trunk lies below the manifest's fall_z, so
    a walk reports the fall (fall detection is the root height, nothing else)."""
    q = body.rt.snapshot().q.copy()
    sc = body.rt.world.scene
    root = sc.free_q(sc.robot_root)
    q[root.start + 2] = 0.2
    q[root.start + 3:root.stop] = [0.0, 1.0, 0.0, 0.0]  # rolled 180 deg
    body.rt.call(lambda w: w.reset(q))
    assert body.base.wait_sim(1.0)
    assert body.base.get_position()[2] < body.cfg.base.fall_z
    assert body.base.walk(0.3, 0.0, 0.0, 0.5) is False


def test_the_lidar_sees_from_its_mount(body):
    if body.lidar is None:
        pytest.skip("no lidar in this manifest")
    assert body.base.set_posture("stand")
    scan = body.base.get_lidar_scan()
    spec = body.cfg.lidar
    assert len(scan.ranges) == spec.n_azimuth and scan.range_max == spec.range
    hits = np.array(body.lidar.points())[:, :3]
    assert len(hits) and (np.linalg.norm(hits[:, :2] - body.pose()[:2], axis=1) < spec.range + 1.0).all()
    assert abs(hits[:, 2]).min() < 0.02  # the beams pitched down hit the flat floor


def test_convex_mpc_is_refused_on_pybullet_with_its_reason():
    cfg = registry.load("go2")
    policy = dataclasses.replace(cfg.base.policy, by_backend={})
    bare = dataclasses.replace(cfg, base=dataclasses.replace(cfg.base, policy=policy))
    pb = backends.get("pybullet")
    if pb.supports(cfg, None) is not None and "install" in pb.supports(cfg, None):
        pytest.skip("the pybullet extra is not installed")
    assert "falls" in pb.supports(bare, None)


def test_the_second_quadruped_is_data_only():
    """quad2 differs from the go2 in spawn, stance, gains and lidar mount, and nothing in
    the package names it."""
    go2, quad2 = registry.load("go2"), _manifest("quad2")
    assert quad2.spawn != go2.spawn and quad2.base.stance != go2.base.stance
    assert (quad2.base.kp, quad2.base.kd) != (go2.base.kp, go2.base.kd) and quad2.lidar.pos != go2.lidar.pos
    import vector_os_nano

    pkg = Path(vector_os_nano.__file__).parent
    assert not [p for p in pkg.rglob("*.py") if "quad2" in p.read_text(encoding="utf-8")]
