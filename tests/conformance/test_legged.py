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
import threading
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.conformance.rig import yaw
from vector_os_nano.embodiments import registry
from vector_os_nano.schemas.robot import EmbodimentConfigError, parse_embodiment_config
from vector_os_nano.interfaces import BaseProtocol
from vector_os_nano.sim import backends
from vector_os_nano.sim.scene.compose import compose
from vector_os_nano.schemas.world import load_world, parse_world
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


def _world(cfg):
    path = SCENES / f"{cfg.world}.yaml"
    if path.is_file():
        return parse_world(yaml.safe_load(path.read_text()), cfg.world, path)
    return dataclasses.replace(load_world(cfg.world), scene=SCENES / "flat.xml", pickables={})


def test_spawn_uses_the_variant_world_on_each_engine(backend):
    cfg = registry.load("go2")
    world = dataclasses.replace(_world(cfg), spawn=(1.25, -2.5, .4, -.7))
    scene = compose(cfg, world)
    rt, base, _lidar = legged_body(cfg, scene, manifest=world, backend=backend)
    try:
        pose = rt.snapshot().q[scene.free_q(scene.robot_root)]
        np.testing.assert_allclose(pose[:3], [1.25, -2.5, .4 + cfg.base.height], atol=1e-12)
        assert yaw(pose[3:]) == pytest.approx(-.7)
    finally:
        base.disconnect()
        rt.close()


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
        world = _world(cfg)
        scene = compose(cfg, world)
        self.cfg = cfg
        self.rt, self.base, self.lidar = legged_body(cfg, scene, manifest=world, backend=backend)
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


def test_odometry_angular_z_is_in_the_world_frame(body):
    sc = body.rt.world.scene
    q, qd = body.rt.snapshot().q.copy(), body.rt.snapshot().qd.copy()
    root, velocity = sc.free_q(sc.robot_root), sc.free_qd(sc.robot_root)
    q[root.start + 2] = 1.
    q[root.start + 3:root.stop] = [math.sqrt(.5), math.sqrt(.5), 0., 0.]
    qd[velocity] = [.1, .2, .3, 0., 1., 0.]
    body.rt.call(lambda w: w.reset(q, qd))
    odom = body.base.get_odometry()
    assert [odom.vx, odom.vy, odom.vz, odom.vyaw] == pytest.approx([.1, .2, .3, 1.])


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


def test_a_walk_from_rest_starts_level_at_every_gait_phase(monkeypatch):
    """convex_mpc: walks from rest that start at 24 instants across one gait cycle
    (1 s at 0.3 m/s, a stop, 2 s of rest each) keep the trunk within 0.1 rad of level.
    Measured with the gait clock counted from the spawn (the trot started mid-cycle):
    0.297 rad at one phase here, and 2 falls and 8 tilts past 0.35 rad in 200 random walks
    from rest in the room (/data/vos-analysis/scratch/vos-S4a-f2/v3); with the clock
    started at each walk: 0.042-0.045 rad at every phase, at most 0.06 rad in 600."""
    monkeypatch.setenv("VECTOR_SIM_LOCKSTEP", "1")
    cfg = registry.load("go2")
    b = Body("mujoco", cfg)
    try:
        period = 1.0 / cfg.base.policy.on("mujoco")[1]["gait_hz"]
        tilts = []
        for k in range(24):
            assert b.base.wait_sim(period * (1 + k / 24))
            b.base.set_velocity(0.3, 0.0, 0.0)
            walk = list(b.run(1.0, every=0.005))
            b.base.stop()
            for p in walk + list(b.run(2.0, every=0.005)):
                tilts.append(math.acos(min(1.0, 1.0 - 2.0 * (p[4] ** 2 + p[5] ** 2))))
        assert max(tilts) < 0.1, max(tilts)
        assert b.upright()
    finally:
        b.close()


def test_a_walk_and_a_navigate_out_of_every_posture_succeed(body):
    """A held posture is not a fall (every manifest posture holds the root below fall_z):
    out of each one a walk reports success and a navigate arrives, upright."""
    assert body.cfg.base.postures
    for name in body.cfg.base.postures:
        assert body.base.set_posture(name)
        assert body.base.get_position()[2] < body.cfg.base.fall_z, name  # the case this pins
        assert body.base.walk(0.3, 0.0, 0.0, 2.0) is True, name
        assert body.upright(), name
        assert body.base.set_posture(name)
        x, y, _ = body.base.get_position()
        h = body.base.get_heading()
        res = body.base.navigate_to(x + 0.6 * math.cos(h), y + 0.6 * math.sin(h), timeout=20.0)
        assert res, (name, dict(res))
        assert body.upright(), name


@pytest.mark.parametrize("motion", ["walk", "navigate"])
def test_a_stop_during_the_rise_out_of_a_posture_is_kept(body, motion):
    """A walk out of a posture first rises to the stance; a stop() from another thread
    0.3 sim-s into that rise ends the walk unmoved (False), not a walk that starts once
    risen (measured before the fix: 0.59 m on the go2)."""
    name = next(iter(body.cfg.base.postures))
    assert body.base.set_posture(name)
    t0, wait, at_stop = body.base.get_sim_time(), body.base.wait_sim, []

    def wait_and_stop(seconds, cancel=None):  # the rise waits one tick at a time
        if not at_stop and body.base.get_sim_time() >= t0 + 0.3:
            at_stop.append(body.pose().copy())
            t = threading.Thread(target=body.base.stop)
            t.start()
            t.join()
        return wait(seconds, cancel)

    body.base.wait_sim = wait_and_stop
    if motion == "walk":
        assert body.base.walk(0.3, 0.0, 0.0, 2.0) is False
    else:
        x, y, _ = body.base.get_position()
        assert body.base.navigate_to(x + 1., y).reason == "aborted"
    assert at_stop and body.base.get_sim_time() < t0 + 1.5  # stopped in the rise, never walked 2 s
    del body.base.wait_sim
    list(body.run(1.0))
    assert math.hypot(*(body.pose()[:2] - at_stop[0][:2])) < 0.1 and body.upright()


def _tip(body) -> None:
    """Roll the body onto its back, legs in the air, and let it settle there."""
    q = body.rt.snapshot().q.copy()
    sc = body.rt.world.scene
    root = sc.free_q(sc.robot_root)
    q[root.start + 2] = 0.2
    q[root.start + 3:root.stop] = [0.0, 1.0, 0.0, 0.0]  # rolled 180 deg
    body.rt.call(lambda w: w.reset(q))
    assert body.base.wait_sim(1.0)
    assert body.base.get_position()[2] < body.cfg.base.fall_z


def test_a_tipped_body_reports_the_fall(body):
    """Started on its back: the trunk lies below the manifest's fall_z, so a walk reports
    the fall, whatever the policy's legs do to the trunk meanwhile (fall detection is the
    root height at the start and the end of the walk), and a navigate ends 'fell'. The
    navigate goes first: the convex MPC's flailing walk can roll the body back onto its
    feet (measured: on its side at z 0.22 after the 0.5 s walk, then upright)."""
    _tip(body)
    x, y, _ = body.base.get_position()
    assert body.base.navigate_to(x + 1.0, y, timeout=5.0).reason == "fell"
    assert body.base.walk(0.3, 0.0, 0.0, 0.5) is False


def test_a_body_tipped_in_a_posture_reports_the_fall(body):
    """Adversarial for the rise out of a posture: tipped while it holds one, the rise to
    the stance does not right it, and a walk and a navigate still report the fall."""
    name = next(iter(body.cfg.base.postures))
    assert body.base.set_posture(name)
    _tip(body)
    assert body.base.walk(0.3, 0.0, 0.0, 0.5) is False
    body.base.set_posture(name)  # held again, still on its back (legs free: it may reach it)
    assert body.base.get_position()[2] < body.cfg.base.fall_z
    x, y, _ = body.base.get_position()
    assert body.base.navigate_to(x + 1.0, y, timeout=5.0).reason == "fell"


def test_the_lidar_sees_from_its_mount(body):
    if body.lidar is None:
        pytest.skip("no lidar in this manifest")
    assert body.base.set_posture("stand")
    scan = body.base.get_lidar_scan()
    spec = body.cfg.lidar
    assert not any(hasattr(body.lidar, name) for name in ("world", "runtime", "gt", "model", "data"))
    assert len(scan.ranges) == spec.n_azimuth and scan.range_max == spec.range
    hits = np.array(body.lidar.points())[:, :3]
    assert len(hits) and (np.linalg.norm(hits[:, :2] - body.pose()[:2], axis=1) < spec.range + 1.0).all()
    assert abs(hits[:, 2]).min() < 0.02  # the beams pitched down hit the flat floor


def test_tilt_is_detected_above_the_fall_height(body):
    q = body.rt.snapshot().q.copy()
    sc = body.rt.world.scene
    root = sc.free_q(sc.robot_root)
    q[root.start + 2] = 1.
    q[root.start + 3:root.stop] = [0., 1., 0., 0.]
    body.rt.call(lambda w: w.reset(q))
    x, y, z = body.base.get_position()
    assert z > body.cfg.base.fall_z
    assert body.base.navigate_to(x + 1, y).reason == "fell"


def test_the_first_mpc_solve_does_not_write_to_the_repl(body, capsys):
    assert body.base.walk(.2, 0., 0., .2)
    assert "QP Init" not in capsys.readouterr().out


def test_a_base_joint_the_model_lacks_names_the_manifest_and_key():
    """A typo in base.joints (stance and postures renamed with it, so the manifest parses)
    fails where the body is built, naming the manifest file, base.joints and the valid set."""
    cfg = registry.load("go2")

    def typo(m):
        return {("FL_hipp_joint" if j == "FL_hip_joint" else j): a for j, a in m.items()}

    base = dataclasses.replace(cfg.base, joints=tuple(typo(dict.fromkeys(cfg.base.joints))),
                               stance=typo(cfg.base.stance),
                               postures={n: typo(q) for n, q in cfg.base.postures.items()})
    with pytest.raises(EmbodimentConfigError,
                       match=r"^go2/robot\.yaml: base\.joints \['FL_hipp_joint'\] are not joints of .*go2\.xml; "
                             r"joints: \[.*'FL_hip_joint'"):
        Body("mujoco", dataclasses.replace(cfg, base=base))


def test_an_arm_joint_the_model_lacks_names_the_manifest_and_key():
    """The same check for arm.joints: a typo in the go2_piper arm fails where the body is
    built, naming the manifest file, arm.joints and the valid set."""
    cfg = registry.load("go2_piper")
    arm = dataclasses.replace(cfg.arm, joints=("piper_joint1x",) + tuple(cfg.arm.joints[1:]))
    with pytest.raises(EmbodimentConfigError,
                       match=r"^go2_piper/robot\.yaml: arm\.joints \['piper_joint1x'\] are not joints of .*go2_piper\.xml; "
                             r"joints: \[.*'piper_joint1'"):
        Body("mujoco", dataclasses.replace(cfg, arm=arm))


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
    assert _world(quad2).spawn != _world(go2).spawn and quad2.base.stance != go2.base.stance
    assert (quad2.base.kp, quad2.base.kd) != (go2.base.kp, go2.base.kd) and quad2.lidar.pos != go2.lidar.pos
    import vector_os_nano

    pkg = Path(vector_os_nano.__file__).parent
    assert not [p for p in pkg.rglob("*.py") if "quad2" in p.read_text(encoding="utf-8")]
