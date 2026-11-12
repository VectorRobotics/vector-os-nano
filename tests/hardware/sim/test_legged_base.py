# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The generic LeggedBase on the go2 (MuJoCo, convex_mpc), as the session builds it.

What the conformance suite (tests/conformance/test_legged.py) does not hold: the room
scene and its lidar, odometry, the velocity commands and their limits, the world-side
reset (``/reset``), the convex-MPC plan and its failures, sim-time motions at any
real-time ratio, the cameras, the scene cache, lockstep determinism and the physics
thread / viewer modes. Moved from the MuJoCoGo2 tests (unit/test_mujoco_go2.py,
hardware/sim/test_go2_runtime.py, unit/test_go2_viewer_pump.py).
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from tests.hardware.sim.dog import stand_up  # noqa: E402
from vector_os_nano.core.types import LaserScan, Odometry  # noqa: E402
from vector_os_nano.sim.session import respawn  # noqa: E402

pytestmark = pytest.mark.sim

_REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def flat():
    dog = stand_up("go2", room=False)
    yield dog
    dog.close()


def _speed(dog) -> float:
    v = dog.base.get_velocity()
    return math.hypot(v[0], v[1], v[2])


# -- lifecycle -----------------------------------------------------------------------


def test_lifecycle_and_physics_thread():
    dog = stand_up("go2", room=False, start=False)
    try:
        with pytest.raises(RuntimeError, match="already connected"):
            dog.base.connect()
        dog.rt.start()
        assert sum(t.name == "go2_physics" for t in threading.enumerate()) == 1
        dog.base.disconnect()
        dog.base.disconnect()  # idempotent
        with pytest.raises(RuntimeError, match="not connected"):
            dog.base.set_posture("stand")
        dog.base.stop()  # never raises
    finally:
        dog.rt.close()
    assert not any(t.name == "go2_physics" for t in threading.enumerate())


def test_state_queries(flat):
    base = flat.base
    pos, vel, h = base.get_position(), base.get_velocity(), base.get_heading()
    assert len(pos) == 3 and len(vel) == 3 and isinstance(h, float)
    assert all(isinstance(v, float) for v in pos + vel)
    odom = base.get_odometry()
    assert isinstance(odom, Odometry) and abs(odom.qw) > 0.5 and odom.timestamp > 0
    assert base.wait_sim(0.1)
    assert base.get_odometry().timestamp > odom.timestamp


def test_the_room_lidar(tmp_path):
    dog = stand_up("go2")
    try:
        scan = dog.base.get_lidar_scan()
        assert isinstance(scan, LaserScan) and len(scan.ranges) == 360 and scan.range_max == 12.0
        assert any(r < scan.range_max for r in scan.ranges), "no wall in the room"
        assert dog.base.get_lidar_scan() is scan  # recast only when 0.2 sim-s old
        assert dog.base.wait_sim(0.25)
        assert dog.base.get_lidar_scan() is not scan
    finally:
        dog.close()


# -- commands ------------------------------------------------------------------------


@pytest.mark.parametrize("cmd, moved", [((0.3, 0.0, 0.0), "x"), ((0.0, 0.2, 0.0), "upright"), ((0.0, 0.0, 0.5), "yaw")])
def test_velocity_commands(flat, cmd, moved):
    base = flat.base
    p0, h0 = base.get_position(), base.get_heading()
    base.set_velocity(*cmd)
    assert base.wait_sim(2.0)
    base.set_velocity(0.0, 0.0, 0.0)
    p1, h1 = base.get_position(), base.get_heading()
    assert p1[2] > flat.cfg.base.fall_z
    if moved == "x":
        assert math.hypot(p1[0] - p0[0], p1[1] - p0[1]) > 0.1
    if moved == "yaw":
        assert abs(math.remainder(h1 - h0, 2 * math.pi)) > 0.3


def test_a_walk_moves_it(flat):
    base = flat.base
    p0 = base.get_position()
    assert base.walk(vx=0.3, duration=2.0)
    p1 = base.get_position()
    assert math.hypot(p1[0] - p0[0], p1[1] - p0[1]) > 0.1 and p1[2] > flat.cfg.base.fall_z


def test_a_zero_command_brings_it_to_rest(flat):
    base = flat.base
    base.set_velocity(0.3, 0.0, 0.0)
    assert base.wait_sim(1.0)
    base.set_velocity(0.0, 0.0, 0.0)
    assert base.wait_sim(0.5)  # SIM seconds: the MPC-to-stance switch rings ~0.35 of them
    assert _speed(flat) < 0.1


def test_commands_are_clipped_to_the_manifest_limits(flat):
    flat.base.set_velocity(100.0, -100.0, 100.0)
    assert flat.base._cmd == flat.cfg.base.limits[:1] + (-flat.cfg.base.limits[1],) + flat.cfg.base.limits[2:]
    flat.base.stop()


def test_a_walk_counts_sim_seconds_at_half_real_time():
    def displacement(rt_factor: float) -> float:
        dog = stand_up("go2", room=False)
        try:
            dog.rt.rt_factor = rt_factor
            p0 = dog.base.get_position()
            assert dog.base.walk(vx=0.5, duration=2.0)
            p1 = dog.base.get_position()
            return math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        finally:
            dog.close()

    full, half = displacement(1.0), displacement(0.5)
    assert full > 0.5, f"walk(0.5 m/s, 2 s) moved only {full:.3f} m"
    assert abs(half - full) <= 0.15 * full, (full, half)


def test_a_walk_holds_the_clock_so_the_load_cannot_move_its_commands(monkeypatch):
    """ledger flake:go2-navigate-fall: under CPU load a follower thread woke late and its
    commands landed on wall-clock-dependent ticks. A walk holds the sim clock between its
    commands, so on a threaded runtime it moves exactly as caller-stepped."""
    def walked(lockstep: bool) -> np.ndarray:
        if lockstep:
            monkeypatch.setenv("VECTOR_SIM_LOCKSTEP", "1")
        else:
            monkeypatch.delenv("VECTOR_SIM_LOCKSTEP", raising=False)
        dog = stand_up("go2", room=False, start=False)
        stop = threading.Event()
        def jitter() -> None:  # grabs the GIL in short bursts at random moments
            rng = random.Random()
            while not stop.is_set():
                sum(range(rng.randrange(20000)))
                time.sleep(rng.random() * 0.002)

        hogs = [threading.Thread(target=jitter, daemon=True) for _ in range(3)]
        try:
            assert dog.rt.threaded is not lockstep
            # Hold the clock at the spawn before the stepper starts: it parks at its first
            # boundary; the posture steps on this thread; the walk holds it from there on.
            dog.rt.wait_sim(0.0, hold=True)
            dog.rt.start()
            assert dog.base.set_posture("stand")
            for t in hogs:  # three threads jittering the walk's and the stepper's GIL turns
                t.start()
            assert dog.base.walk(0.4, 0.0, 0.3, 1.5)
            return dog.rt.snapshot().q.copy()
        finally:
            stop.set()
            dog.close()

    assert np.array_equal(walked(lockstep=True), walked(lockstep=False))


# -- the world-side reset --------------------------------------------------------------


def test_a_reset_stands_the_body_up_where_it_is(flat):
    """ledger flake:go2-reset-velocity-under-load: a reset is ONE step-boundary job, read
    in sim time; the body stands at its stance height (no drop to ring through)."""
    base = flat.base
    base.set_velocity(0.4, 0.0, 0.6)  # walking and turning when the reset lands
    assert base.wait_sim(1.0)
    with flat.rt.paused():
        before, yaw = base.get_position(), base.get_heading()
        respawn(flat.rt, base, flat.cfg)
        p0, yaw0 = base.get_position(), base.get_heading()
        assert _speed(flat) == 0.0 and base._cmd == (0.0, 0.0, 0.0)
    assert math.hypot(p0[0] - before[0], p0[1] - before[1]) < 1e-9
    assert p0[2] == flat.cfg.base.height and abs(math.remainder(yaw0 - yaw, 2 * math.pi)) < 1e-9
    assert base.wait_sim(1.5)
    p1 = base.get_position()
    assert _speed(flat) < 0.1 and math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 0.03
    assert abs(p1[2] - flat.cfg.base.height) < 5e-3


def test_a_reset_under_load_leaves_the_dog_still(flat):
    from vector_os_nano.sim.drivers.camera import Camera

    stop, errors = threading.Event(), []
    camera = Camera(flat.cfg.camera, flat.rt)

    def hammer() -> None:
        while not stop.is_set():
            try:
                flat.base.get_odometry()
                camera.capture(64, 48)
                flat.base.get_lidar_scan()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
                return

    loaders = [threading.Thread(target=hammer) for _ in range(3)]
    for t in loaders:
        t.start()
    try:
        flat.base.set_velocity(0.4, 0.0, 0.6)
        assert flat.base.wait_sim(1.0)
        respawn(flat.rt, flat.base, flat.cfg)
        p0 = flat.base.get_position()
        assert flat.base.wait_sim(1.5)
        p1 = flat.base.get_position()
    finally:
        stop.set()
        for t in loaders:
            t.join()
    assert not errors, errors
    assert math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 0.03, (p0, p1)
    assert sum(1 for t in threading.enumerate() if t.name == "go2_physics") == 1


# -- the convex MPC ------------------------------------------------------------------


def test_the_mpc_model_is_the_go2():
    pytest.importorskip("pinocchio", reason="pin ([go2] extra) not installed")
    from vector_os_nano.third_party.convex_mpc.go2_robot_data import PinGo2Model

    assert PinGo2Model().model.nq == 19


def test_the_mpc_plan_is_dropped_when_idle_and_failures_hold_the_stance(monkeypatch, caplog):
    monkeypatch.setenv("VECTOR_SIM_LOCKSTEP", "1")
    dog = stand_up("go2", room=False)
    base, policy = dog.base, dog.base._policy
    try:
        assert dog.rt.threaded is False
        base.set_velocity(0.3, 0.0, 0.0)
        base.wait_sim(0.2)
        assert policy._plan is not None
        base.set_velocity(0.0, 0.0, 0.0)
        base.wait_sim(0.6)  # the stop brake (MPC to zero velocity, then all four feet down)
        assert policy._plan is None and policy._tick == 0  # no stale horizon

        class _Broken:
            def solve_QP(self, *_a, **_k):
                raise RuntimeError("QP infeasible")

        policy._mpc = _Broken()
        caplog.set_level(logging.WARNING, logger="vector_os_nano.sim.policies.convex_mpc")
        base.set_velocity(0.3, 0.0, 0.0)
        base.wait_sim(1.0)
        warned = [r for r in caplog.records if "failed solves in a row" in r.getMessage()]
        assert len(warned) == 1, [r.getMessage() for r in caplog.records]
        assert policy._plan is None
        assert base.get_position()[2] > 0.2, "fell instead of holding the stance"
    finally:
        dog.close()


def test_a_missing_mpc_solver_fails_the_start_with_the_install_line(monkeypatch):
    import builtins

    real = builtins.__import__

    def no_casadi(name, *a, **k):
        if name == "casadi":
            raise ModuleNotFoundError("No module named 'casadi'")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_casadi)
    with pytest.raises(ImportError, match=r"casadi and pin: uv pip install -e '\.\[go2\]'"):
        stand_up("go2", room=False, start=False)
    assert not any(t.name == "go2_physics" for t in threading.enumerate())


# -- cameras, scenes, determinism ------------------------------------------------------


def test_camera_frames_honour_each_size_with_matching_intrinsics(flat):
    from vector_os_nano.sim.drivers.camera import Camera

    camera = Camera(flat.cfg.camera, flat.rt)
    for w, h in ((320, 240), (640, 480)):
        f = camera.capture(w, h)
        assert f.rgb.shape == (h, w, 3) and f.depth.shape == (h, w) and f.self_mask.shape == (h, w)
        assert (f.K[0, 2], f.K[1, 2]) == (w / 2, h / 2) and not hasattr(f, "seg")


def test_scenes_go_to_the_user_cache_never_the_package(monkeypatch, tmp_path):
    import mujoco

    import vector_os_nano
    from vector_os_nano.embodiments import registry
    from vector_os_nano.hardware.sim.scene_builder import room_scene
    from vector_os_nano.vcli.worlds.world_manifest import load_world

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    pkg = Path(vector_os_nano.__file__).parent
    before = set(pkg.rglob("scene_*.xml"))
    paths = [room_scene(registry.load(b), load_world("apartment")).xml_path for b in ("go2", "go2_piper")]
    for p in paths:
        assert p.is_relative_to(tmp_path), p
        mujoco.MjModel.from_xml_path(str(p))  # loads from the cache dir
    assert set(pkg.rglob("scene_*.xml")) == before
    cfg = registry.load("go2_piper")  # content-addressed: a rebuild reuses the file
    assert room_scene(cfg, load_world("apartment")).xml_path == paths[1]
    assert not list(tmp_path.rglob("*.tmp"))


def test_a_missing_model_file_fails_loud(tmp_path):
    import dataclasses

    from vector_os_nano.embodiments import registry
    from vector_os_nano.hardware.sim.scene_builder import room_scene
    from vector_os_nano.vcli.worlds.world_manifest import load_world

    cfg = registry.load("go2_piper")
    gone = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, path=str(tmp_path / "go2_piper.xml")))
    with pytest.raises(Exception, match="go2_piper.xml"):
        room_scene(gone, load_world("apartment"))


_SCRIPT = r"""
import json
from vector_os_nano.sim.session import SimSession, SimSpec
from vector_os_nano.sim.ground_truth import GroundTruth
s = SimSession.start(SimSpec("go2_piper", gui=False))
try:
    base, arm, grip = s.agent._base, s.agent._arm, s.agent._gripper
    assert s.runtime.threaded is False
    base.walk(vx=-0.3, duration=1.0)
    base.walk(vyaw=0.5, duration=1.0)
    arm.move_joints([0.2, 0.5, -0.4, 0.0, 0.3, 0.0], duration=0.5)
    grip.close(); grip.open()
    s.agent._perception.capture()
    gt = GroundTruth(s.runtime)
    print("GT=" + json.dumps({"base": gt.base_pose(), "objects": gt.object_positions(),
                              "q": s.runtime.snapshot().q.tolist()}))
finally:
    s.stop()
"""


def _lockstep_run() -> dict:
    env = {**os.environ, "VECTOR_SIM_LOCKSTEP": "1", "VECTOR_NAV_STACK": ""}
    out = subprocess.run([sys.executable, "-c", _SCRIPT], cwd=_REPO, env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    line = next(x for x in out.stdout.splitlines() if x.startswith("GT="))
    return json.loads(line[3:])


def test_two_lockstep_runs_end_in_identical_ground_truth():
    """VECTOR_SIM_LOCKSTEP=1: the same script in two fresh processes, same final GT."""
    a, b = _lockstep_run(), _lockstep_run()
    assert a == b
    assert np.linalg.norm(np.array(a["base"][:2]) - [10.0, 3.0]) > 0.2, "the scripted walk did not move the dog"


# -- physics threads and the viewer ----------------------------------------------------


class _RecordingViewer:
    """A passive viewer stand-in (no GL)."""

    def __init__(self) -> None:
        self.cam = types.SimpleNamespace(lookat=np.zeros(3))
        self.sync_threads: list[int] = []

    def sync(self) -> None:
        self.sync_threads.append(threading.get_ident())

    def close(self) -> None:
        pass


def test_a_main_thread_world_runs_no_stepper_and_its_caller_pumps(monkeypatch):
    """A world whose viewer is main-thread only (macOS) gets no stepper thread: walks and
    postures advance physics on the caller's thread."""
    from vector_os_nano.sim.backends.mujoco.world import MuJoCoWorld

    real_init = MuJoCoWorld.__init__

    def main_thread_only(self, *a, **k):
        real_init(self, *a, **k)
        self.caller_thread_only = True

    monkeypatch.setattr(MuJoCoWorld, "__init__", main_thread_only)
    dog = stand_up("go2", room=False)
    try:
        assert dog.rt.threaded is False
        assert not any(t.name == "go2_physics" for t in threading.enumerate())
        t0 = dog.base.get_sim_time()
        dog.base.walk(vx=0.3, duration=0.05)
        assert dog.base.get_sim_time() > t0 + 0.1  # duration + its settle
        assert dog.base.set_posture("sit", duration=0.1) and dog.base.get_sim_time() > t0 + 0.2
        assert not any(t.name == "go2_physics" for t in threading.enumerate())
    finally:
        dog.close()


def test_the_viewer_syncs_on_the_stepping_thread_and_follows_the_robot(flat):
    world = flat.rt.world
    viewer = _RecordingViewer()
    with flat.rt.paused():  # only this thread steps
        world._viewer, world._synced = viewer, 0.0
        try:
            flat.rt.step(1)
        finally:
            world._viewer = None
    assert viewer.sync_threads and all(t == threading.get_ident() for t in viewer.sync_threads)
    x, y, _ = flat.base.get_position()
    assert np.allclose(viewer.cam.lookat, [x, y, 0.3], atol=1e-3)
