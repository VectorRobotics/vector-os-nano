# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""SimRuntime: one PhysicsWorld owner, step-boundary mutations, atomic snapshots; renders
of the world it steps observe one instant."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytestmark = pytest.mark.sim  # renders through EGL

from vector_os_nano.sim.backends.mujoco.world import MuJoCoWorld  # noqa: E402
from vector_os_nano.sim.ground_truth import GroundTruth  # noqa: E402
from vector_os_nano.sim.runtime import SimRuntime  # noqa: E402
from vector_os_nano.sim.scene import compiled_scene  # noqa: E402

# A camera on a body that slides along x at a constant 2 m/s (the controller holds the
# velocity), so x = 2 * t (10 mm per 5 ms tick), looking down +x at a wall whose near
# face is at x = 10. A frame whose depth and pose came from different instants is off
# by >= one tick (10 mm).
_XML = """
<mujoco>
  <option timestep="0.001"/>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="20 20 0.1"/>
    <geom name="wall" type="box" pos="10.05 0 1" size="0.05 5 5" rgba="1 0 0 1"/>
    <body name="ball" pos="0 2 1">
      <freejoint/>
      <geom type="sphere" size="0.1"/>
    </body>
    <body name="rig" pos="0 0 1">
      <joint name="rail" type="slide" axis="1 0 0"/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      <camera name="cam" xyaxes="0 -1 0 0 0 1" fovy="45"/>
    </body>
  </worldbody>
</mujoco>
"""
_SPEED = 2.0
_RAIL_QD = 6  # after the ball's 6 velocity slots


@pytest.fixture(scope="module")
def scene(tmp_path_factory: pytest.TempPathFactory):
    path = tmp_path_factory.mktemp("rt") / "rig.xml"
    path.write_text(_XML)
    return compiled_scene(path)


def _runtime(scene, **kw) -> SimRuntime:
    def ctrl(w):  # the camera rig position is a pure function of sim time
        w.data.qvel[_RAIL_QD] = _SPEED

    kw.setdefault("substeps", 5)
    return SimRuntime(MuJoCoWorld(scene), controller=ctrl, name="test_rt", **kw)


def _physics_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "test_rt_physics"]


@pytest.fixture
def rt(scene):
    r = _runtime(scene)
    r.start()
    yield r
    r.close()


def test_four_threads_x1000_calls_while_stepping(rt) -> None:
    """snapshot/render/submit hammering from 4 threads: no error, one stepper, monotonic
    time (per source: a render observes the world, up to one tick ahead of the snapshot)."""
    errors: list[BaseException] = []

    def worker(k: int) -> None:
        last = {"snap": -1.0, "render": -1.0}
        try:
            for i in range(1000):
                op = (i + k) % 3
                if op == 0:
                    t = rt.snapshot().sim_time
                    assert t >= last["snap"], (t, last)
                    last["snap"] = t
                elif op == 1:
                    q = rt.call(lambda w: w.state()[0])
                    assert q.shape == (8,)
                    rt.submit(lambda w: w.data.qvel.__setitem__(slice(0, 3), 0.0))
                else:
                    f = rt.world.render("cam", 16, 12, ("rgb", "depth"))
                    assert f.rgb.shape == (12, 16, 3) and f.depth.shape == (12, 16)
                    assert f.sim_time >= last["render"]
                    last["render"] = f.sim_time
        except BaseException as exc:  # noqa: BLE001 — surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    t0 = rt.sim_time
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not errors, errors
    assert len(_physics_threads()) == 1
    assert rt.sim_time > t0
    assert rt.wait_sim(0.05), "physics stopped stepping"


def test_rgbd_frame_fields_share_one_sim_time(rt) -> None:
    """Pose, depth and the frame stamp are of ONE instant, while the camera moves."""
    assert rt.wait_sim(0.05)
    for _ in range(20):
        f = rt.world.render("cam", 64, 48, ("rgb", "depth"))
        assert f.cam_pos[0] == pytest.approx(_SPEED * f.sim_time, abs=1e-9)
        centre = float(f.depth[24, 32])
        assert centre == pytest.approx(10.0 - f.cam_pos[0], abs=5e-3)


def test_submit_is_applied_and_published_before_its_result(rt) -> None:
    def teleport(w):
        w.data.qpos[0] = 3.25
        w.data.qvel[:] = 0.0
        return "done"

    assert rt.submit(teleport).result(timeout=5) == "done"
    # the published snapshot already carries the mutation (no tick needed)
    assert rt.snapshot().q[0] == pytest.approx(3.25)


def test_wait_sim_counts_sim_seconds_at_half_real_time(rt) -> None:
    rt.rt_factor = 0.5
    rt.wait_sim(0.1)  # settle onto the slower schedule
    s0, w0 = rt.sim_time, time.monotonic()
    assert rt.wait_sim(0.5)
    sim, wall = rt.sim_time - s0, time.monotonic() - w0
    assert sim >= 0.5
    assert 0.8 < wall < 1.6, f"0.5 sim-s took {wall:.2f} wall-s at rt_factor 0.5"


def test_pause_is_reference_counted_and_never_starts_a_second_stepper(rt) -> None:
    with rt.paused():
        with rt.paused():
            t = rt.sim_time
            time.sleep(0.05)
            assert rt.sim_time == t, "stepped while paused"
            # mutations are still applied by the parked stepper
            assert rt.call(lambda w: w.time()) == t
        time.sleep(0.05)
        assert rt.sim_time == t, "one release resumed a doubly-paused runtime"
        rt.step(4)  # caller-thread steps while parked
        assert rt.sim_time == pytest.approx(t + 4 * rt.tick_dt)
    assert rt.wait_sim(0.05)
    assert len(_physics_threads()) == 1


def test_close_fails_pending_work_and_stops_the_stepper(scene) -> None:
    r = _runtime(scene)
    r.start()
    r.world.render("cam", 16, 12, ("rgb",))
    with r.paused():
        pass
    r.close()
    assert not _physics_threads()
    with pytest.raises(RuntimeError):
        r.submit(lambda w: None).result(timeout=1)
    with pytest.raises(RuntimeError):
        r.world.render("cam", 16, 12)  # the runtime closed its world
    r.close()  # idempotent


def _scripted_run(scene) -> np.ndarray:
    r = _runtime(scene)  # lockstep: no thread; only the script advances physics
    r.start()
    try:
        assert not _physics_threads()
        t = r.sim_time
        time.sleep(0.05)
        assert r.sim_time == t, "lockstep physics moved without being stepped"
        r.call(lambda w: w.data.qvel.__setitem__(slice(0, 3), [0.3, -0.2, 2.0]))
        assert r.wait_sim(0.4)
        r.world.render("cam", 32, 24, ("rgb",))  # a render must not perturb the run
        r.step(17)
        return r.snapshot().q.copy()
    finally:
        r.close()


def test_lockstep_runs_are_bitwise_identical(scene, monkeypatch) -> None:
    monkeypatch.setenv("VECTOR_SIM_LOCKSTEP", "1")
    a, b = _scripted_run(scene), _scripted_run(scene)
    assert np.array_equal(a, b), (a, b)
    assert a[2] < 1.0, "the ball should have fallen"


def test_call_times_out_when_the_stepper_stalls(scene) -> None:
    """sim-runtime:call-has-no-timeout — a stalled stepper (a hung controller) never
    hangs a caller: call() raises, naming the runtime, and the job never runs late."""
    gate, ran = threading.Event(), []
    r = SimRuntime(MuJoCoWorld(scene), controller=lambda w: gate.wait(), name="stalled", paced=False)
    r.start()
    try:
        t0 = time.monotonic()
        with pytest.raises(RuntimeError, match="stalled"):
            r.call(lambda w: ran.append(1), timeout=0.3)
        assert time.monotonic() - t0 < 2.0
        gate.set()
        assert r.wait_sim(0.05) and ran == []  # cancelled, not applied after the caller gave up
    finally:
        gate.set()
        r.close()


def test_a_dead_stepper_fails_every_later_call_loudly(scene) -> None:
    """sim-runtime:stepper-death-is-silent — after the physics thread dies, the runtime
    never degrades to caller-stepped: every call, wait and step raises its reason."""
    def boom(w) -> None:
        raise ValueError("solver exploded")

    r = SimRuntime(MuJoCoWorld(scene), controller=boom, name="dying", paced=False)
    r.start()
    try:
        deadline = time.monotonic() + 5.0
        while r.daemon_alive and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not r.daemon_alive and "solver exploded" in r.dead_reason
        gt = GroundTruth(r)
        # S1V-3: snapshot readers (ground truth among them) fail closed, never serve the
        # last state the dead stepper published as if it were now.
        for poke in (lambda: r.call(lambda w: None), lambda: r.wait_sim(0.01), lambda: r.step(1),
                     lambda: r.sim_time, r.snapshot, gt.object_positions,
                     lambda: gt.clearance("ball"), lambda: gt.joint_positions(("rail",)),
                     lambda: gt.visible_pixels("ball", "cam", 32, 24)):
            with pytest.raises(RuntimeError, match="solver exploded"):
                poke()
    finally:
        r.close()


# -- the clock hold (ledger flake:go2-navigate-fall) -----------------------------------


def _logging_runtime(scene, cmd: list) -> tuple[SimRuntime, list]:
    """A runtime whose controller logs (tick, the command it sees)."""
    log: list = []
    return SimRuntime(MuJoCoWorld(scene), controller=lambda w: log.append((round(w.time() / 0.005), cmd[0])),
                      substeps=5, name="test_rt"), log


def test_a_held_clock_lands_every_command_on_the_tick_after_its_wait(scene) -> None:
    """A control loop that waits with hold sees its commands on exactly the ticks it
    would caller-stepped, while three threads fight it for the GIL."""
    cmd = [None]
    rt, log = _logging_runtime(scene, cmd)
    stop = threading.Event()
    hogs = [threading.Thread(target=lambda: [sum(range(2000)) for _ in iter(stop.is_set, True)], daemon=True)
            for _ in range(3)]
    try:
        assert rt.wait_sim(0.0, hold=True)  # before the stepper starts: it parks at its first boundary
        rt.start()
        for t in hogs:
            t.start()
        for i in range(30):
            cmd[0] = i
            assert rt.wait_sim(0.01, hold=True)  # two ticks, then parked again
        rt.release_clock()
    finally:
        stop.set()
        rt.close()
    assert {t: c for t, c in log if t < 60} == {t: t // 2 for t in range(60)}


def test_a_hold_is_only_its_threads_and_a_forgotten_one_is_dropped(scene, monkeypatch) -> None:
    """Adversarial: another thread's release leaves the hold; a holder that never comes
    back does not freeze the simulation (dropped after _HOLD_WALL_S)."""
    from vector_os_nano.sim import runtime as rt_mod

    monkeypatch.setattr(rt_mod, "_HOLD_WALL_S", 0.6)
    rt, _log = _logging_runtime(scene, [None])
    rt.start()
    try:
        assert rt.wait_sim(0.0, hold=True)  # parked now, by this thread
        t0 = rt.sim_time
        other = threading.Thread(target=rt.release_clock)
        other.start()
        other.join()
        time.sleep(0.3)
        assert rt.sim_time == t0, "another thread's release dropped this thread's hold"
        time.sleep(1.0)
        assert rt.sim_time > t0, "a forgotten hold froze the simulation"
    finally:
        rt.close()


@pytest.mark.parametrize('hold', [False, True])
def test_sensor_backed_cancellation_does_not_hold_the_publication_lock(rt, hold):
    target = rt.sim_time + .03
    class SensorStop:
        def is_set(self):
            return rt.call(lambda world: world.time(), timeout=.5) >= target
    assert not rt.wait_sim(1., cancel=SensorStop(), hold=hold)
    rt.release_clock()
    assert rt.wait_sim(.01), 'the physics publisher must remain responsive after the stop'
