# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""SimRuntime: one PhysicsWorld owner, step-boundary mutations, atomic snapshots/renders."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytestmark = pytest.mark.sim  # renders through EGL

from vector_os_nano.sim.backends.mujoco.world import MuJoCoWorld  # noqa: E402
from vector_os_nano.sim.runtime import SimRuntime  # noqa: E402
from vector_os_nano.sim.scene import compiled_scene  # noqa: E402

# A camera on a mocap body that the controller drives to x = 2 * t (10 mm per 5 ms
# tick), looking down +x at a wall whose near face is at x = 10. A frame whose
# depth and pose came from different instants is off by >= one tick (10 mm).
_XML = """
<mujoco>
  <option timestep="0.001"/>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="20 20 0.1"/>
    <geom name="wall" type="box" pos="10.05 0 1" size="0.05 5 5" rgba="1 0 0 1"/>
    <body name="rig" mocap="true" pos="0 0 1">
      <camera name="cam" xyaxes="0 -1 0 0 0 1" fovy="45"/>
    </body>
    <body name="ball" pos="0 2 1">
      <freejoint/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""
_SPEED = 2.0


@pytest.fixture(scope="module")
def scene(tmp_path_factory: pytest.TempPathFactory):
    path = tmp_path_factory.mktemp("rt") / "rig.xml"
    path.write_text(_XML)
    return compiled_scene(path)


def _runtime(scene, **kw) -> SimRuntime:
    def ctrl(w):  # the camera rig position is a pure function of sim time
        w.data.mocap_pos[0, 0] = _SPEED * w.data.time

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
    """snapshot/render/submit hammering from 4 threads: no error, one stepper, monotonic time."""
    errors: list[BaseException] = []

    def worker(k: int) -> None:
        last = -1.0
        try:
            for i in range(1000):
                op = (i + k) % 3
                if op == 0:
                    t = rt.snapshot().sim_time
                    assert t >= last, (t, last)
                    last = t
                elif op == 1:
                    q = rt.call(lambda w: w.state()[0])
                    assert q.shape == (7,)
                    rt.submit(lambda w: w.data.qvel.__setitem__(slice(0, 3), 0.0))
                else:
                    f = rt.render("cam", 16, 12, ("rgb", "depth"))
                    assert f.rgb.shape == (12, 16, 3) and f.depth.shape == (12, 16)
                    assert f.sim_time >= last
                    last = f.sim_time
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
        f = rt.render("cam", 64, 48, ("rgb", "depth"))
        # the controller set x = 2 t at the start of the tick that ended at sim_time
        assert f.cam_pos[0] == pytest.approx(_SPEED * (f.sim_time - rt.tick_dt), abs=1e-9)
        centre = float(f.depth[24, 32])
        assert centre == pytest.approx(10.0 - f.cam_pos[0], abs=5e-3)


def test_render_honours_each_size_with_matching_intrinsics(rt) -> None:
    for w, h in ((320, 240), (640, 480), (320, 240)):
        f = rt.render("cam", w, h, ("rgb",))
        assert f.rgb.shape == (h, w, 3)
        assert (f.K.width, f.K.height) == (w, h)
        assert f.K.fy == pytest.approx((h / 2) / np.tan(np.radians(45) / 2))


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
    r.render("cam", 16, 12, ("rgb",))
    with r.paused():
        pass
    r.close()
    assert not _physics_threads()
    with pytest.raises(RuntimeError):
        r.submit(lambda w: None).result(timeout=1)
    with pytest.raises(RuntimeError):
        r.render("cam", 16, 12)
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
        r.render("cam", 32, 24, ("rgb",))  # a render must not perturb the run
        r.step(17)
        return r.snapshot().q.copy()
    finally:
        r.close()


def test_lockstep_runs_are_bitwise_identical(scene, monkeypatch) -> None:
    monkeypatch.setenv("VECTOR_SIM_LOCKSTEP", "1")
    a, b = _scripted_run(scene), _scripted_run(scene)
    assert np.array_equal(a, b), (a, b)
    assert a[2] < 1.0, "the ball should have fallen"
