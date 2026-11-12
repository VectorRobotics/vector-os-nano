# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""WP06 — MuJoCoGo2 on SimRuntime (real MuJoCo, headless EGL).

Each test pins a ledger bug that the base driver had:
* reset_pose raced the stepper and left the old velocity command (reset-pose-race);
* walk() counted WALL seconds on a sim that runs < 1x (walltime-walk-navigate);
* an idle->moving transition replayed a stale MPC plan, and a failing QP was held
  silently forever (stale-mpc-plan, sim-locomotion:missed2);
* scenes were rewritten into the package dir (scene-write-into-package);
* every render size got the first caller's renderer (renderer-size-frozen).
"""
from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2  # noqa: E402

pytestmark = pytest.mark.sim

_REPO = Path(__file__).resolve().parents[3]
_PHYS = "mujoco_go2_physics"


@pytest.fixture
def flat_go2():
    go2 = MuJoCoGo2(gui=False, room=False)
    go2.connect()
    go2.stand()
    yield go2
    go2.disconnect()


def _camera(go2: MuJoCoGo2):
    """The go2 head camera (the manifest's), on the dog's runtime."""
    from vector_os_nano.embodiments import registry
    from vector_os_nano.sim.drivers.camera import Camera

    return Camera(registry.load("go2").camera, go2.runtime)


def _hammer(go2: MuJoCoGo2, stop: threading.Event, errors: list) -> None:
    while not stop.is_set():
        try:
            go2.get_odometry()
            _camera(go2).capture(64, 48)
            go2.get_lidar_scan()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            return


def test_reset_pose_under_load_leaves_the_dog_still(flat_go2) -> None:
    go2 = flat_go2
    stop, errors = threading.Event(), []
    loaders = [threading.Thread(target=_hammer, args=(go2, stop, errors)) for _ in range(3)]
    for t in loaders:
        t.start()
    try:
        go2.set_velocity(0.4, 0.0, 0.6)  # walking and turning when the reset lands
        assert go2.wait_sim(1.0)
        yaw = go2.get_heading()
        # Read the reset instant itself: the free-running stepper ticks on right after
        # the mutation resolves, so an unpaused read raced one tick (dz 1.5e-4, ~1/20).
        with go2.runtime.paused():
            go2.reset_pose()
            p0, yaw0 = go2.get_position(), go2.get_heading()
        assert abs(p0[2] - 0.35) < 1e-6  # manifest standing height
        assert abs(math.remainder(yaw0 - yaw, 2 * math.pi)) < 0.05  # heading kept
        assert go2.wait_sim(1.5)
        p1 = go2.get_position()
    finally:
        stop.set()
        for t in loaders:
            t.join()
    assert not errors, errors
    assert go2._cmd_vel == (0.0, 0.0, 0.0)
    assert math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 0.03, (p0, p1)
    assert sum(1 for t in threading.enumerate() if t.name == _PHYS) == 1


def _walk_displacement(rt_factor: float) -> float:
    go2 = MuJoCoGo2(gui=False, room=False)
    go2.connect()
    try:
        go2.stand()
        go2.runtime.rt_factor = rt_factor
        p0 = go2.get_position()
        assert go2.walk(vx=0.5, duration=2.0)
        p1 = go2.get_position()
        return math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    finally:
        go2.disconnect()


def test_walk_counts_sim_seconds_at_half_real_time() -> None:
    full, half = _walk_displacement(1.0), _walk_displacement(0.5)
    assert full > 0.5, f"walk(0.5 m/s, 2 s) moved only {full:.3f} m"
    assert abs(half - full) <= 0.15 * full, (full, half)


def test_camera_frames_honour_each_size_with_matching_intrinsics(flat_go2) -> None:
    for w, h in ((320, 240), (640, 480)):
        f = _camera(flat_go2).capture(w, h)
        assert f.rgb.shape == (h, w, 3) and f.depth.shape == (h, w) and f.self_mask.shape == (h, w)
        assert (f.K[0, 2], f.K[1, 2]) == (w / 2, h / 2) and not hasattr(f, "seg")


def test_mpc_plan_is_dropped_when_idle_and_failures_fall_back_loudly(monkeypatch, caplog) -> None:
    monkeypatch.setenv("VECTOR_SIM_LOCKSTEP", "1")
    go2 = MuJoCoGo2(gui=False, room=False, backend="mpc")
    go2.connect()
    try:
        go2.stand()
        assert go2.runtime.threaded is False
        go2.set_velocity(0.3, 0.0, 0.0)
        go2.wait_sim(0.2)
        assert go2._mpc_U_opt is not None
        go2.set_velocity(0.0, 0.0, 0.0)
        go2.wait_sim(0.6)  # the stop brake (MPC to zero velocity, then all four feet down)
        assert go2._mpc_U_opt is None and go2._mpc_ctrl_i == 0  # no stale horizon

        class _Broken:
            def solve_QP(self, *_a, **_k):
                raise RuntimeError("QP infeasible")

        go2._mpc = _Broken()
        caplog.set_level(logging.WARNING, logger="vector_os_nano.hardware.sim.mujoco_go2")
        go2.set_velocity(0.3, 0.0, 0.0)
        go2.wait_sim(1.0)
        warned = [r for r in caplog.records if "convex-MPC failed" in r.getMessage()]
        assert len(warned) == 1, [r.getMessage() for r in caplog.records]
        assert go2._mpc_U_opt is None
        assert go2.get_position()[2] > 0.2, "fell instead of holding the stance"
    finally:
        go2.disconnect()


def test_scenes_go_to_the_state_cache_never_the_package(monkeypatch, tmp_path) -> None:
    from vector_os_nano.hardware.sim import mujoco_go2

    monkeypatch.setenv("VECTOR_HOME", str(tmp_path))
    pkg = Path(mujoco_go2.__file__).parent
    before = {p for p in pkg.rglob("scene_*.xml")}
    paths = [mujoco_go2._build_flat_scene_xml(),
             mujoco_go2._build_room_scene_xml(with_arm=False),
             mujoco_go2._build_room_scene_xml(with_arm=True)]
    for p in paths:
        assert p.is_relative_to(tmp_path / "cache" / "scenes"), p
        import mujoco

        mujoco.MjModel.from_xml_path(str(p))  # loads from the cache dir
    assert {p for p in pkg.rglob("scene_*.xml")} == before
    # content-addressed: a rebuild reuses the same file, nothing half-written left over
    assert mujoco_go2._build_room_scene_xml(with_arm=True) == paths[2]
    assert not list(tmp_path.rglob("*.tmp"))


_SCRIPT = r"""
import json, sys
from vector_os_nano.sim.session import SimSession, SimSpec
from vector_os_nano.sim.ground_truth import GroundTruth
s = SimSession.start(SimSpec("go2_piper", gui=False))
try:
    base, arm, grip = s.agent._base, s.agent._arm, s.agent._gripper
    assert base.runtime.threaded is False
    base.walk(vx=-0.3, duration=1.0)
    base.walk(vyaw=0.5, duration=1.0)
    arm.move_joints([0.2, 0.5, -0.4, 0.0, 0.3, 0.0], duration=0.5)
    grip.close(); grip.open()
    s.agent._perception.capture()
    gt = GroundTruth(base.runtime)
    print("GT=" + json.dumps({"base": gt.base_pose(), "objects": gt.object_positions(),
                              "q": base.runtime.snapshot().q.tolist()}))
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


def test_two_lockstep_runs_end_in_identical_ground_truth() -> None:
    """VECTOR_SIM_LOCKSTEP=1: the same script in two fresh processes, same final GT."""
    a, b = _lockstep_run(), _lockstep_run()
    assert a == b
    base = np.array(a["base"][:2])
    assert np.linalg.norm(base - [10.0, 3.0]) > 0.2, "the scripted walk did not move the dog"


def test_missing_mpc_stack_falls_back_loudly_and_reports_the_gait(monkeypatch, caplog) -> None:
    def _no_casadi(self):
        raise ModuleNotFoundError("No module named 'casadi'")

    monkeypatch.setattr(MuJoCoGo2, "_init_mpc_stack", _no_casadi)
    caplog.set_level(logging.WARNING, logger="vector_os_nano.hardware.sim.mujoco_go2")
    go2 = MuJoCoGo2(gui=False, room=False, backend="auto")
    go2.connect()
    try:
        assert go2.gait_backend == "sinusoidal"
        msg = " ".join(r.getMessage() for r in caplog.records)
        assert "convex-MPC gait unavailable" in msg and "third_party/convex_mpc" in msg, msg
    finally:
        go2.disconnect()


def test_arm_scene_model_missing_fails_loud(monkeypatch, tmp_path) -> None:
    from vector_os_nano.hardware.sim import mujoco_go2

    monkeypatch.setattr(mujoco_go2, "_GO2_PIPER_XML", tmp_path / "go2_piper.xml")
    with pytest.raises(FileNotFoundError, match="go2\\+arm scene model missing"):
        mujoco_go2._build_room_scene_xml(with_arm=True)
