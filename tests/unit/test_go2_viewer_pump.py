# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""R2-1: MuJoCoGo2 platform-aware physics drive (daemon vs main-thread pump).

These pin the thread-safety contract that lets the go2 window open on macOS
without the cross-thread GLFW segfault:

- headless / Linux / Windows  -> background daemon drives physics (unchanged).
- macOS / mjpython (pump mode) -> NO daemon; the caller pumps step() and the
  viewer is synced only on the caller's thread.

Run headless with stub viewers — no real GL — so they are deterministic and add
nothing to the known MUJOCO_GL cross-test pollution.
"""
import threading
import types

import pytest

pytest.importorskip("mujoco", reason="mujoco not installed")


class _RecordingViewer:
    """Minimal stand-in for a live passive viewer (no GL)."""

    def __init__(self) -> None:
        self.cam = types.SimpleNamespace(
            lookat=[0.0, 0.0, 0.0], type=0, distance=0.0, elevation=0.0, azimuth=0.0
        )
        self.sync_threads: list[int] = []

    def sync(self) -> None:
        self.sync_threads.append(threading.get_ident())

    def close(self) -> None:
        pass


def test_headless_runs_the_background_daemon():
    """gui=False -> headless drive mode -> the daemon physics thread runs."""
    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
    from vector_os_nano.hardware.sim import viewer_mode as vm

    go2 = MuJoCoGo2(gui=False)
    go2.connect()
    try:
        assert go2._drive_mode == vm.HEADLESS
        assert any(t.name == "mujoco_go2_physics" for t in threading.enumerate())
    finally:
        go2.disconnect()
    assert not any(t.name == "mujoco_go2_physics" for t in threading.enumerate())


def test_main_thread_pump_starts_no_daemon(monkeypatch):
    """Pump mode (macOS/mjpython) starts NO background thread; step() advances."""
    from vector_os_nano.hardware.sim import viewer_mode as vm
    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2

    # Force the resolver to the macOS pump decision without a real GL window.
    monkeypatch.setattr(
        vm, "resolve_viewer_drive_mode",
        lambda has_viewer, under_mjpython=None: vm.MAIN_THREAD_PUMP,
    )

    go2 = MuJoCoGo2(gui=False)
    go2.connect()
    try:
        assert go2._drive_mode == vm.MAIN_THREAD_PUMP
        assert not any(t.name == "mujoco_go2_physics" for t in threading.enumerate())
        # Physics is now caller-driven: the pump advances sim time.
        t0 = float(go2._mj.data.time)
        go2.step(5)
        assert float(go2._mj.data.time) > t0
    finally:
        go2.disconnect()


def test_step_syncs_only_on_the_caller_thread():
    """step() must call viewer.sync() on the CALLER's thread (GLFW main-thread)."""
    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2

    go2 = MuJoCoGo2(gui=False)
    go2.connect()
    try:
        with go2.runtime.paused():  # park the stepper so only step() drives sync
            viewer = _RecordingViewer()
            go2._viewer = viewer
            go2._sim_step = 0  # first step hits the viewer-sync decimation boundary
            go2.step(1)
            go2._viewer = None
        assert viewer.sync_threads, "step() did not sync the viewer"
        assert all(tid == threading.get_ident() for tid in viewer.sync_threads)
    finally:
        go2._viewer = None  # avoid disconnect closing the stub twice
        go2.disconnect()


def test_walk_pumps_physics_on_caller_thread_in_pump_mode(monkeypatch):
    """In pump mode walk() drives the gait on the caller thread (no daemon)."""
    from vector_os_nano.hardware.sim import viewer_mode as vm
    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2

    monkeypatch.setattr(
        vm, "resolve_viewer_drive_mode",
        lambda has_viewer, under_mjpython=None: vm.MAIN_THREAD_PUMP,
    )
    go2 = MuJoCoGo2(gui=False)
    go2.connect()
    try:
        assert not any(t.name == "mujoco_go2_physics" for t in threading.enumerate())
        t0 = float(go2._mj.data.time)
        go2.walk(vx=0.3, vy=0.0, vyaw=0.0, duration=0.05)
        # walk() pumped step() for duration + settle -> sim time advanced even
        # though no background daemon exists.
        assert float(go2._mj.data.time) > t0 + 0.1
    finally:
        go2.disconnect()


def test_posture_does_not_spawn_a_thread_in_pump_mode(monkeypatch):
    """A posture (stepped on the caller) must NOT start a stepper under pump mode."""
    from vector_os_nano.hardware.sim import viewer_mode as vm
    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2

    monkeypatch.setattr(
        vm, "resolve_viewer_drive_mode",
        lambda has_viewer, under_mjpython=None: vm.MAIN_THREAD_PUMP,
    )
    go2 = MuJoCoGo2(gui=False)
    go2.connect()
    try:
        t0 = go2.get_sim_time()
        go2.sit(duration=0.1)
        assert go2.get_sim_time() > t0
        assert go2.runtime.threaded is False
        assert not any(t.name == "mujoco_go2_physics" for t in threading.enumerate())
    finally:
        go2.disconnect()
