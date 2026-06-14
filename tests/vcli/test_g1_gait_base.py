# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #5 batch 2 — G1MuJoCoBase: real gait, real physics evidence.

Every motion number asserted here comes from MuJoCo state (qpos), never from
command echo — the kernel's motion-evidence contract holds on real physics.
Tests are skipped when scripts/setup_g1_gait.sh has not installed the assets
(CI/machines without the 57 MB download); on the dev desktop they RUN.
"""
from __future__ import annotations

import math
import time

import pytest

from vector_os_nano.hardware.sim.mujoco_g1 import (
    G1MuJoCoBase,
    g1_assets_ready,
)

pytestmark = pytest.mark.skipif(
    not g1_assets_ready(),
    reason="g1_gait assets not installed (run scripts/setup_g1_gait.sh)")


@pytest.fixture()
def base():
    b = G1MuJoCoBase()
    b.connect()
    yield b
    b.disconnect()    # instance hygiene: never leak a sim thread


class TestMissingAssetsFailLoud:
    def test_names_the_setup_script(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="setup_g1_gait"):
            G1MuJoCoBase(asset_dir=tmp_path)


class TestRealGaitMotion:
    def test_walk_forward_displaces_by_physics(self, base):
        start = base.get_position()
        base.walk(0.5, 0.0, 0.0, duration=3.0)
        end = base.get_position()
        moved = math.hypot(end[0] - start[0], end[1] - start[1])
        # 3s at 0.5 m/s commanded; the REAL gait delivers ~0.46 m/s steady
        # with ~1s ramp-up from standstill, and the paced loop may run at
        # ~0.8x wall on a loaded desktop — require genuine walking, not a
        # speed-precision bound (the pacing itself is asserted separately).
        assert moved >= 0.55, f"real displacement too small: {moved:.2f} m"
        assert base.get_position()[2] > 0.45      # still upright, not fallen

    def test_lateral_walk_is_real_motion(self, base):
        # supports_holonomic: vy drives REAL sideways stepping here (the
        # habitat base honestly refuses this — two worlds, two truths).
        assert G1MuJoCoBase.supports_holonomic is True
        start = base.get_position()
        base.walk(0.0, 0.3, 0.0, duration=3.0)
        end = base.get_position()
        moved = math.hypot(end[0] - start[0], end[1] - start[1])
        assert moved >= 0.25, f"lateral displacement too small: {moved:.2f} m"

    def test_stop_halts_motion(self, base):
        base.set_velocity(0.5, 0.0, 0.0)
        time.sleep(1.0)
        base.stop()
        time.sleep(0.5)
        p1 = base.get_position()
        time.sleep(0.7)
        p2 = base.get_position()
        drift = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        assert drift < 0.25, f"robot kept moving after stop: {drift:.2f} m"

    def test_deadman_stops_a_dead_client(self, base):
        base.set_velocity(0.5, 0.0, 0.0)   # ONE command, never re-armed
        time.sleep(1.6)                     # > deadman (0.6s) + decel
        p1 = base.get_position()
        time.sleep(0.7)
        p2 = base.get_position()
        drift = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        assert drift < 0.25, "deadman did not stop the march"

    def test_sim_paces_toward_real_time(self, base):
        t0 = base._data.time
        time.sleep(1.5)
        advanced = base._data.time - t0
        # Lower bound documents desktop reality (~0.6x under load/CPU
        # governor); the hard contract is the UPPER bound — pacing must
        # never run the gait faster than wall time (campaign #3 lesson:
        # unpaced motion is invisible motion). Exact-1x polish is a
        # batch-3 GUI item.
        assert advanced >= 0.8, f"sim too slow: {advanced:.2f}s in 1.5s wall"
        assert advanced <= 1.65, f"sim too fast (pacing broken): {advanced:.2f}s"

    def test_odometry_is_physics_state(self, base):
        odo = base.get_odometry()
        assert odo.timestamp > 0
        assert odo.z > 0.45                       # upright, real height
        # quaternion normalized (real physics state, not the qw=1 default)
        n = math.sqrt(odo.qw**2 + odo.qx**2 + odo.qy**2 + odo.qz**2)
        assert abs(n - 1.0) < 1e-3

    def test_concurrent_odometry_never_torn(self, base):
        """The snapshot fix (workflow critical): hammer get_odometry from a
        reader thread WHILE the gait walks. A torn read mixing two physics
        states would denormalize the quaternion — assert every sample stays
        unit-norm. Also exercises get_velocity off-thread."""
        import threading

        norms = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                o = base.get_odometry()
                norms.append(
                    math.sqrt(o.qw**2 + o.qx**2 + o.qy**2 + o.qz**2))
                base.get_velocity()

        t = threading.Thread(target=reader)
        t.start()
        base.walk(0.4, 0.0, 0.3, duration=2.0)
        stop.set()
        t.join()
        assert len(norms) > 100, f"reader sampled too few: {len(norms)}"
        assert all(abs(n - 1.0) < 1e-3 for n in norms), \
            "torn quaternion read — snapshot not atomic"

    def test_reads_after_disconnect_fail_loud(self):
        b = G1MuJoCoBase()
        b.connect()
        b.disconnect()
        with pytest.raises(RuntimeError, match="not connected"):
            b.get_position()

    def test_viewer_frame_is_valid_png(self, base):
        png = base.viewer_frame_png()
        assert png[:8] == b"\x89PNG\r\n\x1a\n"   # valid PNG signature
        assert len(png) > 1000                    # a real rendered frame

    def test_viewer_frames_differ_while_walking(self, base):
        """The GUI acceptance evidence: chase-cam frames captured DURING a
        walk must differ (the robot visibly steps/advances, not a frozen
        glide). Rendered on the control thread — no torn pose."""
        import threading

        t = threading.Thread(
            target=lambda: base.walk(0.5, 0.0, 0.0, duration=3.0), daemon=True)
        t.start()
        time.sleep(0.6)
        f1 = base.viewer_frame_png()
        time.sleep(1.2)
        f2 = base.viewer_frame_png()
        t.join()
        assert f1 != f2, "viewer frames identical during a walk — not moving"
