# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""sim_tick_dt — the sim-dt source for the go2 bridge path-follower ramps.

The bridge's _follow_path ticks on a 20 Hz WALL timer but must integrate its
velocity ramps against SIM time (the physics daemon often runs <1x). These
tests pin the contract: nominal fallback when no clock / first tick / reset,
true sim-dt otherwise, 0 when paused, clamped after a stall.
"""

from __future__ import annotations

import pytest

from vector_os_nano.sim.nav.follow import sim_tick_dt

NOMINAL = 1.0 / 20.0


def test_no_sim_clock_falls_back_to_nominal() -> None:
    # Real hardware / unreadable clock: pre-fix wall-tick behavior.
    assert sim_tick_dt(None, 10.0, NOMINAL) == NOMINAL
    assert sim_tick_dt(None, None, NOMINAL) == NOMINAL


def test_first_tick_is_nominal() -> None:
    assert sim_tick_dt(12.34, None, NOMINAL) == NOMINAL


def test_normal_progress_returns_true_sim_dt() -> None:
    # Sim at ~0.65x real-time: one wall tick advances ~0.0325 sim-seconds.
    dt = sim_tick_dt(100.0325, 100.0, NOMINAL)
    assert dt == pytest.approx(0.0325)


def test_realtime_sim_is_byte_identical_to_wall_tick() -> None:
    # sim/wall == 1: dt equals the nominal tick — ramp rates unchanged.
    assert sim_tick_dt(100.05, 100.0, NOMINAL) == pytest.approx(NOMINAL)


def test_paused_sim_freezes_ramps() -> None:
    assert sim_tick_dt(100.0, 100.0, NOMINAL) == 0.0


def test_backwards_jump_means_sim_reset_uses_nominal() -> None:
    assert sim_tick_dt(0.1, 100.0, NOMINAL) == NOMINAL


def test_stall_recovery_is_clamped() -> None:
    # A 2 s hiccup must not slew the ramp 40 ticks at once.
    assert sim_tick_dt(102.0, 100.0, NOMINAL) == pytest.approx(4.0 * NOMINAL)


def test_custom_clamp() -> None:
    assert sim_tick_dt(102.0, 100.0, NOMINAL, max_ticks=2.0) == pytest.approx(
        2.0 * NOMINAL
    )
