# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""WP05 E5 on the go2 (LeggedBase + convex_mpc, flat ground, headless): every stop brakes the body.

Dropping a 0.4 m/s trot straight onto the PD stance mid-stride kicked it to
0.35-0.53 m/s at 0.2 sim-s (measured, 8 trials) — after a stop() and after a walk's
own zero command alike (navigate pauses between its steps, so a Ctrl-C mid-navigate
often lands in that kick). The convex_mpc policy brakes: it keeps the MPC tracking zero
velocity, hands over to the stance with all four feet down, then ramps to the stand pose.
"""
from __future__ import annotations

import math
import time
from typing import Any

import pytest

pytest.importorskip("mujoco")

from tests.hardware.sim.dog import stand_up  # noqa: E402

pytestmark = pytest.mark.sim

# Measured with the brake (in-process, 0.4 m/s trot, 24 trials on flat ground, 8 in the
# room): 0.005-0.12 m/s at 0.2 sim-s, peak <= 0.12 m/s from there to 1.0 sim-s; the
# stance ramp creeps < 0.05 m/s at 1.0 sim-s. Without it: 0.35-0.46 at 0.2, peak 0.53.
_STOPPED_MPS = 0.15


@pytest.fixture(scope="module")
def go2():
    dog = stand_up("go2", room=False)
    assert dog.base.gait_backend == "convex_mpc"
    yield dog.base
    dog.close()


def _wait_sim(dog, seconds: float) -> None:
    t0 = dog.get_sim_time()
    while dog.get_sim_time() - t0 < seconds:
        time.sleep(0.002)


def _speed(dog) -> float:
    v = dog.get_velocity()
    return math.hypot(v[0], v[1])


def _stop_profile(dog, stop: Any) -> tuple[float, float]:
    """(speed at 0.2 sim-s, peak speed from 0.2 to 1.0 sim-s) after *stop*()."""
    t0 = dog.get_sim_time()
    stop()
    peak, at_02 = 0.0, None
    while (dt := dog.get_sim_time() - t0) < 1.0:
        if dt >= 0.2:
            at_02 = _speed(dog) if at_02 is None else at_02
            peak = max(peak, _speed(dog))
        time.sleep(0.002)
    return at_02, peak


@pytest.mark.parametrize("how", ["stop", "zero_command"])
@pytest.mark.parametrize("cmd", [(0.4, 0.0, 0.0), (0.3, 0.0, 0.6)])
def test_a_stop_brakes_a_walking_body_within_0_2_sim_s(go2, cmd, how) -> None:
    go2.set_velocity(*cmd)
    _wait_sim(go2, 1.7)
    assert _speed(go2) > 0.1  # it is walking
    stop = go2.stop if how == "stop" else (lambda: go2.set_velocity(0.0, 0.0, 0.0))
    at_02, peak = _stop_profile(go2, stop)
    assert at_02 < _STOPPED_MPS and peak < _STOPPED_MPS, (at_02, peak)
    _wait_sim(go2, 1.0)  # the stance ramp ends; the dog stands still and upright
    assert _speed(go2) < 0.02 and go2.get_position()[2] > 0.2
    assert not go2._policy.braking and go2.wait_settled(timeout=0.0)


def test_a_standing_body_is_left_alone(go2) -> None:
    _wait_sim(go2, 0.1)
    assert not go2._policy.braking
    go2.stop()  # standing still: nothing to brake
    _wait_sim(go2, 0.05)
    assert not go2._policy.braking and go2.wait_settled(timeout=0.0)


def test_a_new_motion_ends_the_brake(go2) -> None:
    go2.set_velocity(0.3, 0.0, 0.0)
    _wait_sim(go2, 1.0)
    go2.stop()
    _wait_sim(go2, 0.05)
    assert go2._policy.braking and not go2.wait_settled(timeout=0.0)
    go2.set_velocity(0.3, 0.0, 0.0)
    _wait_sim(go2, 0.05)
    assert not go2._policy.braking
    go2.stop()
    _wait_sim(go2, 2.0)
