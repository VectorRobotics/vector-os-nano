# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""A resumed brake keeps its gait phase; a restart from a stance begins feet-down."""
from types import SimpleNamespace

import numpy as np
import pytest

from vector_os_nano.sim.policies.convex_mpc import Policy


@pytest.mark.parametrize("resume_at, expected_phase", [(10.2, .2), (10.6, 0.)])
def test_restart_during_braking_or_stance_ramp(resume_at, expected_phase):
    p = Policy.__new__(Policy)
    p._stance = np.zeros(12)
    p._kp = p._kd = 1.
    p._limit = np.ones(12)
    p._p = {"brake_s": .2, "brake_ramp_s": 1.}
    p._gait = SimpleNamespace(compute_current_mask=lambda t: np.ones(4))
    phases = []

    def torque(obs, vx, vy, vyaw, t):
        phases.append(t)
        return np.zeros(12)

    p._mpc_torque = torque
    p._fresh()

    def tick(t, vx):
        return p.torques(SimpleNamespace(time=t, command=(vx, 0., 0.),
                                        joint_pos=np.zeros(12), joint_vel=np.zeros(12)))

    tick(10., .3)
    tick(10.1, 0.)
    if resume_at > 10.5:
        tick(10.5, 0.)
    tick(resume_at, .3)
    assert phases[-1] == pytest.approx(expected_phase)
