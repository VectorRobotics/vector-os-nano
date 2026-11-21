# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""``sinusoidal``: an open-loop trot for a quadruped, tracked by the manifest's PD gains.

``base.joints`` are four legs of (hip, thigh, calf) in the order FL, FR, RL, RR.
Diagonal legs swing together (FL + RR, then FR + RL). A forward command swings
thigh and calf with an amplitude that grows with vx (full at 0.5 m/s), a yaw command
swings the left and right legs against each other (full at 1 rad/s), a lateral one
sways the hips. At zero command it holds the stance (no brake: the gait has no
momentum to shed). Keys of ``base.policy``: ``freq_hz`` (steps per second),
``thigh_amp``, ``calf_amp``, ``hip_amp`` (rad).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from vector_os_nano.schemas.robot import BaseSpec
from vector_os_nano.interfaces.policy import PolicyObs
from vector_os_nano.sim.policies import pd

_TROT = (0.0, math.pi, math.pi, 0.0)  # leg phase: FL + RR, FR + RL
_LEFT = (0, 2)                        # FL, RL


class Policy:
    braking = False
    balances = False  # its stance is a PD hold: the base's own

    def __init__(self, base: BaseSpec, params: dict[str, float], tau_limit: np.ndarray, folder: Path) -> None:
        if len(base.joints) != 12:
            raise ValueError(f"sinusoidal: a quadruped of 4 x (hip, thigh, calf) joints, got {len(base.joints)}")
        self._p = params
        self._stance = np.array([base.stance[j] for j in base.joints], dtype=np.float64)
        self._vy_max = base.limits[1]
        self._kp, self._kd, self._limit = base.kp, base.kd, tau_limit
        self.tick_hz = base.policy.tick_hz

    def reset(self, obs: PolicyObs) -> None:
        """Stateless: the gait is a function of time and command."""

    def torques(self, obs: PolicyObs) -> np.ndarray:
        vx, vy, vyaw = (float(v) for v in obs.command)
        moving = vx != 0.0 or vy != 0.0 or vyaw != 0.0
        q_des = self.targets(obs.time, vx, vy, vyaw) if moving else self._stance
        return pd(q_des, obs.joint_pos, obs.joint_vel, self._kp, self._kd, self._limit)

    def targets(self, t: float, vx: float, vy: float, vyaw: float) -> np.ndarray:
        """The 12 joint targets of the trot at sim time *t* for the command."""
        p = self._p
        q = self._stance.copy()
        omega = 2.0 * math.pi * p["freq_hz"]
        fwd = float(np.clip(vx / 0.5, -1.0, 1.0)) if abs(vx) > 0.01 else 0.0
        turn = float(np.clip(vyaw / 1.0, -1.0, 1.0)) if abs(vyaw) > 0.01 else 0.0
        for leg in range(4):
            i = leg * 3
            phase = omega * t + _TROT[leg]
            # left legs push back, right legs forward: a counter-clockwise turn
            amp = float(np.clip(fwd + (-turn if leg in _LEFT else turn), -1.5, 1.5))
            if abs(vy) > 0.01:
                q[i] += p["hip_amp"] * (vy / self._vy_max) * math.sin(phase)
            # a negative amplitude pushes backward: same swing, calf half a cycle later
            calf_phase = 0.0 if amp >= 0 else math.pi
            q[i + 1] += p["thigh_amp"] * abs(amp) * math.sin(phase)
            q[i + 2] += p["calf_amp"] * abs(amp) * math.sin(phase + calf_phase)
        return q
