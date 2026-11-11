# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The locomotion-controller contract: a legged base is a manifest plus a policy.

A policy is chosen by ``base.policy.ref`` in ``robot.yaml`` and is a row in
``vector_os_nano.sim.policies.POLICIES``. It is simulator-neutral: it sees a
:class:`PolicyObs` and returns joint torques, nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class PolicyObs:
    """What a policy observes at one control tick. Frames are fixed here, not per policy."""

    time: float                 # sim seconds since the policy was reset
    base_pos: np.ndarray        # (3,) world
    base_quat: np.ndarray       # (4,) w x y z, world <- body
    base_lin_vel: np.ndarray    # (3,) WORLD frame
    base_ang_vel: np.ndarray    # (3,) BODY frame
    joint_pos: np.ndarray       # (n,) in the manifest's base.joints order
    joint_vel: np.ndarray       # (n,)
    command: np.ndarray         # (3,) vx, vy in the body frame (m/s), vyaw (rad/s)


@runtime_checkable
class ControlPolicy(Protocol):
    """Velocity command -> joint torques, called at ``tick_hz``.

    Braking and the hand-over to a stance at zero command are the policy's job.
    Postures (stand / sit / lie) are not: the base driver holds them itself.
    """

    tick_hz: float              # how often torques() is called; physics substeps = physics_hz / tick_hz

    def reset(self, obs: PolicyObs) -> None:
        """Start from *obs* (after a spawn, a posture, or a body reset)."""

    def torques(self, obs: PolicyObs) -> np.ndarray:
        """(n,) joint torques in the manifest's base.joints order, held until the next tick."""
