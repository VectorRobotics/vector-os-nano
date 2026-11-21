# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""``convex_mpc``: the Go2 trot of the vendored centroidal convex MPC
(``third_party/convex_mpc``: a Pinocchio model of the Go2, a gait scheduler, the QP,
swing / stance leg control).

Moving: the MPC force plan, re-solved every ``mpc_dt`` (a sixteenth of the gait
cycle), turned into leg torques. The gait clock starts afresh at each motion from
rest, every foot down (a clock counted from the spawn started the trot mid-cycle: a
foot the schedule had mid-swing lifted on a full swing the schedule cut short and the
MPC loaded it in the air; measured, 2 in 200 walks from rest fell); the leg
controller's swing state follows it, since its first tick records every foot in
stance. Zero
command after moving: it BRAKES, tracking zero velocity for ``brake_s``, hands over to
a PD hold of its four-feet-down pose once all feet are down, then ramps that pose to the
stance over ``brake_ramp_s`` (dropping a trot straight onto the stance mid-stride kicked
the body to 0.35-0.53 m/s). Idle: a PD hold of the stance. A failed solve drops the
plan and holds the last torque; after ``_MAX_FAILS`` in a row it holds the stance and
warns (rate-limited in sim time) until a solve works (tricky-bugs Case 2: never a stale
plan).

Keys of ``base.policy``: ``gait_hz``, ``duty`` (stance fraction), ``z_des`` (trunk
height, m), ``brake_s``, ``brake_ramp_s``. The solver needs casadi and Pinocchio
(``uv pip install -e '.[go2]'``); without them this policy fails at construction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from vector_os_nano.schemas.robot import BaseSpec
from vector_os_nano.interfaces.policy import PolicyObs
from vector_os_nano.sim.policies import pd

logger = logging.getLogger(__name__)

_LEGS = ("FL", "FR", "RL", "RR")  # the Pinocchio model's legs, in base.joints order
_DT_FACTOR = 16                    # MPC steps per gait cycle
_MAX_FAILS = 5
_WARN_EVERY_S = 5.0                # sim seconds between two "MPC failing" warnings
INSTALL_HINT = "convex_mpc needs casadi and pin: uv pip install -e '.[go2]'"


class Policy:
    balances = False  # its stance is a PD hold: the base's own

    def __init__(self, base: BaseSpec, params: dict[str, float], tau_limit: np.ndarray, folder: Path) -> None:
        try:
            # casadi is imported lazily by the QP on its FIRST solve: import it here, so a
            # missing solver fails the start, never every tick (tricky-bugs Case 0)
            import casadi  # noqa: F401
            import pinocchio  # noqa: F401

            from vector_os_nano.third_party.convex_mpc.com_trajectory import ComTraj
            from vector_os_nano.third_party.convex_mpc.gait import Gait
            from vector_os_nano.third_party.convex_mpc.go2_robot_data import PinGo2Model
            from vector_os_nano.third_party.convex_mpc.leg_controller import LegController
        except ImportError as exc:
            raise ImportError(f"{INSTALL_HINT} ({exc})") from exc
        if len(base.joints) != 3 * len(_LEGS):
            raise ValueError(f"convex_mpc drives the Go2's 12 leg joints, got {len(base.joints)}")
        self.tick_hz = base.policy.tick_hz
        self._p = params
        self._stance = np.array([base.stance[j] for j in base.joints], dtype=np.float64)
        self._kp, self._kd, self._limit = base.kp, base.kd, tau_limit
        self._pin = PinGo2Model()
        self._gait = Gait(params["gait_hz"], params["duty"])
        self._traj = ComTraj(self._pin)
        self._legs = LegController()  # its swing state restarts with the gait clock (all feet down at 0)
        self._mpc: Any = None  # built on the first solve
        self._dt = self._gait.gait_period / _DT_FACTOR
        self._solve_every = max(1, int(self.tick_hz // (1.0 / self._dt)))
        self._warned_at = -1e9
        self._fresh()

    def _fresh(self) -> None:
        self._plan: np.ndarray | None = None
        self._tick = 0
        self._fails = 0
        self._tau = np.zeros(len(self._stance))
        self._was_moving = False
        self._t0 = 0.0                          # the gait clock's zero: the last motion from rest
        self._brake_t: float | None = None     # when the command went to zero
        self._brake_q: np.ndarray | None = None  # the four-feet-down pose of the hand-over
        self._brake_ho_t = 0.0

    @property
    def braking(self) -> bool:
        return self._brake_t is not None

    def reset(self, obs: PolicyObs) -> None:
        """A fresh start: no plan, no brake."""
        self._fresh()

    def torques(self, obs: PolicyObs) -> np.ndarray:
        vx, vy, vyaw = (float(v) for v in obs.command)
        moving = vx != 0.0 or vy != 0.0 or vyaw != 0.0
        if moving and not self._was_moving and (self._brake_t is None or self._brake_q is not None):
            self._t0 = obs.time  # from rest: a fresh gait cycle
        t = obs.time - self._t0
        if moving:
            self._brake_t = self._brake_q = None
        elif self._was_moving:
            self._brake_t = t
        self._was_moving = moving
        # braking: the MPC tracks zero velocity until all four feet are down
        braking = self._brake_t is not None and self._brake_q is None and (
            t - self._brake_t < self._p["brake_s"] or not self._gait.compute_current_mask(t).all())
        tau = self._mpc_torque(obs, vx, vy, vyaw, t) if moving or braking else None
        if not (moving or braking):  # a new motion solves a fresh plan, never an old horizon
            self._plan = None
            self._tick = 0
        if tau is None:
            if (moving or braking) and self._fails < _MAX_FAILS:
                tau = self._tau  # one bad tick: keep the last good torque
            else:
                tau = pd(self._stance_q(obs, t), obs.joint_pos, obs.joint_vel, self._kp, self._kd, self._limit)
        self._tau = np.array(tau, dtype=float)
        return self._tau

    def _stance_q(self, obs: PolicyObs, t: float) -> np.ndarray:
        """The stance; after a brake, ramped from the four-feet-down hand-over pose."""
        if self._brake_t is None:
            return self._stance
        if self._brake_q is None:
            self._brake_q = np.array(obs.joint_pos, dtype=np.float64)
            self._brake_ho_t = t
        a = min(1.0, (t - self._brake_ho_t) / self._p["brake_ramp_s"])
        q = self._brake_q + a * a * (3.0 - 2.0 * a) * (self._stance - self._brake_q)
        if a >= 1.0:
            self._brake_t = self._brake_q = None
        return q

    def _mpc_torque(self, obs: PolicyObs, vx: float, vy: float, vyaw: float, t: float) -> np.ndarray | None:
        """Leg torques from the MPC plan, or None when the plan failed."""
        self._update_model(obs)
        try:
            if self._tick % self._solve_every == 0:
                self._traj.generate_traj(self._pin, self._gait, t, vx, vy, self._p["z_des"], vyaw,
                                         time_step=self._dt)
                if self._mpc is None:
                    from vector_os_nano.third_party.convex_mpc.centroidal_mpc import CentroidalMPC

                    self._mpc = CentroidalMPC(self._pin, self._traj)
                sol = self._mpc.solve_QP(self._pin, self._traj, False)
                n = self._traj.N
                self._plan = sol["x"].full().flatten()[12 * n:].reshape((12, n), order="F")
            self._tick += 1
            if self._plan is None:
                raise RuntimeError("no MPC force plan")
            force = self._plan[:, 0]
            tau = np.zeros(len(self._stance))
            for i, leg in enumerate(_LEGS):
                out = self._legs.compute_leg_torque(leg, self._pin, self._gait, force[i * 3:(i + 1) * 3], t)
                tau[i * 3:(i + 1) * 3] = out.tau
        except Exception as exc:  # noqa: BLE001 — counted, then the stance hold (fail loud)
            self._plan = None
            self._tick = 0
            self._fails += 1
            if self._fails >= _MAX_FAILS and t - self._warned_at > _WARN_EVERY_S:
                self._warned_at = t
                logger.warning("convex_mpc: %d failed solves in a row (%s: %s); holding the stance with "
                               "PD until a solve succeeds", self._fails, type(exc).__name__, exc)
            return None
        self._fails = 0
        return np.clip(tau, -self._limit, self._limit)

    def _update_model(self, obs: PolicyObs) -> None:
        """The Pinocchio state from the observation: quaternion x y z w, velocity in the body frame."""
        import pinocchio as pin

        qw, qx, qy, qz = obs.base_quat
        R = pin.Quaternion(qw, qx, qy, qz).toRotationMatrix()
        q = np.concatenate([obs.base_pos, [qx, qy, qz, qw], obs.joint_pos])
        dq = np.concatenate([R.T @ obs.base_lin_vel, obs.base_ang_vel, obs.joint_vel])
        self._pin.update_model(q, dq)
