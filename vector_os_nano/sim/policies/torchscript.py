# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""``torchscript``: a learned walking policy (a TorchScript module, the G1's motion.pt)
run the way the script it was trained under runs it (unitree_rl_gym
deploy/deploy_mujoco): every ``decimation`` ticks the network maps the observation to
an action, and every tick the joints track ``stance + action.scale x action`` with the
manifest's PD gains (kp / kd per joint, torques clipped to the motors' range).

The observation is the manifest's ``obs`` terms in order, each times its ``scale`` (a
number, or one per component), built from ``PolicyObs`` only: ``base_lin_vel`` and
``base_ang_vel`` (body frame), ``projected_gravity`` (the world's down in the body
frame), ``command``, ``joint_pos_rel`` (to the stance), ``joint_vel``, ``last_action``
and ``gait_phase`` (sin, cos of the phase of a ``period``-second clock that starts with
the policy). The policy balances the body itself (``balances``): at zero command it
steps in place, so the stance is its own and a PD hold of it is never used.

Keys of ``base.policy``: ``file`` (relative to the robot's asset folder),
``decimation``, ``obs``, ``action: {scale}``. The module is run once on load: an
observation or an action of the wrong width fails the start. torch is imported here
only, when the policy is built.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from vector_os_nano.embodiments.config import BaseSpec
from vector_os_nano.interfaces.policy import PolicyObs
from vector_os_nano.sim.policies import TERMS, pd

INSTALL_HINT = "the torchscript policy needs torch: uv pip install -e '.[perception]'"


def _to_body(quat: np.ndarray, v: Any) -> np.ndarray:
    """World vector *v* in the body frame of *quat* (w x y z, world <- body)."""
    w, x, y, z = quat
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return r.T @ np.asarray(v, dtype=np.float64)


class Policy:
    braking = False
    balances = True

    def __init__(self, base: BaseSpec, params: dict[str, Any], tau_limit: np.ndarray, folder: Path) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(f"{INSTALL_HINT} ({exc})") from exc
        self._torch = torch
        n = len(base.joints)
        self._n_action = n
        path = Path(folder) / params["file"]
        with warnings.catch_warnings():  # torch 2.x deprecates jit.load; the file is TorchScript
            warnings.simplefilter("ignore", FutureWarning)
            self._net = torch.jit.load(str(path), map_location="cpu").eval()
        self._terms = params["obs"]
        for term, scale, _period in self._terms:
            if isinstance(scale, tuple) and len(scale) != TERMS[term](n):
                raise ValueError(f"{path.name}: obs {term} has {TERMS[term](n)} components, its scale {len(scale)}")
        width = sum(TERMS[t](n) for t, _s, _p in self._terms)
        try:
            out = self._run(np.zeros(width, dtype=np.float32))
        except RuntimeError as exc:
            raise ValueError(f"{path.name} does not take an observation of {width} (the manifest's obs "
                             f"terms): {str(exc).splitlines()[0]}") from None
        self.tick_hz = base.policy.tick_hz
        self._dec = params["decimation"]
        self._scale = params["action"]
        self._stance = np.array([base.stance[j] for j in base.joints], dtype=np.float64)
        self._kp, self._kd, self._limit = base.kp, base.kd, tau_limit
        self.reset(None)

    def _run(self, x: np.ndarray) -> np.ndarray:
        if not np.isfinite(x).all():
            raise ValueError("torchscript observation must be finite")
        with self._torch.no_grad():
            out = self._net(self._torch.from_numpy(x).unsqueeze(0)).numpy()
        if out.shape != (1, self._n_action) or not np.isfinite(out).all():
            raise ValueError(f"torchscript action must be finite with shape (1, {self._n_action}); got {out.shape}")
        return out

    def reset(self, obs: PolicyObs | None) -> None:
        """A fresh episode: no last action, the stance as the target, the clock at 0."""
        self._n = 0
        self._action = np.zeros(len(self._stance), dtype=np.float32)
        self._target = self._stance

    def torques(self, obs: PolicyObs) -> np.ndarray:
        if self._n and self._n % self._dec == 0:
            self._action = self._run(self.observe(obs))[0]
            self._target = self._stance + self._scale * self._action
        self._n += 1
        return pd(self._target, obs.joint_pos, obs.joint_vel, self._kp, self._kd, self._limit)

    def observe(self, obs: PolicyObs) -> np.ndarray:
        """The network's input at this tick (float32)."""
        t = self._n / self.tick_hz
        parts = []
        for term, scale, period in self._terms:
            if term == "base_lin_vel":
                v = _to_body(obs.base_quat, obs.base_lin_vel)
            elif term == "base_ang_vel":
                v = obs.base_ang_vel
            elif term == "projected_gravity":
                v = _to_body(obs.base_quat, (0.0, 0.0, -1.0))
            elif term == "command":
                v = obs.command
            elif term == "joint_pos_rel":
                v = obs.joint_pos - self._stance
            elif term == "joint_vel":
                v = obs.joint_vel
            elif term == "last_action":
                v = self._action
            else:  # gait_phase
                phase = 2.0 * math.pi * (t % period) / period
                v = (math.sin(phase), math.cos(phase))
            parts.append(np.asarray(v, dtype=np.float64) * np.asarray(scale))
        return np.concatenate(parts).astype(np.float32)
