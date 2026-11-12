# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The built-in control policies (``interfaces.ControlPolicy``): one row each.

A legged manifest names its policy in ``base.policy.ref`` (per backend in
``base.policy.by_backend``). A row names the policy's module and the keys of
``base.policy`` it owns; the manifest loader checks both (:func:`owned`), so a bad ref or
key fails at load naming the file. A policy module exposes ``Policy(base, params,
tau_limit)``: *base* is the manifest ``BaseSpec`` (joint order, stance, gains), *params*
its own keys as floats, *tau_limit* the per-joint torque bound. It sees only ``PolicyObs``: no model, no data, no world. Modules are imported
lazily: a policy whose solver is not installed costs nothing until it is chosen.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from vector_os_nano.embodiments.config import BaseSpec
from vector_os_nano.interfaces.policy import ControlPolicy

POLICIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "convex_mpc": ("vector_os_nano.sim.policies.convex_mpc", ("gait_hz", "duty", "z_des", "brake_s", "brake_ramp_s")),
    "sinusoidal": ("vector_os_nano.sim.policies.sinusoidal", ("freq_hz", "thigh_amp", "calf_amp", "hip_amp")),
}


def owned(ref: str, params: dict[str, Any]) -> dict[str, float]:
    """*params* as floats when they are exactly the keys row *ref* owns: a ``KeyError``
    names the valid rows, a ``ValueError`` the row's keys (or the value not a number)."""
    try:
        keys = POLICIES[ref][1]
    except KeyError:
        raise KeyError(f"no control policy {ref!r}; policies: {sorted(POLICIES)}") from None
    if set(params) != set(keys):
        raise ValueError(f"keys {sorted(params)}; {ref} takes exactly {sorted(keys)}")
    out = {}
    for k in keys:
        try:
            out[k] = float(params[k])
        except (TypeError, ValueError):
            raise ValueError(f"{k}: {params[k]!r} is not a number") from None
    return out


def make(ref: str, base: BaseSpec, params: dict[str, Any], tau_limit: np.ndarray) -> ControlPolicy:
    """The policy row *ref* built for *base* with its :func:`owned` *params*."""
    p = owned(ref, params)
    return importlib.import_module(POLICIES[ref][0]).Policy(base, p, tau_limit)


def pd(q_des: np.ndarray, q: np.ndarray, dq: np.ndarray, kp: float, kd: float, limit: np.ndarray) -> np.ndarray:
    """The PD torque toward *q_des*, clipped to +-*limit* (postures and stance holds)."""
    return np.clip(kp * (q_des - q) - kd * dq, -limit, limit)
