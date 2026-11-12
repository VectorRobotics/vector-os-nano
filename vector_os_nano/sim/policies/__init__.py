# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The built-in control policies (``interfaces.ControlPolicy``): one row each.

A legged manifest names its policy in ``base.policy.ref`` (per backend in
``base.policy.by_backend``). A policy module exposes ``Policy(base, params, tau_limit)``:
*base* is the manifest ``BaseSpec`` (joint order, stance, gains), *params* its own keys
of ``base.policy`` (parsed strictly with :func:`owned`), *tau_limit* the per-joint torque
bound. It sees only ``PolicyObs``: no model, no data, no world. Modules are imported
lazily: a policy whose solver is not installed costs nothing until it is chosen.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from vector_os_nano.embodiments.config import BaseSpec
from vector_os_nano.interfaces.policy import ControlPolicy

POLICIES: dict[str, str] = {
    "convex_mpc": "vector_os_nano.sim.policies.convex_mpc",
    "sinusoidal": "vector_os_nano.sim.policies.sinusoidal",
}


def make(ref: str, base: BaseSpec, params: dict[str, Any], tau_limit: np.ndarray) -> ControlPolicy:
    """The policy row *ref* built for *base*; a ``KeyError`` names the valid rows."""
    try:
        module = POLICIES[ref]
    except KeyError:
        raise KeyError(f"no control policy {ref!r}; policies: {sorted(POLICIES)}") from None
    return importlib.import_module(module).Policy(base, params, tau_limit)


def owned(params: dict[str, Any], keys: tuple[str, ...], ref: str) -> dict[str, float]:
    """*params* as floats when it holds exactly *keys* (fail loud naming the others)."""
    if set(params) != set(keys):
        raise ValueError(f"base.policy ({ref}): keys {sorted(params)}; {ref} takes exactly {sorted(keys)}")
    return {k: float(params[k]) for k in keys}


def pd(q_des: np.ndarray, q: np.ndarray, dq: np.ndarray, kp: float, kd: float, limit: np.ndarray) -> np.ndarray:
    """The PD torque toward *q_des*, clipped to +-*limit* (postures and stance holds)."""
    return np.clip(kp * (q_des - q) - kd * dq, -limit, limit)
