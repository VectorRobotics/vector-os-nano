# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The built-in control policies (``interfaces.ControlPolicy``): one row each.

A legged manifest names its policy in ``base.policy.ref`` (per backend in
``base.policy.by_backend``). A row names the policy's module and the keys of
``base.policy`` it owns, each with its parser; the manifest loader checks both
(:func:`owned`), so a bad ref, key or value fails at load naming the file. A policy
module exposes ``Policy(base, params, tau_limit, folder)``: *base* is the manifest
``BaseSpec`` (joint order, stance, gains), *params* its own keys parsed, *tau_limit* the
per-joint torque bound, *folder* where the files it names resolve (the robot's asset
folder). It sees only ``PolicyObs``: no model, no data, no world. Modules are imported
lazily: a policy whose solver is not installed costs nothing until it is chosen.
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from vector_os_nano.schemas.robot import BaseSpec
from vector_os_nano.interfaces.policy import ControlPolicy

# torchscript: an observation term -> its width (n = the number of base joints).
TERMS: dict[str, Callable[[int], int]] = {
    "base_lin_vel": lambda n: 3, "base_ang_vel": lambda n: 3, "projected_gravity": lambda n: 3,
    "command": lambda n: 3, "joint_pos_rel": lambda n: n, "joint_vel": lambda n: n,
    "last_action": lambda n: n, "gait_phase": lambda n: 2,
}


def _count(v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise ValueError(f"{v!r} is not a whole number >= 1")
    return v


def _scale(v: Any) -> float | tuple[float, ...]:
    out = tuple(float(x) for x in v) if isinstance(v, list) else float(v)
    if not all(math.isfinite(x) for x in (out if isinstance(out, tuple) else (out,))):
        raise ValueError("scale must be finite")
    return out


def _obs(v: Any) -> tuple[tuple[str, float | tuple[float, ...], float | None], ...]:
    """``[{term, scale, period?}]`` -> ``((term, scale, period), ...)``; ``period`` (s)
    belongs to ``gait_phase`` and only to it."""
    if not isinstance(v, list) or not v:
        raise ValueError(f"{v!r} is not a list of {{term, scale, period?}}")
    out = []
    for e in v:
        if not isinstance(e, dict) or not {"term", "scale"} <= set(e) <= {"term", "scale", "period"}:
            raise ValueError(f"{e!r}: an obs entry is exactly {{term, scale, period?}}")
        if e["term"] not in TERMS:
            raise ValueError(f"obs term {e['term']!r}; terms: {sorted(TERMS)}")
        if ("period" in e) != (e["term"] == "gait_phase"):
            raise ValueError(f"{e!r}: gait_phase, and only it, takes a period")
        period = float(e["period"]) if "period" in e else None
        if period is not None and (not math.isfinite(period) or period <= 0):
            raise ValueError("gait_phase period must be finite and positive")
        out.append((e["term"], _scale(e["scale"]), period))
    return tuple(out)


def _action(v: Any) -> float:
    if not isinstance(v, dict) or set(v) != {"scale"}:
        raise ValueError(f"{v!r}: action is exactly {{scale}}")
    out = _scale(v["scale"])
    if isinstance(out, tuple):
        raise ValueError("action scale must be a number")
    return out


def _wheels(v: Any) -> tuple[tuple[float, ...], ...]:
    if not isinstance(v, list) or not v:
        raise ValueError('wheels must be a nonempty list of [x, y, radius, lateral_offset]')
    rows = tuple(tuple(float(x) for x in row) for row in v)
    if any(len(row) != 4 or not all(math.isfinite(x) for x in row) or row[2] <= 0 for row in rows):
        raise ValueError('each wheel needs finite [x, y, positive radius, lateral_offset]')
    return rows


POLICIES: dict[str, tuple[str, dict[str, Callable[[Any], Any]]]] = {
    "convex_mpc": ("vector_os_nano.sim.policies.convex_mpc",
                   {"file": str, **dict.fromkeys(("gait_hz", "duty", "z_des", "brake_s", "brake_ramp_s"), float)}),
    "sinusoidal": ("vector_os_nano.sim.policies.sinusoidal",
                   dict.fromkeys(("freq_hz", "thigh_amp", "calf_amp", "hip_amp"), float)),
    "torchscript": ("vector_os_nano.sim.policies.torchscript",
                    {"file": str, "decimation": _count, "obs": _obs, "action": _action}),
    "swerve": ("vector_os_nano.sim.policies.swerve",
               {"wheels": _wheels, "steering_range": _scale,
                **dict.fromkeys(("steering_kp", "steering_kd", "steering_speed", "wheel_kp", "wheel_ki", "wheel_speed",
                                 "accel", "yaw_accel"), float)}),
}


def owned(ref: str, params: dict[str, Any]) -> dict[str, Any]:
    """*params* parsed when they are exactly the keys row *ref* owns: a ``KeyError``
    names the valid rows, a ``ValueError`` the row's keys (or the key whose value its
    parser refuses)."""
    try:
        keys = POLICIES[ref][1]
    except KeyError:
        raise KeyError(f"no control policy {ref!r}; policies: {sorted(POLICIES)}") from None
    if set(params) != set(keys):
        raise ValueError(f"keys {sorted(params)}; {ref} takes exactly {sorted(keys)}")
    out = {}
    for k, parse in keys.items():
        try:
            out[k] = parse(params[k])
        except (TypeError, ValueError) as exc:
            why = f"{params[k]!r} is not a number" if parse is float else str(exc)
            raise ValueError(f"{k}: {why}") from None
    return out


def make(ref: str, base: BaseSpec, params: dict[str, Any], tau_limit: np.ndarray, folder: Path) -> ControlPolicy:
    """The policy row *ref* built for *base* with its :func:`owned` *params*, its files
    resolved in *folder*."""
    p = owned(ref, params)
    return importlib.import_module(POLICIES[ref][0]).Policy(base, p, tau_limit, folder)


def pd(q_des: np.ndarray, q: np.ndarray, dq: np.ndarray, kp: Any, kd: Any, limit: np.ndarray) -> np.ndarray:
    """The PD torque toward *q_des*, clipped to +-*limit* (postures and stance holds);
    *kp* / *kd* one gain or one per joint."""
    return np.clip(np.asarray(kp) * (q_des - q) - np.asarray(kd) * dq, -limit, limit)
