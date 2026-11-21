# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Grasp-owned manifest parameters, validated before a robot starts."""
import math
from typing import Any

from vector_os_nano.schemas.robot import EmbodimentConfig, EmbodimentConfigError, _keys, _require, _floats

_GRASP_POSITIVE = {"pre_grasp_height", "lift_height", "reach_m", "view_duration", "hold_duration"}
_GRASP_NONNEGATIVE = {"grasp_z_above", "approach_back"}
_GRASP_BASE_COUNTS = {"approach_steps", "seat_steps", "scan_steps", "retries"}
_GRASP_BASE_POSITIVE = {"standoff", "view_standoff", "step_speed", "step_duration", "stall_distance", "seat_distance",
                        "lateral_gain", "lateral_speed", "lateral_deadband", "yaw_speed", "yaw_deadband",
                        "scan_step", "scan_range"}


def _positive(raw: Any, ctx: str, *, zero: bool = False) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise EmbodimentConfigError(f"{ctx}: expected a finite number") from None
    if isinstance(raw, bool) or not math.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise EmbodimentConfigError(f"{ctx}: expected a finite {'nonnegative' if zero else 'positive'} number")
    return value


def parse(raw: dict[str, Any], cfg: EmbodimentConfig, gctx: str) -> dict[str, Any]:
    arm = cfg.arm
    if arm is None or arm.gripper is None:
        raise EmbodimentConfigError(f"{gctx}: requires an arm with a gripper")
    g = _keys(raw, gctx, _GRASP_POSITIVE | _GRASP_NONNEGATIVE | {"durations", "base"})
    parsed = {k: _positive(_require(g, k, gctx), f"{gctx}.{k}", zero=k in _GRASP_NONNEGATIVE)
              for k in _GRASP_POSITIVE | _GRASP_NONNEGATIVE}
    durations = _floats(_require(g, "durations", gctx), 4, f"{gctx}.durations")
    if any(d <= 0 for d in durations):
        raise EmbodimentConfigError(f"{gctx}.durations: all four motion durations must be positive")
    parsed["durations"] = durations
    if "base" in g:
        bctx = f"{gctx}.base"
        if cfg.base is None:
            raise EmbodimentConfigError(f"{bctx}: requires a base")
        b = _keys(g["base"], bctx, _GRASP_BASE_COUNTS | _GRASP_BASE_POSITIVE)
        values = {k: _positive(_require(b, k, bctx), f"{bctx}.{k}", zero=k in _GRASP_BASE_COUNTS)
                  for k in _GRASP_BASE_COUNTS | _GRASP_BASE_POSITIVE}
        for k in _GRASP_BASE_COUNTS:
            if not values[k].is_integer():
                raise EmbodimentConfigError(f"{bctx}.{k}: expected a nonnegative integer")
            values[k] = int(values[k])
        parsed["base"] = values
    return parsed
