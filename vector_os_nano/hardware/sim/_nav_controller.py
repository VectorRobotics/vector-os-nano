# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""World-agnostic closed-loop navigation controller (campaign #11 R5).

ONE source for the face->walk->capture->settle drive + g1_vgraph waypoint routing
that G1 and Go2 previously duplicated (~180 lines each, rule 7/3 debt). The control
logic is MEASURED (campaign #6) — extracted here VERBATIM and parameterized only by
what genuinely differs per embodiment:

  - constants     -> a frozen NavConsts pack (G1 fall_z=0.4 vs Go2 0.20, etc.)
  - time advance  -> tick_fn (G1._advance / Go2._drive_for)
  - control token -> ctrl_token factory (G1 = no-op nullcontext; Go2 grabs its
    skill-ctrl token so its thread-gated set_velocity isn't dropped)

The pose/command primitives are injected as callables, so this module imports no
embodiment and touches no kernel/BaseProtocol (rule 2/7). Both bases keep a thin
navigate_to / _navigate_point / geodesic_distance that delegates here; the
three-value result dict is single-sourced (rule 3)."""
from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class NavConsts:
    """Per-embodiment closed-loop nav tuning (all measured; see each base)."""
    fall_z: float           # base height below this = fallen
    tol_floor: float        # gait COM-oscillation floor — tol can't beat this
    speed: float            # default forward cmd
    vyaw_max: float         # yaw cmd ceiling
    k_yaw: float            # proportional heading gain
    face_tol: float         # rad: pivot-in-place until aligned within this
    yaw_deadband: float     # rad: stop steering when aligned (anti-hunt)
    capture_r: float        # m: inside, freeze steering + drive straight
    tick_s: float           # control tick
    settle_s: float         # hold stance + re-sample before deciding arrival
    timeout_s: float
    stall_window_s: float   # forward-walk no-progress window
    stall_min_m: float      # min net progress within the window
    inflation: float        # obstacle inflation (vgraph routing)
    waypoint_tol: float     # loose tol for intermediate waypoints


def _null_token() -> Any:
    return contextlib.nullcontext()


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _nan_result() -> dict:
    return {"reached": False, "already_there": False, "moved_m": 0.0,
            "elapsed_s": 0.0, "remaining": float("inf"),
            "reason": "bad_params_nan", "transport": "sim_oracle"}


def drive_to_point(
    *, x: float, y: float, tol: float, speed: float,
    get_position: Callable[[], "list[float]"],
    get_heading: Callable[[], float],
    set_velocity: Callable[[float, float, float], None],
    stop: Callable[[], None],
    tick_fn: Callable[[float], None],
    consts: NavConsts,
    ctrl_token: Callable[[], Any] = _null_token,
) -> dict:
    """Closed-loop face->walk->capture->settle drive to a SINGLE world (x, y).

    Verbatim port of the campaign-#6 G1 controller; the only per-embodiment
    inputs are consts / tick_fn / ctrl_token. Returns the three-value contract
    {reached, already_there, moved_m, net_m, elapsed_s, remaining, pos,
    effective_tol, reason, transport}. ``ctrl_token`` is a zero-arg factory
    returning a context manager held across the drive+settle (Go2 acquires its
    skill-ctrl token; G1 gets a no-op nullcontext, so its path is unchanged)."""
    if not all(math.isfinite(v) for v in (x, y, tol, speed)):
        return _nan_result()
    tx, ty = float(x), float(y)
    eff_tol = max(float(tol), consts.tol_floor)
    speed = max(0.1, float(speed))

    sx, sy, sz = get_position()
    start_dist = math.hypot(tx - sx, ty - sy)
    if start_dist <= eff_tol:
        return {"reached": True, "already_there": True, "moved_m": 0.0,
                "net_m": 0.0, "elapsed_s": 0.0, "remaining": start_dist,
                "pos": [sx, sy, sz], "effective_tol": eff_tol,
                "reason": "already_within_tol", "transport": "sim_oracle"}

    t0 = time.monotonic()
    px, py = sx, sy
    moved = 0.0
    best_dist = start_dist
    best_t = t0
    reason = "timeout"

    with ctrl_token():
        try:
            while time.monotonic() - t0 < consts.timeout_s:
                cx, cy, cz = get_position()
                yaw = get_heading()
                moved += math.hypot(cx - px, cy - py)
                px, py = cx, cy
                dist = math.hypot(tx - cx, ty - cy)

                if cz < consts.fall_z:
                    reason = "fell"
                    break
                if dist <= eff_tol:
                    reason = "arrived"
                    break

                err = _wrap(math.atan2(ty - cy, tx - cx) - yaw)
                if dist < consts.capture_r:
                    vx, vyaw = max(0.15, 0.4 * speed), 0.0
                elif abs(err) > consts.face_tol:
                    vx = 0.0
                    vyaw = max(-consts.vyaw_max, min(consts.vyaw_max, consts.k_yaw * err))
                else:
                    vx = speed
                    vyaw = (0.0 if abs(err) < consts.yaw_deadband
                            else max(-consts.vyaw_max,
                                     min(consts.vyaw_max, consts.k_yaw * err)))
                set_velocity(vx, 0.0, vyaw)

                if vx > 0.0:
                    if dist < best_dist - consts.stall_min_m:
                        best_dist, best_t = dist, time.monotonic()
                    elif time.monotonic() - best_t > consts.stall_window_s:
                        reason = "stalled_no_progress"
                        break
                else:
                    best_t = time.monotonic()
                tick_fn(consts.tick_s)
        finally:
            stop()

        # Settle: hold stance ~1 gait period and re-sample the AUTHORITATIVE pose
        # (the robot coasts past the in-flight break sample). moved accrues here.
        settle_end = time.monotonic() + consts.settle_s
        while time.monotonic() < settle_end:
            set_velocity(0.0, 0.0, 0.0)
            cx, cy, _cz = get_position()
            moved += math.hypot(cx - px, cy - py)
            px, py = cx, cy
            tick_fn(consts.tick_s)

    fx, fy, fz = get_position()
    remaining = math.hypot(tx - fx, ty - fy)
    reached = bool(remaining <= eff_tol and fz >= consts.fall_z)
    if reached:
        reason = "ok"
    return {
        "reached": reached, "already_there": False,
        "moved_m": round(moved, 3),
        "net_m": round(math.hypot(fx - sx, fy - sy), 3),
        "elapsed_s": round(time.monotonic() - t0, 3),
        "remaining": round(remaining, 3),
        "pos": [fx, fy, fz], "effective_tol": eff_tol,
        "reason": reason, "transport": "sim_oracle",
    }


def route_and_drive(
    *, x: float, y: float, tol: float, speed: float, obstacles: list,
    get_position: Callable[[], "list[float]"],
    get_heading: Callable[[], float],
    set_velocity: Callable[[float, float, float], None],
    stop: Callable[[], None],
    tick_fn: Callable[[float], None],
    consts: NavConsts,
    ctrl_token: Callable[[], Any] = _null_token,
) -> dict:
    """Route around ``obstacles`` (g1_vgraph) and drive each leg via
    drive_to_point. ctrl_token is acquired/released PER LEG (drive_to_point
    enters its CM each call), preserving the per-leg token semantics. Returns
    the three-value dict + 'waypoints'. Honest 'unreachable' on a boxed-in goal
    (rule 5)."""
    if not all(math.isfinite(v) for v in (x, y, tol, speed)):
        return _nan_result()
    from vector_os_nano.hardware.sim import g1_vgraph  # noqa: PLC0415
    sx, sy, _sz = get_position()
    waypoints, _length = g1_vgraph.plan_path(
        (sx, sy), (float(x), float(y)), obstacles, consts.inflation)
    if waypoints is None:
        return {"reached": False, "already_there": False, "moved_m": 0.0,
                "elapsed_s": 0.0, "remaining": float("inf"),
                "reason": "unreachable", "transport": "sim_oracle"}
    t0 = time.monotonic()
    total_moved = 0.0
    legs = waypoints[1:]
    last = None
    for i, (wx, wy) in enumerate(legs):
        is_final = (i == len(legs) - 1)
        leg_tol = tol if is_final else consts.waypoint_tol
        last = drive_to_point(
            x=wx, y=wy, tol=leg_tol, speed=speed,
            get_position=get_position, get_heading=get_heading,
            set_velocity=set_velocity, stop=stop, tick_fn=tick_fn,
            consts=consts, ctrl_token=ctrl_token)
        total_moved += float(last.get("moved_m", 0.0))
        # Break on any TERMINAL leg failure: a fall, a stall, OR a per-leg
        # timeout (the leg consumed its full independent budget without reaching
        # even waypoint_tol — driving the next leg from a not-yet-reached
        # waypoint just compounds path error and burns the global budget).
        if last.get("reason") in ("fell", "stalled_no_progress", "timeout"):
            break
    fx, fy, fz = get_position()
    remaining = math.hypot(float(x) - fx, float(y) - fy)
    eff_tol = max(float(tol), consts.tol_floor)
    reached = bool(remaining <= eff_tol and fz >= consts.fall_z)
    return {
        "reached": reached, "already_there": False,
        "moved_m": round(total_moved, 3),
        "net_m": round(math.hypot(fx - sx, fy - sy), 3),
        "elapsed_s": round(time.monotonic() - t0, 3),
        "remaining": round(remaining, 3),
        "pos": [fx, fy, fz], "effective_tol": eff_tol,
        "waypoints": [[round(p[0], 2), round(p[1], 2)] for p in waypoints],
        "reason": "ok" if reached else (last or {}).get("reason", "timeout"),
        "transport": "sim_oracle",
    }


def path_length_geodesic(a: "list[float]", b: "list[float]", obstacles: list,
                         inflation: float) -> float:
    """Visibility-graph path length around obstacles, else straight line."""
    if obstacles:
        from vector_os_nano.hardware.sim import g1_vgraph  # noqa: PLC0415
        return g1_vgraph.path_length(
            (float(a[0]), float(a[1])), (float(b[0]), float(b[1])),
            obstacles, inflation)
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
