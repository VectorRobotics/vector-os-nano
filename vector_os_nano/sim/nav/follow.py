# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The ONE path follower every in-process base navigates with (Rule 11).

``follow(base, goal, obstacles, body=...)`` plans with the vgraph (goal snapped
out of furniture, start escaped out of the inflation), then streams velocity at
``tick_s`` of SIM time — never stop-and-go — with pure pursuit (a point
``lookahead_m`` ahead on the tracked segment, so a drifting gait is pulled back
onto the clear line). A waypoint is passed once the robot is within
``capture_ratio x inflation`` of it or has a clear line of sight (against the
inflated obstacles) to the next one, so the executed path never cuts a corner
the planner kept clear. Each tick it
checks the cancel signal (a stop), a fall, the sim-time budget and progress: no
``stall_progress_m`` of progress in ``stall_window_s`` triggers a reverse +
sidestep + replan recovery, ``max_recoveries`` times, then an honest "stalled".

The result is a :class:`NavResult` (a dict whose truth is "reached") with a
``reason`` from :data:`NAV_REASONS` — never a bare bool (Rule 4). The base is an
``interfaces.BaseProtocol`` (pose, heading, ``holonomic``, ``set_velocity``,
``stop``, ``get_odometry``, ``get_sim_time`` and ``wait_sim``); a sim clock that stops advancing ends
the follow as "timeout", never as an endless loop. Tuning comes from
``config/nav.yaml`` (``follower`` + ``navigation.arrival_radius``) and the body's
manifest ``base.nav`` block (radius, speed, yaw rate); the base says when it has fallen.
"""
from __future__ import annotations

import collections
import functools
import math
import threading
from pathlib import Path
from typing import Any, Callable

import yaml

from vector_os_nano.schemas.robot import NavSpec
from vector_os_nano.sim.nav import vgraph as vg

NAV_REASONS: tuple[str, ...] = ("arrived", "unreachable", "timeout", "aborted", "stalled", "fell")

_NAV_YAML = Path(__file__).resolve().parents[3] / "config" / "nav.yaml"


@functools.lru_cache(maxsize=1)
def nav_config() -> dict[str, Any]:
    """``config/nav.yaml`` (read once). Missing keys fail loud at the reader."""
    return yaml.safe_load(_NAV_YAML.read_text()) or {}


def arrival_radius() -> float:
    """The one arrival radius (navigate 'arrived' and the at_position default tol)."""
    return float(nav_config()["navigation"]["arrival_radius"])


class NavResult(dict):
    """Navigation telemetry; ``bool(result)`` is True only when the target was reached.

    Keys: reached, reason (one of NAV_REASONS), goal (asked), target (planned,
    the snapped goal), snapped, pos, dist (to target), moved_m, net_m, plan_m,
    elapsed_s (sim), recoveries.
    """

    __slots__ = ()

    def __bool__(self) -> bool:
        return bool(self.get("reached", False))

    @property
    def reason(self) -> str:
        return str(self["reason"])


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _xyz(base: Any) -> tuple[float, float, float]:
    p = base.get_position()
    return float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0


def _drive(base: Any, vx: float, vy: float, vyaw: float, seconds: float, cancel: Any) -> None:
    """Hold one command for *seconds* of sim time (a recovery manoeuvre)."""
    base.set_velocity(vx, vy, vyaw)
    base.wait_sim(seconds, cancel)
    base.set_velocity(0.0, 0.0, 0.0)


def _pursuit_point(p, a, b, lookahead: float) -> "tuple[float, float]":
    """The point *lookahead* m past p's projection onto segment a->b (clamped at b)."""
    ex, ey = b[0] - a[0], b[1] - a[1]
    le = math.hypot(ex, ey)
    if le < 1e-6:
        return b
    t = ((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / (le * le)
    s = min(1.0, max(0.0, t) + lookahead / le)
    return (a[0] + s * ex, a[1] + s * ey)


def _cross_track(p, a, b) -> "tuple[float, tuple[float, float]]":
    """(signed distance of p left of line a->b, the line's unit left normal)."""
    ex, ey = b[0] - a[0], b[1] - a[1]
    le = math.hypot(ex, ey)
    if le < 1e-6:
        return 0.0, (0.0, 0.0)
    nx, ny = -ey / le, ex / le
    return (p[0] - a[0]) * nx + (p[1] - a[1]) * ny, (nx, ny)


def _remaining(pos: tuple[float, float], path: list, i: int) -> float:
    return math.dist(pos, path[i]) + sum(math.dist(a, b) for a, b in zip(path[i:], path[i + 1:]))


def follow(
    base: Any,
    goal: "tuple[float, float]",
    obstacles: "list[list[tuple[float, float]]]",
    *,
    nav: NavSpec,
    fallen: Callable[[], bool],
    inflation: float | None = None,
    tol: float | None = None,
    speed: float | None = None,
    timeout: float | None = None,
    cancel: Any = None,
    on_progress: Callable[[float, float], None] | None = None,
) -> NavResult:
    """Drive *base* to *goal* around *obstacles*; see the module docstring.

    *nav* is the manifest ``base.nav`` block; *fallen* says whether the body is down
    (the base's own rule). *inflation* defaults to ``nav.body_radius``;
    *tol* to ``nav.goal_tol`` else ``follower.goal_tol``, and never exceeds
    the arrival radius; *timeout* (sim s) to ``budget_factor x geodesic / speed +
    budget_pad_s``. *cancel* is anything
    with ``is_set()``; the caller supplies its run's cancellation token.
    """
    cfg = nav_config()["follower"]
    r = float(nav.body_radius if inflation is None else inflation)
    v_max = float(speed or nav.speed)
    vyaw_max = nav.vyaw_max
    tol = min(float(tol if tol is not None else nav.goal_tol if nav.goal_tol is not None else cfg["goal_tol"]),
              arrival_radius())
    cancel = cancel if cancel is not None else threading.Event()
    tick = float(cfg["tick_s"])
    inflated = [vg.inflate_polygon(p, r) for p in obstacles]

    x0, y0, _ = _xyz(base)
    t0 = base.get_sim_time()
    goal = (float(goal[0]), float(goal[1]))
    path, plan_m = vg.plan_path((x0, y0), goal, obstacles, r, snap_goal=True)
    moved = [0.0]
    last = [(x0, y0)]
    recoveries = 0

    def result(reason: str) -> NavResult:
        x, y, zz = _xyz(base)
        target = path[-1] if path else goal
        d = math.dist((x, y), target)
        return NavResult(
            reached=reason == "arrived", reason=reason, goal=list(goal),
            target=[round(target[0], 3), round(target[1], 3)],
            snapped=bool(path) and math.dist(target, goal) > 1e-6,
            pos=[x, y, zz], dist=d, moved_m=moved[0], net_m=math.dist((x, y), (x0, y0)),
            plan_m=plan_m, elapsed_s=base.get_sim_time() - t0, recoveries=recoveries,
        )

    if path is None:
        return result("unreachable")
    budget = timeout if timeout is not None else (
        float(cfg["budget_factor"]) * plan_m / v_max + float(cfg["budget_pad_s"]))
    deadline = t0 + float(budget)
    capture = float(cfg["capture_ratio"]) * r
    i = 1
    seg_start = (x0, y0)  # the segment the robot tracks: seg_start -> path[i] (clear)
    lookahead = float(cfg["lookahead_m"])
    # Gait slip: a legged body walks a few degrees off its heading (the MPC go2
    # held a steady 0.12 m off a wall-side line). Estimate the course-vs-heading
    # offset from the last second of straight walking and steer the course.
    hist: collections.deque = collections.deque(maxlen=max(2, round(float(cfg["slip_window_s"]) / tick)))
    slip, straight = 0.0, False
    best, best_t = _remaining((x0, y0), path, i), t0
    next_progress = t0
    holonomic = base.holonomic
    settled_since = None
    braking = False
    try:
        while True:
            if cancel.is_set():
                return result("aborted")
            x, y, _ = _xyz(base)
            now = base.get_sim_time()
            moved[0] += math.dist((x, y), last[0])
            last[0] = (x, y)
            if fallen():
                return result("fell")
            target = path[-1]
            if now >= deadline:
                return result("timeout")
            if braking or math.dist((x, y), target) <= tol:
                # Zero command is not zero motion: keep control through braking.
                braking = True
                base.set_velocity(0.0, 0.0, 0.0)
                odom = base.get_odometry()
                if math.hypot(odom.vx, odom.vy) <= .03 and abs(odom.vyaw) <= .05:
                    settled_since = now if settled_since is None else settled_since
                    if now - settled_since >= .2:
                        if math.dist((x, y), target) <= tol:
                            return result("arrived")
                        braking, settled_since = False, None
                        best, best_t = _remaining((x, y), path, i), now
                        continue
                else:
                    settled_since = None
                if not base.wait_sim(min(tick, deadline-now), cancel):
                    return result("aborted" if cancel.is_set() else "timeout")
                continue
            settled_since = None
            while i < len(path) - 1 and (
                    math.dist((x, y), path[i]) <= capture
                    or vg.segment_clear((x, y), path[i + 1], inflated)):
                i += 1
                seg_start = (x, y)
            rem = _remaining((x, y), path, i)
            if on_progress is not None and now >= next_progress:
                next_progress = now + float(cfg["progress_every_s"])
                on_progress(rem, now - t0)
            if rem < best - float(cfg["stall_progress_m"]):
                best, best_t = rem, now
            elif now - best_t > float(cfg["stall_window_s"]):
                if recoveries >= int(cfg["max_recoveries"]):
                    return result("stalled")
                recoveries += 1
                _recover(base, inflated, cfg, cancel)
                x, y, _ = _xyz(base)
                new_path, new_m = vg.plan_path((x, y), goal, obstacles, r, snap_goal=True)
                if new_path is None:
                    return result("unreachable")
                path, plan_m, i, seg_start = new_path, new_m, 1, (x, y)
                best, best_t = _remaining((x, y), path, i), base.get_sim_time()
                continue
            # pure pursuit on the tracked segment (pulls a drifting gait back onto the
            # clear line), streamed: no settle between ticks
            wx, wy = _pursuit_point((x, y), seg_start, path[i], lookahead)
            heading = base.get_heading()
            hist.append((x, y, heading))
            if not straight:
                hist.clear()
                hist.append((x, y, heading))
            elif len(hist) == hist.maxlen and math.dist(hist[0][:2], (x, y)) > 0.15:
                course = math.atan2(y - hist[0][1], x - hist[0][0])
                slip += float(cfg["slip_gain"]) * (_wrap(course - heading) - slip)
                slip = max(-float(cfg["slip_max_rad"]), min(float(cfg["slip_max_rad"]), slip))
            err = _wrap(math.atan2(wy - y, wx - x) - heading - slip)
            vyaw = max(-vyaw_max, min(vyaw_max, float(cfg["k_yaw"]) * err))
            if abs(err) > float(cfg["face_tol_rad"]):
                vx = 0.0
                best_t = now  # a pivot in place is progress the distance cannot see
            else:
                d_goal = math.dist((x, y), target)
                taper = min(1.0, d_goal / float(cfg["slow_radius"])) if i == len(path) - 1 else 1.0
                vx = max(float(cfg["min_speed"]), v_max * taper) * math.cos(err)
            vy = 0.0
            if vx > 0.0 and holonomic:  # strafe back onto the tracked line
                e, (nx, ny) = _cross_track((x, y), seg_start, path[i])
                lim = float(cfg["cross_track_max"])
                corr = max(-lim, min(lim, -float(cfg["cross_track_gain"]) * e))
                vy = corr * (nx * -math.sin(heading) + ny * math.cos(heading))
            # only plain straight walking measures the slip (a pivot or strafe would bias it)
            straight = abs(err) < 0.15 and abs(vy) < 0.05
            base.set_velocity(vx, vy, vyaw)
            if not base.wait_sim(tick, cancel) and not cancel.is_set():
                return result("timeout")  # the sim clock stopped: never loop on it
    finally:
        base.stop()


def _recover(base: Any, inflated: list, cfg: dict, cancel: Any) -> None:
    """Back off, then (a holonomic base) step sideways toward the freer side."""
    _drive(base, -float(cfg["reverse_speed"]), 0.0, 0.0, float(cfg["reverse_s"]), cancel)
    if not base.holonomic or cancel.is_set():
        return
    x, y, _ = _xyz(base)
    h = base.get_heading()
    probe = 0.5

    def clearance(sign: float) -> float:
        q = (x - sign * probe * math.sin(h), y + sign * probe * math.cos(h))
        return min((vg.distance_to_polygon(q, p) for p in inflated), default=math.inf)

    side = 1.0 if clearance(1.0) >= clearance(-1.0) else -1.0
    _drive(base, 0.0, side * float(cfg["sidestep_speed"]), 0.0, float(cfg["sidestep_s"]), cancel)


def turn_by(base: Any, angle: float, *, vyaw: float, tol: float,
            cancel: Any = None, corrections: int = 3) -> tuple[bool, float]:
    """Rotate in place by *angle* rad, closed-loop on the measured heading.

    One open-loop ``walk`` for the whole angle, then up to *corrections* short
    walks on the residual. Returns (within tol, residual rad). Works on any base
    with ``walk(vx, vy, vyaw, duration)`` + ``get_heading()``.
    """
    cancel = cancel if cancel is not None else threading.Event()
    target = _wrap(base.get_heading() + angle)
    remaining = float(angle)
    for _ in range(1 + corrections):
        if abs(remaining) <= tol or cancel.is_set():
            break
        base.walk(0.0, 0.0, math.copysign(vyaw, remaining), abs(remaining) / vyaw)
        remaining = _wrap(target - base.get_heading())
    return abs(remaining) <= tol, remaining


def sim_tick_dt(
    now_sim: float | None,
    last_sim: float | None,
    nominal_dt: float,
    max_ticks: float = 4.0,
) -> float:
    """Sim-time elapsed between two controller ticks, for dt-scaled ramps.

    - ``now_sim is None`` (no readable sim clock — real hardware, clock
      error) or ``last_sim is None`` (first tick) -> ``nominal_dt``
      (the original wall-tick behavior).
    - Backwards jump (sim reset) -> ``nominal_dt``.
    - Paused/stalled sim -> ``0.0`` (ramps freeze along with the gait).
    - Clamped to ``max_ticks * nominal_dt`` so a stall/hiccup cannot slew
      commands violently on the tick after it resolves.
    """
    if now_sim is None or last_sim is None:
        return nominal_dt
    dt = now_sim - last_sim
    if dt < 0.0:
        return nominal_dt
    return min(dt, max_ticks * nominal_dt)
