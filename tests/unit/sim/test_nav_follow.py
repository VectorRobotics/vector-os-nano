# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The one path follower (sim/nav/follow.py) on a kinematic base, no MuJoCo.

Every NavResult reason is produced by a scenario here; stall recovery, the sim-time
budget, the cancel signal, a fall and Ctrl-C (the base is always stopped) are each
pinned. The scene-level clearance replay lives in test_nav_apartment.py.
"""
from __future__ import annotations

import math
import threading

import pytest

from tests.harness.kin_base import KinBase
from vector_os_nano.sim.nav import vgraph as vg
from vector_os_nano.sim.nav.follow import NAV_REASONS, NavResult, arrival_radius, follow, turn_by

GO2 = {"body_radius": 0.28, "speed": 0.4, "vyaw_max": 0.8, "fall_z": 0.15}


class Never:
    @staticmethod
    def is_set():
        return False


class After:
    """A cancel signal that fires on its n-th poll."""

    def __init__(self, n):
        self.n = n

    def is_set(self):
        self.n -= 1
        return self.n < 0


WALL = vg.box_polygon(2.0, 0.0, 0.1, 1.0)  # blocks the straight line (0,0)->(4,0)
BOXED = [vg.box_polygon(10, 12, 3, 0.2), vg.box_polygon(10, 8, 3, 0.2),
         vg.box_polygon(12, 10, 0.2, 3), vg.box_polygon(8, 10, 0.2, 3)]


def _run(reason: str) -> NavResult:
    """One scenario per NavResult reason."""
    if reason == "arrived":
        return follow(KinBase(), (4, 0), [WALL], body=GO2, cancel=Never())
    if reason == "unreachable":
        return follow(KinBase(), (10, 10), BOXED, body=GO2, cancel=Never())
    if reason == "timeout":
        return follow(KinBase(), (4, 0), [WALL], body=GO2, timeout=1.0, cancel=Never())
    if reason == "aborted":
        return follow(KinBase(), (4, 0), [WALL], body=GO2, cancel=After(5))
    if reason == "stalled":  # an obstacle the map lacks, forever
        base = KinBase(blocked=lambda x, y, t: 1.0 < x < 1.4)
        return follow(base, (3, 0), [], body=GO2, cancel=Never())
    if reason == "fell":
        return follow(KinBase(z_at=lambda t: 0.3 if t < 1.0 else 0.05), (4, 0), [], body=GO2,
                      cancel=Never())
    raise AssertionError(reason)


@pytest.mark.parametrize("reason", NAV_REASONS)
def test_every_reason_is_reachable_and_honest(reason):
    res = _run(reason)
    assert isinstance(res, NavResult) and res.reason == reason
    assert bool(res) is (reason == "arrived") is res["reached"]
    if reason == "arrived":
        assert res["dist"] <= 0.25 and res["moved_m"] > 4.0  # went around the wall


def test_arrival_never_looser_than_the_arrival_radius():
    res = follow(KinBase(), (3, 0), [], body=GO2, tol=5.0, cancel=Never())
    assert res and math.dist(res["pos"][:2], (3, 0)) <= arrival_radius()


def test_goal_in_furniture_is_snapped_and_reported():
    table = vg.box_polygon(3, 0, 0.84, 0.45)
    res = follow(KinBase(), (3, 0), [table], body=GO2, cancel=Never())
    assert res and res["snapped"] and res["goal"] == [3.0, 0.0]
    assert not vg.point_in_polygon(tuple(res["target"]), vg.inflate_polygon(table, 0.28))


def test_stall_recovers_by_backing_off_and_replanning():
    # skills-nav:inprocess-nav-no-abort-stall: a bump the map lacks (for 8 sim-s)
    base = KinBase(blocked=lambda x, y, t: t < 8.0 and 1.0 < x < 1.4)
    res = follow(base, (3, 0), [], body=GO2, cancel=Never())
    assert res and res["recoveries"] >= 1


def test_the_base_is_stopped_on_every_exit_even_ctrl_c():
    base = KinBase()

    def boom(seconds, cancel=None):
        raise KeyboardInterrupt

    base.wait_sim = boom
    with pytest.raises(KeyboardInterrupt):
        follow(base, (4, 0), [], body=GO2, cancel=Never())
    assert base.stops == 1 and base.cmd == (0.0, 0.0, 0.0)


def test_default_cancel_is_the_run_abort_signal():
    from vector_os_nano.vcli.cognitive import abort

    base = KinBase()
    orig = base.wait_sim

    def wait_then_stop(seconds, cancel=None):
        if base.t > 1.0:
            abort.request_abort()
        return orig(seconds, cancel)

    base.wait_sim = wait_then_stop
    try:
        res = follow(base, (4, 0), [], body=GO2)
    finally:
        abort.clear_abort()
    assert res.reason == "aborted" and base.cmd == (0.0, 0.0, 0.0)


def test_velocity_is_streamed_not_stop_and_go():
    # skills-nav:stop-and-go-gait-timeout: chained blocking walks zeroed the command
    # every 0.7 s; the follower streams one command per tick and never settles.
    base = KinBase()
    commands = []
    orig = base.set_velocity

    def record(vx, vy, vyaw):
        commands.append(vx)
        orig(vx, vy, vyaw)

    base.set_velocity = record
    res = follow(base, (6, 0), [], body=GO2, cancel=Never())
    assert res
    assert all(v > 0 for v in commands)  # no zero command until the final stop()
    assert res["elapsed_s"] < 6.0 / 0.4 + 2.0  # ~ the commanded speed, no halving


def test_budget_is_one_global_sim_time_deadline():
    # sim-locomotion:g1-nav-budget-not-global: the per-waypoint budget gave every
    # waypoint the full timeout; a many-corner path must still stop at the deadline.
    zig = [vg.box_polygon(2, 0.6, 0.1, 1.0), vg.box_polygon(4, -0.6, 0.1, 1.0),
           vg.box_polygon(6, 0.6, 0.1, 1.0)]
    base = KinBase()
    res = follow(base, (8, 0), zig, body=GO2, timeout=6.0, cancel=Never())
    assert res.reason == "timeout" and res["elapsed_s"] <= 6.0 + 0.11


def test_turn_by_closes_the_loop_on_heading():
    class Sloppy(KinBase):  # the gait only delivers 70 % of the commanded yaw
        def walk(self, vx, vy, vyaw, duration):
            return super().walk(vx, vy, 0.7 * vyaw, duration)

    base = Sloppy()
    ok, residual = turn_by(base, math.pi / 2, vyaw=0.5, tol=math.radians(5), cancel=Never())
    assert ok and abs(base.yaw - math.pi / 2) <= math.radians(5)


def test_follower_can_run_on_another_thread_and_be_cancelled():
    stop = threading.Event()
    base = KinBase()
    orig = base.wait_sim

    def wait(seconds, cancel=None):
        if base.t > 0.5:
            stop.set()
        return orig(seconds, cancel)

    base.wait_sim = wait
    out = {}
    th = threading.Thread(target=lambda: out.update(r=follow(base, (9, 0), [], body=GO2, cancel=stop)))
    th.start()
    th.join(10)
    assert out["r"].reason == "aborted"


class Drifting(KinBase):
    """A gait that slides sideways (+y world) while walking, like the MPC go2 does."""

    def wait_sim(self, seconds, cancel=None):
        self.y += 0.06 * seconds if abs(self.cmd[0]) > 0 else 0.0
        return super().wait_sim(seconds, cancel)


def test_pure_pursuit_pulls_a_drifting_gait_back_onto_the_line():
    # Real go2 tour (MPC): aiming at a waypoint 3.7 m away corrected a 0.13 m sideways
    # drift too slowly and the body grazed the hallway wall (centre clearance 0.15 m).
    wall = vg.box_polygon(4.0, 0.4, 4.0, 0.05)  # its inflated face sits at y = 0.07
    base = Drifting()
    res = follow(base, (8.0, 0.0), [wall], body=GO2, cancel=Never())
    assert res
    assert min(vg.distance_to_polygon(p, wall) for p in base.trace) >= 0.19 + 0.05
