# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Sim test: LeggedBase.navigate_to on the go2 in the apartment — obstacle-aware planning.

1. navigate_to(10.46, 3.0) arrives within 0.25 m, and the dog did not end inside the
   pick_table footprint (x in [10.80, 11.10], y in [2.75, 3.25]): it stopped in front.
2. The planner reads the static scene's obstacles at an instant (vgraph.scene_obstacles).
3. With the Piper mounted (go2_piper) navigate_to returns in bounded wall time, moving.

The body stands up as the session builds it (connect, then the stand posture), and a
walk or navigate holds the sim clock between its commands, so a loaded box changes
nothing about the motion (ledger flake:go2-navigate-fall).
"""
from __future__ import annotations

import math
import time

import pytest

pytest.importorskip("mujoco")

from tests.hardware.sim.dog import stand_up  # noqa: E402
from vector_os_nano.sim.nav import vgraph as vg  # noqa: E402

pytestmark = pytest.mark.sim

_TABLE_X = (10.80, 11.10)
_TABLE_Y = (2.75, 3.25)
_GOAL = (10.46, 3.0)
_NAV_TOL = 0.25


@pytest.fixture(scope="module")
def dog():
    d = stand_up("go2")
    yield d
    d.close()


def test_the_planner_reads_the_static_scene(dog) -> None:
    polys = vg.scene_obstacles(dog.rt.world.scene, dog.rt.snapshot().q)
    assert len(polys) > 10  # furniture, walls, the pick table
    table = [p for p in polys if all(_TABLE_X[0] - 0.01 <= x <= _TABLE_X[1] + 0.01
                                       and _TABLE_Y[0] - 0.01 <= y <= _TABLE_Y[1] + 0.01 for x, y in p)]
    assert table, "the pick table is not an obstacle"


def test_navigate_to_arrives_and_avoids_table(dog) -> None:
    res = dog.base.navigate_to(*_GOAL, tol=_NAV_TOL, timeout=90.0)
    fx, fy, fz = dog.base.get_position()
    dist = math.hypot(fx - _GOAL[0], fy - _GOAL[1])
    print(f"\n[test] navigate_to{_GOAL}: {res.reason} in {res['elapsed_s']:.2f} sim-s, "
          f"final ({fx:.3f}, {fy:.3f}, {fz:.3f}), {dist:.3f} m from the goal")
    assert res, f"{res.reason}: final ({fx:.3f}, {fy:.3f}, {fz:.3f})"
    assert dist <= _NAV_TOL
    assert not (_TABLE_X[0] <= fx <= _TABLE_X[1] and _TABLE_Y[0] <= fy <= _TABLE_Y[1]), "rammed the table"


def test_navigate_to_with_piper_attached_does_not_crash() -> None:
    """REGRESSION (R12): navigate_to with the Piper attached and a tight tol must not
    segfault or hang (the original raced the stepper with mj_forward on the live data)."""
    from vector_os_nano.sim.drivers.arm import Arm

    d = stand_up("go2_piper")
    try:
        Arm(d.cfg.arm, d.rt, wait_mount=d.base.wait_settled)
        start = d.base.get_position()
        t0 = time.time()
        d.base.navigate_to(*_GOAL, tol=0.12, timeout=30.0)
        elapsed = time.time() - t0
        end = d.base.get_position()
        assert elapsed < 60.0, f"navigate_to took {elapsed:.1f} s: a hang"
        assert math.hypot(end[0] - start[0], end[1] - start[1]) > 0.15, "the dog did not move"
    finally:
        d.close()
