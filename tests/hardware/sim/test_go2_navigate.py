# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Sim test: MuJoCoGo2.navigate_to — obstacle-aware path planning.

Verifies that:
  1. navigate_to(10.46, 3.0) returns True (dog arrived within tolerance).
  2. The dog's final position is within 0.25 m of the goal.
  3. The dog did NOT end up inside the pick_table footprint
     (x ∈ [10.80, 11.10], y ∈ [2.75, 3.25]) — i.e. it stopped in front,
     not rammed through.
  4. obstacles_from_model is importable for the go2 model (generic function).

Run with:
  MUJOCO_GL=egl PATH=/usr/bin:$PATH .venv/bin/python -m pytest \
      tests/hardware/sim/test_go2_navigate.py -q -s -m sim
"""
from __future__ import annotations

import math
import time

import pytest

mujoco = pytest.importorskip("mujoco")

from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
from vector_os_nano.hardware.sim.mujoco_g1 import obstacles_from_model

# Pick-table footprint (x ∈ [10.80, 11.10], y ∈ [2.75, 3.25])
_TABLE_X_MIN: float = 10.80
_TABLE_X_MAX: float = 11.10
_TABLE_Y_MIN: float = 2.75
_TABLE_Y_MAX: float = 3.25

# Navigate goal: clear point in front of the pick_table
_GOAL_X: float = 10.46
_GOAL_Y: float = 3.0
_NAV_TOL: float = 0.25


# ---------------------------------------------------------------------------
# Session-scope fixture: one Go2 sim for all tests in this module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def go2_sim():
    """Headless MPC Go2 in the room scene. Torn down after module."""
    go2 = MuJoCoGo2(gui=False, room=True, backend="mpc")
    go2.connect()
    # Allow physics to settle before navigating
    time.sleep(0.5)
    try:
        yield go2
    finally:
        go2.disconnect()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.sim
def test_obstacles_from_model_importable_for_go2(go2_sim: MuJoCoGo2) -> None:
    """obstacles_from_model is generic and works with the go2 scene."""
    mj = mujoco
    mj.mj_forward(go2_sim._mj.model, go2_sim._mj.data)
    polys = obstacles_from_model(
        go2_sim._mj.model,
        go2_sim._mj.data,
        robot_geom_ids=go2_sim._mj.layout.robot_geom_ids,
    )
    # The go2 room has furniture, walls, and pick_table — expect multiple polygons
    assert len(polys) > 0, (
        f"Expected obstacles in the room scene, got {len(polys)}"
    )
    print(f"\n[test] obstacles_from_model: {len(polys)} polygons enumerated")


@pytest.mark.sim
def test_navigate_to_arrives_and_avoids_table(go2_sim: MuJoCoGo2) -> None:
    """navigate_to reaches the goal in front of the table without ramming through.

    Acceptance criteria:
      - Returns True (arrived within tol).
      - Final position within _NAV_TOL of (_GOAL_X, _GOAL_Y).
      - Final position NOT inside the table footprint.
    """
    print(
        f"\n[test] navigate_to({_GOAL_X}, {_GOAL_Y}) — "
        f"start pos: {go2_sim.get_position()}"
    )

    arrived = go2_sim.navigate_to(_GOAL_X, _GOAL_Y, tol=_NAV_TOL, timeout=90.0)

    final_pos = go2_sim.get_position()
    fx, fy = final_pos[0], final_pos[1]
    dist_to_goal = math.hypot(fx - _GOAL_X, fy - _GOAL_Y)

    print(
        f"[test] navigate_to returned: {arrived}, "
        f"final pos=({fx:.3f}, {fy:.3f}), "
        f"dist_to_goal={dist_to_goal:.3f} m"
    )

    # --- Assertion 1: function reported arrival ---
    assert arrived, (
        f"navigate_to returned False — dog did not arrive within {_NAV_TOL} m. "
        f"Final pos=({fx:.3f}, {fy:.3f}), dist={dist_to_goal:.3f} m"
    )

    # --- Assertion 2: distance to goal ---
    assert dist_to_goal <= _NAV_TOL, (
        f"Dog ended {dist_to_goal:.3f} m from goal ({_GOAL_X}, {_GOAL_Y}), "
        f"tolerance={_NAV_TOL} m. Final pos=({fx:.3f}, {fy:.3f})"
    )

    # --- Assertion 3: dog did NOT end inside the table footprint ---
    inside_table = (
        _TABLE_X_MIN <= fx <= _TABLE_X_MAX
        and _TABLE_Y_MIN <= fy <= _TABLE_Y_MAX
    )
    assert not inside_table, (
        f"Dog ended INSIDE the pick_table footprint at ({fx:.3f}, {fy:.3f})! "
        f"Table footprint: x∈[{_TABLE_X_MIN},{_TABLE_X_MAX}], "
        f"y∈[{_TABLE_Y_MIN},{_TABLE_Y_MAX}]. "
        "The dog rammed through instead of stopping in front."
    )

    print(
        f"[test] PASS — arrived={arrived}, dist={dist_to_goal:.3f} m, "
        f"inside_table={inside_table}"
    )


@pytest.mark.sim
def test_navigate_to_with_piper_attached_does_not_crash(monkeypatch) -> None:
    """REGRESSION (R12): navigate_to must not segfault/hang with the Piper attached.

    The grasp path runs navigate_to with a MuJoCoPiper connected and a tight tol. The
    original navigate_to called mj_forward to snapshot the pose — that WRITES derived
    fields and RACED the 1 kHz gait daemon's mj_step, segfaulting/hanging (the green
    grasp "hang"). The fix reads the pose via the thread-safe accessor and never calls
    mj_forward on the live model. This test reproduces the crashing configuration
    (arm-enabled scene + Piper attached + tight tol) and asserts navigate_to returns
    in bounded wall-time having moved the dog. Self-contained (own arm-enabled sim).
    """
    monkeypatch.setenv("VECTOR_SIM_WITH_ARM", "1")  # never leaks past this test
    from vector_os_nano.hardware.sim.mujoco_piper import MuJoCoPiper

    go2 = MuJoCoGo2(gui=False, room=True, backend="mpc")
    try:
        go2.connect()
        piper = MuJoCoPiper(go2)
        piper.connect()
        time.sleep(0.5)
        start = go2.get_position()
        t0 = time.time()
        # Tight tol (the grasp-context value) toward the table-front standoff.
        arrived = go2.navigate_to(10.46, 3.0, tol=0.12, timeout=30.0)
        elapsed = time.time() - t0
        end = go2.get_position()
        moved = math.hypot(end[0] - start[0], end[1] - start[1])
        print(
            f"\n[test] navigate_to(+piper) -> {arrived} in {elapsed:.1f}s, "
            f"moved {moved:.2f} m, end=({end[0]:.2f},{end[1]:.2f})"
        )
        # Bounded wall-time (no hang/segfault — the test completing at all proves no
        # crash; this also rules out a hang).
        assert elapsed < 60.0, f"navigate_to took {elapsed:.1f}s — possible hang"
        # Dog actually advanced toward the goal (didn't no-op).
        assert moved > 0.15, f"dog barely moved ({moved:.2f} m)"
    finally:
        go2.disconnect()  # the Piper rides on the go2 sim (its disconnect is a flag)
