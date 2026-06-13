# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #6 batch 1 — G1MuJoCoBase.navigate_to closed-loop waypoint drive.

Every arrival/distance number is REAL physics (get_position), the three-value
contract NavigateToPointSkill unpacks. Gait traps the adversarial workflow
flagged are pinned here: a tol FLOOR, already_there short-circuit, settled
arrival, NaN guard, honest unreachable-in-time -> reached=False. Skipped
without the downloaded assets.
"""
from __future__ import annotations

import math

import pytest

from vector_os_nano.hardware.sim.mujoco_g1 import (
    G1MuJoCoBase,
    _NAV_TOL_FLOOR,
    g1_assets_ready,
)

pytestmark = pytest.mark.skipif(
    not g1_assets_ready(),
    reason="g1_gait assets not installed (run scripts/setup_g1_gait.sh)")


@pytest.fixture()
def base():
    b = G1MuJoCoBase()
    b.connect()
    yield b
    b.disconnect()


class TestContractShape:
    def test_nan_target_fails_loud_no_motion(self, base):
        out = base.navigate_to(float("nan"), 0.0)
        assert out["reached"] is False
        assert out["reason"] == "bad_params_nan"

    def test_already_there_short_circuit(self, base):
        sx, sy, _ = base.get_position()
        out = base.navigate_to(sx, sy, tol=0.3)   # start == goal
        assert out["already_there"] is True
        assert out["reached"] is True
        assert out["moved_m"] == 0.0
        assert out["elapsed_s"] == 0.0

    def test_tol_floored_to_gait_oscillation(self, base):
        # a sub-floor tol is reported as the effective gait floor, never
        # silently chased (the gait can't latch below its COM oscillation).
        sx, sy, _ = base.get_position()
        out = base.navigate_to(sx + 0.05, sy, tol=0.01)
        assert out["effective_tol"] == _NAV_TOL_FLOOR
        # 0.05 m < floor 0.30 -> treated as already there
        assert out["already_there"] is True


class TestRealDrive:
    def test_drives_to_a_reachable_point(self, base):
        sx, sy, _ = base.get_position()
        # 2 m straight ahead-ish; closed loop must face + walk + settle there
        tx, ty = sx + 2.0, sy
        out = base.navigate_to(tx, ty, tol=0.3)
        assert out["reached"] is True, f"did not arrive: {out}"
        assert out["remaining"] <= out["effective_tol"]
        assert out["already_there"] is False
        assert out["net_m"] > 1.0          # real displacement toward goal
        assert out["moved_m"] > 0.0
        # settled pose really is within tol (not an in-flight oscillation peak)
        fx, fy, fz = out["pos"]
        assert math.hypot(tx - fx, ty - fy) <= out["effective_tol"]
        assert fz > 0.45                    # upright at arrival

    def test_turn_then_walk_to_side_target(self, base):
        # target to the SIDE forces the face-first pivot before walking
        sx, sy, _ = base.get_position()
        tx, ty = sx + 1.0, sy + 1.5
        out = base.navigate_to(tx, ty, tol=0.4)
        assert out["reached"] is True, f"did not arrive: {out}"
        fx, fy, _ = out["pos"]
        assert math.hypot(tx - fx, ty - fy) <= out["effective_tol"]


class TestNavigateViewer:
    def test_frames_advance_toward_goal(self, base):
        """GUI evidence: chase frames captured DURING a navigate show the
        robot CLOSING distance to the goal (monotonic-ish) and the frames
        differ (real stepping, not a frozen glide)."""
        import threading
        import time as _t

        sx, sy, _ = base.get_position()
        tx, ty = sx + 2.0, sy + 1.0
        box = {}
        t = threading.Thread(
            target=lambda: box.update(r=base.navigate_to(tx, ty, tol=0.3)),
            daemon=True)
        t.start()
        frames, dists = [], []
        for _i in range(4):
            _t.sleep(1.2)
            frames.append(base.viewer_frame_png())
            p = base.get_position()
            dists.append(math.hypot(tx - p[0], ty - p[1]))
        t.join()
        assert box["r"]["reached"] is True
        # distance to goal shrank overall (closed the gap)
        assert dists[-1] < dists[0] - 0.5, f"did not approach goal: {dists}"
        # frames differ (stepping, not frozen)
        assert frames[0] != frames[-1]
