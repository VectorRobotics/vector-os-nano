# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Shared fixtures for the Go2 locomotion harness.

All harness tests require mujoco. convex_mpc is intentionally excluded.
Fixtures are scope="function" so each test gets a clean simulation state.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path so vector_os_nano is importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Skip entire harness if mujoco is unavailable
pytest.importorskip("mujoco", reason="mujoco not installed — skipping locomotion harness")


def _import_mujoco_go2():
    """The ONE MuJoCoGo2 class (a by-path re-load made a second copy in sys.modules)."""
    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2

    return MuJoCoGo2


@pytest.fixture
def go2_flat():
    """Connect a headless MuJoCoGo2 on flat ground (room=False).

    Yields the connected instance; disconnects after test completes.
    """
    MuJoCoGo2 = _import_mujoco_go2()
    robot = MuJoCoGo2(gui=False, room=False)
    robot.connect()
    yield robot
    robot.disconnect()


@pytest.fixture
def go2_standing(go2_flat):
    """Connect and stand the robot before the test.

    Builds on go2_flat; calls stand(duration=2.0) so tests start from
    a stable standing posture.
    """
    go2_flat.stand(duration=2.0)
    return go2_flat
