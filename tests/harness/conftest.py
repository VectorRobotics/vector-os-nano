# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Shared fixtures for the Go2 locomotion harness.

All harness tests require mujoco. The go2 runs its manifest's policy (convex_mpc).
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


@pytest.fixture
def go2_flat():
    """The go2's generic LeggedBase on flat ground, headless, as the session builds it.

    Yields the connected base (its runtime running); closes both after the test.
    """
    from tests.hardware.sim.dog import stand_up

    dog = stand_up("go2", room=False, start=False)
    dog.rt.start()
    yield dog.base
    dog.close()


@pytest.fixture
def go2_standing(go2_flat):
    """The base after its stand posture (2 s), so tests start from a stable stance."""
    assert go2_flat.set_posture("stand", duration=2.0)
    return go2_flat
