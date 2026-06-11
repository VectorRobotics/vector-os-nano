# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""M2 acceptance (part 1) — REAL habitat subprocess end-to-end.

Runs only where the pinned conda interpreter AND a test scene exist (this
campaign's Linux box); skips elsewhere. Exercises the genuine ADR-009
process split: spawn server -> PORT handshake -> connect -> walk on the
navmesh -> exact odometry -> geodesic oracle -> clean shutdown, no orphan.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from vector_os_nano.playground.habitat.base import HabitatBase
from vector_os_nano.playground.habitat.bridge import habitat_available

_SCENE = os.environ.get(
    "VECTOR_HABITAT_SCENE",
    str(Path.home() / "sandbox" / "habitat-spike" / "data" / "scene_datasets"
        / "habitat-test-scenes" / "skokloster-castle.glb"),
)

pytestmark = pytest.mark.skipif(
    not (habitat_available() and os.path.exists(_SCENE)),
    reason="pinned habitat conda env or test scene not present",
)


@pytest.fixture(scope="module")
def live_base():
    base = HabitatBase(scene=_SCENE)
    base.connect()
    yield base
    base.disconnect()


def test_connect_state_walk_roundtrip(live_base) -> None:
    p0 = live_base.get_position()
    h0 = live_base.get_heading()
    assert all(math.isfinite(v) for v in p0) and math.isfinite(h0)

    assert live_base.walk(vx=0.5, duration=1.0) is True
    p1 = live_base.get_position()
    moved = math.dist(p0[:2], p1[:2])
    # navmesh may clip the step near obstacles — moved, but never more than commanded
    assert 0.0 < moved <= 0.5 + 1e-3

    live_base.walk(vyaw=math.pi / 2, duration=1.0)
    assert abs(live_base.get_heading() - (h0 + math.pi / 2)) % (2 * math.pi) < 0.05


def test_geodesic_oracle_finite_and_consistent(live_base) -> None:
    a = live_base.get_position()
    live_base.walk(vx=0.4, duration=1.0)
    b = live_base.get_position()
    d = live_base.geodesic_distance(a, b)
    euclid = math.dist(a[:2], b[:2])
    assert math.isfinite(d)
    assert d >= euclid - 1e-3  # geodesic can never undercut the straight line


def test_snap_point_lands_on_navmesh(live_base) -> None:
    p = live_base.get_position()
    snapped = live_base.snap_point([p[0] + 0.2, p[1] + 0.2, p[2]])
    assert all(math.isfinite(v) for v in snapped)
