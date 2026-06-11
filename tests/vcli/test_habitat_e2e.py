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


# ---------------------------------------------------------------------------
# M2 acceptance (part 2) — verify predicates through the REAL sandbox
# ---------------------------------------------------------------------------

_APARTMENT = os.environ.get(
    "VECTOR_HABITAT_APARTMENT",
    str(Path.home() / "sandbox" / "habitat-spike" / "data" / "scene_datasets"
        / "habitat-test-scenes" / "apartment_1.glb"),
)


@pytest.fixture(scope="module")
def apartment_world_and_agent():
    """The full M2 stack: registry world + real habitat base + agent stub."""
    import dataclasses
    from types import SimpleNamespace

    from vector_os_nano.playground.catalog import get_scenario
    from vector_os_nano.playground.world import PlaygroundWorld

    scenario = dataclasses.replace(get_scenario("apartment"), scene_ref=_APARTMENT)
    world = PlaygroundWorld(scenario)
    base = HabitatBase(scene=_APARTMENT)
    base.connect()
    agent = SimpleNamespace(_base=base)
    yield world, agent, base
    base.disconnect()


@pytest.mark.skipif(
    not os.path.exists(_APARTMENT), reason="apartment_1 test scene not present"
)
def test_m2_acceptance_connect_walk_verify(apartment_world_and_agent) -> None:
    """campaign M2 判据: headless connect→walk→verify 走通 (真谓词+真沙箱)."""
    from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier

    world, agent, base = apartment_world_and_agent
    verifier = GoalVerifier(world.build_verify_namespace(agent))

    # Walk somewhere, then verify AT the actual position via the sandbox.
    base.walk(vx=0.4, duration=1.5)
    x, y, _ = base.get_position()
    ok, _val = verifier.evaluate(f"at_position({x:.3f}, {y:.3f}, 0.5)")
    assert ok is True
    ok_far, _ = verifier.evaluate(f"at_position({x + 5.0:.3f}, {y + 5.0:.3f}, 0.5)")
    assert ok_far is False  # the predicate discriminates, not rubber-stamps

    # Geodesic oracle through the sandbox: near target small, far target larger.
    ok_geo, near = verifier.evaluate(f"geodesic_dist({x:.3f}, {y:.3f})")
    assert ok_geo is False or near < 0.25  # bool(0.0-ish) may be False; value is what matters
    assert near == pytest.approx(0.0, abs=0.25)

    h0 = base.get_heading()
    ok_face, _ = verifier.evaluate(f"facing({h0:.4f}, 0.2)")
    assert ok_face is True


# ---------------------------------------------------------------------------
# M3 muscle layer — shortest-path navigation + egocentric render (live)
# ---------------------------------------------------------------------------


def test_navigate_to_far_point_and_render(apartment_world_and_agent) -> None:
    """Navigate across the apartment to a far navmesh point; verify with the
    geodesic predicate (the M3 success criterion); render an egocentric frame."""
    world, agent, base = apartment_world_and_agent

    # Pick a target meaningfully far away: probe a few snapped offsets and
    # keep the one with the largest finite geodesic distance from here.
    here = base.get_position()
    best, best_d = None, 0.0
    for dx, dy in ((4, 0), (-4, 0), (0, 4), (0, -4), (3, 3), (-3, -3)):
        cand = base.snap_point([here[0] + dx, here[1] + dy, here[2]])
        d = base.geodesic_distance(here, cand)
        import math as _m
        if _m.isfinite(d) and d > best_d:
            best, best_d = cand, d
    assert best is not None and best_d > 1.0, "no far navigable target found"

    out = base.navigate_to(best[0], best[1])
    assert out["reached"] is True, f"navigation failed: {out}"

    from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier

    verifier = GoalVerifier(world.build_verify_namespace(agent))
    ok, dist = verifier.evaluate(f"geodesic_dist({best[0]:.3f}, {best[1]:.3f}) < 0.5")
    assert ok is True, f"geodesic verify failed (dist check returned {dist})"

    png = base.render_rgb_png()
    assert png.startswith(b"\x89PNG") and len(png) > 2000  # a real image, not a stub


def test_pano_pair_is_real_and_pose_synced(apartment_world_and_agent) -> None:
    """M4: equirect color+depth pair with the pose captured in the same frame."""
    import numpy as np

    _, _, base = apartment_world_and_agent
    pano = base.get_pano()
    assert pano["rgb"].shape == (960, 1920, 3)
    assert pano["depth"].shape == (960, 1920)
    finite = pano["depth"][np.isfinite(pano["depth"]) & (pano["depth"] > 0)]
    assert finite.size > 10000 and 0.2 < float(finite.min()) < float(finite.max()) < 50.0
    assert float(pano["rgb"].std()) > 5.0  # a real scene, not a blank buffer
    assert len(pano["pos"]) == 3 and np.isfinite(pano["heading"])


def test_unprojected_pano_cloud_is_plausible(apartment_world_and_agent) -> None:
    """M4: the /registered_scan source — world-frame cloud around the agent."""
    import numpy as np

    from vector_os_nano.playground.habitat.sysnav_bridge import (
        unproject_equirect_depth,
    )

    _, _, base = apartment_world_and_agent
    pano = base.get_pano()
    pts = unproject_equirect_depth(pano["depth"], pano["pos"], pano["heading"])
    assert len(pts) > 5000
    center = np.array([pano["pos"][0], pano["pos"][1], pano["pos"][2] + 1.2])
    dists = np.linalg.norm(pts - center, axis=1)
    assert 0.05 < float(dists.min()) and float(np.median(dists)) < 10.0
