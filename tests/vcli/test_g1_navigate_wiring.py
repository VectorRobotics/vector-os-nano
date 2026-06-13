# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #6 R2 — NavigateToPointSkill wired to the G1 agent + verify story.

The skill is base-generic and works UNCHANGED against G1MuJoCoBase: it calls
base.navigate_to and reads the three-value contract. The flat-scene verify
story is honest — G1MuJoCoBase.geodesic_distance returns the euclidean
distance (geodesic == straight line on an obstacle-free scene), so the
kernel's geodesic_dist(x, y) coordinate-goal predicate binds instead of
fail-safing to inf.
"""
from __future__ import annotations

import math

import pytest

from vector_os_nano.hardware.sim.mujoco_g1 import G1MuJoCoBase, g1_assets_ready
from vector_os_nano.playground.catalog import get_scenario
from vector_os_nano.playground.world import PlaygroundWorld


def test_geodesic_distance_is_euclidean_on_flat():
    # no assets needed — pure math contract
    b = G1MuJoCoBase.__new__(G1MuJoCoBase)
    d = b.geodesic_distance([0.0, 0.0, 0.7], [3.0, 4.0, 0.7])
    assert abs(d - 5.0) < 1e-6


@pytest.mark.skipif(
    not g1_assets_ready(),
    reason="g1_gait assets not installed (run scripts/setup_g1_gait.sh)")
class TestNavigateWiring:
    def test_registry_has_navigate_to(self):
        from vector_os_nano.vcli.g1_runtime import boot_g1_agent

        agent = boot_g1_agent(PlaygroundWorld(get_scenario("g1_flat")))
        try:
            skills = set(agent._skill_registry.list_skills())
            assert "navigate_to" in skills
            assert {"walk", "turn", "stop"} <= skills
        finally:
            agent._base.disconnect()

    def test_skill_drives_g1_and_reports_reached(self):
        from types import SimpleNamespace

        from vector_os_nano.skills.navigate_to_point import NavigateToPointSkill
        from vector_os_nano.vcli.g1_runtime import boot_g1_agent

        agent = boot_g1_agent(PlaygroundWorld(get_scenario("g1_flat")))
        try:
            sx, sy, _ = agent._base.get_position()
            ctx = SimpleNamespace(base=agent._base, world_model=None, agent=agent)
            res = NavigateToPointSkill().execute(
                {"x": sx + 2.0, "y": sy, "tol": 0.3}, ctx)
            assert res.success is True, f"skill did not reach: {res.error_message}"
            rd = res.result_data
            assert rd["reached"] is True
            assert rd["goal_kind"] == "coordinate"
            assert rd["moved_m"] > 0.0
        finally:
            agent._base.disconnect()

    def test_geodesic_dist_predicate_binds_after_arrival(self):
        # the verify story: after a real drive, the kernel's geodesic_dist
        # predicate (euclidean on G1) reads < tol — NOT inf.
        from vector_os_nano.vcli.g1_runtime import boot_g1_agent

        agent = boot_g1_agent(PlaygroundWorld(get_scenario("g1_flat")))
        try:
            sx, sy, _ = agent._base.get_position()
            tx, ty = sx + 1.5, sy
            agent._base.navigate_to(tx, ty, tol=0.3)
            ns = PlaygroundWorld(get_scenario("g1_flat")).build_verify_namespace(agent)
            gd = ns["geodesic_dist"](tx, ty)
            assert math.isfinite(gd), "geodesic_dist fail-safed to inf on G1"
            assert gd < 0.8, f"not arrived by the coordinate verify: {gd}"
        finally:
            agent._base.disconnect()
