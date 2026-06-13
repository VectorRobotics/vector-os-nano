# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Campaign #5 R3b — g1_flat scenario registration + boot wiring.

The scenario resolves to a base PlaygroundWorld with the robot persona;
boot_g1_agent attaches a real G1MuJoCoBase with the base-only skill set,
and WalkSkill produces real-physics moved_m against it. Boot/walk tests
skip without the downloaded assets (scripts/setup_g1_gait.sh).
"""
from __future__ import annotations


import pytest

from vector_os_nano.hardware.sim.mujoco_g1 import g1_assets_ready
from vector_os_nano.playground.catalog import get_scenario
from vector_os_nano.playground.world import PlaygroundWorld


class TestScenarioRegistration:
    """No assets needed — pure registry/world contract."""

    def test_g1_flat_registered_with_g1_embodiment(self):
        sc = get_scenario("g1_flat")
        assert sc.embodiment == "g1"
        assert sc.sim_backend == "mujoco"     # in-process, not habitat
        assert sc.scene_xml.endswith("assets/g1_gait/scene.xml")

    def test_g1_world_is_a_base_world(self):
        w = PlaygroundWorld(get_scenario("g1_flat"))
        assert w.has_base() is True

    def test_g1_world_uses_robot_persona_not_habitat(self):
        # mujoco backend → robot persona (NOT the habitat "already running"
        # persona, which would send the planner hunting for launch scripts).
        w = PlaygroundWorld(get_scenario("g1_flat"))
        role, _tools = w.persona_blocks()
        from vector_os_nano.vcli.prompt import ROBOT_ROLE_PROMPT
        assert role == ROBOT_ROLE_PROMPT

    def test_g1_world_base_verify_namespace(self):
        # base predicates present; geodesic fails safe (no navmesh on flat).
        w = PlaygroundWorld(get_scenario("g1_flat"))
        ns = w.build_verify_namespace(agent=None)
        assert "at_position" in ns and "facing" in ns


@pytest.mark.skipif(
    not g1_assets_ready(),
    reason="g1_gait assets not installed (run scripts/setup_g1_gait.sh)")
class TestBootWiring:
    def test_boot_attaches_g1_base_and_base_skills(self):
        from vector_os_nano.vcli.g1_runtime import boot_g1_agent

        agent = boot_g1_agent(PlaygroundWorld(get_scenario("g1_flat")))
        try:
            from vector_os_nano.hardware.sim.mujoco_g1 import G1MuJoCoBase
            assert isinstance(agent._base, G1MuJoCoBase)
            skills = set(agent._skill_registry.list_skills())
            assert {"walk", "turn", "stop"} <= skills
            # base-only: no arm skills taught to a baseless-of-arm humanoid
            assert "pick" not in skills and "place" not in skills
        finally:
            agent._base.disconnect()

    def test_walk_skill_real_physics_moved_m(self):
        from types import SimpleNamespace

        from vector_os_nano.skills.go2.walk import WalkSkill
        from vector_os_nano.vcli.g1_runtime import boot_g1_agent

        agent = boot_g1_agent(PlaygroundWorld(get_scenario("g1_flat")))
        try:
            ctx = SimpleNamespace(
                base=agent._base, world_model=None, agent=agent)
            res = WalkSkill().execute(
                {"direction": "forward", "distance": 1.0}, ctx)
            assert res.success is True
            moved = res.result_data.get("moved_m")
            assert moved is not None and moved > 0.3, \
                f"walk produced no real motion: {moved}"
        finally:
            agent._base.disconnect()
