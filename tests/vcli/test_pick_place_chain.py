# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Pick -> place handoff regression (headless sim).

Exercises the manipulation handoff a "grab X then put it down" plan relies on:
pick(mode='hold') leaves the object in the gripper (holding_object() True), then
place releases it (holding_object() False) and the object ends up moved + resting.
This guards both the pick->place capability and place's grounded verify_hint
(`released()`, WP07), which replaced the trivially-true `placed_count() >= 1` and the
never-grounding negation `not holding_object()`.
"""
from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")  # headless sim oracle; skip if unavailable


def _agent():
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.core.skill import SkillRegistry
    from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm
    from vector_os_nano.hardware.sim.mujoco_gripper import MuJoCoGripper
    from vector_os_nano.hardware.sim.mujoco_perception import MuJoCoPerception
    from vector_os_nano.skills import get_default_skills
    from vector_os_nano.skills.pick import SIM_PICK_CONFIG

    arm = MuJoCoArm(gui=False)
    arm.connect()
    agent = Agent(
        arm=arm,
        gripper=MuJoCoGripper(arm),
        perception=MuJoCoPerception(arm),
        config={"skills": {"pick": dict(SIM_PICK_CONFIG)}},
    )
    reg = SkillRegistry()
    for s in get_default_skills():
        reg.register(s)
    agent._skill_registry = reg
    return agent, arm


def test_pick_hold_then_place_releases_and_moves_object():
    from vector_os_nano.vcli.worlds.arm_sim_oracle import make_holding_object, make_released

    agent, arm = _agent()
    try:
        holding = make_holding_object(agent)
        released = make_released(agent)
        x0, y0, _ = arm.get_object_positions()["banana"]

        assert holding() is False  # gripper starts empty

        pick = agent.execute_skill("pick", {"object_label": "banana", "mode": "hold"})
        assert pick.success
        assert holding() is True, "pick(mode='hold') must leave the object held"
        assert released() is False and released("banana") is False

        place = agent.execute_skill("place", {"location": "left"})
        assert place.success
        # place's grounded verify: the gripper released the object.
        assert holding() is False, "place must release the held object"
        assert released() is True and released("banana") is True  # place's verify_hint

        x1, y1, _ = arm.get_object_positions()["banana"]
        moved = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        assert moved > 0.05, f"placed object should have moved (>5cm); moved {moved:.3f}m"
    finally:
        arm.disconnect()
