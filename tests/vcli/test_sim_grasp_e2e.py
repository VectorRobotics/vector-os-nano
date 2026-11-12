# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Headless sim grasp regression — Step 4 FIX 1 validation.

Asserts that SIM_PICK_CONFIG (z_offset=0.0) produces a REAL grasp in the
MuJoCo arm sim: holding_object() transitions from False to True, and the
target object is physically lifted (z > initial_z + 0.05).

Single object, single attempt, ~3-5s wall time.
"""
from __future__ import annotations

import pytest

# Skip the entire module if mujoco is not installed.
pytest.importorskip("mujoco", reason="mujoco not installed")

from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm
from vector_os_nano.hardware.sim.mujoco_gripper import MuJoCoGripper
from vector_os_nano.hardware.sim.mujoco_perception import MuJoCoPerception
from vector_os_nano.core.agent import Agent
from vector_os_nano.skills import get_so101_skills
from vector_os_nano.skills.pick import SIM_PICK_CONFIG
from vector_os_nano.vcli.worlds.sim_oracles import rig_namespace


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sim_agent():
    """Connected headless arm Agent with SIM_PICK_CONFIG, module-scoped for speed."""
    arm = MuJoCoArm(gui=False)
    arm.connect()
    gripper = MuJoCoGripper(arm)
    perception = MuJoCoPerception(arm)
    agent = Agent(
        arm=arm,
        gripper=gripper,
        perception=perception,
        config={"skills": {"pick": dict(SIM_PICK_CONFIG)}},
    )
    # Re-register skills so the fixture-built agent uses the same set as CLI.
    reg = get_so101_skills()
    for s in reg:
        agent._skill_registry.register(s)
    yield agent
    arm.disconnect()


# ---------------------------------------------------------------------------
# SIM_PICK_CONFIG contract
# ---------------------------------------------------------------------------

class TestSimPickConfig:
    """Static contract for SIM_PICK_CONFIG — the single-source constant."""

    def test_z_offset_is_zero(self):
        assert SIM_PICK_CONFIG["z_offset"] == 0.0

    def test_hardware_offsets_disabled(self):
        assert SIM_PICK_CONFIG["hardware_offsets"] is False


# ---------------------------------------------------------------------------
# Headless grasp regression
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSimGraspE2E:
    """Real headless grasp: banana must be physically lifted after pick."""

    def test_not_holding_before_pick(self, sim_agent: Agent):
        """Pre-condition: gripper is empty before the pick."""
        holding = rig_namespace(sim_agent._arm.runtime, "so101")["holding_object"]
        assert holding() is False, "Expected gripper empty before pick"

    def test_banana_grasp_and_lift(self, sim_agent: Agent):
        """After pick(banana, hold), object is held AND lifted > 5cm."""
        arm = sim_agent._arm

        # Record initial banana z.
        initial_positions = arm.get_object_positions()
        assert "banana" in initial_positions, "banana must be present in MuJoCo scene"
        initial_z = initial_positions["banana"][2]

        # Execute the pick skill.
        result = sim_agent.execute_skill(
            "pick",
            {"object_label": "banana", "mode": "hold"},
        )
        assert result.success, f"pick skill failed: {getattr(result, 'failure_reason', result)}"

        # Check holding predicate.
        holding = rig_namespace(sim_agent._arm.runtime, "so101")["holding_object"]
        assert holding() is True, "holding_object() must be True after successful pick(hold)"

        # Check the banana is physically lifted.
        final_positions = arm.get_object_positions()
        final_z = final_positions.get("banana", [0.0, 0.0, 0.0])[2]
        lift = final_z - initial_z
        assert lift > 0.05, (
            f"banana should be lifted > 5cm above initial z={initial_z:.3f}m, "
            f"got final z={final_z:.3f}m (lift={lift*100:.1f}cm)"
        )
