# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Pick binds its target from the call, never from the simulator's oracle.

Two scenarios:
  (a) execute_skill("pick", {}) — no object_label → the skill refuses (no_target)
      and nothing moves; it used to grasp the nearest scene object (banana) read
      from get_object_positions.
  (b) execute_skill("pick", {"object_label": "mug"}) — bound target → the mug
      is gripped (NOT the nearest banana).

The test is headless (gui=False) and deterministic — no LLM, no network.
Requires mujoco; entire module is skipped when mujoco is absent.
"""
from __future__ import annotations


import pytest

# Guard: skip when mujoco is not installed (hardware or CI without sim deps).
pytest.importorskip("mujoco", reason="mujoco not installed")

from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm
from vector_os_nano.hardware.sim.mujoco_gripper import MuJoCoGripper
from vector_os_nano.hardware.sim.mujoco_perception import MuJoCoPerception
from vector_os_nano.core.agent import Agent
from vector_os_nano.skills import get_so101_skills
from vector_os_nano.skills.pick import SIM_PICK_CONFIG
from vector_os_nano.vcli.worlds.sim_oracles import rig_namespace


# ---------------------------------------------------------------------------
# Scene geometry (matches so101_mujoco.xml free bodies at sim start)
# All distances are XY from origin; banana wins nearest.
# ---------------------------------------------------------------------------

# Confirmed nearest by computing sqrt(x²+y²) for each free body:
#   banana (0.12, 0.12) → 0.170
#   duck   (0.15,-0.10) → 0.180
#   mug    (0.22, 0.05) → 0.226   ← bound-target test
_NEAREST_OBJECT = "banana"
_BOUND_TARGET = "mug"


# ---------------------------------------------------------------------------
# Fixtures — module-scoped for speed (one sim per module, two tests share it)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim_agent_unbound():
    """Headless agent wired with SIM_PICK_CONFIG — for the unbound-target pick."""
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


@pytest.fixture(scope="module")
def sim_agent_bound():
    """Separate headless agent for the bound-target pick (fresh sim state)."""
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
    reg = get_so101_skills()
    for s in reg:
        agent._skill_registry.register(s)
    yield agent
    arm.disconnect()


# ---------------------------------------------------------------------------
# (a) Unbound pick — no object_label → nearest object (banana) is lifted
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPickUnboundRefused:
    """R2: pick with no target asks for one; it never reads the nearest object from
    the simulator's oracle (skills-manip:pick-sim-oracle-nearest-object)."""

    def test_unbound_pick_fails_and_names_why(self, sim_agent_unbound: Agent) -> None:
        result = sim_agent_unbound.execute_skill("pick", {"mode": "hold"})
        assert not result.success and result.result_data["diagnosis"] == "no_target"

    def test_nothing_is_held_or_lifted(self, sim_agent_unbound: Agent) -> None:
        holding = rig_namespace(sim_agent_unbound._arm.runtime, "so101")["holding_object"]
        assert holding() is False
        z = sim_agent_unbound._arm.get_object_positions()[_NEAREST_OBJECT][2]
        assert z < 0.15, f"{_NEAREST_OBJECT!r} must stay on the table, got z={z:.3f}m"


# ---------------------------------------------------------------------------
# (b) Bound-target pick — object_label="mug" → mug is gripped, not banana
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPickBoundTargetUnchanged:
    """Bound target must NOT be overridden by the nearest-object logic."""

    def test_gripper_empty_before_bound_pick(self, sim_agent_bound: Agent) -> None:
        holding = rig_namespace(sim_agent_bound._arm.runtime, "so101")["holding_object"]
        assert holding() is False, "Expected empty gripper before bound pick"

    def test_bound_pick_succeeds(self, sim_agent_bound: Agent) -> None:
        result = sim_agent_bound.execute_skill(
            "pick", {"object_label": _BOUND_TARGET, "mode": "hold"}
        )
        assert result.success, (
            f"Bound pick({_BOUND_TARGET!r}) failed: "
            f"{getattr(result, 'failure_reason', result)}"
        )

    def test_holding_after_bound_pick(self, sim_agent_bound: Agent) -> None:
        holding = rig_namespace(sim_agent_bound._arm.runtime, "so101")["holding_object"]
        assert holding() is True, "holding_object() must be True after bound pick"

    def test_mug_is_lifted_not_banana(self, sim_agent_bound: Agent) -> None:
        """The bound object (mug) is lifted; banana stays on the table."""
        arm = sim_agent_bound._arm
        positions = arm.get_object_positions()
        mug_z = positions.get(_BOUND_TARGET, [0.0, 0.0, 0.0])[2]
        assert mug_z > 0.05, (
            f"{_BOUND_TARGET!r} should be lifted > 5cm, got z={mug_z:.3f}m"
        )
        # Banana should still be near table level (not lifted by this pick).
        banana_z = positions.get(_NEAREST_OBJECT, [0.0, 0.0, 0.06])[2]
        assert banana_z < 0.15, (
            f"{_NEAREST_OBJECT!r} should stay on table, got z={banana_z:.3f}m"
        )
