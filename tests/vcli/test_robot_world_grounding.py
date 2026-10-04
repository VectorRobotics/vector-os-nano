# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""RobotWorld sim-oracle grounding (Step 2 GROUNDING).

The plain RobotWorld (the normal "open the so101 sim -> grasp the banana" CLI
path) must contribute the SAME deterministic sim-oracle verify predicates the
PlaygroundWorld has, so verify stops failing on the engine's empty perception
stubs (detect_objects->[], describe_scene->"") and the planner's verify allowlist
gains real arm predicates.

Covers:
- RobotWorld.build_verify_namespace(agent) returns the five sim-oracle predicates
  when a SIM arm is present, with the right shapes/behaviour;
- it stays byte-identical ({}) on the no-arm / real-hardware path;
- the ENGINE end-to-end wiring: the world predicates REPLACE the engine's empty
  stubs in the merged verifier namespace.

Uses a REAL headless MuJoCoArm (gui=False) — connect() is deterministic (~0.3s)
and needs no GL. Skips cleanly if mujoco is unavailable.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vector_os_nano.vcli.worlds import RobotWorld

# The sim oracle requires a real MuJoCo arm. Skip the whole module (rather than
# error) if the sim stack cannot import on this host.
mujoco = pytest.importorskip("mujoco")


@pytest.fixture()
def sim_agent():
    """A connected headless MuJoCoArm + gripper wrapped in a real Agent.

    connect() loads the bundled tabletop scene deterministically (no GL); the
    arm's get_object_positions is the oracle the predicates read.
    """
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm
    from vector_os_nano.hardware.sim.mujoco_gripper import MuJoCoGripper

    arm = MuJoCoArm(gui=False)
    arm.connect()
    gripper = MuJoCoGripper(arm)
    agent = Agent(arm=arm, gripper=gripper)
    try:
        yield agent
    finally:
        try:
            arm.disconnect()
        except Exception:  # noqa: BLE001 — teardown best-effort
            pass


# ---------------------------------------------------------------------------
# build_verify_namespace — sim arm present
# ---------------------------------------------------------------------------


def test_namespace_exposes_sim_oracle_predicates(sim_agent) -> None:
    ns = RobotWorld().build_verify_namespace(sim_agent)
    assert set(ns) == {
        "detect_objects",
        "describe_scene",
        "holding_object",
        "arm_at_home",
        "placed_count",
        "released",  # WP07: the gripper fall-transition oracle (place's verify_hint)
    }
    # Every contributed binding must be callable (drops straight into verify ns).
    assert all(callable(fn) for fn in ns.values())


def test_detect_objects_filters_and_returns_all(sim_agent) -> None:
    ns = RobotWorld().build_verify_namespace(sim_agent)
    detect = ns["detect_objects"]

    banana = detect("banana")
    assert len(banana) == 1
    assert banana[0]["name"] == "banana"
    # Same {"name","x","y","z"} shape the engine stub returns, just non-empty.
    assert {"name", "x", "y", "z"} <= set(banana[0])

    # Empty query => all scene objects (the bundled tabletop scene has 6).
    everything = detect("")
    assert len(everything) == 6
    assert {o["name"] for o in everything} == {
        "banana",
        "bottle",
        "duck",
        "lego",
        "mug",
        "screwdriver",
    }


def test_describe_scene_non_empty(sim_agent) -> None:
    describe = RobotWorld().build_verify_namespace(sim_agent)["describe_scene"]
    summary = describe()
    assert isinstance(summary, str)
    assert "banana" in summary


def test_holding_and_home_predicates_are_bools(sim_agent) -> None:
    ns = RobotWorld().build_verify_namespace(sim_agent)
    # Nothing grasped yet — not holding.
    assert ns["holding_object"]() is False
    # arm_at_home is a coarse joint check; just assert the contract (a bool).
    assert isinstance(ns["arm_at_home"](), bool)
    # placed_count returns an int (resting objects); fails safe, never raises.
    assert isinstance(ns["placed_count"](), int)


# ---------------------------------------------------------------------------
# build_verify_namespace — no-arm / real-hardware path stays byte-identical
# ---------------------------------------------------------------------------


def test_namespace_empty_without_agent() -> None:
    assert RobotWorld().build_verify_namespace(None) == {}


def test_namespace_empty_when_arm_lacks_oracle() -> None:
    """A non-sim arm (no get_object_positions) contributes nothing."""

    class _RealArmStub:
        # Deliberately no get_object_positions — represents real hardware.
        def get_joint_positions(self):  # pragma: no cover - never called
            return [0.0] * 5

    agent = MagicMock()
    agent._arm = _RealArmStub()
    agent._base = None  # real-hardware arm, no sim base (else MagicMock leaks a base-like mock)
    assert RobotWorld().build_verify_namespace(agent) == {}


# ---------------------------------------------------------------------------
# Engine end-to-end wiring — world predicates replace the empty stubs
# ---------------------------------------------------------------------------


def test_engine_merges_robot_world_grounding(sim_agent) -> None:
    """The merged verifier namespace must carry the REAL detect_objects.

    Proves the engine's empty perception stub (detect_objects -> []) is REPLACED
    by RobotWorld's sim-oracle binding once the world is wired in.
    """
    from vector_os_nano.vcli.engine import VectorEngine

    engine = VectorEngine(backend=MagicMock(), intent_router=MagicMock())
    engine._world = RobotWorld()
    engine._vgg_agent = sim_agent

    ns = engine._build_verifier_namespace(sim_agent)
    assert "detect_objects" in ns
    assert "holding_object" in ns
    # The stub would return [] for any query; the real oracle finds the banana.
    assert ns["detect_objects"]("banana"), "stub was not replaced by the sim oracle"
