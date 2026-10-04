# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Playground (INC3) — deterministic sim-oracle verify predicates.

Covers:
- the home pose single source (manifest stance == the home skill default),
- the world registry wiring (resolve_world_named -> PlaygroundWorld),
- a kernel-integration test: PlaygroundWorld wired into VectorEngine, the merged
  verifier namespace evaluated THROUGH the real GoalVerifier (the "test with
  vector-os-nano" gate).

No MuJoCo / network — the arm is a deterministic fake.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vector_os_nano.playground import PlaygroundWorld, register_scenarios
from vector_os_nano.playground.catalog import TABLETOP
from tests.harness.fake_gt import bind_fake_sim
from vector_os_nano.embodiments import registry

# The SO-101 home pose (its manifest stance; the arm_at_home oracle reads it there).
_HOME_JOINTS = tuple(registry.load("so101").stance.values())


# ---------------------------------------------------------------------------
# Deterministic fakes (no MuJoCo)
# ---------------------------------------------------------------------------


class FakeArm:
    """A deterministic stand-in for MuJoCoArm's oracle surface.

    Exposes the three oracle methods the predicates read: get_object_positions,
    get_joint_positions, fk. EE position is supplied directly (FK is a pure
    lookup here — the predicates only need a deterministic xyz).
    """

    def __init__(
        self,
        objects: dict[str, list[float]],
        joints: list[float],
        ee: list[float],
        connected: bool = True,
    ) -> None:
        self._objects = objects
        self._joints = list(joints)
        self._ee = list(ee)
        self._connected = connected

    def get_object_positions(self) -> dict[str, list[float]]:
        return {k: list(v) for k, v in self._objects.items()}

    def get_joint_positions(self) -> list[float]:
        return list(self._joints)

    def fk(self, joint_positions: list[float]):
        # Deterministic: EE is fixed regardless of the queried joints.
        return list(self._ee), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


class FakeGripper:
    def __init__(self, holding: bool) -> None:
        self._holding = holding

    def is_holding(self) -> bool:
        return self._holding


def _agent(arm: Any = None, gripper: Any = None) -> SimpleNamespace:
    agent = SimpleNamespace(_arm=arm, _gripper=gripper)
    bind_fake_sim(agent, "so101")  # the live sim session's GT oracles
    return agent


_HOME = list(_HOME_JOINTS)


# ---------------------------------------------------------------------------
# Home-pose single-source agreement
# ---------------------------------------------------------------------------


def test_home_pose_matches_home_skill_default() -> None:
    """The verify home pose must agree with the home SKILL's motion default."""
    from vector_os_nano.skills.home import _DEFAULT_HOME_JOINTS

    assert list(_HOME_JOINTS) == list(_DEFAULT_HOME_JOINTS)












# ---------------------------------------------------------------------------
# World registry wiring (lazy hook)
# ---------------------------------------------------------------------------


class TestRegistryWiring:
    def test_tabletop_resolvable_by_name(self) -> None:
        from vector_os_nano.vcli.worlds.registry import resolve_world_named

        register_scenarios()  # idempotent
        world = resolve_world_named("tabletop")
        assert isinstance(world, PlaygroundWorld)
        assert world.name == "tabletop"
        assert world.is_robot() is True
        assert world.scenario.object_names == TABLETOP.object_names

    def test_register_scenarios_idempotent(self) -> None:
        from vector_os_nano.vcli.worlds.registry import get_world_registry

        register_scenarios()
        register_scenarios()  # must not raise on the second pass
        assert "tabletop" in get_world_registry().names()

    def test_decompose_vocab_single_sourced(self) -> None:
        world = PlaygroundWorld()
        assert world.decompose_vocab() is None
        assert world.derive_vocab_from_registry() is True


# ---------------------------------------------------------------------------
# Kernel integration: predicates through the real GoalVerifier (the gate)
# ---------------------------------------------------------------------------


def _make_engine():
    from vector_os_nano.vcli.engine import VectorEngine
    from vector_os_nano.vcli.intent_router import IntentRouter
    from vector_os_nano.vcli.tools.base import CategorizedToolRegistry

    class _MockBackend:
        def call(self, messages, tools, system, max_tokens, on_text=None):
            class _R:
                text = "{}"

            return _R()

    return VectorEngine(
        backend=_MockBackend(),
        registry=CategorizedToolRegistry(),
        system_prompt=[],
        intent_router=IntentRouter(),
    )


class TestKernelIntegration:
    def test_playground_predicates_via_goal_verifier(self) -> None:
        """PlaygroundWorld wired into the engine; predicates evaluate through the
        real GoalVerifier off the merged verifier namespace."""
        from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier

        arm = FakeArm(
            objects={"mug": [0.22, 0.05, 0.06], "banana": [0.12, 0.12, 0.06]},
            joints=_HOME,
            ee=[0.2, 0.0, 0.2],
        )
        agent = _agent(arm)
        eng = _make_engine()
        eng._world = PlaygroundWorld()
        ns = eng._build_verifier_namespace(agent)

        # The world binds the session's GT oracles (vcli/worlds/sim_oracles).
        assert "holding_object" in ns and "object_on" in ns
        assert "detect_objects" not in ns and "placed_count" not in ns  # vacuous, gone
        gv = GoalVerifier(ns)
        assert gv.verify("arm_at_home()") is True
        assert gv.verify("object_on('mug', 'table') and object_on('banana', 'table')") is True
        assert gv.verify("holding_object()") is False  # no gripper holding

    def test_engine_resolves_playground_when_wired(self) -> None:
        """Wiring the playground makes the engine namespace carry its GT oracles."""
        from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier

        arm = FakeArm(
            objects={"mug": [0.22, 0.05, 0.06]}, joints=_HOME, ee=[0.2, 0.0, 0.2]
        )
        eng = _make_engine()
        eng._world = PlaygroundWorld()
        ns = eng._build_verifier_namespace(_agent(arm))
        gv = GoalVerifier(ns)
        assert gv.verify("object_on('mug', 'table')") is True
        assert gv.verify("object_on('mug', 'dining_table')") is False  # not a tabletop surface
