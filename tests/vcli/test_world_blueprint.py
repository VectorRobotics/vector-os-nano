# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""W3.1 — frozen WorldBlueprint value object + parity with the imperative seam.

A blueprint makes world+scenario configuration declarative, diffable, and
replayable the way GoalTree already is — WITHOUT replacing the imperative
registration: ``blueprint_of(world)`` derives a blueprint from any existing
World (single-source, no hand-authoring), and ``BlueprintWorld`` adapts a
blueprint back into a full World the engine consumes UNCHANGED — so parity
is structural, not a parallel code path.
"""
from __future__ import annotations

import dataclasses

import pytest

from vector_os_nano.playground.scenario import Scenario
from vector_os_nano.playground.world import PlaygroundWorld
from vector_os_nano.vcli.worlds.blueprint import (
    BlueprintWorld,
    WorldBlueprint,
    blueprint_of,
)
from vector_os_nano.vcli.worlds.dev import DevWorld
from vector_os_nano.vcli.worlds.robot import RobotWorld


def _go2_scenario() -> Scenario:
    return Scenario(
        id="bp_go2", embodiment="go2", scene_xml="", task_hint="t",
        rooms={"kitchen": (0.0, 0.0, 2.0, 2.0)},
    )


class TestBlueprintValueObject:
    def test_frozen(self) -> None:
        bp = blueprint_of(DevWorld())
        with pytest.raises(dataclasses.FrozenInstanceError):
            bp.name = "x"  # type: ignore[misc]

    def test_with_capability_returns_new_instance(self) -> None:
        bp = blueprint_of(DevWorld())
        factory = lambda registry, agent, backend: None  # noqa: E731
        bp2 = bp.with_capability(factory)
        assert bp2 is not bp
        assert len(bp2.capability_factories) == len(bp.capability_factories) + 1
        assert factory in bp2.capability_factories
        assert factory not in bp.capability_factories  # original untouched

    def test_disabled_returns_inert_variant(self) -> None:
        w = PlaygroundWorld(_go2_scenario())
        bp = blueprint_of(w).disabled()
        adapted = BlueprintWorld(bp)
        assert adapted.build_verify_namespace(agent=None) == {}
        assert adapted.register_tools(registry=None, agent=None) is None


class TestParityWithImperativeSeam:
    @pytest.mark.parametrize(
        "world",
        [DevWorld(), RobotWorld(), PlaygroundWorld(_go2_scenario())],
        ids=["dev", "robot", "playground-go2"],
    )
    def test_blueprint_world_matches_source_world(self, world) -> None:
        adapted = BlueprintWorld(blueprint_of(world))

        assert adapted.name == world.name
        assert adapted.is_robot() == world.is_robot()
        assert adapted.persona_blocks() == world.persona_blocks()
        assert adapted.decompose_vocab() == world.decompose_vocab()
        assert (
            adapted.derive_vocab_from_registry()
            == world.derive_vocab_from_registry()
        )
        # has_base: optional in the protocol — parity includes its absence.
        assert getattr(adapted, "has_base", lambda: False)() == getattr(
            world, "has_base", lambda: False
        )()
        # Verify namespace: same predicate NAMES from the same factory
        # (values are per-call closures — identity is not the contract).
        assert set(adapted.build_verify_namespace(agent=None)) == set(
            world.build_verify_namespace(agent=None)
        )

    def test_playground_step_primitives_parity(self) -> None:
        w = PlaygroundWorld(_go2_scenario())
        adapted = BlueprintWorld(blueprint_of(w))
        assert set(adapted.build_step_primitives(agent=None)) == set(
            w.build_step_primitives(agent=None)
        )

    def test_adapted_world_satisfies_protocol_surface(self) -> None:
        """The engine consumes a World; the adapter must satisfy the seam."""
        from vector_os_nano.vcli.worlds.base import World

        adapted = BlueprintWorld(blueprint_of(DevWorld()))
        assert isinstance(adapted, World)  # runtime_checkable structural check

    def test_adapted_world_registers_in_registry(self) -> None:
        from vector_os_nano.vcli.worlds import WorldRegistry

        registry = WorldRegistry()
        bp = blueprint_of(PlaygroundWorld(_go2_scenario()))
        registry.register("bp_go2", lambda: BlueprintWorld(bp))
        resolved = registry.resolve("bp_go2")
        assert resolved.name == "bp_go2"
        assert resolved.has_base() is True
