# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""M1 seam evolution — the Scenario DTO and world registry are sim-backend-aware.

The third-world campaign (ADR-009) adds a non-MuJoCo backend. The seam must
let such a world register and resolve WITHOUT the kernel importing any
simulator: ``Scenario`` carries ``sim_backend`` / ``scene_ref`` additively
(frozen-dataclass rule: new fields last, with defaults), and ``WorldRegistry``
resolves a world for a non-MJCF scenario exactly like any other.
"""
from __future__ import annotations

import dataclasses

import pytest

from vector_os_nano.playground.catalog import SCENARIOS
from vector_os_nano.playground.scenario import Scenario
from vector_os_nano.playground.world import PlaygroundWorld
from vector_os_nano.vcli.worlds import WorldRegistry


# ---------------------------------------------------------------------------
# Scenario DTO — additive backend fields
# ---------------------------------------------------------------------------


class TestScenarioBackendFields:
    def test_preexisting_mjcf_presets_untouched(self) -> None:
        """Additive evolution: the original MJCF presets are untouched by the
        backend fields — sim_backend defaults to 'mujoco', scene_ref to ''.
        (The M2 'apartment' preset is the first NON-default user and is
        asserted separately in test_habitat_world.py.)"""
        for scenario_id in ("tabletop", "tabletop_tray", "go2_room"):
            scenario = SCENARIOS[scenario_id]
            assert scenario.sim_backend == "mujoco", scenario_id
            assert scenario.scene_ref == "", scenario_id

    def test_habitat_backend_scenario_constructs(self) -> None:
        s = Scenario(
            id="hm3d_house",
            embodiment="go2",
            scene_xml="",  # not an MJCF scene — scene_ref carries the reference
            task_hint="navigate a photoreal scanned house",
            sim_backend="habitat",
            scene_ref="hm3d/minival/00800-TEEsavR23oF",
            rooms={"kitchen": (0.0, 0.0, 3.0, 3.0)},
        )
        assert s.sim_backend == "habitat"
        assert s.scene_ref == "hm3d/minival/00800-TEEsavR23oF"

    def test_scenario_stays_frozen(self) -> None:
        s = Scenario(
            id="x", embodiment="arm", scene_xml="a.xml", task_hint="t",
            sim_backend="habitat",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.sim_backend = "mujoco"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Non-MJCF world registration through the kernel seam
# ---------------------------------------------------------------------------


def _habitat_scenario() -> Scenario:
    return Scenario(
        id="hm3d_house",
        embodiment="go2",
        scene_xml="",
        task_hint="photoreal nav world",
        sim_backend="habitat",
        scene_ref="hm3d/minival/00800-TEEsavR23oF",
        rooms={"kitchen": (0.0, 0.0, 3.0, 3.0)},
    )


class TestNonMjcfWorldRegistration:
    def test_registers_and_resolves_by_name(self) -> None:
        registry = WorldRegistry()  # isolated, not the process-wide one
        registry.register("hm3d_house", lambda: PlaygroundWorld(_habitat_scenario()))
        world = registry.resolve("hm3d_house")
        assert world.name == "hm3d_house"
        assert world.scenario.sim_backend == "habitat"
        assert world.scenario.scene_ref.startswith("hm3d/")

    def test_resolved_world_satisfies_engine_surface(self) -> None:
        """The engine touches exactly the four-registration surface — a
        non-MJCF world must satisfy it without any sim/agent present."""
        registry = WorldRegistry()
        registry.register("hm3d_house", lambda: PlaygroundWorld(_habitat_scenario()))
        world = registry.resolve("hm3d_house")
        role, tools = world.persona_blocks()
        assert role and tools
        assert world.is_robot() is True
        assert world.has_base() is True  # go2 embodiment => base predicates
        assert world.derive_vocab_from_registry() is True
        ns = world.build_verify_namespace(agent=None)
        assert callable(ns["at_position"]) and callable(ns["visited"])

    def test_unknown_name_fails_loud_with_valid_set(self) -> None:
        registry = WorldRegistry()
        registry.register("hm3d_house", lambda: PlaygroundWorld(_habitat_scenario()))
        with pytest.raises(KeyError, match="hm3d_house"):
            registry.resolve("does_not_exist")
