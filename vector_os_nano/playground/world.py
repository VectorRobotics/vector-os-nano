# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""PlaygroundWorld — a parallel-track world for tabletop arm + Go2 base scenarios.

The playground is a SEPARATE world track (ADR-008). It integrates with the
kernel ONLY across the versioned public contract: the four registrations (tools,
verify namespace, decompose vocab, persona) plus the verified-loop observation
surface. The dependency edge is strictly ONE-WAY: this package imports the kernel
(``vcli.worlds.base``) + ``hardware``/``skills``; the kernel never imports the
playground except through the world registry's lazy hook.

The playground reuses the robot persona and single-sources its decompose
vocabulary from the skill registry (like ``RobotWorld``). Its distinct, owned
contribution is the verify namespace: deterministic sim-oracle predicates. The
predicate set is chosen by the scenario's embodiment — ARM scenarios (tabletop)
contribute arm/scene predicates over the connected arm + known objects; the GO2
scenario (a mobile-base quadruped, ``has_base``) contributes base predicates
(at_position / facing / visited) over the connected base. This proves the seam
generalizes across embodiments with no embodiment-specific code in the kernel.
"""

from __future__ import annotations

from typing import Any

from vector_os_nano.playground.catalog import TABLETOP
from vector_os_nano.playground.scenario import Scenario
from vector_os_nano.playground.verify.arm_predicates import (
    make_arm_at_home,
    make_holding_object,
    make_placed_count,
)
from vector_os_nano.playground.verify.base_predicates import (
    make_at_position,
    make_facing,
    make_rooms_producer,
    make_visited,
)
from vector_os_nano.playground.verify.scene_predicates import (
    make_detect_objects,
    make_detect_producer,
    make_describe_scene,
)
from vector_os_nano.vcli.prompt import ROBOT_ROLE_PROMPT, ROBOT_TOOL_INSTRUCTIONS


class PlaygroundWorld:
    """Tabletop arm world for a playground preset scenario.

    Defaults to the bundled ``tabletop`` scenario. A different scenario can be
    injected at construction (the named registry resolves preset scenes later).
    """

    def __init__(self, scenario: Scenario | None = None) -> None:
        self._scenario: Scenario = scenario if scenario is not None else TABLETOP
        # The registry name doubles as the scenario id so resolution is 1:1.
        self.name: str = self._scenario.id

    @property
    def scenario(self) -> Scenario:
        return self._scenario

    @property
    def embodiment(self) -> str:
        """The hardware family this scenario targets (e.g. ``"arm"``/``"go2"``)."""
        return self._scenario.embodiment

    # Embodiments that carry a mobile base. "go2" is the MuJoCo quadruped;
    # "mobile" is a backend-agnostic kinematic base (habitat third world, M2).
    _BASE_EMBODIMENTS = ("go2", "mobile")

    def has_base(self) -> bool:
        """True for a mobile-base scenario, False for an arm scenario.

        Reported from the scenario's embodiment so callers/tests can inspect the
        world's intent. NOTE: the engine still gates the base primitives in the
        decompose vocab from the *connected agent* (``agent._base``), not from
        the world — keeping the mechanism world-agnostic. A mobile scenario
        backed by an agent that has a ``_base`` therefore puts walk_forward/
        turn/scan_360 + the base skills in vocab and enables base verify
        predicates.
        """
        return self._scenario.embodiment in self._BASE_EMBODIMENTS

    def is_robot(self) -> bool:
        # Both arm and base scenarios drive (simulated) robot hardware.
        return True

    def persona_blocks(self) -> tuple[str, str]:
        # ADR-006 thing #4: the world owns its persona. Habitat scenarios get
        # the habitat persona (the world is ALREADY running — the MuJoCo
        # launch guidance would send the LLM hunting for launch scripts);
        # MuJoCo scenarios keep the robot persona unchanged.
        if getattr(self._scenario, "sim_backend", "mujoco") == "habitat":
            from vector_os_nano.vcli.prompt import (
                HABITAT_ROLE_PROMPT,
                HABITAT_TOOL_INSTRUCTIONS,
            )

            return HABITAT_ROLE_PROMPT, HABITAT_TOOL_INSTRUCTIONS
        return ROBOT_ROLE_PROMPT, ROBOT_TOOL_INSTRUCTIONS

    def register_tools(self, registry: Any, agent: Any) -> None:
        # Arm/skill tools are registered by the CLI from discover_* + wrap_skills,
        # exactly as the robot path. No playground-specific tools yet.
        return None

    def build_verify_namespace(self, agent: Any) -> dict[str, Any]:
        """Return the playground's deterministic sim-oracle verify predicates.

        The engine (INC2) merges these on top of its dev/robot bindings and the
        empty perception stubs additively. The predicate set is selected by the
        scenario's embodiment:

        - ARM scenarios (tabletop) contribute arm/scene predicates over the
          connected arm + the scenario's known objects; ``detect_objects`` /
          ``describe_scene`` REPLACE the engine's empty perception stubs.
        - the GO2 scenario contributes base predicates (at_position / facing /
          visited) over the connected base + the scenario's named rooms.

        All predicates are bound to *agent* and fail safe when the hardware is
        unavailable (never raise into the GoalVerifier sandbox).
        """
        if self.has_base():
            return self._base_verify_namespace(agent)
        return self._arm_verify_namespace(agent)

    def _arm_verify_namespace(self, agent: Any) -> dict[str, Any]:
        objects = self._scenario.object_names
        return {
            "detect_objects": make_detect_objects(agent, objects),
            "describe_scene": make_describe_scene(agent, objects),
            "holding_object": make_holding_object(agent),
            "arm_at_home": make_arm_at_home(agent),
            # The scenario's drop-zone (when defined) becomes the DEFAULT region
            # for placed_count(), so a sub-goal can verify against the scene-named
            # region without hand-passing raw coordinates.
            "placed_count": make_placed_count(
                agent, default_region=self._scenario.place_region
            ),
        }

    def _base_verify_namespace(self, agent: Any) -> dict[str, Any]:
        # The scenario's named rooms become the source of truth for visited(),
        # so a navigation sub-goal verifies "reached <room>" by scene name
        # without hand-passing raw coordinates.
        from vector_os_nano.playground.verify.base_predicates import (
            make_geodesic_dist,
        )

        rooms = self._scenario.rooms
        return {
            "at_position": make_at_position(agent),
            "facing": make_facing(agent),
            "visited": make_visited(agent, rooms),
            # M2: navmesh geodesic distance (the VLN success criterion) — binds
            # for every base embodiment, fails safe to inf when the connected
            # base has no geodesic oracle (e.g. the MuJoCo go2).
            "geodesic_dist": make_geodesic_dist(agent),
        }

    # The strategy name a decompose plan emits for the detect PRODUCING step. It
    # is distinct from the ``detect_objects`` verify PREDICATE: the predicate is
    # a verify-namespace callable returning a bare list, while this is an executor
    # primitive whose result_data ({"objects": [...]}) is captured to the
    # Blackboard so a downstream foreach source_step resolves the real list.
    DETECT_STRATEGY: str = "detect_objects_skill"

    # The strategy name a Go2 decompose plan emits for the rooms PRODUCING step —
    # the base counterpart of DETECT_STRATEGY. It "locates" the scenario's named
    # rooms and writes a ``{"rooms": [...]}`` list whose result_data the executor
    # captures to the Blackboard, so a "visit each room one by one" foreach whose
    # source_step points at it resolves the REAL room list (pure path traversal).
    # Distinct from the ``visited``/``at_position`` verify PREDICATES: those are
    # verify-namespace callables; this is an executor primitive producing a list.
    ROOMS_STRATEGY: str = "locate_rooms_skill"

    def build_step_primitives(self, agent: Any) -> dict[str, Any]:
        """Return executor PRIMITIVES this world provides as producing steps.

        Distinct from :meth:`build_verify_namespace` (verify predicates): these
        are run by the GoalExecutor as strategy steps, and their structured
        output is captured to the Blackboard so later steps (e.g. a ``foreach``)
        can consume it.

        - ARM scenarios provide a single detect producer that performs the
          deterministic sim-oracle detection and writes an ``{"objects": [...]}``
          list — the real perception output a grab-everything foreach iterates.
        - the GO2 scenario provides a rooms producer that "locates" the scenario's
          named rooms and writes a ``{"rooms": [...]}`` list — the producing step
          a "visit each room one by one" foreach iterates. The room set is the
          SAME scenario-owned source of truth ``visited`` reads, so the producer
          and the verifier never diverge.

        Keyed by the strategy name a plan emits (DETECT_STRATEGY / ROOMS_STRATEGY).
        """
        if self.has_base():
            return {self.ROOMS_STRATEGY: make_rooms_producer(self._scenario.rooms)}
        objects = self._scenario.object_names
        return {self.DETECT_STRATEGY: make_detect_producer(agent, objects)}

    def register_capabilities(self, registry: Any, agent: Any, backend: Any) -> None:
        # No-op for now: the playground keeps the kernel's skill/primitive routing.
        return None

    def decompose_vocab(self) -> None:
        # None => the engine derives the vocab from the live skill registry
        # (see derive_vocab_from_registry), single-sourcing the action space.
        return None

    def derive_vocab_from_registry(self) -> bool:
        # Single-source the decompose vocabulary from the skill registry, like
        # RobotWorld — no hand-authored, split-brain vocab.
        return True
