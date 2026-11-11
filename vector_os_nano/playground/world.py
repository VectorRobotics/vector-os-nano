# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""PlaygroundWorld — a parallel-track world for tabletop arm + Go2 base scenarios.

The playground is a SEPARATE world track (ADR-008). It integrates with the
kernel ONLY across the versioned public contract: the four registrations (tools,
verify namespace, decompose vocab, persona) plus the verified-loop observation
surface. The dependency edge is strictly ONE-WAY: this package imports the kernel
(``vcli.worlds.base``) + ``hardware``/``skills``; the kernel never imports the
playground except through the world registry's lazy hook.

The playground reuses the robot persona, single-sources its decompose vocabulary
from the skill registry and verifies with the SAME ground-truth oracles as the robot
world (``vcli/worlds/sim_oracles``, bound to the live sim session). What it owns is
the scenario: which scene, its known objects / rooms, and the producing-step
primitives (a deterministic object listing for an arm scene, the room list for the
Go2 scene) that a foreach plan iterates.
"""

from __future__ import annotations

from typing import Any

from vector_os_nano.playground.catalog import TABLETOP
from vector_os_nano.playground.scenario import Scenario
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

    def has_base(self) -> bool:
        """True for a mobile-base scenario (Go2), False for an arm scenario.

        Reported from the scenario's embodiment so callers/tests can inspect the
        world's intent. NOTE: the engine still gates the base primitives in the
        decompose vocab from the *connected agent* (``agent._base``), not from
        the world — keeping the mechanism world-agnostic. A go2 scenario backed
        by an agent that has a ``_base`` therefore puts walk_forward/turn/
        scan_360 + the go2 skills in vocab and enables base verify predicates.
        """
        return self._scenario.embodiment == "go2"

    def is_robot(self) -> bool:
        # Both arm and base scenarios drive (simulated) robot hardware.
        return True

    def persona_blocks(self, agent: Any = None) -> tuple[str, str]:
        # Reuse the robot persona for now; a playground-specific persona is a
        # later increment if the tabletop task needs distinct tool instructions.
        return ROBOT_ROLE_PROMPT, ROBOT_TOOL_INSTRUCTIONS

    def register_tools(self, registry: Any, agent: Any) -> None:
        # Arm/skill tools are registered by the CLI from discover_* + wrap_skills,
        # exactly as the robot path. No playground-specific tools yet.
        return None

    def build_verify_namespace(self, agent: Any) -> dict[str, Any]:
        """The same ground-truth oracles as the robot world (``vcli/worlds/sim_oracles``),
        bound to the live sim session that owns *agent*: the scenario's rooms and
        surfaces are the world manifest's (``in_room`` / ``object_on``)."""
        from vector_os_nano.vcli.worlds.robot import RobotWorld

        return RobotWorld().build_verify_namespace(agent)

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
            return {self.ROOMS_STRATEGY: _rooms_producer(self._scenario.rooms)}
        return {self.DETECT_STRATEGY: _detect_producer(agent, self._scenario.object_names)}

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


def _rooms_producer(rooms: dict[str, tuple[float, float, float, float]]):
    """Executor primitive: the scenario's rooms as ``{"rooms": [{name, x, y}], "count"}``
    (box centres, sorted by name) for a "visit each room" foreach."""
    out = [
        {"name": n, "x": (b[0] + b[2]) / 2.0, "y": (b[1] + b[3]) / 2.0}
        for n, b in sorted((rooms or {}).items())
    ]

    def rooms_producer(**_: Any) -> dict[str, Any]:
        return {"rooms": list(out), "count": len(out)}

    return rooms_producer


def _detect_producer(agent: Any, object_names: tuple[str, ...]):
    """Executor primitive: the scenario's objects (the arm's scene listing, filtered to
    *object_names*; empty = all) as ``{"objects": [{name, x, y, z}], "count"}`` for a
    "grab everything" foreach. An actor-side listing — never a verify oracle."""

    def detect_producer(query: str = "", **_: Any) -> dict[str, Any]:
        try:
            positions = getattr(agent, "_arm", None).get_object_positions()
        except Exception:  # noqa: BLE001 — no arm / no sim: nothing detected
            positions = {}
        q = (query or "").strip().lower()
        objects = [
            {"name": n, "x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
            for n, p in sorted(positions.items())
            if (not object_names or n in object_names) and (not q or q in n.lower())
        ]
        return {"objects": objects, "count": len(objects)}

    return detect_producer
