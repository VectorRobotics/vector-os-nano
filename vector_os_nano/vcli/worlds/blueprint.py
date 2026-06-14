# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""WorldBlueprint — a frozen, declarative value object for the world seam (W3.1).

Makes world+scenario configuration declarative, diffable, and replayable the
way ``GoalTree`` already is, WITHOUT replacing the imperative registration:

- :func:`blueprint_of` derives a blueprint from ANY existing :class:`World`
  by capturing its seam methods as factories — single-source, never
  hand-authored twice.
- :class:`WorldBlueprint` is a frozen dataclass with fluent builders
  (``with_capability`` / ``disabled``) that return NEW instances
  (replace-immutability, rule 6: additive-only evolution).
- :class:`BlueprintWorld` adapts a blueprint back into a full ``World`` the
  engine consumes UNCHANGED — "build-from-blueprint" without a parallel
  engine code path, so parity with the imperative seam is structural.

The imperative path stays authoritative; ``.blueprint()`` on a world is
optional sugar (``blueprint_of`` works on every world).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from vector_os_nano.vcli.worlds.base import DecomposeVocab


def _noop_tools(registry: Any, agent: Any) -> None:
    return None


def _empty_namespace(agent: Any) -> dict[str, Any]:
    return {}


def _no_primitives(agent: Any) -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class WorldBlueprint:
    """Declarative snapshot of everything a world contributes at the seam.

    Factories (not values) are stored for the per-agent pieces so a blueprint
    stays reusable across agents/sessions, like the imperative methods it
    mirrors. Frozen — evolve with ``dataclasses.replace`` or the fluent
    builders, never in place.
    """

    name: str
    is_robot: bool
    persona: tuple[str, str]
    tools_factory: Callable[[Any, Any], None] = _noop_tools
    verify_namespace_factory: Callable[[Any], dict[str, Any]] = _empty_namespace
    step_primitives_factory: Callable[[Any], dict[str, Any]] = _no_primitives
    capability_factories: tuple[Callable[[Any, Any, Any], None], ...] = ()
    decompose_vocab: "DecomposeVocab | None" = None
    derive_vocab_from_registry: bool = False
    has_base: bool = False

    # ------------------------------------------------------------------
    # Fluent builders (replace-immutability)
    # ------------------------------------------------------------------

    def with_capability(
        self, factory: Callable[[Any, Any, Any], None]
    ) -> "WorldBlueprint":
        """Return a NEW blueprint with *factory* appended to the capability set."""
        return replace(
            self, capability_factories=(*self.capability_factories, factory)
        )

    def disabled(self) -> "WorldBlueprint":
        """Return an inert variant: persona kept, every contribution emptied.

        Useful for declarative stack variants (e.g. a dry-run world that
        plans but registers no tools/predicates).
        """
        return replace(
            self,
            tools_factory=_noop_tools,
            verify_namespace_factory=_empty_namespace,
            step_primitives_factory=_no_primitives,
            capability_factories=(),
        )


def blueprint_of(world: Any) -> WorldBlueprint:
    """Derive a blueprint from any World — the single-source constructor.

    Captures the world's seam methods as factories (closures over *world*),
    so the blueprint can never drift from the world that emitted it.
    """
    has_base_fn = getattr(world, "has_base", None)
    step_prims_fn = getattr(world, "build_step_primitives", None)
    return WorldBlueprint(
        name=str(world.name),
        is_robot=bool(world.is_robot()),
        persona=tuple(world.persona_blocks()),  # type: ignore[arg-type]
        tools_factory=world.register_tools,
        verify_namespace_factory=world.build_verify_namespace,
        step_primitives_factory=(
            step_prims_fn if callable(step_prims_fn) else _no_primitives
        ),
        capability_factories=(world.register_capabilities,)
        if callable(getattr(world, "register_capabilities", None))
        else (),
        decompose_vocab=world.decompose_vocab(),
        derive_vocab_from_registry=bool(world.derive_vocab_from_registry()),
        has_base=bool(has_base_fn()) if callable(has_base_fn) else False,
    )


class BlueprintWorld:
    """A full ``World`` built from a :class:`WorldBlueprint`.

    The engine consumes this exactly like any imperative world — this adapter
    IS build-from-blueprint, with no parallel engine code path to drift.
    """

    def __init__(self, blueprint: WorldBlueprint) -> None:
        self._bp = blueprint
        self.name: str = blueprint.name

    @property
    def blueprint(self) -> WorldBlueprint:
        return self._bp

    def is_robot(self) -> bool:
        return self._bp.is_robot

    def persona_blocks(self) -> tuple[str, str]:
        return self._bp.persona

    def register_tools(self, registry: Any, agent: Any) -> None:
        return self._bp.tools_factory(registry, agent)

    def build_verify_namespace(self, agent: Any) -> dict[str, Any]:
        return self._bp.verify_namespace_factory(agent)

    def build_step_primitives(self, agent: Any) -> dict[str, Any]:
        return self._bp.step_primitives_factory(agent)

    def register_capabilities(self, registry: Any, agent: Any, backend: Any) -> None:
        for factory in self._bp.capability_factories:
            factory(registry, agent, backend)

    def decompose_vocab(self) -> "DecomposeVocab | None":
        return self._bp.decompose_vocab

    def derive_vocab_from_registry(self) -> bool:
        return self._bp.derive_vocab_from_registry

    def has_base(self) -> bool:
        return self._bp.has_base
