# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Scenario — an immutable description of a playground preset scene.

A *scenario* is the static data a playground world needs to ground its verify
predicates and (later) drive a visible sim: which embodiment it targets, which
MJCF scene to load, a one-line task hint, and the known graspable object names.
The object names are the contract the deterministic sim-oracle predicates use to
read ground truth (``get_object_positions`` returns only the free bodies present
in that scene).

This is a frozen DTO — it carries no behaviour and imports nothing from the
kernel, so it is safe for both tracks to depend on. Frozen dataclasses change
additively only (new field last, with a default).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Scenario:
    """An immutable playground preset scene.

    Args:
        id: stable scenario id (the world-registry name, e.g. ``"tabletop"``).
        embodiment: the hardware family this scene targets (e.g. ``"arm"``).
        scene_xml: path to the MJCF scene file (relative to the repo root or
            absolute) the sim loads for this scenario.
        task_hint: a short human-readable description of the scene's task.
        object_names: the graspable object body names present in the scene; the
            sim-oracle verify predicates read ground truth for exactly these.
        place_region: an optional named drop-zone as an axis-aligned bounding box
            ``(x_min, y_min, x_max, y_max)``. When set, the playground binds it as
            the DEFAULT region for ``placed_count()`` so a sub-goal can verify
            "placed in the scene's drop-zone" without hand-passing raw coordinates.
            ``None`` (the default) means the scene defines no region and
            ``placed_count()`` counts every resting object.
        rooms: an optional mapping of room name -> axis-aligned bounding box
            ``(x_min, y_min, x_max, y_max)`` for a mobile-base (Go2) scene. The
            base ``visited(room)`` verify predicate checks the base's planar
            position against the named box, so a navigation sub-goal can verify
            "reached the kitchen" by scene name without hand-passing coordinates.
            Empty (the default) means the scene declares no named rooms.
        sim_backend: the simulator backend this scenario targets. ``"mujoco"``
            (the default — every pre-existing preset) loads ``scene_xml`` as
            MJCF; other backends (e.g. ``"habitat"``, ADR-009) interpret
            ``scene_ref`` instead. The kernel never imports a backend from this
            field — it is data the world/boot layer dispatches on.
        scene_ref: backend-generic scene reference for non-MJCF backends
            (e.g. a dataset-relative scene id such as
            ``"hm3d/minival/00800-TEEsavR23oF"``). Empty for MJCF scenarios,
            which keep using ``scene_xml``.
    """

    id: str
    embodiment: str
    scene_xml: str
    task_hint: str
    object_names: tuple[str, ...] = field(default_factory=tuple)
    place_region: tuple[float, float, float, float] | None = None
    rooms: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    sim_backend: str = "mujoco"
    scene_ref: str = ""
