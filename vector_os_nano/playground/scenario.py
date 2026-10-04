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
        object_names: the graspable object body names present in the scene (the
            scenario's object listing; empty = every free body).
        place_region: an optional drop-zone ``(x_min, y_min, x_max, y_max)``; the
            world manifest's matching surface (``tray`` on the tabletop) is what the
            ``object_on`` oracle grades a place against.
        rooms: an optional mapping of room name -> ``(x_min, y_min, x_max, y_max)``
            for a mobile-base scene (the world manifest's room boxes) — the rooms a
            "visit each room" plan iterates; ``in_room`` grades arrival.
    """

    id: str
    embodiment: str
    scene_xml: str
    task_hint: str
    object_names: tuple[str, ...] = field(default_factory=tuple)
    place_region: tuple[float, float, float, float] | None = None
    rooms: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
