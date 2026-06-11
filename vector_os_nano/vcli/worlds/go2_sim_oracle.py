# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Deterministic sim-oracle verify predicates over a connected mobile base.

The go2 counterpart of ``arm_sim_oracle``: these are the callables a base
sub-goal's ``verify`` expression evaluates against. They read the sim's
DETERMINISTIC ground truth off the connected base — ``get_position`` /
``get_heading`` — never the VLM (ADR-008: generator and verifier independent).

Single-sourced HERE in the kernel so BOTH the playground go2 world AND the plain
``RobotWorld`` can consume them without the kernel importing the playground
(ADR-008 / kernel rule 2: the dependency edge is one-way, playground -> kernel).
``playground/verify/base_predicates.py`` is a thin re-export shim.

Grounding contract:
- The base is reached from the agent via ``getattr(agent, "_base", None)`` (the
  same accessor the engine's SkillContext builder and robot_context use). When
  the base is absent or not connected, every predicate FAILS SAFE (returns
  ``False``) — it must NEVER raise into the GoalVerifier sandbox.
- Each predicate is a thin factory bound to the connected ``agent`` (and, for
  ``visited``, the scenario's named rooms), so a world can drop them straight
  into the verify namespace.

The predicates are side-effect-free: position / heading reads do not advance the
sim. A "room" is an axis-aligned bounding box owned by the scenario; ``visited``
checks the base's current xy against a named room's box.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Tolerances (metres / radians). Deliberately generous: verify is a coarse gate
# on "did the step reach roughly the intended state", not a precision check.
_AT_POSITION_TOL_M: float = 0.5
_FACING_TOL_RAD: float = math.radians(20.0)


def _get_base(agent: Any) -> Any | None:
    """Return the connected base reachable from *agent*, or None (fail-safe).

    Mirrors the kernel accessor ``getattr(agent, "_base", None)``. Returns None
    when no agent, no base, or the base reports itself disconnected — so callers
    can fail safe without raising.
    """
    if agent is None:
        return None
    base = getattr(agent, "_base", None)
    if base is None:
        return None
    # Respect an explicit connected flag when present (MuJoCoGo2 exposes
    # ``_connected``); absence of the attr means the base has no such notion, so
    # treat it as usable.
    if getattr(base, "_connected", True) is False:
        return None
    return base


def _base_position(base: Any) -> list[float] | None:
    """Return the base's current xyz, or None (fail-safe)."""
    try:
        pos = base.get_position()
        return [float(pos[0]), float(pos[1]), float(pos[2])]
    except Exception as exc:  # noqa: BLE001
        logger.debug("sim-oracle base.get_position failed: %s", exc)
        return None


def _base_heading(base: Any) -> float | None:
    """Return the base's current yaw (radians), or None (fail-safe)."""
    try:
        return float(base.get_heading())
    except Exception as exc:  # noqa: BLE001
        logger.debug("sim-oracle base.get_heading failed: %s", exc)
        return None


def _angle_delta(a: float, b: float) -> float:
    """Smallest absolute difference between two angles (radians), in [0, pi]."""
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def make_at_position(agent: Any) -> Callable[..., bool]:
    """Build ``at_position(x, y, tol=...)`` bound to *agent*.

    True when the base's planar (xy) position is within ``tol`` metres of the
    target ``(x, y)``. ``tol`` defaults to ``_AT_POSITION_TOL_M``. Reads
    deterministic ground truth; fails safe to ``False`` (bad args or no base).
    """

    def at_position(x: Any, y: Any, tol: Any = _AT_POSITION_TOL_M) -> bool:
        base = _get_base(agent)
        if base is None:
            return False
        try:
            tx, ty, t = float(x), float(y), float(tol)
        except (TypeError, ValueError):
            return False
        pos = _base_position(base)
        if pos is None:
            return False
        return math.dist((pos[0], pos[1]), (tx, ty)) <= t

    return at_position


def make_facing(agent: Any) -> Callable[..., bool]:
    """Build ``facing(heading, tol=...)`` bound to *agent*.

    True when the base's yaw is within ``tol`` radians of the target ``heading``
    (radians), wrapping correctly across the +/-pi seam. ``tol`` defaults to
    ``_FACING_TOL_RAD``. Reads deterministic ground truth; fails safe to
    ``False`` (bad args or no base).
    """

    def facing(heading: Any, tol: Any = _FACING_TOL_RAD) -> bool:
        base = _get_base(agent)
        if base is None:
            return False
        try:
            target, t = float(heading), float(tol)
        except (TypeError, ValueError):
            return False
        yaw = _base_heading(base)
        if yaw is None:
            return False
        return _angle_delta(yaw, target) <= t

    return facing


def make_visited(agent: Any, rooms: dict[str, tuple[float, float, float, float]]) -> Callable[..., bool]:
    """Build ``visited(room)`` bound to *agent* + the scenario's named rooms.

    True when the base's current planar position lies inside the named room's
    axis-aligned bounding box ``(x_min, y_min, x_max, y_max)``. An unknown room
    name fails safe to ``False`` (it is not silently treated as "anywhere").
    Reads deterministic ground truth; fails safe to ``False`` when the base is
    unavailable. ``rooms`` is the scenario-owned source of truth for box names.
    """

    room_boxes = {
        str(name): tuple(float(v) for v in box)
        for name, box in (rooms or {}).items()
        if _is_box(box)
    }

    def visited(room: Any) -> bool:
        base = _get_base(agent)
        if base is None:
            return False
        box = room_boxes.get(str(room))
        if box is None:
            return False
        pos = _base_position(base)
        if pos is None:
            return False
        x_min, y_min, x_max, y_max = box
        return x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max

    return visited


def _is_box(box: Any) -> bool:
    """True if *box* coerces to a 4-tuple of floats ``(x_min, y_min, x_max, y_max)``."""
    try:
        x_min, y_min, x_max, y_max = (float(v) for v in box)
    except (TypeError, ValueError):
        return False
    return True


def make_rooms_producer(
    rooms: dict[str, tuple[float, float, float, float]],
) -> Callable[..., dict[str, Any]]:
    """Build a rooms PRODUCING-STEP callable over the scenario's named rooms.

    The go2 counterpart of ``arm_sim_oracle.make_detect_producer``: an EXECUTOR
    primitive (NOT a verify predicate) that "locates" the rooms a navigation chain
    should visit and wraps them as a producing step's structured output —
    ``{"rooms": [{"name", "x", "y"}, ...], "count": N}``. The executor captures
    that dict to the run Blackboard under the step name, so a downstream
    ``foreach`` whose ``source_step`` points at this step resolves
    ``source_step.rooms`` to the REAL room list (pure path traversal, never eval).

    Each emitted room carries its NAME (the contract the ``visited(room)`` verify
    predicate reads) plus the centre ``(x, y)`` of its axis-aligned box, so a body
    template can navigate by name (``visited('${room.name}')``) or by coordinate
    (``at_position(${room.x}, ${room.y})``). The room set is the SAME
    scenario-owned source of truth ``visited`` reads, so producer and verifier
    never diverge.

    Deterministic and fail-safe: an empty / malformed room map yields
    ``{"rooms": [], "count": 0}`` — never raises into the executor. Rooms are
    emitted in sorted name order so the produced list is stable across runs.
    """

    room_boxes = {
        str(name): tuple(float(v) for v in box)
        for name, box in (rooms or {}).items()
        if _is_box(box)
    }

    def rooms_producer(**_: Any) -> dict[str, Any]:
        out: list[dict[str, Any]] = []
        for name in sorted(room_boxes):
            x_min, y_min, x_max, y_max = room_boxes[name]
            out.append(
                {
                    "name": name,
                    "x": (x_min + x_max) / 2.0,
                    "y": (y_min + y_max) / 2.0,
                }
            )
        return {"rooms": out, "count": len(out)}

    return rooms_producer


def make_geodesic_dist(agent: Any) -> Callable[..., float]:
    """Build ``geodesic_dist(x, y)`` bound to *agent* (M2, habitat oracle).

    Returns the NAVMESH geodesic distance (metres) from the base's current
    position to the target ``(x, y)`` when the connected base exposes a
    ``geodesic_distance`` oracle (HabitatBase does); a verify expression like
    ``geodesic_dist(3.0, 1.0) < 0.5`` is the standard VLN success criterion.
    Fails safe to ``float('inf')`` — no base, no oracle support, bad args, or
    an unreachable target all read as "infinitely far", so a < threshold
    check is honestly False and nothing raises into the sandbox.
    """

    def geodesic_dist(x: Any, y: Any) -> float:
        base = _get_base(agent)
        if base is None or not callable(getattr(base, "geodesic_distance", None)):
            return float("inf")
        pos = _base_position(base)
        if pos is None:
            return float("inf")
        try:
            tx, ty = float(x), float(y)
            d = float(base.geodesic_distance(list(pos), [tx, ty, pos[2]]))
        except (TypeError, ValueError):
            return float("inf")
        except Exception:  # noqa: BLE001 — oracle hiccup reads as unreachable
            return float("inf")
        if not math.isfinite(d):
            return float("inf")
        return d

    return geodesic_dist
