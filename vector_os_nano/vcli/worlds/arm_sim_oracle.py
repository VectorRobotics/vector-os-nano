# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Single-source deterministic sim-oracle grounding (ADR-008 C1).

This module is the ONE place the arm/scene verify predicates live. It is consumed
by BOTH the PlaygroundWorld (preset tabletop scenarios) and the plain RobotWorld
(the normal "open the sim -> grasp the banana" CLI path), so the grounding logic
is never duplicated (kernel rule 3: no split-brain). It lives in the kernel side
of the seam (``vcli.worlds``) because the kernel/RobotWorld must NOT import the
playground package — the dependency edge is one-way (playground -> kernel only,
ADR-008 / kernel rule 2). The playground's ``verify/arm_predicates`` and
``verify/scene_predicates`` modules are now thin re-export shims over this file.

These are the callables a sub-goal's ``verify`` expression evaluates against. They
read the sim's DETERMINISTIC ground truth — ``get_object_positions`` /
``get_joint_positions`` / ``fk`` on the connected arm — never the VLM. The VLM
detect/describe pipeline stays the agent's perception *skill*; the verifier checks
the oracle (ADR-008: generator and verifier independent).

Grounding contract:
- WORLD-AGNOSTIC: the arm is reached from the agent only via duck-typing —
  ``getattr(agent, "_arm", None)`` and ``getattr(agent, "_gripper", None)`` (the
  same accessors the CLI / sim tool use). The module never imports a concrete
  world or embodiment.
- FAIL-SAFE: when the arm is absent or not connected, every predicate FAILS SAFE
  (returns ``False`` / ``0`` / ``[]`` / ``""``) — it must NEVER raise into the
  GoalVerifier sandbox.
- Each predicate is a thin factory bound to the connected ``agent`` so the engine
  can drop them straight into the verify namespace.

The predicates are intentionally side-effect-free: ``fk`` and joint reads do not
advance the sim, and object reads are pure ground-truth lookups.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable

from vector_os_nano.vcli.cognitive.oracle_registry import oracle

logger = logging.getLogger(__name__)

# Home joint configuration for the SO-101 arm (matches skills/home.py default).
# A separate source would risk drift, but home.py owns the *motion* default while
# this owns the *verify* tolerance; both must agree, so the value is asserted in
# tests against the skill default.
_HOME_JOINTS: tuple[float, ...] = (-0.014, -1.238, 0.562, 0.858, 0.311)

# Tolerances (radians / metres). Deliberately generous: verify is a coarse gate
# on "did the step reach roughly the intended state", not a precision check.
_HOME_TOL_RAD: float = 0.10
_LIFT_MIN_Z: float = 0.10  # an object is "lifted" above this table-clearance z
_NEAR_EE_RADIUS: float = 0.08  # an object within this of the EE is "at the gripper"


def _get_arm(agent: Any) -> Any | None:
    """Return the connected arm reachable from *agent*, or None (fail-safe).

    Mirrors the kernel accessor ``getattr(agent, "_arm", None)``. Returns None
    when no agent, no arm, or the arm reports itself disconnected — so callers
    can fail safe without raising.
    """
    if agent is None:
        return None
    arm = getattr(agent, "_arm", None)
    if arm is None:
        return None
    # Respect an explicit connected flag when present; absence of the attr means
    # the arm has no such notion, so treat it as usable.
    if getattr(arm, "_connected", True) is False:
        return None
    return arm


def _gripper_is_holding(agent: Any) -> bool:
    """True if the agent's gripper reports holding an object (fail-safe)."""
    gripper = getattr(agent, "_gripper", None)
    if gripper is None:
        return False
    is_holding = getattr(gripper, "is_holding", None)
    if not callable(is_holding):
        return False
    try:
        return bool(is_holding())
    except Exception as exc:  # noqa: BLE001
        logger.debug("sim-oracle gripper.is_holding failed: %s", exc)
        return False


def _ee_position(arm: Any) -> list[float] | None:
    """Return the current end-effector xyz via FK, or None (fail-safe)."""
    try:
        joints = arm.get_joint_positions()
        ee_pos, _rot = arm.fk(joints)
        return [float(c) for c in ee_pos]
    except Exception as exc:  # noqa: BLE001
        logger.debug("sim-oracle fk/get_joint_positions failed: %s", exc)
        return None


def make_holding_object(agent: Any) -> Callable[..., bool]:
    """Build ``holding_object(target=None)`` bound to *agent*.

    True when the gripper reports holding AND a qualifying scene object is both
    lifted above the table-clearance height and within grasp radius of the EE.

    ``target`` (optional) makes the check TARGET-AWARE: when given a scene object
    name (or id), only that object counts — so a verify can assert "holding the
    REQUESTED object", not merely "holding SOMETHING" (R2-7: an unbound pick that
    grabbed the nearest, wrong object must NOT verify True against the named
    target). The match is structural and language-neutral (case-insensitive
    exact match on the scene name the oracle owns); the caller passes the
    resolved scene name, never a raw NL query. ``target=None`` preserves the
    original "holding anything" semantics (all existing callers unchanged).

    Reads only deterministic ground truth; fails safe to ``False``.
    """

    @oracle(scope="robot", channel="gripper", transition="rise")
    def holding_object(target: Any = None) -> bool:
        arm = _get_arm(agent)
        if arm is None:
            return False
        if not _gripper_is_holding(agent):
            return False
        ee = _ee_position(arm)
        if ee is None:
            return False
        try:
            objects = arm.get_object_positions()
        except Exception as exc:  # noqa: BLE001
            logger.debug("sim-oracle get_object_positions failed: %s", exc)
            return False
        want = None if target is None else str(target).strip().lower()
        for name, pos in objects.items():
            try:
                x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            except (TypeError, ValueError, IndexError):
                continue
            if z < _LIFT_MIN_Z:
                continue
            if want is not None and str(name).strip().lower() != want:
                continue
            dist = math.dist((x, y, z), (ee[0], ee[1], ee[2]))
            if dist <= _NEAR_EE_RADIUS:
                return True
        return False

    return holding_object


def make_arm_at_home(agent: Any) -> Callable[[], bool]:
    """Build ``arm_at_home()`` bound to *agent*.

    True when every arm joint is within ``_HOME_TOL_RAD`` of the home pose.
    Reads deterministic joint ground truth; fails safe to ``False``.
    """

    @oracle(scope="robot", channel="arm", transition="rise")
    def arm_at_home() -> bool:
        arm = _get_arm(agent)
        if arm is None:
            return False
        try:
            joints = arm.get_joint_positions()
        except Exception as exc:  # noqa: BLE001
            logger.debug("sim-oracle get_joint_positions failed: %s", exc)
            return False
        if len(joints) != len(_HOME_JOINTS):
            return False
        return all(
            abs(float(j) - h) <= _HOME_TOL_RAD
            for j, h in zip(joints, _HOME_JOINTS)
        )

    return arm_at_home


def make_placed_count(agent: Any, default_region: Any = None) -> Callable[..., int]:
    """Build ``placed_count(target_region=...)`` bound to *agent*.

    Counts scene objects resting (below the lift height) whose xy lies inside an
    axis-aligned ``target_region`` ``(x_min, y_min, x_max, y_max)``. When the call
    passes no ``target_region``, the scenario's ``default_region`` (a scene-defined
    drop-zone, when present) is used instead; with neither, counts all resting
    objects. An explicit ``target_region`` always overrides the default. Reads
    deterministic ground truth; fails safe to ``0``.
    """

    scene_region = _parse_region(default_region)

    @oracle(scope="robot", channel="gripper")
    def placed_count(target_region: Any = None) -> int:
        arm = _get_arm(agent)
        if arm is None:
            return 0
        try:
            objects = arm.get_object_positions()
        except Exception as exc:  # noqa: BLE001
            logger.debug("sim-oracle get_object_positions failed: %s", exc)
            return 0
        if target_region is None:
            region = scene_region  # not passed -> scenario default (None = count all)
        else:
            region = _parse_region(target_region)
            if region is None:
                # Explicit but malformed: the caller asked for a specific region we
                # cannot honor. Do NOT silently fall back to the scene default — that
                # would verify against a different region than asked. Fail safe to 0
                # (never raise into the verifier).
                logger.debug(
                    "placed_count: malformed explicit target_region %r; returning 0",
                    target_region,
                )
                return 0
        count = 0
        for pos in objects.values():
            try:
                x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            except (TypeError, ValueError, IndexError):
                continue
            if z >= _LIFT_MIN_Z:
                continue  # still lifted / in flight, not yet placed
            if region is not None:
                x_min, y_min, x_max, y_max = region
                if not (x_min <= x <= x_max and y_min <= y <= y_max):
                    continue
            count += 1
        return count

    return placed_count


def _parse_region(region: Any) -> tuple[float, float, float, float] | None:
    """Coerce a 4-tuple/list ``(x_min, y_min, x_max, y_max)`` or return None."""
    if region is None:
        return None
    try:
        x_min, y_min, x_max, y_max = (float(v) for v in region)
    except (TypeError, ValueError):
        return None
    return x_min, y_min, x_max, y_max


# ---------------------------------------------------------------------------
# Scene predicates: detect_objects / describe_scene (perception-stub replacements)
# ---------------------------------------------------------------------------
#
# These replace the engine's empty perception STUBS (``detect_objects -> []``,
# ``describe_scene -> ""``) when a world contributes them. They source ground
# truth from the connected arm's ``get_object_positions`` — restricted to a
# scenario's known ``object_names`` (an EMPTY tuple means "ALL objects in the
# scene": the known-set filter is skipped) — and return the SAME shapes the stubs
# use:
#   - ``detect_objects(query="") -> list[dict]``  (each: ``{"name", "x", "y", "z"}``)
#   - ``describe_scene() -> str``
# Same fail-safe / VLM-independence contract as the arm predicates above.


def _scene_objects(agent: Any, object_names: tuple[str, ...]) -> dict[str, list[float]]:
    """Return ground-truth positions for the scenario's known objects (fail-safe).

    Reads ``arm.get_object_positions()`` and filters to ``object_names`` so the
    oracle only reports the scene's declared graspables. An EMPTY ``object_names``
    means "all objects in the scene" (the filter is skipped). Empty dict on any
    failure or when the arm is unavailable.
    """
    arm = _get_arm(agent)
    if arm is None:
        return {}
    try:
        positions = arm.get_object_positions()
    except Exception as exc:  # noqa: BLE001
        logger.debug("sim-oracle get_object_positions failed: %s", exc)
        return {}
    known = set(object_names)
    out: dict[str, list[float]] = {}
    for name, pos in positions.items():
        if known and name not in known:
            continue
        try:
            out[name] = [float(pos[0]), float(pos[1]), float(pos[2])]
        except (TypeError, ValueError, IndexError):
            continue
    return out


def make_detect_objects(
    agent: Any, object_names: tuple[str, ...]
) -> Callable[..., list[dict[str, Any]]]:
    """Build ``detect_objects(query="")`` bound to *agent* + scenario objects.

    Returns a list of ``{"name", "x", "y", "z"}`` dicts (same list-of-dict shape
    the engine stub returns, just non-empty). A non-empty ``query`` filters by
    case-insensitive substring match on the object name. Fails safe to ``[]``.
    """

    @oracle(scope="robot")
    def detect_objects(query: str = "") -> list[dict[str, Any]]:
        objects = _scene_objects(agent, object_names)
        q = (query or "").strip().lower()
        result: list[dict[str, Any]] = []
        for name in sorted(objects):
            if q and q not in name.lower():
                continue
            x, y, z = objects[name]
            result.append({"name": name, "x": x, "y": y, "z": z})
        return result

    return detect_objects


def make_detect_producer(
    agent: Any, object_names: tuple[str, ...]
) -> Callable[..., dict[str, Any]]:
    """Build a detect PRODUCING-STEP callable bound to *agent* + scenario objects.

    Unlike ``detect_objects`` (a verify-namespace PREDICATE that returns a bare
    list), this is an EXECUTOR primitive: it runs the SAME deterministic
    sim-oracle detection but wraps the result as a producing step's structured
    output — ``{"objects": [...], "count": N}``. The executor captures that dict
    to the run Blackboard under the step name, so a downstream ``foreach`` whose
    ``source_step`` points at this step resolves ``source_step.objects`` to the
    REAL detected list (pure path traversal, never eval). This closes the gap
    where a foreach previously needed a fabricated detect primitive: a real
    detect-producing step now carries the objects list.

    Deterministic (sim oracle, no VLM) and fail-safe: an absent/unavailable arm
    yields ``{"objects": [], "count": 0}`` — never raises into the executor. The
    ``query`` argument (default ``""``) filters identically to ``detect_objects``.
    """

    detect = make_detect_objects(agent, object_names)

    def detect_producer(query: str = "", **_: Any) -> dict[str, Any]:
        objects = detect(query)
        return {"objects": objects, "count": len(objects)}

    return detect_producer


def make_describe_scene(
    agent: Any, object_names: tuple[str, ...]
) -> Callable[[], str]:
    """Build ``describe_scene()`` bound to *agent* + scenario objects.

    Returns a deterministic one-line summary of the objects present and their
    positions (same ``str`` shape the engine stub returns, just non-empty).
    Fails safe to ``""`` when no objects are observable.
    """

    @oracle(scope="robot")
    def describe_scene() -> str:
        objects = _scene_objects(agent, object_names)
        if not objects:
            return ""
        parts = [
            f"{name} at ({x:.2f}, {y:.2f}, {z:.2f})"
            for name, (x, y, z) in sorted(objects.items())
        ]
        return "Tabletop scene: " + "; ".join(parts) + "."

    return describe_scene
