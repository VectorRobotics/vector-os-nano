# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""actor_causation — grade whether the ACTOR caused a robot step's state change.

R2b. The R1 evidence gate proves a step's verify predicate is GROUNDED (consumes a
real world oracle, non-tautologically, with ``verify_result=True``). But a GROUNDED
predicate alone does NOT prove the *actor* (the robot under command) caused the
post-state — a NO-OP step whose target was already satisfied at baseline, or a code
sub-goal that teleports ``qpos`` with no motor command, both make ``at_position(...)``
true without the actor walking there. This module adds the missing conjunct: a
GROUNDED robot-predicate step is DOWNGRADED to RAN unless the actor actually
COMMANDED motion via the instrumented control path AND the pose changed.

SCOPED HONESTY CLAIM (verbatim — red-team THIS, not "closes teleport"):

    R2b grades that the actor COMMANDED motion via the instrumented ctrl path AND
    the pose changed; it does NOT prove the change was caused ONLY by that command.
    A skill that BOTH commands motion AND pokes qpos is NOT caught — deferred to
    shadow-MjData re-step (state-level independence, OUT OF SCOPE).

What it CLOSES: the NO-OP (predicate already true at baseline, no commanded motion)
and the pure-TELEPORT (qpos poked, zero commanded motion) false-greens — both now
grade UNCAUSED and the step downgrades GROUNDED -> RAN.
What it DEFERS: a step that simultaneously commands real motion AND pokes qpos
would still grade CAUSED (the command signal fires); proving caused-ONLY-by requires
a shadow MjData re-step (re-simulate from the baseline applying only the recorded
commands and compare) — explicitly OUT OF SCOPE this round.

ALSO OUT OF SCOPE (documented, deferred): the live go2 ``navigate`` path drives via
a ROS2 bridge whose ``cmd_vel`` runs on another thread and is GATED OUT before the
instrumented counter (``mujoco_go2.set_velocity`` returns at the ``_gated`` guard
when the caller is not the skill-token thread). Only ``walk()``/``walk_forward``
(which set ``_skill_ctrl_tid`` so their ``set_velocity`` is ungated + counted) carry
an honest actor-causation signal. Grading the bridge/nav route is deferred.

Single-sourced into ``classify_step_evidence`` (trace_store) so the actor-causation
verdict flows to BOTH gates and the VECTOR_VERDICT report — no split-brain.
"""
from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# Cumulative |cmd| (m/s + rad/s summed over the step) below which we treat the
# actor as having issued NO real motion command — a terminal ``set_velocity(0,0,0)``
# stop must NOT satisfy causation, so we gate on cumulative magnitude, not on the
# raw write COUNT (a stop is a write but carries zero command).
MOTION_EPS: float = 1e-3

# Planar (xy) + yaw displacement below which the pose is treated as UNCHANGED.
# Generous: a few mm of jitter from the gait settle must not read as "moved".
DISPLACEMENT_EPS: float = 0.02

# Heading change (rad) below which a base rotation is treated as gait drift, not an
# intentional turn — so a forward walk's yaw-wobble cannot launder a satisfied-at-
# baseline facing(heading) no-op to CAUSED (review fix STEP-12: cross-channel leak).
# ~10deg: well above forward-walk drift, well below an intentional turn-to-face (the
# facing oracle tol is 20deg and the gait under-rotates, so honest turns command far
# more).
_YAW_CAUSE_EPS: float = math.radians(10.0)


class ActorCaused(enum.Enum):
    """Tri-state actor-causation grade for a StepRecord (R2b).

    Deliberately a tri-state, NOT ``bool | None`` with None=fail-closed: a
    ``NOT_GRADED`` step classifies EXACTLY as it did before R2b (zero regression
    for legacy / hand-built traces and every non-robot-predicate step). Only an
    explicit ``UNCAUSED`` downgrades a GROUNDED robot-predicate step to RAN.

    - CAUSED      — the actor commanded motion via the instrumented ctrl path AND
                    the relevant pose changed (the displacement matched the command).
    - UNCAUSED    — graded, but the actor did NOT cause the state change (no
                    commanded motion, or the pose did not move while the predicate
                    is true — a satisfied-at-baseline NO-OP, or a teleport).
    - NOT_GRADED  — actor-causation was not evaluated for this step (default). The
                    executor never touched it, or it is not a GROUNDED-capable /
                    robot-predicate step. Classifies as before R2b.
    """

    CAUSED = "CAUSED"
    UNCAUSED = "UNCAUSED"
    NOT_GRADED = "NOT_GRADED"


def from_name(name: Any) -> "ActorCaused":
    """Deserialize an ``ActorCaused`` from its ``.value`` string, fail-safe.

    An unknown / missing value (older on-disk traces predating R2b) maps to
    ``NOT_GRADED`` — i.e. legacy-equivalent classification, never a spurious
    downgrade. Never raises into the (de)serializer.
    """
    if isinstance(name, ActorCaused):
        return name
    try:
        return ActorCaused(str(name))
    except (ValueError, TypeError):
        return ActorCaused.NOT_GRADED


# ---------------------------------------------------------------------------
# Baseline snapshot — a FROZEN actor-state capture taken BEFORE a step runs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActorBaseline:
    """A frozen snapshot of the actor's commanded-motion counters + pose.

    Captured BEFORE a step executes (``capture(agent)``); compared with a fresh
    ``capture`` taken AFTER ``_verify_and_value`` to grade causation. Every field
    is a SNAPSHOT VALUE (a plain float / tuple), never a live handle, so advancing
    the live robot after capture cannot mutate the baseline (the staleness trap
    fix 5 guards against).

    ``None`` fields mean "this actor channel was unavailable at capture" (no base /
    arm / gripper, or a read raised) — graded fail-safe.
    """

    # Base (go2) — cumulative commanded |cmd| and pose snapshot.
    base_cmd_motion: float | None = None
    base_pos: tuple[float, float, float] | None = None
    base_heading: float | None = None

    # Arm — cumulative commanded |ctrl delta| and joint snapshot.
    arm_ctrl_motion: float | None = None
    arm_joints: tuple[float, ...] | None = None

    # Gripper — per-weld {welded-body-name -> active(bool)} snapshot for the 0->1
    # transition that signals a fresh grasp.
    gripper_welds: dict[str, bool] | None = None


def _read_float(fn: Any) -> float | None:
    """Call *fn* and coerce to float, fail-safe to None."""
    if not callable(fn):
        return None
    try:
        return float(fn())
    except Exception as exc:  # noqa: BLE001
        logger.debug("actor_causation: float read failed: %s", exc)
        return None


def _read_tuple(fn: Any) -> tuple[float, ...] | None:
    """Call *fn* and coerce to a float tuple, fail-safe to None."""
    if not callable(fn):
        return None
    try:
        return tuple(float(v) for v in fn())
    except Exception as exc:  # noqa: BLE001
        logger.debug("actor_causation: tuple read failed: %s", exc)
        return None


def _capture_base(base: Any) -> tuple[float | None, tuple[float, float, float] | None, float | None]:
    """Snapshot the base's commanded-motion counter + pose (all fail-safe)."""
    if base is None:
        return None, None, None
    if getattr(base, "_connected", True) is False:
        return None, None, None
    cmd_motion = _read_float(getattr(base, "cmd_motion", None))
    pos_t = _read_tuple(getattr(base, "get_position", None))
    pos = (pos_t[0], pos_t[1], pos_t[2]) if pos_t is not None and len(pos_t) >= 3 else None
    heading = _read_float(getattr(base, "get_heading", None))
    return cmd_motion, pos, heading


def _capture_arm(arm: Any) -> tuple[float | None, tuple[float, ...] | None]:
    """Snapshot the arm's commanded-motion counter + joints (all fail-safe)."""
    if arm is None:
        return None, None
    if getattr(arm, "_connected", True) is False:
        return None, None
    ctrl_motion = _read_float(getattr(arm, "ctrl_motion", None))
    joints = _read_tuple(getattr(arm, "get_joint_positions", None))
    return ctrl_motion, joints


def _capture_gripper(gripper: Any) -> dict[str, bool] | None:
    """Snapshot the gripper's per-weld {body -> active} map (fail-safe)."""
    if gripper is None:
        return None
    reader = getattr(gripper, "weld_is_active", None)
    if not callable(reader):
        return None
    try:
        welds = reader()
    except Exception as exc:  # noqa: BLE001
        logger.debug("actor_causation: gripper weld read failed: %s", exc)
        return None
    if not isinstance(welds, dict):
        return None
    return {str(k): bool(v) for k, v in welds.items()}


def capture(agent: Any) -> ActorBaseline:
    """Capture a FROZEN actor-causation baseline from *agent* (duck-typed).

    Reads ``agent._base`` / ``agent._arm`` / ``agent._gripper`` — the SAME kernel
    accessors the verify oracles use — and snapshots each channel's cumulative
    commanded-motion counter plus its pose. Every value is read NOW and stored as a
    plain float/tuple/dict, so the returned baseline is immutable to subsequent
    live-robot motion (no stale-handle trap). A missing agent / channel yields a
    baseline whose fields are None (graded fail-safe). NEVER raises.
    """
    if agent is None:
        return ActorBaseline()
    base = getattr(agent, "_base", None)
    arm = getattr(agent, "_arm", None)
    gripper = getattr(agent, "_gripper", None)
    base_cmd_motion, base_pos, base_heading = _capture_base(base)
    arm_ctrl_motion, arm_joints = _capture_arm(arm)
    gripper_welds = _capture_gripper(gripper)
    return ActorBaseline(
        base_cmd_motion=base_cmd_motion,
        base_pos=base_pos,
        base_heading=base_heading,
        arm_ctrl_motion=arm_ctrl_motion,
        arm_joints=arm_joints,
        gripper_welds=gripper_welds,
    )


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

# Predicate-name -> the actor channel it implicates. A verify expression naming
# any of these is a "robot-predicate" step whose causation we grade; the channel
# decides which counter / displacement to consult.
_BASE_PREDICATES: frozenset[str] = frozenset(
    {"at_position", "facing", "visited", "in_room", "near_object", "in_view"}
)
_ARM_PREDICATES: frozenset[str] = frozenset({"arm_at_home"})
_GRIPPER_PREDICATES: frozenset[str] = frozenset(
    {"holding_object", "is_holding", "holding", "lifted", "kept_holding"}
)
# Placement predicates: the actor caused them only by RELEASING an object this step
# (a weld 1->0), so an object that was already resting there grades UNCAUSED.
_RELEASE_PREDICATES: frozenset[str] = frozenset({"released", "object_at", "object_on"})


def _names_in_expr(verify: str, names: frozenset[str]) -> bool:
    """True iff *verify* references any name in *names* as a bare identifier."""
    import re

    if not verify:
        return False
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", verify))
    return bool(tokens & names)


def _base_channel_caused(
    baseline: ActorBaseline, post: ActorBaseline, verify: str
) -> bool | None:
    """True iff the base moved in the dimension *verify*'s predicate constrains.

    A base predicate's causation must come from the channel it actually measures,
    NOT a blended ``max(planar, yaw)`` (review fix STEP-12: a forward walk's planar
    move must NOT satisfy a ``facing(heading)`` no-op, and a pure turn's yaw move must
    NOT satisfy ``at_position``/``visited``):

      - ``facing``                    -> heading channel: ``dyaw >= _YAW_CAUSE_EPS``
      - ``at_position`` / ``visited`` -> planar channel:  ``dxy  >= DISPLACEMENT_EPS``

    Returns None when the relevant snapshot is unavailable (grade fail-closes to
    UNCAUSED).
    """
    if baseline.base_pos is None or post.base_pos is None:
        return None
    if _names_in_expr(verify, frozenset({"facing"})):
        if baseline.base_heading is None or post.base_heading is None:
            return None
        dyaw = abs(
            math.atan2(
                math.sin(post.base_heading - baseline.base_heading),
                math.cos(post.base_heading - baseline.base_heading),
            )
        )
        return dyaw >= _YAW_CAUSE_EPS
    moved = math.dist(baseline.base_pos[:2], post.base_pos[:2]) >= DISPLACEMENT_EPS
    if _names_in_expr(verify, frozenset({"in_view"})) and not moved:
        # The camera rides the base: turning in place brings an object into view too.
        if baseline.base_heading is None or post.base_heading is None:
            return None
        return abs(math.atan2(
            math.sin(post.base_heading - baseline.base_heading),
            math.cos(post.base_heading - baseline.base_heading),
        )) >= _YAW_CAUSE_EPS
    return moved


def _arm_displacement(baseline: ActorBaseline, post: ActorBaseline) -> float | None:
    """Max per-joint |delta| between baseline and post arm joints, or None."""
    if baseline.arm_joints is None or post.arm_joints is None:
        return None
    if len(baseline.arm_joints) != len(post.arm_joints):
        return None
    if not baseline.arm_joints:
        return 0.0
    return max(abs(p - b) for b, p in zip(baseline.arm_joints, post.arm_joints))


def _gripper_fresh_grasp(baseline: ActorBaseline, post: ActorBaseline) -> bool | None:
    """True iff some weld went 0->1 between baseline and post, or None if ungradable."""
    if baseline.gripper_welds is None or post.gripper_welds is None:
        return None
    for body, active_now in post.gripper_welds.items():
        was = bool(baseline.gripper_welds.get(body, False))
        if active_now and not was:
            return True
    return False


def _gripper_fresh_release(baseline: ActorBaseline, post: ActorBaseline) -> bool | None:
    """True iff some weld went 1->0 between baseline and post, or None if ungradable."""
    if baseline.gripper_welds is None or post.gripper_welds is None:
        return None
    return any(
        was and not post.gripper_welds.get(body, False)
        for body, was in baseline.gripper_welds.items()
    )


def grade(
    baseline: ActorBaseline | None,
    post: ActorBaseline,
    verify: str,
    oracle_names: frozenset[str],
) -> ActorCaused:
    """Grade whether the actor CAUSED the step's GROUNDED state change.

    Called by the executor ONCE per step, after ``_verify_and_value``. The caller
    only consults the result for a step the R1 gate already deems GROUNDED-capable
    (a robot predicate over a real oracle); this function decides CAUSED vs UNCAUSED
    for such a step. Rules (fail-CLOSED — when in doubt, UNCAUSED, never CAUSED):

    - *baseline is None* (capture never ran) -> UNCAUSED (fail closed).
    - The verify names a BASE predicate (at_position / facing / visited):
        CAUSED iff cumulative commanded |cmd| advanced by >= MOTION_EPS AND the base
        pose displaced by >= DISPLACEMENT_EPS. A NO-OP (no commanded motion) or a
        TELEPORT (no commanded motion, pose jumps) both -> UNCAUSED.
    - The verify names an ARM predicate (arm_at_home): CAUSED iff arm ctrl motion
        advanced AND a joint moved.
    - The verify names a GRIPPER predicate (holding_object): CAUSED iff a weld
        transitioned 0->1 this step (a fresh grasp). [holding grading — see fix 6.]
    - Any other / ungradable case -> UNCAUSED (fail closed). The executor only
        downgrades on UNCAUSED, and only consults this for GROUNDED-capable robot
        steps, so a non-robot predicate never reaches here as UNCAUSED in practice.

    Keys on the ACTOR signal (commanded-motion counter + pose displacement), NEVER
    on sim_time advancing (which the go2 daemon ticks regardless — the stale trap).
    """
    if baseline is None:
        return ActorCaused.UNCAUSED

    if _names_in_expr(verify, _BASE_PREDICATES):
        if baseline.base_cmd_motion is None or post.base_cmd_motion is None:
            return ActorCaused.UNCAUSED
        commanded = post.base_cmd_motion - baseline.base_cmd_motion
        moved = _base_channel_caused(baseline, post, verify)
        if moved is None:
            return ActorCaused.UNCAUSED
        if commanded >= MOTION_EPS and moved:
            return ActorCaused.CAUSED
        return ActorCaused.UNCAUSED

    if _names_in_expr(verify, _ARM_PREDICATES):
        if baseline.arm_ctrl_motion is None or post.arm_ctrl_motion is None:
            return ActorCaused.UNCAUSED
        commanded = post.arm_ctrl_motion - baseline.arm_ctrl_motion
        disp = _arm_displacement(baseline, post)
        if disp is None:
            return ActorCaused.UNCAUSED
        if commanded >= MOTION_EPS and disp >= DISPLACEMENT_EPS:
            return ActorCaused.CAUSED
        return ActorCaused.UNCAUSED

    if _names_in_expr(verify, _GRIPPER_PREDICATES):
        fresh = _gripper_fresh_grasp(baseline, post)
        return ActorCaused.CAUSED if fresh else ActorCaused.UNCAUSED

    if _names_in_expr(verify, _RELEASE_PREDICATES):
        fresh = _gripper_fresh_release(baseline, post)
        return ActorCaused.CAUSED if fresh else ActorCaused.UNCAUSED

    return ActorCaused.UNCAUSED


def is_robot_predicate(verify: str, oracle_names: frozenset[str]) -> bool:
    """True iff *verify* names a graded robot predicate present in *oracle_names*.

    The executor uses this to decide whether to GRADE a step's causation at all: a
    step whose verify names no robot predicate (or names one absent from the live
    oracle set) is left ``NOT_GRADED`` (legacy-equivalent), so non-robot worlds
    (dev predicates, answer steps) are byte-unaffected.
    """
    graded = (
        _BASE_PREDICATES | _ARM_PREDICATES | _GRIPPER_PREDICATES | _RELEASE_PREDICATES
    ) & oracle_names
    return _names_in_expr(verify, frozenset(graded))
