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

WP07 (spine S2): every robot-channel oracle call in a verify is graded on its OWN
channel, with the channel and transition read off its ``@oracle`` tag (no first
match): base/arm need commanded motion + displacement, the gripper needs the weld
transition the call's polarity asks for (0->1 for ``holding_object``, 1->0 under a
``not``), and a base target already met at the baseline pose is UNCAUSED (F->T).
Commanded motion is the ACTOR's only while a producer-dispatched tool runs
(``sim.runtime.ACTOR_EPOCH``): the in-process ``navigate`` follower counts, while a
TARE/bridge follower or a background explore driving the base between tools never
does. The producer grades a re-verify that retried (or only re-checked) its attempt
group's latest attempt against that attempt's baseline, any other on its own step.

S1 (charter 3.2): the pose half of a baseline is GROUND TRUTH — the base pose, the arm
joints and the attachments come from the sim session's GroundTruth, never from what
the actor's own devices report; only the commanded-motion counters come from the
drivers. A baseline carries its session's id, and one taken in another session (a
body switched mid-turn) never causes.

Single-sourced into ``classify_step_evidence`` (trace_store) so the actor-causation
verdict flows to BOTH gates and the VECTOR_VERDICT report — no split-brain.
"""
from __future__ import annotations

import ast
import enum
import logging
import math
from collections.abc import Mapping
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



# ---------------------------------------------------------------------------
# Baseline snapshot — a FROZEN actor-state capture taken BEFORE a step runs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActorBaseline:
    """A frozen snapshot of the actor's commanded-motion counters + its true pose.

    Captured BEFORE a step executes (``capture(agent, session)``); compared with a fresh
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

    # Gripper — {attached object -> True} (ground truth) for the 0->1 transition that
    # signals a fresh grasp.
    gripper_welds: dict[str, bool] | None = None

    # Perception — the belief's count of LOCALIZED observations (SceneGraph
    # ``observation_count``): it advances only when the actor's perception put an
    # object at a position this step.
    belief_obs: int | None = None

    # The sim session the pose half was read in (its ``uid``): a baseline from another
    # session never causes.
    session: Any = None
    joint_groups: dict[str, tuple[float | None, tuple[float, ...] | None]] | None = None


def _read_float(fn: Any) -> float | None:
    """Call *fn* and coerce to float, fail-safe to None."""
    if not callable(fn):
        return None
    try:
        return float(fn())
    except Exception as exc:  # noqa: BLE001
        logger.debug("actor_causation: float read failed: %s", exc)
        return None


def _read_truth(what: str, fn: Any) -> Any:
    """A ground-truth read, fail-safe to None (an unreadable channel never causes)."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        logger.debug("actor_causation: ground-truth %s read failed: %s", what, exc)
        return None


def _connected(device: Any) -> bool:
    return device is not None and getattr(device, "_connected", True) is not False


def capture(agent: Any, session: Any = None) -> ActorBaseline:
    """Capture a FROZEN actor-causation baseline of *agent* in its sim *session*.

    The commanded-motion counters come from the drivers (``agent._base.cmd_motion``,
    ``agent._arm.ctrl_motion``); the base pose, the arm joints (by the arm's joint
    names) and the attachments come from ``session.gt`` — never from what the devices
    report. Every value is read NOW and stored as a plain float/tuple/dict, so the
    returned baseline is immutable to subsequent live-robot motion. A missing agent,
    session or channel yields None fields (graded fail-safe). NEVER raises.
    """
    if agent is None:
        return ActorBaseline()
    base = getattr(agent, "_base", None)
    arm = getattr(agent, "_arm", None)
    gripper = getattr(agent, "_gripper", None)
    gt = getattr(session, "gt", None)
    base_pose = None
    if _connected(base) and gt is not None and gt.has_base:
        base_pose = _read_truth("base pose", gt.base_pose)
    arm_joints = None
    if _connected(arm) and gt is not None:
        joints = _read_truth("arm joints", lambda: gt.joint_positions(tuple(arm.joint_names)))
        arm_joints = None if joints is None else tuple(float(v) for v in joints)
    held = _read_truth("attachments", gt.attachments) if gripper is not None and gt is not None else None
    obs = getattr(getattr(agent, "_spatial_memory", None), "observation_count", None)
    groups = {}
    for name, device in getattr(agent, '_services', {}).get('joint_groups', {}).items():
        q = _read_truth('joint group', lambda: tuple(gt.joint_positions(device.joint_names))) if gt is not None else None
        groups[name] = (_read_float(getattr(device, 'ctrl_motion', None)), q)
    return ActorBaseline(
        base_cmd_motion=_read_float(getattr(base, "cmd_motion", None)) if _connected(base) else None,
        base_pos=None if base_pose is None else tuple(float(v) for v in base_pose[:3]),
        base_heading=None if base_pose is None else float(base_pose[3]),
        arm_ctrl_motion=_read_float(getattr(arm, "ctrl_motion", None)) if _connected(arm) else None,
        arm_joints=arm_joints,
        gripper_welds=None if held is None else {str(n): True for n in held},
        belief_obs=obs if isinstance(obs, int) and not isinstance(obs, bool) else None,
        session=getattr(session, "uid", None),
        joint_groups=groups,
    )


# ---------------------------------------------------------------------------
# Grading — per channel, from the oracles' own @oracle tags
# ---------------------------------------------------------------------------

# The actor channels causation is graded on ("none" oracles name no actor state: a
# step verified only by them is never graded here). "perception" is the actor's belief
# judged against GT (``localized``): caused only by a localized observation this step.
_ROBOT_CHANNELS: frozenset[str] = frozenset({"base", "arm", "gripper", "perception", "joints"})
# ``not <pred>`` asks for the opposite transition of ``<pred>``.
_FLIP = {"rise": "fall", "fall": "rise", "level": "level"}


def _tag(oracles: Any, name: str) -> Any:
    """The OracleTag of *name* in *oracles* ({name: OracleTag | tagged callable})."""
    from vector_os_nano.vcli.cognitive.oracle_registry import OracleTag

    v = oracles.get(name) if isinstance(oracles, Mapping) else None
    return v if isinstance(v, OracleTag) else getattr(v, "__oracle__", None)


def robot_calls(verify: str, oracles: Any) -> list[tuple[str, ast.Call, str, str]]:
    """``(name, call, channel, wanted transition)`` for every robot-channel oracle call
    in *verify*. The channel and the transition come from the oracle's tag; under a
    ``not`` the wanted transition flips (``not holding_object()`` asks for a release)."""
    try:
        tree = ast.parse((verify or "").strip(), mode="eval")
    except SyntaxError:
        return []
    out: list[tuple[str, ast.Call, str, str]] = []

    def walk(node: ast.AST, neg: bool) -> None:
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            walk(node.operand, not neg)
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            tag = _tag(oracles, node.func.id)
            if tag is not None and tag.channel in _ROBOT_CHANNELS:
                want = _FLIP[tag.transition] if neg else tag.transition
                out.append((node.func.id, node, tag.channel, want))
        for child in ast.iter_child_nodes(node):
            walk(child, neg)

    walk(tree.body, False)
    return out


def _call_number(call: ast.Call, index: int, key: str) -> float | None:
    from vector_os_nano.vcli.cognitive.coord_goal import _const_number

    if len(call.args) > index:
        return _const_number(call.args[index])
    kw = next((k.value for k in call.keywords if k.arg == key), None)
    return None if kw is None else _const_number(kw)


def _facing_tol() -> float:
    """The facing oracle's own default tolerance (re-read, never re-authored; no
    fallback, so a moved oracle fails loud instead of grading against a stale copy)."""
    from vector_os_nano.vcli.worlds.sim_oracles import _FACING_TOL_RAD

    return float(_FACING_TOL_RAD)


def _base_truth_at(name: str, call: ast.Call, b: ActorBaseline) -> bool | None:
    """Was ``at_position`` / ``facing`` with these literal args already true at the
    baseline pose? None when unknowable (another predicate, non-literal args)."""
    from vector_os_nano.vcli.cognitive import coord_goal as cg

    if name == "at_position" and b.base_pos is not None:
        x, y, tol = _call_number(call, 0, "x"), _call_number(call, 1, "y"), _call_number(call, 2, "tol")
        if x is None or y is None:
            return None
        tol = min(cg._at_position_tol() if tol is None else tol, cg._MAX_ARRIVAL_TOL_M)
        return math.dist(b.base_pos[:2], (x, y)) <= tol
    if name == "facing" and b.base_heading is not None:
        h, tol = _call_number(call, 0, "heading"), _call_number(call, 1, "tol")
        if h is None:
            return None
        tol = min(_facing_tol() if tol is None else tol, cg._MAX_FACING_TOL_RAD)
        return abs(math.atan2(math.sin(b.base_heading - h), math.cos(b.base_heading - h))) <= tol
    return None


def _moved(b: ActorBaseline, p: ActorBaseline, name: str) -> bool:
    """The base moved in the dimension the predicate measures: yaw for ``facing``, the
    planar position otherwise (a walk's wobble never causes a facing no-op, and a turn
    in place never causes an at_position). ``in_view`` takes either: the camera rides
    the base, so turning in place brings an object into view too."""
    if b.base_pos is None or p.base_pos is None:
        return False
    turned = False
    if b.base_heading is not None and p.base_heading is not None:
        d = p.base_heading - b.base_heading
        turned = abs(math.atan2(math.sin(d), math.cos(d))) >= _YAW_CAUSE_EPS
    if name == "facing":
        return turned
    moved = math.dist(b.base_pos[:2], p.base_pos[:2]) >= DISPLACEMENT_EPS
    return moved or (name == "in_view" and turned)


def _commanded(before: float | None, after: float | None) -> bool:
    return before is not None and after is not None and after - before >= MOTION_EPS


def commanded_nothing(agent: Any, before: ActorBaseline, after: ActorBaseline) -> bool:
    """Charter 3.5: *agent* commanded no motion between the captures *before* and *after*
    of one call — every device it has (base, arm, gripper) was read both times and its
    commanded-motion counter or its attachments did not change at all. A body with no
    such device, a device that could not be read, or a session switch is no evidence:
    False (the call is then a failed attempt, never refused)."""
    channels = [(b, a) for dev, b, a in (
        ("_base", before.base_cmd_motion, after.base_cmd_motion),
        ("_arm", before.arm_ctrl_motion, after.arm_ctrl_motion),
        ("_gripper", before.gripper_welds, after.gripper_welds),
    ) if getattr(agent, dev, None) is not None]
    for name in getattr(agent, '_services', {}).get('joint_groups', {}):
        channels.append(((before.joint_groups or {}).get(name, (None, None))[0],
                         (after.joint_groups or {}).get(name, (None, None))[0]))
    return (bool(channels) and before.session == after.session
            and all(b is not None and b == a for b, a in channels))


def _gripper_caused(want: str, call: ast.Call, b: ActorBaseline, p: ActorBaseline, world: Any = None) -> bool:
    """A weld transition of the wanted polarity: 0->1 for ``rise`` (a grasp), 1->0 for
    ``fall`` (a release), either for ``level``. Target-aware when the call names an
    object: the name resolves through the world manifest *world* (an alias, "green
    bottle") or is a weld key itself, and only that body's weld counts
    (``holding_object('mug')`` is not caused by a fresh banana weld); a name that
    matches no welded body is never caused (I4F1-D2: an alias was caused by ANY weld)."""
    if b.gripper_welds is None or p.gripper_welds is None:
        return False
    target = call.args[0].value if call.args and isinstance(call.args[0], ast.Constant) else None
    keys = set(b.gripper_welds) | set(p.gripper_welds)
    if isinstance(target, str):
        name = ((world.resolve(target) if world is not None else None) or target).strip().lower()
        keys = {k for k in keys if k.lower() == name}
    rose = any(p.gripper_welds.get(k) and not b.gripper_welds.get(k) for k in keys)
    fell = any(b.gripper_welds.get(k) and not p.gripper_welds.get(k) for k in keys)
    return {"rise": rose, "fall": fell}.get(want, rose or fell)


def _call_caused(name: str, call: ast.Call, channel: str, want: str,
                 b: ActorBaseline, p: ActorBaseline, world: Any = None) -> bool:
    if channel == "gripper":
        return _gripper_caused(want, call, b, p, world)
    if channel == "perception":
        # The belief gained a localized observation this step (WP11 observation_count);
        # memory from before the step, or a step that localized nothing, is UNCAUSED.
        return b.belief_obs is not None and p.belief_obs is not None and p.belief_obs > b.belief_obs
    if channel == 'joints':
        arg = call.args[0] if call.args else next((k.value for k in call.keywords if k.arg == 'group'), None)
        if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
            return False
        bc, bq = (b.joint_groups or {}).get(arg.value, (None, None))
        pc, pq = (p.joint_groups or {}).get(arg.value, (None, None))
        return (_commanded(bc, pc) and bq is not None and pq is not None and len(bq) == len(pq)
                and max((abs(x-y) for x, y in zip(bq, pq)), default=0.) >= DISPLACEMENT_EPS)
    if channel == "arm":
        if not _commanded(b.arm_ctrl_motion, p.arm_ctrl_motion):
            return False
        if b.arm_joints is None or p.arm_joints is None or len(b.arm_joints) != len(p.arm_joints):
            return False
        return max((abs(x - y) for x, y in zip(b.arm_joints, p.arm_joints)), default=0.0) >= DISPLACEMENT_EPS
    # base: commanded motion inside the actor's tools, moved in the measured dimension,
    # and (F->T) the literal target was not already met at the baseline pose.
    if not (_commanded(b.base_cmd_motion, p.base_cmd_motion) and _moved(b, p, name)):
        return False
    was = _base_truth_at(name, call, b)
    return not ((want == "rise" and was is True) or (want == "fall" and was is False))


def grade_channels(
    baseline: ActorBaseline | None, post: ActorBaseline, verify: str, oracles: Any,
    turn_baseline: ActorBaseline | None = None, world: Any = None,
) -> tuple[ActorCaused, tuple[str, ...]]:
    """Grade EVERY robot-channel oracle call in *verify* (no first match).

    CAUSED iff each call was caused on its own channel between *baseline* and *post*
    (``_call_caused``): the base/arm commanded motion inside the actor's tools and
    moved, the gripper made the weld transition the call's polarity asks for, and a
    base target was not already met at the baseline (F->T). A call whose tag has
    ``window="turn"`` (kept_holding) is judged from *turn_baseline* (captured when the
    turn started) instead: a grasp anywhere in this turn causes it, an object held
    before the turn never does. The per-call detail
    (``"gripper:holding_object:CAUSED"``) is returned for the audit row. *oracles*
    maps the live evidence names to their ``@oracle`` tags (or the tagged callables).
    Fail closed: no baseline, no robot call, an unreadable channel, or a baseline taken
    in another sim session grade UNCAUSED.
    """
    calls = robot_calls(verify, oracles)
    if baseline is None:
        return ActorCaused.UNCAUSED, ("no baseline",)
    if baseline.session != post.session:
        return ActorCaused.UNCAUSED, ("baseline from another session",)
    if not calls:
        return ActorCaused.UNCAUSED, ()

    def caused(name: str, call: ast.Call, channel: str, want: str) -> bool:
        b = turn_baseline if getattr(_tag(oracles, name), "window", "step") == "turn" else baseline
        return b is not None and b.session == post.session and _call_caused(name, call, channel, want, b, post, world)

    detail = tuple(
        f"{channel}:{name}:{'CAUSED' if caused(name, call, channel, want) else 'UNCAUSED'}"
        for name, call, channel, want in calls
    )
    ok = all(d.endswith(":CAUSED") for d in detail)
    return (ActorCaused.CAUSED if ok else ActorCaused.UNCAUSED), detail


def grade(
    baseline: ActorBaseline | None, post: ActorBaseline, verify: str, oracles: Any,
    turn_baseline: ActorBaseline | None = None, world: Any = None,
) -> ActorCaused:
    """``grade_channels`` without the per-call detail."""
    return grade_channels(baseline, post, verify, oracles, turn_baseline, world)[0]


def is_robot_predicate(verify: str, oracles: Any) -> bool:
    """True iff *verify* calls an oracle tagged with an actor channel (base / arm /
    gripper). Anything else is not causation-graded (dev predicates, answer steps)."""
    return bool(robot_calls(verify, oracles))
