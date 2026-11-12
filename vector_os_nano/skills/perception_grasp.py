# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""PerceptionGraspSkill — the honest, perception-driven Go2+Piper grasp.

This is the skill the North Star calls for: the robot LOCALIZES the target the
way a real robot must — perceive it — instead of reading the simulator's
omniscient object table.

Pipeline (the goal, verbatim):
    VLM 识别 (Moondream bbox)
      -> EdgeTAM 分割 (binary mask)
      -> pointcloud 取中点 (mask + REAL rendered depth -> camera-frame centroid
         -> world via the exact MuJoCo cam->world transform)   [grasp_point.py]
      -> IK + grasp motion (REUSES the proven PickTopDownSkill via its
         params['target_xyz'] seam — no re-implemented motion)
      -> verify  holding_object('<name>')

Contrast with PickTopDownSkill (skills/pick_top_down.py): that skill reads the
object's world pose from a pre-populated world model (ground truth). This skill
NEVER does — the 3D grasp point comes only from depth + mask. If perception
fails (no detection / empty mask / no depth points) it FAILS LOUD with a precise
diagnosis; it must NEVER silently substitute a ground-truth pose to stay green
(that re-fakes perception, the exact thing the goal forbids).

The verify oracle holding_object('<name>') stays ground-truth and BYTE-UNCHANGED
(it is the moat oracle the actor cannot author): the skill only EMITS the verify
string, it never imports/evaluates the oracle or feeds the centroid into it.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.perception.grasp_point import grasp_point_from_rgbd
from vector_os_nano.skills.utils.approach_pose import compute_approach_pose

logger = logging.getLogger(__name__)

# Perception-backend surface this skill consumes (a Go2GraspPerception or any
# adapter giving ONE RGB-D snapshot, detection on a given frame, and a segmenter).
_REQUIRED_PERCEPTION = ("capture", "detect_in", "segment")
# A producer-passed box is stale once the camera moved this far since its frame.
_STALE_BOX_M, _STALE_BOX_RAD = 0.02, 0.035


def _fail(diagnosis: str, message: str, **extra: Any) -> SkillResult:
    data = {"diagnosis": diagnosis, "perceived": False}
    data.update(extra)
    return SkillResult(success=False, error_message=message, result_data=data)


# Deictic words: "the thing in front", "that one" — a spatial reference, no object name.
_DEICTIC_ZH = ("前面", "前方", "面前", "眼前", "前边", "前", "那个", "这个")
_DEICTIC_EN = re.compile(r"\b(?:in front|front|ahead|nearest|this|that|one)\b")
# Request words around a CJK target ("抓起那个罐子"), and the 色 of "红色": never an object.
_FILLER_ZH = re.compile(r"[抓拿取捡起帮我把给请的一下吧个色]|东西|物体|物品")


def _target_words(query: str) -> str:
    """*query* without its deictic words ("grab that can" -> "grab can")."""
    q = (query or "").strip().lower()
    for tok in _DEICTIC_ZH:
        q = q.replace(tok, " ")
    return " ".join(_DEICTIC_EN.sub(" ", q).split())


def _names_object(query: str) -> bool:
    """True when *query* names a concrete object (-> the learned detector route).

    Distinguishes a NAMED object ("罐子"/"red can"/"that can", the shared world
    vocabulary) from a pure-colour ("抓红色的") or deictic ("前面的东西") query; those
    resolve on the classical front_object resolver (by colour or by space), never by
    name. A deictic word never hides a name: "抓前面那个罐子" goes to the detector.
    """
    from vector_os_nano.perception.vocab import vocab

    return vocab().names_object(_target_words(query))



# Colour words the colour resolver has no hue band for (the world vocabulary only
# holds the ones it can see): a query naming one is refused, never grasped as "any".
_UNSEEN_COLOURS = ("white", "black", "grey", "gray", "pink", "brown", "silver", "gold",
                   "白色", "黑色", "灰色", "粉色", "棕色", "银色", "金色")


def _unsupported_target(query: str) -> SkillResult | None:
    """Why the grasp cannot honestly resolve *query*, or None.

    A colour it cannot see, words the world vocabulary cannot read (untranslatable
    CJK left once the deictic, request and colour words are gone: "抓前面那个苹果",
    "抓红色的苹果"), or an unknown noun beside a colour ("the red unicorn": the colour
    route would ignore the noun) are refused before any motion: the robot never
    substitutes the front-most blob, the only object of that colour, or a same-noun
    object of another colour for what was named.
    """
    from vector_os_nano.perception.vocab import _has, vocab

    q = (query or "").lower()
    unseen = next((c for c in _UNSEEN_COLOURS if _has(q, c)), None)
    if unseen is not None:
        return _fail("unsupported_colour",
                     f"{query!r} names the colour {unseen!r}, which this robot cannot tell apart; "
                     "name the object without it, or by a colour it sees "
                     "(red, orange, yellow, green, blue, purple).", query=query)
    v, rest = vocab(), _target_words(query)
    bare = v.without_colours(rest)
    said = v.english(bare)
    if v.noun(rest) is None and ((said and v.colour(rest)) or (said is None and _FILLER_ZH.sub("", bare).strip())):
        nouns = sorted({c for _, c in v.nouns})
        return _fail("unknown_object",
                     f"{query!r} names no object the world vocabulary knows; name one of {nouns} "
                     "(or say 'the thing in front').", query=query, known_objects=nouns)
    return None


def _scene_name(context: SkillContext, query: str) -> str | None:
    """The world-manifest pickable *query* names ("绿色的瓶子" -> pickable_bottle_green),
    or None. Only the verify LABEL comes from it; the grasp POINT is always perceived."""
    world_id = (context.config or {}).get("world")
    if not world_id:
        return None
    from vector_os_nano.vcli.worlds.world_manifest import load_world

    return load_world(world_id).resolve(query)


# Dog-to-object planar distance (m) at which the Piper top-down envelope reaches the
# object. MEASURED R17: the top-down EE reaches only ~0.22m forward of the dog centre
# the dog's SENSOR (0.3 m forward of body origin) must sit within this distance
# of the grasp point so that the body origin is ≥ 0.40 m behind the target —
# the Piper's IK feasibility boundary (measured against the live body pose).
# 0.18 m → sensor at (target.x − 0.18), body at (target.x − 0.48), EE within
# 20-22 mm of the bottle (< 60 mm weld radius) across all tested body_z values.
_GRASP_REACH_M = 0.05
# Minimum plausible z for a back-projected grasp point (world frame, metres).
# Real cans sit at z≈0.32; anything below this is the table-top/floor surface
# (a degraded mask whose depth hits the table, not the object). FAIL LOUD rather
# than accepting a floor-level grasp (the R39 t3 z=0.039 bug). Both back-projection
# sites (_perceive_grasp_point and _grasp_point_from_passed_box) use this guard.
_MIN_GRASP_Z = 0.12
# Forward progress (m) below which the dog is treated as STALLED (jammed against
# the pick-table edge) over one approach step — its closest stable standoff. The
# gait advances ~6-12 cm per 0.8 s step when free, so 3 cm cleanly separates a
# real jam from a slow-but-advancing step.
_STALL_EPS = 0.03
# Final "seat" phase after the approach loop: press forward up to this many firm
# short steps until the dog truly stops advancing (two consecutive sub-_SEAT_EPS
# steps), so it ends at its MAXIMALLY-forward standoff every run (the Piper's
# forward reach band is thin — a few cm of standoff is reach-vs-unreachable).
_SEAT_PRESSES = 6
_SEAT_EPS = 0.015
# Lateral (vy) tracking of the object's y-line during the approach — the gait
# drifts sideways several cm over an approach, and the Piper's lateral grasp window
# is narrow, so we close that y-error with a proportional sidestep. Deadband avoids
# hunting once aligned.
_LATERAL_GAIN = 1.2
_LATERAL_VMAX = 0.25
_LATERAL_DEADBAND = 0.03
# Pre-grasp clearance above the object used for the reach (IK) check.
_PRE_GRASP_H = 0.08
# Post-approach forward nudge: if IK still fails after _approach_object (the
# approach stall can fire 5-10 cm short of the true maximum-reach standoff due
# to gait dynamics — measured from ~20% MISS runs), press forward with pure vx
# (no lateral/yaw) up to this many times so the dog closes the last 5-10 cm.
# Pure-vx steps are used because the lateral correction in the seat phase
# competes with forward advance and can prevent progress below _SEAT_EPS.
# 5 presses × 0.8 s @ vx=0.4 → up to ~15 cm commanded; effective ~8-12 cm on
# the MPC gait — enough to close the typical 5 cm gap that causes IK FAIL.
_POST_APPROACH_NUDGE_N = 5
_POST_APPROACH_NUDGE_V = 0.4    # m/s forward
_POST_APPROACH_NUDGE_DUR = 0.8  # s per press
_POST_APPROACH_NUDGE_VY = 0.2    # m/s lateral cap (correct y-drift toward the object)
# Go2+Piper-specific pick geometry handed to PickTopDownSkill. The arm's
# forward-reach envelope at the table standoff is a THIN z-band (it reaches far
# forward only at z~0.32 and cannot also hover 5 cm higher there), so a SHALLOW
# pre-grasp hover keeps both the pre-grasp and the grasp inside the reachable band
# (a tall hover IK-fails -> the whole grasp bails). The EE extends straight to the
# object's reachable height and the jaws close around the object from there.
_GO2_PRE_GRASP_H = 0.02
_GO2_GRASP_Z_ABOVE = 0.0
# Post-grasp lift: after the weld fires, raise the Piper shoulder (joint2) by this
# much to haul the welded object clear of the table top (z rises several cm) —
# proof of a real pick, not a weld-in-place. Runs ONLY after gripper.is_holding().
_LIFT_SHOULDER_DELTA = 0.5   # rad
_LIFT_DURATION = 2.0         # s
# The arm AT REST sits across the forward head-camera FOV (the gripper bar occludes
# the table), corrupting the front-object mask. Lift the shoulder (joint2) to raise
# the arm ABOVE the FOV before perceiving — empirically clears the view (R13: front
# mask 115px@12cm -> 812px@2cm). The grasp IK then moves the arm down to the object.
_STOW_FOR_VIEW: list[float] = [0.0, 1.2, 0.0, 0.0, 0.0, 0.0]


def _approach_object(
    base: Any, target_xy: tuple[float, float], *,
    reach_m: float = _GRASP_REACH_M, step_v: float = 0.4,
    max_walks: int = 12, on_progress: Any = None,
) -> bool:
    """Walk the base FORWARD toward target_xy to its CLOSEST stable standoff.

    A scripted open-loop forward walk (the base `walk` primitive — NOT the parked FAR
    nav-stack): "前面的东西" is straight ahead, so a heading-aligned forward step closes
    the gap. Two stop conditions, whichever fires first:

      1. within ``reach_m`` of the target (the nominal standoff), OR
      2. the dog STALLS — its front jams against the pick-table edge and stops
         advancing (forward progress < _STALL_EPS over a step). This is the
         furthest-forward, most-repeatable standoff and is what makes the grasp
         robust: the Piper's top-down reach SATURATES forward, so the dog being
         maximally forward (jammed) maximises EE reach, and the jam pose is
         kinematically defined (table edge) — independent of perceived-x noise.

    Either way the dog ends at its closest stable pose, then we align heading so the
    forward-mounted arm points AT the object. Returns True iff the dog is within a
    graspable band of the target at the end (reach_m + margin).
    """
    import math

    def _state() -> tuple[float, float, float, float]:
        """(planar distance, heading error toward target, forward-x, lateral error).

        ``lateral`` is the target's offset PERPENDICULAR to the dog's heading (body
        +y, left): >0 means the target is to the dog's left. Used to command ``vy``
        so the dog tracks the object's y-line instead of drifting laterally with the
        gait (the gait accumulates several cm of side-drift over an approach, which
        — with the Piper's narrow lateral grasp window — pushes the IK target out of
        reach in y; runs failed at y-drift ~0.22 m)."""
        pos = base.get_position()
        dx, dy = target_xy[0] - pos[0], target_xy[1] - pos[1]
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        try:
            hd = float(base.get_heading())
        except Exception:  # noqa: BLE001 — no heading -> skip steering
            return dist, 0.0, float(pos[0]), 0.0
        yaw_err = math.atan2(math.sin(bearing - hd), math.cos(bearing - hd))
        # Perpendicular offset in the body frame: rotate (dx,dy) by -heading, take y.
        lateral = -dx * math.sin(hd) + dy * math.cos(hd)
        return dist, yaw_err, float(pos[0]), lateral

    def _vy(lateral: float) -> float:
        """Lateral set-point: track the object's y-line. Clamped, with a deadband."""
        if abs(lateral) < _LATERAL_DEADBAND:
            return 0.0
        return max(-_LATERAL_VMAX, min(_LATERAL_VMAX, lateral * _LATERAL_GAIN))

    prev_x: float | None = None
    stalls = 0
    jammed = False
    for _ in range(max_walks):
        dist, yaw_err, cur_x, lateral = _state()
        if dist <= reach_m:
            break
        # STALL check — the dog jammed against the table and is no longer
        # advancing. Require TWO consecutive low-progress steps (a single slow step
        # can happen mid-gait or during a heading correction; a real table jam stays
        # stalled) so the dog is driven to its FULL forward standoff (max EE reach),
        # not stopped a few cm short on a momentary slowdown. Only count a stall once
        # a real step has happened (prev_x set) and while roughly on-heading.
        if prev_x is not None and (cur_x - prev_x) < _STALL_EPS and abs(yaw_err) < 0.3:
            stalls += 1
            if stalls >= 2:
                jammed = True
                if on_progress:
                    on_progress(f"approach: jammed at x={cur_x:.2f} (stall x2) — standoff reached")
                break
        else:
            stalls = 0
        prev_x = cur_x
        gap = dist - reach_m
        dur = max(0.6, min(1.6, gap / max(step_v, 1e-3)))
        # STEER toward the target each step — the open-loop forward walk drifts in
        # BOTH heading (gait curvature) and lateral position. A proportional yaw
        # correction keeps the dog pointed on the bearing; a proportional ``vy``
        # correction keeps it ON the object's y-line (so the forward-mounted arm
        # ends up laterally aligned). If badly mis-headed, turn more, creep less.
        vyaw = max(-0.6, min(0.6, yaw_err * 1.5))
        vx = step_v if abs(yaw_err) < 0.5 else step_v * 0.3
        if on_progress:
            on_progress(f"approach: {dist:.2f}m, yaw {math.degrees(yaw_err):.0f}deg, "
                        f"lat {lateral:.2f}m — walk {dur:.1f}s")
        try:
            base.walk(vx=vx, vy=_vy(lateral), vyaw=vyaw, duration=dur)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[PGRASP] approach walk raised: %s", exc)
            return False

    # SEAT firmly against the table. The stall above can trip a few cm short of the
    # true jam (the gait slows before it fully seats), and the Piper's forward reach
    # band is thin — a 5 cm standoff difference is reach-vs-unreachable. So press
    # forward a few firm short steps until the dog truly stops advancing (two
    # consecutive sub-_SEAT_EPS steps), driving it to its MAXIMALLY-forward, most
    # repeatable standoff every run. Heading AND lateral are corrected each press so
    # the dog seats square on the object's y-line, not drifted to a neighbour's.
    # Only seat when the loop ended on a JAM (stall) — if the dog reached the nominal
    # reach_m standoff without jamming, it is already correctly placed and pressing
    # further would walk it INTO / past the object.
    seat_prev: float | None = None
    seat_stalls = 0
    for _ in range(_SEAT_PRESSES if jammed else 0):
        _, yaw_err, cur_x, lateral = _state()
        if seat_prev is not None and (cur_x - seat_prev) < _SEAT_EPS:
            seat_stalls += 1
            if seat_stalls >= 2:
                break
        else:
            seat_stalls = 0
        seat_prev = cur_x
        try:
            base.walk(vx=step_v, vy=_vy(lateral),
                      vyaw=max(-0.4, min(0.4, yaw_err * 1.5)), duration=0.7)
        except Exception:  # noqa: BLE001
            break

    # Final alignment: correct any residual lateral offset (sidestep onto the y-line)
    # then heading, so the forward-mounted arm points AT the object in BOTH axes.
    _, yaw_err, _, lateral = _state()
    if abs(lateral) > _LATERAL_DEADBAND:
        try:
            base.walk(vx=0.0, vy=_vy(lateral), vyaw=0.0, duration=0.8)
        except Exception:  # noqa: BLE001
            pass
    _, yaw_err, _, _ = _state()
    if abs(yaw_err) > 0.08:
        try:
            base.walk(vx=0.0, vy=0.0, vyaw=max(-0.6, min(0.6, yaw_err * 1.5)), duration=0.8)
        except Exception:  # noqa: BLE001
            pass
    dist, _, _, _ = _state()
    return dist <= reach_m + 0.20


# Re-pose tuning (R38). The seam that converts FAR's "roughly near the table at
# any heading" terminal pose into the head-on, correctly-facing precondition the
# scripted _approach_object already handles.
# Clearance the dog should stand at from the object before the jam/seat. 0.45 m
# keeps the subsequent jam/seat landing the dog at ~x=10.45 (the proven standoff)
# while staying inside the FAR arrival_radius so the move is short.
_REPOSE_CLEARANCE_M = 0.45
# Turn-in-place: yaw error below this is treated as "already facing the object"
# (a no-op). Above it the dog turns toward the facing yaw before any forward creep.
# Generous (~9 deg) so a head-on spawn pose (D34-D51) never triggers a real turn.
_REPOSE_YAW_DEADBAND = 0.16     # rad (~9 deg)
_REPOSE_TURN_VYAW = 0.8         # rad/s turn-in-place speed
_REPOSE_TURN_MAX_S = 4.0        # cap a single turn command's duration
# Short approach toward the computed standoff after turning. Only fires when the
# dog is materially off the standoff; the existing jam/seat/nudge does the rest.
_REPOSE_MOVE_DEADBAND = 0.12    # m
_REPOSE_MOVE_VX = 0.4           # m/s
_REPOSE_MOVE_MAX_S = 3.0        # cap a single move command's duration
# Pre-perceive orientation scan (R38). After a FAR navigate the dog can arrive
# facing AWAY from the table; a single frame then localizes a far wall, not the
# can. A PLAUSIBLE grasp target is a near-table object: its planar distance from
# the dog must be below this radius (the dog navigated to the table standoff, so
# the can is within ~1.2 m; a 7 m "object" is the wrong wall). The scan turns in
# place by _SCAN_STEP_RAD up to _SCAN_MAX_STEPS times, perceiving at each heading,
# and keeps the CLOSEST plausible perception. The dog returns to the best heading.
_SCAN_MAX_LOCAL_M = 1.6
_SCAN_STEP_RAD = 0.6            # ~34 deg per scan step
_SCAN_MAX_STEPS = 6             # up to ~200 deg of sweep
_SCAN_TURN_VYAW = 0.8           # rad/s


def _grasp_ready_repose(
    base: Any, can_xy: tuple[float, float], *,
    clearance: float = _REPOSE_CLEARANCE_M, on_progress: Any = None,
) -> bool:
    """Re-pose the base to a head-on, facing standoff before the scripted grasp.

    FAR's nav primitive (``base.navigate_to``) has no terminal-heading control: it
    stops within ~0.8 m of (x, y) at whatever heading the last way_point left, and
    may overshoot. The scripted ``_approach_object`` below assumes a +X head-on
    spawn pose, so from a FAR arrival (heading ~169 deg off, overshot in y/x) it
    would creep the WRONG way. This deterministic sandwich fixes that:

        compute_approach_pose(clearance)  -> a standoff (ax, ay) on the dog's side
                                             with a yaw that FACES the can
        turn-in-place to that facing yaw  -> the MISSING terminal-heading step
        short forward move toward (ax,ay) -> close the standoff gap
        (then the caller hands off to _approach_object/jam/seat/nudge)

    Idempotent / benign when the dog is ALREADY head-on (yaw error below the
    deadband, already at the standoff): it issues no large turn and no move, so the
    scripted-from-spawn grasp (D34-D51) is preserved. Reuses the proven
    ``compute_approach_pose``. World-agnostic — uses only the base's
    ``walk`` / ``get_position`` / ``get_heading`` duck-typed surface.

    Returns True if the re-pose ran (or was a benign no-op); False if the base
    lacks the required surface (caller then proceeds as before — no regression).
    """
    import math

    walk = getattr(base, "walk", None)
    get_pos = getattr(base, "get_position", None)
    get_hd = getattr(base, "get_heading", None)
    if not callable(walk) or not callable(get_pos) or not callable(get_hd):
        # Missing the surface we need (e.g. a base without get_heading) — skip the
        # re-pose gracefully; the legacy approach still runs. No +X assumption.
        return False

    try:
        pos = get_pos()
        heading = float(get_hd())
    except Exception as exc:  # noqa: BLE001 — no live pose -> skip, never assume +x
        logger.debug("[PGRASP] re-pose skipped (no live pose/heading): %s", exc)
        return False

    dog_pose = (float(pos[0]), float(pos[1]), heading)
    can3 = (float(can_xy[0]), float(can_xy[1]), 0.0)
    try:
        ax, ay, ayaw = compute_approach_pose(can3, dog_pose, clearance=clearance)
    except ValueError:
        # Dog is essentially on top of the can — face the can directly instead.
        ax, ay = dog_pose[0], dog_pose[1]
        ayaw = math.atan2(can3[1] - ay, can3[0] - ax)

    # --- 1. turn in place to the facing yaw (the missing terminal-heading step) ---
    yaw_err = math.atan2(math.sin(ayaw - heading), math.cos(ayaw - heading))
    if abs(yaw_err) > _REPOSE_YAW_DEADBAND:
        dur = min(_REPOSE_TURN_MAX_S, abs(yaw_err) / _REPOSE_TURN_VYAW)
        vyaw = _REPOSE_TURN_VYAW if yaw_err > 0 else -_REPOSE_TURN_VYAW
        if on_progress:
            on_progress(f"re-pose: turn {math.degrees(yaw_err):.0f}deg to face object")
        logger.info("[PGRASP] re-pose turn %.0fdeg (heading %.2f -> %.2f)",
                    math.degrees(yaw_err), heading, ayaw)
        try:
            walk(vx=0.0, vy=0.0, vyaw=vyaw, duration=dur)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[PGRASP] re-pose turn raised: %s", exc)
            return True

    # --- 2. short forward move toward the standoff (ax, ay) -------------------
    try:
        pos = get_pos()
    except Exception:  # noqa: BLE001
        return True
    move_gap = math.hypot(ax - pos[0], ay - pos[1])
    if move_gap > _REPOSE_MOVE_DEADBAND:
        dur = min(_REPOSE_MOVE_MAX_S, move_gap / max(_REPOSE_MOVE_VX, 1e-3))
        if on_progress:
            on_progress(f"re-pose: move {move_gap:.2f}m to standoff")
        logger.info("[PGRASP] re-pose move %.2fm toward standoff (%.2f, %.2f)",
                    move_gap, ax, ay)
        try:
            walk(vx=_REPOSE_MOVE_VX, vy=0.0, vyaw=0.0, duration=dur)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[PGRASP] re-pose move raised: %s", exc)
    return True


# Final heading-alignment tolerance for _face_object (the Piper top-down IK envelope
# is sized for a dog standing HEAD-ON, facing the object). Tighter than the
# approach-loop yaw band so the forward-mounted arm points square at the target.
_FACE_TOL_RAD = 0.08
_FACE_VYAW_MAX = 0.6


def _face_object(base: Any, target_xy: tuple[float, float], *, max_turns: int = 4) -> None:
    """Turn in place so the dog FACES target_xy (final heading align before the grasp)."""
    import math

    walk = getattr(base, "walk", None)
    get_pos = getattr(base, "get_position", None)
    get_hd = getattr(base, "get_heading", None)
    if not (callable(walk) and callable(get_pos) and callable(get_hd)):
        return
    for _ in range(max_turns):
        try:
            pos = get_pos()
            hd = float(get_hd())
        except Exception:  # noqa: BLE001
            return
        bearing = math.atan2(target_xy[1] - pos[1], target_xy[0] - pos[0])
        yaw_err = math.atan2(math.sin(bearing - hd), math.cos(bearing - hd))
        if abs(yaw_err) < _FACE_TOL_RAD:
            return
        try:
            walk(vx=0.0, vy=0.0,
                 vyaw=max(-_FACE_VYAW_MAX, min(_FACE_VYAW_MAX, yaw_err * 1.5)),
                 duration=0.6)
        except Exception:  # noqa: BLE001
            return


def _approach_and_seat(base: Any, grasp_xy: tuple[float, float]) -> bool:
    """FINAL-HOP approach: scripted re-pose + stall-seating creep (D95).

    The dog reaches here already routed head-on to the perceive standoff (~0.95 m) by
    ``navigate_to_object``'s vgraph plan, with clear line-of-sight to the table. The
    remaining short hop to the IK-reachable standoff is the FLAKIEST link when handed
    back to the vgraph planner (the prior ``_approach_via_nav`` path): targeting a
    standoff this close to the inflated table obstacle, ``navigate_to`` intermittently
    (a) returns False and dumps the dog metres PAST the object, or (b) returns True at a
    sub-optimally-seated pose a few cm short of the THIN Piper forward-reach band → IK
    still fails. The scripted forward creep ``_approach_object`` instead stall-SEATS the
    dog against the table edge — a kinematically-pinned standoff, repeatable run-to-run
    and independent of planner variance — the proven, most-reliable terminal placement
    (it is exactly what the no-navigate_to path already used, D34-D51).

    So the scripted seat-creep is now the PRIMARY (and only) final-hop mechanism for
    every base — unifying the two former approach branches onto ONE impl (Rule 11):
        1. ``_grasp_ready_repose`` — turn to FACE the object + close any standoff gap
           (a benign no-op when already head-on, as it is after navigate_to_object).
        2. ``_approach_object``    — open-loop forward creep that stall-seats at the
           table edge (steers heading + lateral each step; the proven repeatable pose).
        3. ``_face_object``        — final tight heading align before the grasp.
    World-agnostic: uses only the base's walk / get_position / get_heading surface;
    each step gracefully no-ops if that surface is absent. Always returns True (the
    approach ran best-effort; the caller's post-approach nudge + IK gate are the
    acceptance, not this return value).
    """
    gx, gy = float(grasp_xy[0]), float(grasp_xy[1])
    logger.info("[PGRASP] seat-approach: scripted re-pose + stall-seat creep to (%.2f,%.2f)",
                gx, gy)
    _grasp_ready_repose(base, (gx, gy), clearance=_REPOSE_CLEARANCE_M)
    _approach_object(base, (gx, gy), max_walks=30)
    _face_object(base, (gx, gy))
    return True


@skill(
    aliases=[
        # Perception-natural grasp phrasings. Registered LAST (after PickTopDown)
        # so these win on the bare-cli / empty-world-model honest path.
        "抓", "拿", "抓起", "抓住", "抓取", "拿起", "取",
        "grab", "grasp", "pick up", "pick",
    ],
)
class PerceptionGraspSkill:
    """Perceive a target (VLM->EdgeTAM->depth pointcloud) and grasp it."""

    name: str = "perception_grasp"
    description: str = (
        "Grasp an object the robot must FIND first: detect it with the VLM, "
        "segment it with EdgeTAM, compute the 3D grasp point from the depth "
        "camera + mask, then top-down grasp with the Piper arm. Use this when "
        "the object is NOT already in the world model (the normal case). The "
        "grasp point is perceived, never read from ground truth. If an earlier "
        "detect step already produced boxes, pass them via 'detections' (or a "
        "single 'bbox') and this skill CONSUMES them — back-projecting the box to "
        "a 3D grasp point — instead of re-perceiving (true producer->consumer)."
    )
    # Auto-verified after the grasp, bound from the call's query (an object the world
    # manifest cannot resolve binds no auto-verify; the model's own verify names it).
    verify_hint: str = "holding_object({query})"
    parameters: dict = {
        "query": {
            "type": "string",
            "required": True,
            "description": "What to grasp, in natural language (e.g. 'banana', 'red can', 'the bottle').",
        },
        "detections": {
            "type": "array",
            "required": False,
            "description": (
                "Optional. Detection dicts from a producer detect step "
                "(${detect.output.detections}); when present the grasp CONSUMES the "
                "matching box (colour-selected) and does NOT re-perceive."
            ),
        },
        "bbox": {
            "type": "array",
            "required": False,
            "description": (
                "Optional. A single (x1,y1,x2,y2) pixel box from a producer; "
                "consumed in place of re-perceiving (back-projected to 3D)."
            ),
        },
    }
    preconditions: list[str] = ["gripper_empty"]
    postconditions: list[str] = []
    effects: dict = {"gripper_state": "holding"}
    failure_modes: list[str] = [
        "no_arm", "no_gripper", "arm_unsupported", "no_perception", "no_camera",
        "no_detections", "perception_failed", "segmentation_failed", "no_depth_points",
        "ik_unreachable", "move_failed", "not_holding",
    ]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        import numpy as np  # local — keep import light

        arm = context.arm
        gripper = context.gripper
        if arm is None:
            return _fail("no_arm", "No arm connected")
        if gripper is None:
            return _fail("no_gripper", "No gripper connected")
        if not hasattr(arm, "ik_top_down"):
            return _fail(
                "arm_unsupported",
                f"Arm {getattr(arm, 'name', type(arm).__name__)!r} lacks ik_top_down; "
                "PerceptionGraspSkill needs a 6-DoF arm with top-down IK (e.g. MuJoCoPiper).",
            )

        perception = context.perception
        query = str(
            params.get("query") or params.get("object_label")
            or params.get("target") or params.get("object_id") or ""
        ).strip()
        if perception is None:
            return _fail("no_perception", "No perception backend available; cannot perceive the target")
        missing = [m for m in _REQUIRED_PERCEPTION if not hasattr(perception, m)]
        if missing:
            return _fail(
                "no_camera",
                f"Perception backend {type(perception).__name__} lacks RGB-D grasp surface: {missing}",
            )

        unsupported = _unsupported_target(query)
        if unsupported is not None:
            return unsupported

        # --- stow the arm OUT of the camera FOV before perceiving (the arm at rest
        # occludes the forward head camera and corrupts the front-object mask) -----
        if hasattr(arm, "move_joints"):
            try:  # move_joints returns once the arm has settled at stow
                arm.move_joints(_STOW_FOR_VIEW, duration=1.2)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[PGRASP] stow-for-view move failed: %s", exc)

        # --- ATTRIBUTE grasp (D47): parse a colour from the query. When present, the
        # perception resolver selects the blob of THAT colour (not the front-most) and
        # the verify LABEL is the world-manifest body the query names (the grasp
        # POINT is still perceived from depth+mask). A deictic "前面的东西" parses no colour → unchanged.
        from vector_os_nano.perception.front_object import parse_color
        color = parse_color(query)
        scene = _scene_name(context, query)

        # --- PRODUCER->CONSUMER composition (R37): if an earlier `detect` step
        # already produced boxes (flowed in via ${detect.output.detections} /
        # ${detect.output.boxes}), CONSUME the matching box and back-project it to a
        # 3D grasp point — instead of re-perceiving. The grasp's target then came
        # FROM the routed detector (true composition), not a fresh independent
        # re-perceive. The 3D math is identical (segment -> grasp_point_from_rgbd);
        # only the box SOURCE differs (passed vs self-detected). FAIL LOUD if the
        # passed box yields no depth points (never a GT fallback).
        passed_box = self._resolve_passed_box(params, color)
        box_frame = self._passed_box_frame(params)
        consumed_bbox = passed_box is not None
        if consumed_bbox:
            logger.info(
                "[PGRASP] CONSUMING producer box %s (colour=%s) — re-perceive SUPPRESSED",
                passed_box, color,
            )
            gp, resolved, fail = self._grasp_point_from_passed_box(
                perception, query, passed_box, color=color, scene=scene, box_frame=box_frame,
                base=context.base, confidence=self._passed_box_confidence(params, color),
            )
        else:
            # --- perceive the target's 3D grasp point (real depth + mask, never GT).
            # R38: after a FAR navigate the dog arrives at an ARBITRARY heading and
            # may be facing AWAY from the table (observed: heading ~117deg, camera on
            # a far wall -> a garbage grasp point 7 m away). So when a steerable base
            # is present, perceive WITH A SCAN: turn in place sampling the camera and
            # keep the heading whose perceived grasp point is the closest PLAUSIBLE
            # near-table object (within a sane local radius of the dog). This honestly
            # localizes the can by LOOKING for it, never by reading GT. On the
            # scripted-from-spawn path (can dead-ahead) the first sample is already
            # plausible, so the scan returns immediately — no D34-D51 regression.
            gp, resolved, fail = self._perceive_with_scan(
                perception, context.base, query, color=color, scene=scene
            )
        if fail is not None:
            return fail
        approached = False

        # --- approach if out of reach (scripted forward walk; D27: NON-gated, it is
        # the base `walk` primitive, not the parked FAR nav-stack). "前面的东西" can be
        # ~0.84m ahead at spawn (out of the Piper ~0.34m top-down envelope, R5); drive
        # the dog forward to standoff. We do NOT re-perceive after the approach: the
        # object's WORLD coordinate is invariant under the dog's motion, and the
        # forward head camera is well-framed from AFAR but looks OVER the table when
        # close (poor framing) — so the afar grasp point (R3: ~6.9cm) is the accurate
        # one; the approach simply brings the dog within reach of that fixed point.
        base = context.base
        # R38 — GRASP-READY RE-POSE. The base may have arrived here under FAR
        # navigation (the producer's navigate step), whose terminal pose has NO
        # heading control: ~169 deg off, overshot in x/y (probe-confirmed). Before
        # the scripted +X-head-on approach below, deterministically turn-in-place to
        # FACE the object and close the standoff gap, so _approach_object starts from
        # the precondition it assumes. Benign no-op when the dog is already head-on
        # (scripted-from-spawn grasp D34-D51 preserved). World-agnostic; reuses the
        # proven compute_approach_pose. Runs only when out of reach (a perfectly
        # placed dog needs no re-pose).
        if base is not None and arm.ik_top_down((gp.x, gp.y, gp.z + _PRE_GRASP_H)) is None:
            # D95 — the dog arrived head-on at the ~0.95 m perceive standoff (routed by
            # navigate_to_object's vgraph), out of IK reach. Close the final hop with the
            # SCRIPTED re-pose + stall-seating creep — the proven, planner-variance-free
            # terminal placement — for EVERY base (one impl, Rule 11). This replaces the
            # flaky vgraph nav-approach for the last hop (it intermittently dumped the dog
            # metres past the object or seated a few cm short of the thin reach band).
            logger.info("[PGRASP] %s out of reach @ (%.2f,%.2f) — scripted seat-approach",
                        resolved, gp.x, gp.y)
            _approach_and_seat(base, (gp.x, gp.y))
            approached = True

        # Post-approach nudge: if the approach stall fired short of the true reach
        # standoff (gait false-stall), IK still fails. Nudge the dog TOWARD the
        # object in 2D until IK succeeds or _POST_APPROACH_NUDGE_N presses exhaust.
        # D42/D41: the residual ~20% MISS was LATERAL y-drift the old pure-vx nudge
        # could not fix -> correct BOTH the forward gap (vx, only while a gap remains
        # so we do not drive into the table) AND the lateral error (vy toward the
        # object y-line, body-frame via heading).
        if base is not None and approached:
            for _n in range(_POST_APPROACH_NUDGE_N):
                if arm.ik_top_down((gp.x, gp.y, gp.z + _PRE_GRASP_H)) is not None:
                    break
                pos = base.get_position()
                dx, dy = gp.x - pos[0], gp.y - pos[1]
                # R38 — use the LIVE heading; the old ``except: th = 0.0  # assume
                # +x`` fallback silently mis-fired from a FAR arrival heading (it
                # is true only at spawn). If the base genuinely cannot report a
                # heading, we cannot steer a body-frame nudge safely — STOP nudging
                # rather than driving in a fabricated +X direction.
                try:
                    th = float(base.get_heading())
                except Exception as exc:  # noqa: BLE001 -- no heading: cannot steer
                    logger.info("[PGRASP] post-approach nudge: no heading (%s) — "
                                "skipping (no +X assumption)", exc)
                    break
                fwd = dx * math.cos(th) + dy * math.sin(th)
                lat = -dx * math.sin(th) + dy * math.cos(th)
                vx = _POST_APPROACH_NUDGE_V if fwd > 0.03 else 0.0
                vy = max(-_POST_APPROACH_NUDGE_VY, min(_POST_APPROACH_NUDGE_VY, lat * 2.0))
                if vx == 0.0 and abs(vy) < 0.02:
                    break
                logger.info("[PGRASP] post-approach nudge %d/%d (IK fails) vx=%.2f vy=%.2f (fwd=%.2f lat=%.2f)",
                            _n + 1, _POST_APPROACH_NUDGE_N, vx, vy, fwd, lat)
                try:
                    base.walk(vx=vx, vy=vy, vyaw=0.0, duration=_POST_APPROACH_NUDGE_DUR)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[PGRASP] nudge walk raised: %s", exc)
                    break

        logger.info("[PGRASP] %s -> grasp_world=(%.3f, %.3f, %.3f) approached=%s",
                    resolved, gp.x, gp.y, gp.z, approached)

        # --- delegate the proven top-down motion via the target_xyz seam ------
        # The Piper's forward-reach envelope at full extension is a THIN z-band: at
        # the far standoff the object sits at (it reaches ~0.49 m forward only when
        # reaching to z~0.32, and cannot also hover 5 cm HIGHER there — measured
        # by a probe: pre-grasp at obj_z+0.05 IK-fails while the grasp at
        # obj_z IK-succeeds). So we tell PickTopDownSkill to use a SHALLOW pre-grasp
        # hover (obj_z + _GO2_PRE_GRASP_H) that stays inside that band — the EE
        # extends straight to the object's reachable height and the jaws close from
        # there, rather than first reaching an unreachable
        # high hover and bailing ik_unreachable. World-specific config injection (the
        # perception skill owns the Go2+Piper reach knowledge), not a kernel change.
        ctx_cfg = dict(context.config or {})
        skills_cfg = dict(ctx_cfg.get("skills", {}))
        ptd_cfg = dict(skills_cfg.get("pick_top_down", {}))
        ptd_cfg.setdefault("pre_grasp_height", _GO2_PRE_GRASP_H)
        ptd_cfg.setdefault("grasp_z_above", _GO2_GRASP_Z_ABOVE)
        skills_cfg["pick_top_down"] = ptd_cfg
        ctx_cfg["skills"] = skills_cfg
        context.config = ctx_cfg

        from vector_os_nano.skills.pick_top_down import PickTopDownSkill
        pick_params = dict(params)
        pick_params["target_xyz"] = [gp.x, gp.y, gp.z]
        pick_params["object_id"] = resolved
        res = PickTopDownSkill().execute(pick_params, context)

        # --- LIFT the grasped object clear of the table -----------------------
        # PickTopDownSkill's lift only returns to the (shallow) pre-grasp hover, so
        # with the Go2+Piper's thin forward z-band the object rises barely ~1 cm —
        # not a convincing pick. Once the weld has actually fired (gripper holding),
        # retract the shoulder UP (joint2) so the welded object is hauled clear of
        # the table (z rises several cm). This is honest: it only runs AFTER a real
        # weld, and the object moves because it is physically attached to link6.
        try:
            if gripper.is_holding() and hasattr(arm, "get_joint_positions") and hasattr(arm, "move_joints"):
                cur = list(arm.get_joint_positions())
                lift_q = list(cur)
                lift_q[1] = max(0.0, cur[1] - _LIFT_SHOULDER_DELTA)  # raise shoulder -> EE up
                arm.move_joints(lift_q, duration=_LIFT_DURATION)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[PGRASP] post-grasp lift failed: %s", exc)

        rd = dict(res.result_data or {})
        rd.update({
            "perceived": True,
            "grasp_world": [gp.x, gp.y, gp.z],
            "detection_label": resolved,
            "query": query,
            "approached": approached,
            # Composition evidence (R37): True iff the grasp target came from a
            # producer-passed box (re-perceive suppressed); False iff this skill
            # ran its own detect/front_object resolve.
            "consumed_bbox": consumed_bbox,
            "reperceived": not consumed_bbox,
            # which segmenter made the mask the grasp point came from
            "segmenter": getattr(perception, "last_segmenter", None),
        })
        if not res.success and "diagnosis" not in rd:
            rd["diagnosis"] = "grasp_failed"
        return SkillResult(
            success=res.success,
            error_message=res.error_message,
            result_data=rd,
        )

    @staticmethod
    def _resolve_passed_box(params: dict, color: str | None):
        """Extract a single (x1,y1,x2,y2) box passed in by a producer, or None.

        Two producer seams (rule 4, ${step.path} binding):
          - ``detections``: a list of detection dicts (``${detect.output.detections}``)
            — each ``{"label", "bbox", "confidence"}``. The colour-matching box is
            selected (D49 _select_detection logic on dicts) so a colour query still
            targets the right object; absent a colour, the highest-confidence box.
          - ``bbox``: a single ``[x1,y1,x2,y2]`` (``${detect.output.boxes.0}`` or a
            pre-resolved box). Used directly.
        A bare ``boxes`` list (``${detect.output.boxes}``) is also accepted, picking
        the first box (no per-box label to colour-select on). Returns None when no
        usable box is present — the caller then re-perceives (back-compatible).
        """
        # Single explicit box wins if well-formed.
        bbox = params.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                return tuple(float(v) for v in bbox)
            except (TypeError, ValueError):
                return None

        # Full detection dicts -> colour-select, then take the bbox.
        dets = params.get("detections")
        if isinstance(dets, list) and dets:
            det = PerceptionGraspSkill._select_detection_dict(dets, color)
            box = det.get("bbox") if isinstance(det, dict) else None
            if isinstance(box, (list, tuple)) and len(box) == 4:
                try:
                    return tuple(float(v) for v in box)
                except (TypeError, ValueError):
                    return None

        # Bare boxes list (no labels) -> first box (cannot colour-select).
        boxes = params.get("boxes")
        if isinstance(boxes, list) and boxes:
            first = boxes[0]
            if isinstance(first, (list, tuple)) and len(first) == 4:
                try:
                    return tuple(float(v) for v in first)
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _passed_box_confidence(params: dict, color: str | None) -> float:
        """The producer's score for the passed box (1.0 for a bare box: nothing to gate)."""
        dets = params.get("detections")
        if params.get("bbox") is None and isinstance(dets, list) and dets:
            det = PerceptionGraspSkill._select_detection_dict(dets, color)
            try:
                return float(det.get("confidence", 1.0))
            except (TypeError, ValueError, AttributeError):
                return 0.0
        return 1.0

    @staticmethod
    def _passed_box_frame(params: dict) -> dict | None:
        """The ``frame`` provenance (size, camera pose) a producer's detect output carries."""
        dets = params.get("detections")
        frame = params.get("frame") or (dets[0].get("frame") if isinstance(dets, list) and dets
                                         and isinstance(dets[0], dict) else None)
        return frame if isinstance(frame, dict) else None

    @staticmethod
    def _box_is_stale(box_frame: dict | None, rgb: Any, cam_xpos: Any, cam_xmat: Any) -> str | None:
        """Why a producer box no longer belongs to the current frame, or None."""
        import numpy as np

        if not box_frame:
            return None
        try:
            if [int(v) for v in box_frame["size"]] != [int(rgb.shape[1]), int(rgb.shape[0])]:
                return f"frame size {box_frame['size']} != {[rgb.shape[1], rgb.shape[0]]}"
            moved = float(np.linalg.norm(np.asarray(box_frame["cam_pos"], float) - np.asarray(cam_xpos, float)))
            Ra = np.asarray(box_frame["cam_mat"], float).reshape(3, 3)
            Rb = np.asarray(cam_xmat, float).reshape(3, 3)
            turned = float(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1.0) / 2.0, -1.0, 1.0)))
        except (KeyError, TypeError, ValueError) as exc:
            return f"unreadable box provenance ({exc})"
        if moved > _STALE_BOX_M or turned > _STALE_BOX_RAD:
            return f"camera moved {moved:.3f} m / {math.degrees(turned):.1f} deg since the box"
        return None

    @staticmethod
    def _select_detection_dict(detections: list[dict], color: str | None) -> dict:
        """_select_detection (D49) on detection DICTS instead of Detection objects.

        Same perceptual-colour-preference rule: when a colour is requested, prefer a
        box whose label NAMES that colour over raw max-confidence; among matches take
        the highest confidence; else fall back to plain max-confidence.
        """
        def _conf(d: dict) -> float:
            try:
                return float(d.get("confidence", 0.0))
            except (TypeError, ValueError):
                return 0.0

        if color:
            matching = [
                d for d in detections
                if isinstance(d, dict) and color in str(d.get("label", "") or "").lower()
            ]
            if matching:
                return max(matching, key=_conf)
        return max(detections, key=_conf)

    def _grasp_point_from_passed_box(
        self, perception: Any, query: str, box: tuple, *, color: str | None = None,
        scene: str | None = None, box_frame: dict | None = None, base: Any = None,
        confidence: float = 1.0,
    ):
        """Back-project a PRODUCER-passed pixel box to a WORLD grasp point.

        The composition path (R37): the box came from the routed detector, NOT a
        fresh perceive. We acquire ONLY the RGB-D frame + geometry (no detect /
        front_object call), segment the passed box (same EdgeTAM->box-rect path the
        self-detect route uses), then run the IDENTICAL grasp_point_from_rgbd math.
        FAIL LOUD if the box yields no depth points — never a GT substitute. The box
        must also be a plausible object on THIS frame (the localizer's gates: size,
        range, height, standing out, colour) within reach of the base; else refused.

        Returns ``(gp | None, resolved_label, fail_result | None)``.
        """
        import numpy as np
        from vector_os_nano.perception.object_localizer import capture_rgbd
        try:
            rgb, depth, intrinsics, cam_xpos, cam_xmat = capture_rgbd(perception)
        except Exception as exc:  # noqa: BLE001
            return None, (query or "front object"), _fail(
                "no_camera", f"Failed to read RGB-D frame: {exc}")

        # Verify LABEL convention matches the self-detect route: the world manifest's
        # body name when the query names one; otherwise the query text.
        resolved = scene or (query or "front object")

        # A box from an earlier frame (the camera moved since: a navigate, the arm
        # stow) is re-detected on THIS frame rather than segmented where it no longer is.
        stale = self._box_is_stale(box_frame, rgb, cam_xpos, cam_xmat)
        if stale:
            logger.info("[PGRASP] producer box %s is stale (%s) — re-detecting %r", box, stale, query)
            try:
                from vector_os_nano.perception.object_localizer import detect_on
                dets = detect_on(perception, rgb, query)
            except Exception as exc:  # noqa: BLE001
                return None, resolved, _fail(
                    "perception_failed", f"The producer box for {resolved!r} is stale ({stale}) and "
                    f"the detector failed re-detecting it: {exc}",
                    detection_label=resolved, query=query, consumed_bbox=True, cause=str(exc),
                )
            if not dets:
                return None, resolved, _fail(
                    "stale_box", f"The producer box for {resolved!r} is stale ({stale}) and "
                    "re-detecting on the current frame found nothing.",
                    detection_label=resolved, query=query, consumed_bbox=True,
                )
            box = tuple(float(v) for v in self._select_detection(dets, color).bbox)

        # Segment the PASSED box — reuse the perception segmenter (EdgeTAM; a box-rect
        # mask when it finds nothing). NO perception.detect / front_object_mask call
        # here (that is the re-perceive we are suppressing).
        try:
            mask = perception.segment(rgb, box)
        except Exception as exc:  # noqa: BLE001 — the segmenter's own failure, with its cause
            return None, resolved, _fail(
                "perception_failed",
                f"The segmenter failed on producer box {box} ({resolved!r}): {type(exc).__name__}: {exc}",
                detection_label=resolved, query=query, consumed_bbox=True,
            )
        if mask is None or int(np.count_nonzero(mask)) == 0:
            return None, resolved, _fail(
                "segmentation_failed",
                f"Segmentation produced no mask for producer box {box} ({resolved!r}).",
                detection_label=resolved, query=query, consumed_bbox=True,
            )

        from types import SimpleNamespace

        from vector_os_nano.perception.grasp_point import grasp_point_from_points, object_points
        from vector_os_nano.perception.object_localizer import implausible
        from vector_os_nano.perception.vocab import vocab

        pts = object_points(depth, rgb, mask, intrinsics, cam_xpos, cam_xmat)
        gp = None if pts is None else grasp_point_from_points(*pts, cam_xpos, cam_xmat)
        why = None if pts is None else implausible(
            SimpleNamespace(bbox=box, confidence=confidence), mask, *pts, depth, rgb,
            noun=vocab().noun(query), colour=color,
        )
        if why is None and gp is not None and base is not None:
            pos = base.get_position()
            far = math.hypot(gp.x - pos[0], gp.y - pos[1])
            if far > _SCAN_MAX_LOCAL_M:
                why = f"{far:.1f} m from the robot (beyond {_SCAN_MAX_LOCAL_M} m)"
        if why is not None:
            return None, resolved, _fail(
                "implausible_box",
                f"The producer box {box} is not a plausible {resolved!r} on the current frame: "
                f"{why}. Look again rather than grasp it.",
                detection_label=resolved, query=query, consumed_bbox=True, why=why,
            )
        if gp is None:
            return None, resolved, _fail(
                "no_depth_points",
                f"No valid depth points under the producer box {box} ({resolved!r}); "
                "cannot localize a grasp point. FAIL LOUD — not substituting a "
                "ground-truth pose.",
                detection_label=resolved, query=query, consumed_bbox=True,
            )
        # CHANGE 2 (R40): low-z collapse guard (same rule as the self-detect site).
        # A back-projected z below _MIN_GRASP_Z is the table/floor surface, not the
        # object top. FAIL LOUD so the step grades RAN, never a false GROUNDED.
        if gp.z < _MIN_GRASP_Z:
            return None, resolved, _fail(
                "low_z_backprojection",
                f"Back-projected z={gp.z:.3f}m < _MIN_GRASP_Z={_MIN_GRASP_Z}m "
                f"for producer box {box} ({resolved!r}); the mask hit the table/floor. "
                "FAIL LOUD — rejecting a floor-level grasp point.",
                detection_label=resolved, query=query, consumed_bbox=True, actual_z=gp.z,
            )
        logger.info(
            "[PGRASP] consumed producer box %s -> grasp_world=(%.3f, %.3f, %.3f)",
            box, gp.x, gp.y, gp.z,
        )
        return gp, resolved, None

    @staticmethod
    def _select_detection(detections: list, color: str | None):
        """Pick the box to grasp; PERCEPTUAL colour preference over max-confidence.

        grounding-dino labels each box with the matched prompt phrase (e.g.
        "a red can" → label "red can"), so when *color* is requested we prefer a box
        whose label NAMES that colour over the raw highest-confidence box — this is
        what makes colour selection perceptual rather than centrality-staged (the
        scene authors green dead-centre, so plain max-confidence + staging would
        always pick green). Among colour-matching boxes, take the highest confidence;
        if none name the colour, fall back to plain max-confidence (the adjective may
        have under-grounded). No colour → plain max-confidence (unchanged).
        """
        def _conf(d) -> float:
            return float(getattr(d, "confidence", 0.0))

        if color:
            matching = [
                d for d in detections
                if color in str(getattr(d, "label", "") or "").lower()
            ]
            if matching:
                return max(matching, key=_conf)
        return max(detections, key=_conf)

    def _perceive_with_scan(
        self, perception: Any, base: Any, query: str, *, color: str | None = None,
        scene: str | None = None,
    ):
        """Perceive the grasp point, turning in place to FIND the target if needed.

        R38 — robust to a FAR arrival heading. Perceives at the current heading
        first; if that yields no localizable object OR an IMPLAUSIBLE one (planar
        distance from the dog > _SCAN_MAX_LOCAL_M — i.e. a far wall, not the
        near-table can), it rotates the base in place by _SCAN_STEP_RAD up to
        _SCAN_MAX_STEPS times, perceiving at each heading, and keeps the CLOSEST
        plausible perception. The dog is left at the best heading. Returns the same
        ``(gp, resolved, fail)`` tuple as ``_perceive_grasp_point``.

        Honest: every candidate point comes from real depth+mask (never GT); the
        scan only decides WHICH camera direction to trust by how near/plausible the
        perceived object is. No steerable base (or no get_position/get_heading) ->
        a single perceive, byte-identical to the pre-R38 path (so the in-process
        spawn grasp is unchanged).
        """
        import math

        walk = getattr(base, "walk", None) if base is not None else None
        get_pos = getattr(base, "get_position", None) if base is not None else None
        get_hd = getattr(base, "get_heading", None) if base is not None else None
        steerable = callable(walk) and callable(get_pos) and callable(get_hd)

        def _plausible(gp: Any) -> tuple[bool, float]:
            """(is a near-table object, planar distance from the dog)."""
            if gp is None or not steerable:
                return (gp is not None, 0.0)
            try:
                pos = get_pos()
                d = math.hypot(gp.x - pos[0], gp.y - pos[1])
            except Exception:  # noqa: BLE001
                return (True, 0.0)
            return (d <= _SCAN_MAX_LOCAL_M, d)

        # First look at the current heading.
        gp, resolved, fail = self._perceive_grasp_point(perception, query, color=color, scene=scene)
        ok, dist = _plausible(gp)
        if ok or not steerable:
            if gp is not None:
                logger.info("[PGRASP] perceive (no scan): plausible target d=%.2fm", dist)
            return gp, resolved, fail

        logger.info("[PGRASP] perceive at arrival heading implausible "
                    "(d=%.2fm > %.2fm) — scanning to find the target",
                    dist if gp is not None else -1.0, _SCAN_MAX_LOCAL_M)

        best = (gp, resolved, fail, dist if gp is not None else float("inf"))
        for step in range(_SCAN_MAX_STEPS):
            try:
                walk(vx=0.0, vy=0.0, vyaw=_SCAN_TURN_VYAW,
                     duration=_SCAN_STEP_RAD / _SCAN_TURN_VYAW)
                import time as _t
                _t.sleep(0.3)  # let the camera settle on the new heading
            except Exception as exc:  # noqa: BLE001
                logger.warning("[PGRASP] scan turn raised: %s", exc)
                break
            g2, r2, f2 = self._perceive_grasp_point(perception, query, color=color, scene=scene)
            ok2, d2 = _plausible(g2)
            logger.info("[PGRASP] scan step %d/%d: %s d=%.2fm plausible=%s",
                        step + 1, _SCAN_MAX_STEPS,
                        "found" if g2 is not None else "none",
                        d2 if g2 is not None else -1.0, ok2)
            if g2 is not None and d2 < best[3]:
                best = (g2, r2, f2, d2)
            if ok2:
                return g2, r2, f2  # a plausible near-table object — take it

        # No PLAUSIBLE near-table target found in the full sweep. Do NOT return a far
        # phantom point: the downstream re-pose/approach would chase it across the
        # room (observed: a 8 m back-projection drove the dog 4.8 m away). FAIL LOUD
        # instead — the target is not perceivable from this arrival framing (the
        # can is out of the d435 FOV / occluded). This is honest: no GT substitute,
        # and it surfaces the real cause (perception, not IK) to the replan context.
        if best[0] is None or best[3] > _SCAN_MAX_LOCAL_M:
            return None, best[1], _fail(
                "no_detections",
                f"Scanned {_SCAN_MAX_STEPS} headings; no plausible near-table target "
                f"for {query!r} within {_SCAN_MAX_LOCAL_M:.1f} m (closest seen "
                f"{best[3]:.1f} m). The object is not perceivable from this arrival "
                "framing — not chasing a far phantom point.",
                query=query,
            )
        return best[0], best[1], best[2]

    def _perceive_grasp_point(
        self, perception: Any, query: str, *, color: str | None = None, scene: str | None = None,
    ):
        """Acquire RGB-D, resolve the target mask, compute the WORLD grasp point.

        Returns ``(gp | None, resolved_label, fail_result | None)``. Pure perception:
        the 3D point is from real depth + mask, NEVER a ground-truth pose. Callable
        twice (before/after an approach) so the point tracks the live camera pose.

        ATTRIBUTE grasp (D47): when *color* is given, the front-object resolver selects
        the salient blob of that colour (FAIL LOUD if none). The resolved LABEL is
        *scene* (the world-manifest body the query names) when given — the grasp POINT
        remains from depth+mask.
        """
        import numpy as np
        from vector_os_nano.perception.object_localizer import capture_rgbd
        try:
            rgb, depth, intrinsics, cam_xpos, cam_xmat = capture_rgbd(perception)
        except Exception as exc:  # noqa: BLE001
            return None, (query or "front object"), _fail("no_camera", f"Failed to read RGB-D frame: {exc}")

        # MODEL ROUTING (route each instruction to the right model+skill):
        #   - COLOUR query ("绿色的瓶子"/pure "抓红色的", color is not None)
        #     -> classical front_object HSV colour resolver (CHANGE 1, R40).
        #     The D47 HSV resolver had 100% colour selection among 3 close cans;
        #     grounding-dino is intermittent on a dense, close-packed scene.
        #     A colour target disambiguates by hue — use the right cheap tool.
        #   - DEICTIC ("前面的东西"/generic, no noun, no colour) -> classical
        #     front_object resolver (geometry / depth).
        #   - NAMED no-colour ("罐子") -> the LEARNED DETECTOR (grounding-dino).
        # When both a noun AND a colour are present ("绿色的瓶子"), the colour wins:
        # use_front_resolver=True so the HSV path is taken regardless of the noun.
        named = _names_object(query)
        deictic = not named
        have_front = hasattr(perception, "front_object_mask")
        # CHANGE 1 (R40): colour query forces the HSV resolver route even when a
        # noun is also present — colour disambiguates better than text grounding on
        # a dense, same-category scene. Keep the existing deictic path unchanged.
        use_front_resolver = have_front and (deictic or color is not None)
        classical = deictic  # kept for the mask-fallback and diagnostics below
        mask = None
        # A query that names a world-manifest object labels the verify with its body
        # name (the grasp POINT is still perceived); otherwise the query text.
        resolved = scene or (query or "front object")
        detection_found = False
        if use_front_resolver:
            try:
                mask = perception.front_object_mask(rgb, depth, color=color)
                _mask_px = int(np.count_nonzero(mask)) if mask is not None else 0
                _d_valid = int((depth > 0).sum()) if depth is not None else -1
                _d_near = int(((depth > 0) & (depth <= 2.0)).sum()) if depth is not None else -1
                _d_min = float(depth[depth > 0].min()) if depth is not None and (depth > 0).any() else -1
                _d_med = float(np.median(depth[depth > 0])) if depth is not None and (depth > 0).any() else -1
                logger.info(
                    "[PGRASP] front_object_mask: mask_px=%d valid_depth=%d "
                    "near_depth(<=2m)=%d d_min=%.3f d_med=%.3f",
                    _mask_px, _d_valid, _d_near, _d_min, _d_med,
                )
            except Exception as exc:  # noqa: BLE001
                return None, resolved, _fail(
                    "perception_failed", f"The front-object resolver failed: {exc}",
                    query=query, cause=str(exc))
        else:
            try:
                from vector_os_nano.perception.object_localizer import detect_on
                detections = detect_on(perception, rgb, query)
            except Exception as exc:  # noqa: BLE001 — a missing model is not "found nothing"
                return None, resolved, _fail(
                    "perception_failed",
                    f"The object detector failed on {query!r}: {type(exc).__name__}: {exc}. "
                    "Nothing was grasped (never a substitute target).", query=query, cause=str(exc))
            if detections:
                detection_found = True
                # PERCEPTUAL colour selection (D48 caveat 1): grounding-dino is
                # colour-conditioned, so when a colour is named, prefer the box whose
                # detector LABEL actually contains that colour over raw max-confidence
                # (so "红色的罐子" picks the RED box even if a green box scores higher —
                # selection by perception, not by scene centrality). Among the boxes
                # whose label matches the colour, take the highest-confidence one. If
                # none match the colour (the prompt may have under-grounded the
                # adjective), fall back to plain max-confidence. No colour → plain
                # max-confidence, unchanged.
                det = self._select_detection(detections, color)
                # Verify LABEL: when a colour is named ("红色的罐子") the verify oracle
                # grades the manifest body (pickable_can_red); otherwise the
                # detector's own label. The grasp POINT is still from depth+mask.
                resolved = scene or str(getattr(det, "label", query) or query)
                try:
                    mask = perception.segment(rgb, det.bbox)
                except Exception as exc:  # noqa: BLE001 — the segmenter's own failure, with its cause
                    return None, resolved, _fail("perception_failed", f"The segmenter failed for {resolved!r}: "
                                                 f"{type(exc).__name__}: {exc}",
                                                 detection_label=resolved, query=query, cause=str(exc))

        if mask is None or int(np.count_nonzero(mask)) == 0:
            if detection_found:
                return None, resolved, _fail("segmentation_failed",
                                             f"Segmentation produced no mask for {resolved!r}.",
                                             detection_label=resolved, query=query)
            return None, resolved, _fail("no_detections",
                                         f"Nothing localizable for {query!r} "
                                         f"({'no salient object in front' if deictic else 'VLM found nothing'}).",
                                         query=query)

        gp = grasp_point_from_rgbd(depth, rgb, mask, intrinsics, cam_xpos, cam_xmat)
        if gp is not None:
            logger.info(
                "[PGRASP] grasp_point_from_rgbd → xyz=(%.3f, %.3f, %.3f) "
                "cam_xpos=(%.3f, %.3f, %.3f)",
                gp.x, gp.y, gp.z,
                float(cam_xpos[0]), float(cam_xpos[1]), float(cam_xpos[2]),
            )
        if gp is None:
            return None, resolved, _fail(
                "no_depth_points",
                f"No valid depth points under the {resolved!r} mask; cannot localize a grasp point. "
                "FAIL LOUD — not substituting a ground-truth pose.",
                detection_label=resolved, query=query,
            )
        # CHANGE 2 (R40): low-z collapse guard. A back-projected z below _MIN_GRASP_Z
        # is the table/floor surface, NOT the top of a can (cans sit at z≈0.32). A
        # degraded mask whose depth hits the table produces this and must FAIL LOUD so
        # the step grades RAN, never a false GROUNDED (R39 t3: z=0.039 accepted).
        if gp.z < _MIN_GRASP_Z:
            return None, resolved, _fail(
                "low_z_backprojection",
                f"Back-projected z={gp.z:.3f}m < _MIN_GRASP_Z={_MIN_GRASP_Z}m "
                f"for {resolved!r}; the mask hit the table/floor, not the object top. "
                "FAIL LOUD — rejecting a floor-level grasp point.",
                detection_label=resolved, query=query, actual_z=gp.z,
            )
        return gp, resolved, None
