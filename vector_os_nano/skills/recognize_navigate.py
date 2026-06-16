# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""RecognizeNavigateSkill — reliable arrival: VLM recognise → navigate (#9 R3).

The recognize→navigate pivot (tricky Case 21): visually servoing the last metres
on a slow/noisy VLM is flaky for every embodiment. Instead:

  1. RECOGNISE — a real VLM grounds the object class and gives its bearing.
  2. LOCATE   — the lidar gives the range to the nearest surface in that bearing;
                bearing + range → a world (x, y) (sensor-derived, never GT).
  3. NAVIGATE — the base's RELIABLE navigate_to (odometry + planner, obstacle-
                aware, collision-stop) drives there. No pixel servoing.

If the object is beyond lidar range (it can be recognised farther than it can be
ranged), the robot briefly ADVANCES toward the recognised bearing until the lidar
locks on, then hands off to navigate_to. Honest (rule 5): the VLM is really called;
the position is sensed, not GT; a never-recognised target FAILS; the kernel's
at_position(object, tol) verify is the ground-truth judge.

Base-generic: any base exposing get_camera_frame + get_lidar_scan + get_position +
get_heading + navigate_to can run it. Bases without navigate_to fail loudly.
"""
from __future__ import annotations

import logging
import math
import time

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.perception.target_locate import (
    locate_from_bearing,
    locate_from_depth,
)
# Import the MODULE so VlmRateLimitError is resolved LIVE in the `except`
# (importlib.reload(vlm_go2) rebinds the class; a cached binding would miss it).
from vector_os_nano.perception import vlm_go2 as _vlm_go2
from vector_os_nano.perception.vlm_targets import VlmTargetDetector

logger = logging.getLogger(__name__)

_ALIGN_TOL = 0.18        # |x_norm| within this = aligned enough to advance
_TURN_VYAW = 0.5
_FWD_VX = 0.45
_ADVANCE_DUR = 1.0       # short advance burst to close into lidar range
_TURN_DUR = 0.5
_MISS_BEFORE_SWEEP = 10  # the VLM detects a far/small target only intermittently
                         # — on a miss, HOLD heading and re-query (the target is
                         # still where it was) MANY times before moving; turning
                         # away on the first miss never re-acquires it (R3)
_ARRIVE_TOL = 0.8        # navigate_to standoff to a located object surface (the
                         # collidable object halts the base at a body radius; a
                         # tight tol would falsely report not-reached)
_STANDOFF_M = 0.7        # navigate to a point this far IN FRONT of the located
                         # surface — the surface itself can be in the planner's
                         # inflated wall/object zone (no path → inf); a standoff
                         # in front is reachable and leaves the robot at the object
_MAX_BLIND_ADVANCES = 4  # cap forward advances taken WITHOUT a lidar lock so a
                         # never-detected run cannot blind-walk past the target
_LOCATE_AGREE_M = 1.2    # two located estimates within this confirm the target —
                         # a single bad VLM bbox gives an outlier estimate that
                         # will NOT repeat, so require agreement before navigating
_MAX_RL_STREAK = 3       # consecutive VLM 429s (rate-limit) before giving up —
                         # a throttled endpoint is NOT "object absent", so fail
                         # honestly (vlm_unavailable) WITHOUT blind-walking the
                         # robot or hammering the endpoint for the full max_iters
_RL_BACKOFF_S = 2.0      # backoff between consecutive rate-limit retries (mirrors
                         # recognize_pick._DETECT_BACKOFF_S); single sleep site


@skill(
    aliases=["recognise and go", "认物体走过去", "找到走到", "去找并走到"],
    direct=False,
)
class RecognizeNavigateSkill:
    """Recognise an object with the VLM, locate it with the lidar, and navigate
    there reliably (no visual servoing)."""

    name: str = "recognize_navigate"
    description: str = (
        "COMPLETE one-step 'find a furniture object and go to it': this single "
        "skill RECOGNISES the object with a real vision-language model AND "
        "navigates all the way to it (depth-located + reliable navigate_to). "
        "Use this ALONE for any 'find/recognise the chair/sofa/plant and walk/go "
        "to it / 认出X走过去' request — do NOT add a separate recognition or "
        "vlm_seek step before it (it already recognises internally). Preferred "
        "over vlm_seek for actually ARRIVING at a furniture object."
    )
    typical_duration_sec: float = 200.0
    verify_hint: str = (
        "visited('<label>') with the recognised object's class name (e.g. "
        "visited('chair')) — pass ONLY the label string, NOT coordinates. The "
        "predicate resolves the object's coordinates from the world model itself "
        "and is true when the robot is within 1.6 m of it. This is the ground-"
        "truth judge that the robot reached the object the VLM recognised — the "
        "planner must NOT leave x/y unbound (use visited(label), not "
        "at_position(x, y), so there is no coordinate to bind)."
    )
    parameters: dict = {
        "label": {"type": "string", "required": True,
                  "description": "Target object class (e.g. 'chair' / '椅子')."},
        "max_iters": {"type": "number", "required": False, "default": 16,
                      "description": "Max recognise/advance cycles before navigate."},
    }
    preconditions: list = ["base with camera + lidar + navigate_to (room mode)"]
    postconditions: list = []
    effects: dict = {}
    failure_modes: list = ["no_base", "no_camera", "no_navigate", "not_found",
                           "not_located", "vlm_unavailable"]

    def __init__(self, detector: "VlmTargetDetector | None" = None) -> None:
        self._detector = detector or VlmTargetDetector()

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        base = context.base
        if base is None:
            return SkillResult(success=False, error_message="no mobile base connected",
                               result_data={"diagnosis": "no_base"})
        if not callable(getattr(base, "get_camera_frame", None)):
            return SkillResult(success=False,
                               error_message="base has no camera (needs a room scene)",
                               result_data={"diagnosis": "no_camera"})
        if not callable(getattr(base, "navigate_to", None)):
            return SkillResult(
                success=False,
                error_message=f"base {type(base).__name__} has no navigate_to — "
                              "recognize_navigate needs the reliable controller",
                result_data={"diagnosis": "no_navigate"})
        query = str(params.get("label", "")).strip()
        if not query:
            return SkillResult(success=False, error_message="needs a target label",
                               result_data={"diagnosis": "bad_params"})
        max_iters = int(params.get("max_iters", 16) or 16)
        _heading = getattr(base, "get_heading", None)

        # Prefer an atomic observation (rgb + depth + cam pose) so the target can
        # be located by DEPTH-AT-BBOX (semantic — the object's own pixels, skips
        # an intervening obstacle, Case 22). Bases without depth fall back to the
        # lidar bearing+range (which can mis-pick an obstacle).
        _has_obs = callable(getattr(base, "get_camera_observation", None))
        located_by = None
        seen = False
        located = None
        prev_located = None     # last estimate, awaiting a consistent confirm
        miss_streak = 0
        rl_streak = 0           # consecutive VLM rate-limit (429) errors
        blind_advances = 0      # forward advances taken without a lidar lock
        for _ in range(max_iters):
            try:
                if _has_obs:
                    obs = base.get_camera_observation()
                    frame = obs["rgb"]
                else:
                    obs = None
                    frame = base.get_camera_frame()
            except Exception as exc:  # noqa: BLE001
                return SkillResult(success=False, error_message=f"camera failed: {exc}",
                                   result_data={"diagnosis": "no_camera"})
            try:
                dets = self._detector.detect_targets(
                    frame, query, raise_on_rate_limit=True)
            except _vlm_go2.VlmRateLimitError:
                # The VLM endpoint is throttled — NOT "object not in frame".
                # Do NOT blind-walk or hammer the endpoint: count consecutive
                # 429s, back off, and after _MAX_RL_STREAK give up honestly.
                rl_streak += 1
                if rl_streak >= _MAX_RL_STREAK:
                    break
                time.sleep(_RL_BACKOFF_S)
                continue        # skip det/miss/walk handling → ZERO walk on 429
            rl_streak = 0       # any non-raising return resets the streak
            det = max(dets, key=lambda d: d["area_frac"]) if dets else None
            if det is None:
                # The VLM detects a far/small target only intermittently. HOLD
                # heading and re-query for a few frames (the target hasn't moved);
                # then ADVANCE FORWARD to close distance and grow the view (the
                # furnished-room targets are ahead of the spawn) — a far object
                # becomes detectable + rangeable as the robot nears it. A sweep
                # every few advances catches a genuinely off-axis target. Turning
                # away on the first miss would lose a dead-ahead target (R3).
                miss_streak += 1
                if miss_streak >= _MISS_BEFORE_SWEEP:
                    if miss_streak % 3 == 0 or blind_advances >= _MAX_BLIND_ADVANCES:
                        # sweep (and once the blind-advance budget is spent, ONLY
                        # sweep — never blind-walk past where the target should be)
                        base.walk(0.0, 0.0, _TURN_VYAW, duration=_TURN_DUR)
                    else:
                        base.walk(_FWD_VX, 0.0, 0.0, duration=_ADVANCE_DUR)
                        blind_advances += 1
                continue
            miss_streak = 0
            seen = True
            x_norm = float(det["x_norm"])
            y_norm = float(det.get("y_norm", 0.0))
            pos = base.get_position()
            heading = float(_heading()) if callable(_heading) else 0.0
            # 1) DEPTH-AT-BBOX (preferred, semantic): depth at the recognised
            #    pixel = the object's distance, skipping intervening obstacles.
            located = None
            if obs is not None and obs.get("depth") is not None:
                located = locate_from_depth(
                    x_norm, y_norm, obs["depth"], obs["cam_pos"],
                    obs["cam_mat"], obs["fovy"])
                if located is not None:
                    located_by = "depth"
            # 2) FALLBACK: lidar range at the recognised bearing (Case 22 risk —
            #    may pick an obstacle; only used when no depth is available).
            if located is None:
                scan = base.get_lidar_scan() if callable(
                    getattr(base, "get_lidar_scan", None)) else None
                pts = getattr(scan, "points", None) if scan is not None else None
                located = locate_from_bearing((pos[0], pos[1]), heading, x_norm, pts)
                if located is not None:
                    located_by = "lidar"
            if located is not None:
                # Confirm with a second AGREEING estimate — a single bad VLM bbox
                # yields an outlier that will not repeat, so don't navigate to it.
                if prev_located is not None and math.hypot(
                        located[0] - prev_located[0],
                        located[1] - prev_located[1]) <= _LOCATE_AGREE_M:
                    located = ((located[0] + prev_located[0]) / 2.0,
                               (located[1] + prev_located[1]) / 2.0)
                    break                   # confirmed → go navigate
                prev_located = located      # first/outlier estimate — re-confirm
                located = None
                continue
            # recognised but beyond lidar range: face the bearing and advance a
            # short burst to close in, then re-recognise + re-range.
            if abs(x_norm) > _ALIGN_TOL:
                base.walk(0.0, 0.0,
                          -_TURN_VYAW if x_norm > 0 else _TURN_VYAW, duration=_TURN_DUR)
            else:
                base.walk(_FWD_VX, 0.0, 0.0, duration=_ADVANCE_DUR)
        try:
            base.stop()
        except Exception as exc:  # noqa: BLE001
            logger.debug("recognize_navigate teardown stop() failed: %s", exc)

        # Honest giveup on a throttled VLM (returns BEFORE the not_found branch so
        # a rate-limit is never mislabelled "object absent"). The robot is halted
        # (base.stop() above), never left mid-gait.
        if rl_streak >= _MAX_RL_STREAK and located is None:
            return SkillResult(
                success=False,
                error_message=(f"VLM unavailable (rate limited {rl_streak}x) — "
                               f"could not recognise the {query}; not blind-walking"),
                result_data={"diagnosis": "vlm_unavailable", "seen": seen,
                             "query": query, "rl_streak": rl_streak})

        if located is None:
            return SkillResult(
                success=False,
                error_message=(f"could not find a {query} by camera (VLM)" if not seen
                               else f"recognised the {query} but could not range it"),
                result_data={"diagnosis": "not_found" if not seen else "not_located",
                             "seen": seen, "query": query})

        # Reliable arrival: navigate to a STANDOFF point just IN FRONT of the
        # located surface, not the surface itself. The located point sits on the
        # object (often near the back wall behind it); navigating onto it can be
        # unreachable (the planner inflates the wall/object → no path → inf). A
        # point _STANDOFF_M back toward the robot is reachable and leaves the
        # robot at the object (within the at_position(.,1.6) verify intent).
        rp = base.get_position()
        dx, dy = float(located[0]) - rp[0], float(located[1]) - rp[1]
        dlen = math.hypot(dx, dy)
        if dlen > _STANDOFF_M:
            gx = float(located[0]) - _STANDOFF_M * dx / dlen
            gy = float(located[1]) - _STANDOFF_M * dy / dlen
        else:
            gx, gy = rp[0], rp[1]           # already within standoff of the object
        out = base.navigate_to(gx, gy, _ARRIVE_TOL)
        reached = bool(out.get("reached", False))
        remaining = float(out.get("remaining", float("inf")))
        pos = list(base.get_position()) if callable(getattr(base, "get_position", None)) else None
        return SkillResult(
            success=reached,
            error_message="" if reached else (
                f"navigation stopped {remaining:.2f}m short of the located {query}"),
            result_data={"query": query, "seen": True, "located": list(located),
                         "located_by": located_by,
                         "reached": reached, "remaining_m": remaining,
                         "position": pos, "transport": "recognize_navigate",
                         "diagnosis": "ok" if reached else "nav_stuck"},
        )
