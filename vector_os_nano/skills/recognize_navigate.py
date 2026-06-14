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

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.perception.target_locate import locate_from_bearing
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
_MAX_BLIND_ADVANCES = 4  # cap forward advances taken WITHOUT a lidar lock so a
                         # never-detected run cannot blind-walk past the target


@skill(
    aliases=["recognise and go", "认物体走过去", "找到走到", "去找并走到"],
    direct=False,
)
class RecognizeNavigateSkill:
    """Recognise an object with the VLM, locate it with the lidar, and navigate
    there reliably (no visual servoing)."""

    name: str = "recognize_navigate"
    description: str = (
        "Reliably find a SEMANTIC object (chair/sofa/potted plant/...) and walk "
        "to it: recognise it with a real vision-language model, locate it with "
        "the lidar, then drive there with the reliable navigate_to controller. "
        "Prefer this over vlm_seek for getting all the way to a furniture object "
        "— recognition gates the goal, navigation makes arrival robust."
    )
    typical_duration_sec: float = 200.0
    verify_hint: str = (
        "at_position(x, y, 1.6) with the named object's coordinates — the "
        "ground-truth judge that the robot reached the object the VLM "
        "recognised and the lidar located (recognition + sensing are the means)"
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
    failure_modes: list = ["no_base", "no_camera", "no_navigate", "not_found", "not_located"]

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

        seen = False
        located = None
        miss_streak = 0
        blind_advances = 0      # forward advances taken without a lidar lock
        for _ in range(max_iters):
            try:
                frame = base.get_camera_frame()
            except Exception as exc:  # noqa: BLE001
                return SkillResult(success=False, error_message=f"camera failed: {exc}",
                                   result_data={"diagnosis": "no_camera"})
            dets = self._detector.detect_targets(frame, query)
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
            pos = base.get_position()
            heading = float(_heading()) if callable(_heading) else 0.0
            scan = base.get_lidar_scan() if callable(
                getattr(base, "get_lidar_scan", None)) else None
            pts = getattr(scan, "points", None) if scan is not None else None
            located = locate_from_bearing((pos[0], pos[1]), heading, x_norm, pts)
            if located is not None:
                break                       # lidar locked on → go navigate
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

        if located is None:
            return SkillResult(
                success=False,
                error_message=(f"could not find a {query} by camera (VLM)" if not seen
                               else f"recognised the {query} but could not range it"),
                result_data={"diagnosis": "not_found" if not seen else "not_located",
                             "seen": seen, "query": query})

        # Reliable arrival: hand the sensed location to navigate_to. The located
        # point is the object's FRONT SURFACE (a lidar hit), and the object is
        # collidable — the base halts at a body-radius standoff, so a tight tol
        # would report not-reached though the robot is AT the object. Use an
        # object STANDOFF tol (matches the at_position(.,1.6) verify intent).
        out = base.navigate_to(float(located[0]), float(located[1]), _ARRIVE_TOL)
        reached = bool(out.get("reached", False))
        remaining = float(out.get("remaining", float("inf")))
        pos = list(base.get_position()) if callable(getattr(base, "get_position", None)) else None
        return SkillResult(
            success=reached,
            error_message="" if reached else (
                f"navigation stopped {remaining:.2f}m short of the located {query}"),
            result_data={"query": query, "seen": True, "located": list(located),
                         "reached": reached, "remaining_m": remaining,
                         "position": pos, "transport": "recognize_navigate",
                         "diagnosis": "ok" if reached else "nav_stuck"},
        )
