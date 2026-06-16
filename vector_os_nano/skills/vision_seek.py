# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""VisionSeekSkill — find & approach a coloured target by RECOGNITION (R10).

The last piece of req #5: the robot LOCATES the target with its camera (not
ground-truth coordinates) and drives to it. The loop is camera → detect_targets
→ decide → actuate:
  - target not seen   → scan (turn in place to sweep the view)
  - seen, off-centre  → turn toward it (by the blob bearing x_norm)
  - seen, centred     → walk forward
  - seen, large       → arrived (the blob fills enough of the frame = close)
Honest (rule 5): if the target is never recognised it FAILS, never falls back
to GT coordinates. The kernel's at_position(<real target>, tol) verify is the
ground-truth judge; recognition is only the MEANS of getting there.
"""
from __future__ import annotations

import logging
import math

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.perception.color_targets import detect_targets

logger = logging.getLogger(__name__)

_ALIGN_TOL = 0.18      # |x_norm| within this = centred enough to walk
# A 0.24 m target box fills ~6-8% of the frame at ~0.7 m — the standoff at
# which we call it 'arrived' (well inside the at_position 1.6 m verify tol).
_ARRIVE_AREA = 0.04
_TURN_VYAW = 0.5       # rad-ish command for aligning / scanning
_FWD_VX = 0.45
_STALL_WINDOW_STEPS = 8    # progress-stall arrival: window of recent positions
_STALL_PROGRESS_M = 0.6    # net travel below this over the window = arrived (near-target orbit)
_MIN_APPROACH_M = 0.6    # robot must translate this far from start before
                         # a progress-stall can count as arrival (vs scan-in-place)
_SCAN_ADVANCE_EVERY = 5   # every Nth scan tick, step forward (not turn) so a
                         # far target enters the down-pitched FOV
_TURN_STEP = 0.5         # deadman re-arm for a turn/scan tick (short — a small
                         # incremental bearing correction, then re-perceive;
                         # forward uses the loop's longer step_duration)
_COAST_MISSES = 4        # keep heading toward the last seen bearing for this
                         # many consecutive missed frames before search-scanning
# Closed-loop heading control (campaign #9 R2/R3, Go2 only — gated by the base's
# `seek_heading_hold` attr). A blocking sinusoidal trot yaw-drifts ~40-60°/2s
# with no feedback, so open-loop fixed-duration turns over/under-shoot and the
# robot wanders. Fix: latch the target's ABSOLUTE world heading at detection,
# then P-correct on base.get_heading() every tick (cheap, runs between the slow
# VLM calls). g1 has no `seek_heading_hold` → keeps the open-loop path verbatim.
_CAM_HALF_FOV = 0.80     # rad — half the recog cam's 75° FOV (g1_room RECOG_CAM);
                         # maps x_norm∈[-1,1] to a bearing offset
_TURN_ONLY_RAD = 0.35    # |heading error| above this = pure turn (too off to walk);
                         # at/below it, walk forward WHILE carrying the yaw trim so
                         # the robot makes progress and curves in (a pure-P turn has
                         # steady-state error under persistent gait drift — walking
                         # while correcting nets forward progress where halting stalls)
_K_YAW = 1.8             # P gain (rad/s per rad of heading error)
_VYAW_CAP = 0.8          # max corrective yaw — never exceed the gait's authority
_FWD_BURST_S = 0.6       # heading-hold forward burst cap → re-sample heading
                         # ~3× faster than the gait drift accrues
_ALIASES = {"红": "red", "蓝": "blue", "绿": "green"}


def _seek_action(det: "dict | None",
                 arrive_area: float = _ARRIVE_AREA) -> "tuple[str, float, float]":
    """Pure decision: given the target detection (or None), return
    (action, vx, vyaw). vyaw>0 = turn left, vyaw<0 = turn right (G1 convention).
    x_norm>0 = target on the robot's right → turn right (vyaw<0).

    ``arrive_area`` is the frame-fill fraction that counts as 'arrived'. The
    colour detector's blob area is a clean range proxy (default 0.04). A VLM
    bounding box is NOT — it is noisy and occasionally oversized at range (a
    0.046 box at 3.6 m, campaign #9 R1), so the VLM path raises this bar near 1
    and leans on the physical progress-stall (collision-blocked) for arrival."""
    if det is None:
        return ("scan", 0.0, _TURN_VYAW)          # sweep left to find it
    if det["area_frac"] >= arrive_area:
        return ("arrived", 0.0, 0.0)
    x = float(det["x_norm"])
    if abs(x) > _ALIGN_TOL:
        return ("turn", 0.0, -_TURN_VYAW if x > 0 else _TURN_VYAW)
    return ("forward", _FWD_VX, 0.0)


def _resolve_color(label: str) -> str:
    key = (label or "").strip().lower()
    for zh, en in _ALIASES.items():
        if zh in key:
            return en
    for c in ("red", "blue", "green"):
        if c in key:
            return c
    return key


def _seek_loop(base, perceive, max_iters: int, step_duration: float = 0.4,
               arrive_area: float = _ARRIVE_AREA
               ) -> "tuple[bool, str, list | None]":
    """Shared perceive→decide→act loop (campaign #9 R1 — detector-agnostic).

    ``perceive(frame) -> det | None`` is the ONLY pluggable part: the colour
    detector filters detect_targets by colour; the VLM detector grounds a
    semantic class. The state machine (scan-with-advance / turn / forward /
    progress-stall arrival) is identical for both — recognition is the means,
    arrival is the truth (rule 5: a never-seen target FAILS, no GT fallback).

    ``step_duration`` is the walk deadman re-arm per perceive-act tick. It MUST
    exceed the perceive latency or the gait stutters: a ~2 s VLM call with a
    0.4 s deadman leaves the robot stopped ~1.6 s of every tick, so 8 ticks
    cover <0.6 m of genuine forward walking and the progress-stall fires a FALSE
    arrival metres short (the campaign #9 R1 bug). Instant detectors (colour)
    keep 0.4 s; the VLM detector passes ~3 s so motion stays continuous across
    the call and the robot only stalls when physically blocked at the target.
    Returns (seen, reason, final_position)."""
    seen = False
    reason = "not_found"
    _has_pos = callable(getattr(base, "get_position", None))
    start_pos = base.get_position() if _has_pos else [0.0, 0.0, 0.0]
    window: list = []      # recent positions, for progress-stall arrival
    scan_steps = 0         # consecutive scan (target-not-seen) ticks
    last_bearing = None    # last seen x_norm — bridges intermittent detection
    miss_streak = 0        # consecutive frames with no detection
    # Closed-loop heading control is enabled ONLY for a base that opts in via
    # `seek_heading_hold` (Go2). The gate is the ATTR, NOT get_heading() — g1
    # ALSO exposes get_heading(), so keying on the method would regress g1's
    # open-loop orbit-arrival (tricky Case 19). theta_goal is the latched
    # absolute world heading toward the target.
    _hold = (bool(getattr(base, "seek_heading_hold", False))
             and callable(getattr(base, "get_heading", None)))
    theta_goal = None
    for _ in range(max_iters):
        frame = base.get_camera_frame()
        det = perceive(frame)
        if det is not None:
            seen = True
            miss_streak = 0
            last_bearing = float(det["x_norm"])
            if _hold:
                # Latch the target's ABSOLUTE world heading: x_norm>0 = target on
                # the right = lower yaw (yaw+ = left). Robust to the ~2 s VLM
                # latency — feedback runs off the latched goal + cheap heading.
                theta_goal = base.get_heading() - float(det["x_norm"]) * _CAM_HALF_FOV
        else:
            miss_streak += 1

        if _hold and theta_goal is not None and (
                det is None or det["area_frac"] < arrive_area):
            # P-controller on measured base yaw: command shrinks as the error
            # shrinks (no overshoot/spin-past), corrected every tick between the
            # slow perceives. Turn until aligned, then walk forward carrying a
            # small live yaw trim so gait drift during the forward burst is
            # continuously corrected.
            err = math.atan2(math.sin(theta_goal - base.get_heading()),
                             math.cos(theta_goal - base.get_heading()))
            vyaw = max(-_VYAW_CAP, min(_VYAW_CAP, _K_YAW * err))
            if abs(err) > _TURN_ONLY_RAD:
                action, vx = "turn", 0.0
            else:
                action, vx = "forward", _FWD_VX   # walk forward + live yaw trim
        elif det is not None:
            action, vx, vyaw = _seek_action(det, arrive_area)
        else:
            # COAST (campaign #9 R1): a slow VLM detects the target only
            # intermittently — gait sway / motion blur drops frames even while
            # the target is dead ahead. Instead of immediately spinning away
            # (which loses a target that is really still in front), keep heading
            # toward the LAST seen bearing for a few missed frames; only fall
            # back to a search-scan once the target is genuinely lost.
            if seen and last_bearing is not None and miss_streak <= _COAST_MISSES:
                if abs(last_bearing) > _ALIGN_TOL:
                    action = "turn"
                    vx, vyaw = 0.0, (-_TURN_VYAW if last_bearing > 0 else _TURN_VYAW)
                else:
                    action, vx, vyaw = "forward", _FWD_VX, 0.0
            else:
                action, vx, vyaw = _seek_action(None)   # ("scan", 0, +vyaw)
        if action == "arrived":     # blob filled the frame (close)
            reason = "arrived"
            break
        # Scan that sweeps AND advances: a far target straight ahead can sit
        # outside the down-pitched camera's vertical FOV — turning in place
        # never brings it in. Every _SCAN_ADVANCE_EVERY scan ticks, step
        # forward instead of turning so far/low targets enter the view.
        if action == "scan":
            scan_steps += 1
            if scan_steps % _SCAN_ADVANCE_EVERY == 0:
                action, vx, vyaw = "forward", _FWD_VX, 0.0
        else:
            scan_steps = 0
        # Progress-stall arrival: a small/low target clips at the frame edge up
        # close, so pixel/box area never crosses the threshold. Honest signal is
        # PHYSICAL: once the robot has SEEN the target AND APPROACHED it
        # (translated _MIN_APPROACH_M from start — so scanning in place far away
        # can NEVER falsely arrive) AND then stops making net progress, it has
        # walked up to the object → arrived.
        # Sampling is EMBODIMENT-SCOPED (Case 19): g1 (open-loop, _hold=False)
        # samples EVERY tick — it arrives by ORBITING the target, so turn ticks
        # near it MUST count (the R1 0.58 m arrival). A heading-hold base (Go2,
        # _hold=True) yaw-drifts and turns far from the target during approach,
        # so it samples ONLY forward ticks — else the non-translating turn/scan
        # ticks fill the window and fire a FALSE arrival metres short.
        _stall_sample = (not _hold) or (action == "forward")
        if _has_pos and seen and _stall_sample:
            p = base.get_position()
            window.append((float(p[0]), float(p[1])))
            approached = math.hypot(p[0] - start_pos[0],
                                    p[1] - start_pos[1]) > _MIN_APPROACH_M
            if approached and len(window) > _STALL_WINDOW_STEPS:
                window.pop(0)
                net = math.hypot(window[-1][0] - window[0][0],
                                 window[-1][1] - window[0][1])
                if net < _STALL_PROGRESS_M:   # no net progress → reached
                    reason = "arrived"
                    break
        # Per-action deadman: only FORWARD wants the long re-arm (keep walking
        # across the perceive latency). TURN/SCAN must stay SHORT — a long turn
        # at _TURN_VYAW overshoots a small bearing error by tens of degrees and
        # flings the target off-screen (the campaign #9 R1 over-rotate bug); a
        # short turn makes an incremental correction and re-perceives.
        dur = step_duration if action == "forward" else min(step_duration, _TURN_STEP)
        if _hold and action == "forward":
            dur = min(dur, _FWD_BURST_S)   # short bursts → re-sample heading often
        base.walk(vx, 0.0, vyaw, duration=dur)
    try:
        base.stop()
    except Exception as exc:  # noqa: BLE001
        logger.debug("seek teardown stop() failed: %s", exc)
    pos = list(base.get_position()) if _has_pos else None
    return seen, reason, pos


@skill(
    aliases=["find", "找到", "找", "看到并走到", "找到并走到", "seek"],
    direct=False,
)
class VisionSeekSkill:
    """Find a coloured target with the camera and walk to it."""

    name: str = "vision_seek"
    description: str = (
        "Visually find a coloured object (red/blue/green) with the robot's "
        "camera and walk to it — for '找到并走到X物体 / find the X object'. "
        "Locates the target by RECOGNITION (camera), not coordinates."
    )
    typical_duration_sec: float = 180.0
    verify_hint: str = (
        "at_position(x, y, 1.6) with the named target's coordinates — the "
        "ground-truth judge that the robot actually reached the object it "
        "recognised (recognition is the means, arrival is the truth)"
    )
    parameters: dict = {
        "label": {
            "type": "string", "required": True,
            "description": "Target colour/object (e.g. 'red' / '红色物体').",
        },
        "max_iters": {
            "type": "number", "required": False, "default": 70,
            "description": "Max perceive-act cycles before giving up.",
        },
    }
    preconditions: list = ["base connected with a camera (room mode)"]
    postconditions: list = []
    effects: dict = {}
    failure_modes: list = ["no_base", "no_camera", "not_found"]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        base = context.base
        if base is None:
            return SkillResult(success=False, error_message="no mobile base connected",
                               result_data={"diagnosis": "no_base"})
        if not callable(getattr(base, "get_camera_frame", None)):
            return SkillResult(
                success=False,
                error_message=("base has no camera — vision seek needs a room "
                               "scene (--scenario g1_room)"),
                result_data={"diagnosis": "no_camera"})
        color = _resolve_color(str(params.get("label", "")))
        max_iters = int(params.get("max_iters", 70) or 70)

        def perceive(frame):
            dets = [d for d in detect_targets(frame) if d["label"] == color]
            return max(dets, key=lambda d: d["area_frac"]) if dets else None

        try:
            seen, reason, pos = _seek_loop(base, perceive, max_iters)
        except Exception as exc:  # noqa: BLE001 — camera/render failure
            return SkillResult(
                success=False, error_message=f"camera frame failed: {exc}",
                result_data={"diagnosis": "no_camera"})
        ok = reason == "arrived"
        return SkillResult(
            success=ok,
            error_message="" if ok else (
                f"could not find the {color} object by camera"
                if not seen else f"saw the {color} object but did not reach it"),
            result_data={"color": color, "seen": seen, "reason": reason,
                         "position": pos, "transport": "vision"},
        )
