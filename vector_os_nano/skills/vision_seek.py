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
_ALIASES = {"红": "red", "蓝": "blue", "绿": "green"}


def _seek_action(det: "dict | None") -> "tuple[str, float, float]":
    """Pure decision: given the target detection (or None), return
    (action, vx, vyaw). vyaw>0 = turn left, vyaw<0 = turn right (G1 convention).
    x_norm>0 = target on the robot's right → turn right (vyaw<0)."""
    if det is None:
        return ("scan", 0.0, _TURN_VYAW)          # sweep left to find it
    if det["area_frac"] >= _ARRIVE_AREA:
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

        seen = False
        reason = "not_found"
        _has_pos = callable(getattr(base, "get_position", None))
        window: list = []      # recent positions, for progress-stall arrival
        for _ in range(max_iters):
            try:
                frame = base.get_camera_frame()
            except Exception as exc:  # noqa: BLE001
                return SkillResult(
                    success=False, error_message=f"camera frame failed: {exc}",
                    result_data={"diagnosis": "no_camera"})
            dets = [d for d in detect_targets(frame) if d["label"] == color]
            det = max(dets, key=lambda d: d["area_frac"]) if dets else None
            if det is not None:
                seen = True
            action, vx, vyaw = _seek_action(det)
            if action == "arrived":     # blob filled the frame (close)
                reason = "arrived"
                break
            # Progress-stall arrival: a small low target clips at the frame edge
            # up close, so pixel area never crosses the threshold and detection
            # flickers. The honest signal is PHYSICAL: once the robot has
            # recognised the target and then stops making net progress (it has
            # walked up to it and is circling/blocked), it has arrived.
            if _has_pos and seen:
                p = base.get_position()
                window.append((float(p[0]), float(p[1])))
                if len(window) > 8:
                    window.pop(0)
                    net = math.hypot(window[-1][0] - window[0][0],
                                     window[-1][1] - window[0][1])
                    if net < 0.45:      # no net progress over 8 steps → reached
                        reason = "arrived"
                        break
            # one short perceive-act step (re-arms deadman each loop)
            base.walk(vx, 0.0, vyaw, duration=0.4)
        try:
            base.stop()
        except Exception:  # noqa: BLE001
            pass
        pos = list(base.get_position()) if callable(getattr(base, "get_position", None)) else None
        ok = reason == "arrived"
        return SkillResult(
            success=ok,
            error_message="" if ok else (
                f"could not find the {color} object by camera"
                if not seen else f"saw the {color} object but did not reach it"),
            result_data={"color": color, "seen": seen, "reason": reason,
                         "position": pos, "transport": "vision"},
        )
