# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""VlmSeekSkill — find & approach a SEMANTIC object by VLM recognition (#9 R1).

The track-A upgrade of VisionSeekSkill: instead of colour segmentation, a REAL
vision-language model (Qwen-VL via OpenRouter) grounds the target class — "go
find the chair / 去找椅子" — in the first-person camera frame, and the SAME seek
state machine (shared ``_seek_loop``) drives the robot to it.

Honest (CLAUDE.md rule 5): the VLM is really called; the detector NEVER reads
ground-truth coordinates. If the model never localises the target the skill
FAILS. The kernel's at_position(<real object>, tol) verify is the ground-truth
judge — recognition is only the MEANS of getting there.
"""
from __future__ import annotations

import logging

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.perception.vlm_targets import VlmTargetDetector
from vector_os_nano.skills.vision_seek import _seek_loop

logger = logging.getLogger(__name__)


@skill(
    aliases=["vlm find", "认物体", "找物体", "去找", "找椅子", "找沙发"],
    direct=False,
)
class VlmSeekSkill:
    """Find a semantic object class with a real VLM and walk to it.

    ``detector`` is injectable (tests pass a deterministic fake); the default is
    the production Qwen-VL/OpenRouter detector, built lazily on first frame so
    constructing the skill never needs a key or network.
    """

    name: str = "vlm_seek"
    description: str = (
        "Visually find a SEMANTIC object (chair, sofa, potted plant, table, "
        "...) with the robot's camera using a real vision-language model and "
        "walk to it — for 'find the chair / 去找椅子 / 认出沙发并走过去'. Locates "
        "the target by VLM RECOGNITION (camera), not coordinates. Prefer this "
        "over vision_seek when the target is a furniture/object class rather "
        "than a bare colour."
    )
    typical_duration_sec: float = 220.0
    verify_hint: str = (
        "at_position(x, y, 1.6) with the named object's coordinates — the "
        "ground-truth judge that the robot actually reached the object the VLM "
        "recognised (recognition is the means, arrival is the truth)"
    )
    parameters: dict = {
        "label": {
            "type": "string", "required": True,
            "description": ("Target object class (e.g. 'chair' / '椅子' / "
                            "'potted plant'). Free text — passed to the VLM."),
        },
        "max_iters": {
            "type": "number", "required": False, "default": 70,
            "description": "Max perceive-act cycles before giving up.",
        },
    }
    preconditions: list = ["base connected with a camera (furnished room mode)"]
    postconditions: list = []
    effects: dict = {}
    failure_modes: list = ["no_base", "no_camera", "not_found"]

    def __init__(self, detector: "VlmTargetDetector | None" = None) -> None:
        # DEBT-1: the 70-iter seek loop servos on bearing/area of a target it is
        # actively approaching (not a far precise locate), so it opts the VLM image
        # DOWN to 256px — ~4x less payload/latency per tick than the LOCATE path's
        # 512 (which recognize_navigate keeps). 256 >> the proven-bad 160.
        self._detector = detector or VlmTargetDetector(max_dim=256)

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        base = context.base
        if base is None:
            return SkillResult(success=False, error_message="no mobile base connected",
                               result_data={"diagnosis": "no_base"})
        if not callable(getattr(base, "get_camera_frame", None)):
            return SkillResult(
                success=False,
                error_message=("base has no camera — VLM seek needs a room "
                               "scene (--scenario g1_room_vlm)"),
                result_data={"diagnosis": "no_camera"})
        query = str(params.get("label", "")).strip()
        if not query:
            return SkillResult(success=False,
                               error_message="vlm_seek requires a target label",
                               result_data={"diagnosis": "bad_params"})
        max_iters = int(params.get("max_iters", 70) or 70)

        def perceive(frame):
            dets = self._detector.detect_targets(frame, query)
            return max(dets, key=lambda d: d["area_frac"]) if dets else None

        try:
            # step_duration covers the ~2 s VLM call: G1's async-deadman gait
            # needs 3.0 s to keep walking ACROSS the call (else it stutters and
            # progress-stall fires short — Case 16); Go2's walk() is BLOCKING so
            # a short step (its seek_step_duration hint) is correct. arrive_area
            # high → arrival is the physical collision-blocked progress-stall.
            step = float(getattr(base, "seek_step_duration", 3.0))
            seen, reason, pos = _seek_loop(
                base, perceive, max_iters, step_duration=step, arrive_area=0.55)
        except Exception as exc:  # noqa: BLE001 — camera/render failure
            return SkillResult(
                success=False, error_message=f"camera frame failed: {exc}",
                result_data={"diagnosis": "no_camera"})
        ok = reason == "arrived"
        return SkillResult(
            success=ok,
            error_message="" if ok else (
                f"could not find a {query} by camera (VLM)"
                if not seen else f"saw the {query} but did not reach it"),
            result_data={"query": query, "seen": seen, "reason": reason,
                         "position": pos, "transport": "vlm_vision"},
        )
