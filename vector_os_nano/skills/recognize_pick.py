# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""RecognizePickSkill — the honest photoreal-perception-driven grasp (campaign #10).

The grasp analog of ``RecognizeNavigateSkill``: a real VLM recognises the object
on the (photoreal co-sim) camera frame, depth back-projects its bbox centre to a
3D world point, and that PERCEPTION-derived point — never the simulator's
ground-truth object pose (rule 5) — is handed to ``PickTopDownSkill`` as
``target_xyz``. Reuses the proven detector + depth back-projection + top-down
grasp; adds no new grasp mechanics.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.perception.target_locate import (
    locate_on_plane,
    locate_xyz_from_depth,
)
from vector_os_nano.perception.vlm_targets import VlmTargetDetector
from vector_os_nano.skills.pick_top_down import PickTopDownSkill

logger = logging.getLogger(__name__)


def _observe(base: Any) -> "dict | None":
    """A ``{rgb, depth, cam_pos, cam_mat, fovy}`` observation, world-agnostic.

    Prefers an atomic ``get_camera_observation`` (g1); otherwise assembles it
    from the Go2-style ``get_camera_frame`` + ``get_depth_frame`` +
    ``get_camera_pose`` (+ the d435 vertical FOV). Returns None if the base
    cannot supply rgb + depth + pose.
    """
    if callable(getattr(base, "get_camera_observation", None)):
        try:
            obs = base.get_camera_observation()
            if obs is not None and obs.get("depth") is not None:
                return obs
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_camera_observation failed: %s", exc)
    if not (callable(getattr(base, "get_camera_frame", None))
            and callable(getattr(base, "get_depth_frame", None))
            and callable(getattr(base, "get_camera_pose", None))):
        return None
    rgb = base.get_camera_frame()
    depth = base.get_depth_frame()
    cam_pos, cam_mat = base.get_camera_pose()
    fovy = float(getattr(base, "camera_fovy", lambda: 42.0)()) \
        if callable(getattr(base, "camera_fovy", None)) else 42.0
    return {"rgb": rgb, "depth": depth, "cam_pos": cam_pos,
            "cam_mat": cam_mat, "fovy": fovy}


@skill(
    aliases=["recognise and pick", "认物体抓起来", "找到抓起", "认出抓取"],
    direct=False,
)
class RecognizePickSkill:
    """Recognise an object with the VLM on the (photoreal) frame, depth-locate it
    in 3D, and grasp it top-down — no ground-truth pose (rule 5)."""

    name: str = "recognize_pick"
    description: str = (
        "COMPLETE one-step 'find an object and pick it up': RECOGNISES the object "
        "with a real vision-language model on the camera frame, depth-locates its "
        "3D position, AND grasps it top-down. Use this ALONE for any 'recognise/"
        "find the <object> and pick it up / 认出X抓起来' request — do NOT add a "
        "separate recognition, detect, or look step before it (it recognises "
        "internally). The grasp target comes from PERCEPTION, not ground truth."
    )
    typical_duration_sec: float = 60.0
    verify_hint: str = (
        "grasped() — true when the gripper holds the recognised object. The grasp "
        "target is perception-derived (VLM + depth), never a ground-truth teleport."
    )
    parameters: dict = {
        "label": {"type": "string", "required": True,
                  "description": "Target object class (e.g. 'red can' / '红色的罐子')."},
    }
    preconditions: list = ["base with camera + depth", "arm + gripper"]
    postconditions: list = []
    effects: dict = {}
    failure_modes: list = ["no_base", "no_camera", "not_found", "not_located"]

    # The remote VLM is intermittently rate-limited / flaky (transient 429 or a
    # garbled response → empty). A few re-queries ride out the transient before
    # honestly reporting not-found (the object hasn't moved).
    _DETECT_TRIES: int = 4
    _DETECT_BACKOFF_S: float = 2.0

    def __init__(self, detector: "VlmTargetDetector | None" = None,
                 pick: "PickTopDownSkill | None" = None) -> None:
        self._detector = detector or VlmTargetDetector()
        self._pick = pick or PickTopDownSkill()

    def _detect_with_retry(self, rgb, query: str) -> list:
        """Collect up to 3 successful detections (riding out transient 429/garble)
        and return ONE det at the MEDIAN bbox centre — the remote VLM's centroid
        jitters a few % between calls, and a median rejects the occasional far-off
        bbox that would push the grasp target out of reach (R15)."""
        good: list = []
        tries = 0
        while len(good) < 3 and tries < self._DETECT_TRIES + 2:
            tries += 1
            try:
                dets = self._detector.detect_targets(rgb, query)
            except Exception as exc:  # noqa: BLE001 — flaky remote VLM
                logger.debug("detect_targets raised (try %d): %s", tries, exc)
                dets = []
            if dets:
                good.append(max(dets, key=lambda d: d.get("area_frac", 0.0)))
            elif tries < self._DETECT_TRIES + 2:
                time.sleep(self._DETECT_BACKOFF_S)
        if not good:
            return []
        mx = float(np.median([d["x_norm"] for d in good]))
        my = float(np.median([d["y_norm"] for d in good]))
        out = dict(max(good, key=lambda d: d.get("area_frac", 0.0)))
        out["x_norm"], out["y_norm"] = mx, my
        logger.info("[RECOGNIZE-PICK] median bbox over %d detections: "
                    "x_norm=%.3f y_norm=%.3f", len(good), mx, my)
        return [out]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        base = context.base
        if base is None:
            return SkillResult(success=False, error_message="no base connected",
                               result_data={"diagnosis": "no_base"})
        query = str(params.get("label", "")).strip()
        if not query:
            return SkillResult(success=False, error_message="needs a target label",
                               result_data={"diagnosis": "bad_params"})
        obs = _observe(base)
        if obs is None:
            return SkillResult(
                success=False,
                error_message="base has no camera+depth observation for perception",
                result_data={"diagnosis": "no_camera"})

        dets = self._detect_with_retry(obs["rgb"], query)
        det = max(dets, key=lambda d: d.get("area_frac", 0.0)) if dets else None
        if det is None:
            return SkillResult(
                success=False,
                error_message=f"VLM did not recognise {query!r} on the camera frame",
                result_data={"diagnosis": "not_found"})

        # Top-down pick off a known support surface: ray-to-plane localizes the
        # object's (x,y) without a depth frame (sidesteps the RGB/depth scene
        # mismatch, R15). support_z = the grasp height on that surface (a scene
        # constant — the table the robot picks from, NOT the object's GT pose).
        support_z = params.get("support_z")
        if support_z is not None:
            h, w = obs["rgb"].shape[:2]
            located = locate_on_plane(
                float(det["x_norm"]), float(det["y_norm"]),
                obs["cam_pos"], obs["cam_mat"], float(obs["fovy"]),
                float(support_z), aspect=(w / h if h else 4.0 / 3.0))
        else:
            located = locate_xyz_from_depth(
                float(det["x_norm"]), float(det["y_norm"]), obs["depth"],
                obs["cam_pos"], obs["cam_mat"], float(obs["fovy"]))
        if located is None:
            return SkillResult(
                success=False,
                error_message=f"no depth at the recognised {query!r} bbox",
                result_data={"diagnosis": "not_located"})

        x, y, z = located
        logger.info("[RECOGNIZE-PICK] %r perception-located at world xyz=(%.3f, "
                    "%.3f, %.3f) — grasping", query, x, y, z)
        return self._pick.execute(
            {"target_xyz": [float(x), float(y), float(z)], "object_id": query},
            context)
