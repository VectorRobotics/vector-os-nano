# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""MuJoCoPerception — the SO-101 skill family's object source: simulator truth.

``detect`` / ``track`` / ``caption`` answer from the scene's object poses (no
detector, no tracker); the image is the body's generic Camera, so what it shows,
its depth and its intrinsics are the real camera's. This is declared ground-truth
debt: R4 moves the SO-101 skills onto the model perception the other arms use and
deletes this module.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable

from vector_os_nano.core.types import Detection, Pose3D, TrackedObject
from vector_os_nano.interfaces.devices import CameraFrame, CameraProtocol
from vector_os_nano.perception.calibration import Calibration

logger = logging.getLogger(__name__)

# Object name → friendly labels for natural language matching
_OBJECT_ALIASES: dict[str, list[str]] = {
    "banana": ["banana", "香蕉", "黄色", "黄"],
    "mug": ["mug", "cup", "杯子", "杯", "马克杯", "红色", "红"],
    "bottle": ["bottle", "瓶子", "瓶", "蓝色", "蓝", "水瓶"],
    "screwdriver": ["screwdriver", "螺丝刀", "起子", "绿色", "绿"],
    "duck": ["duck", "鸭子", "小鸭", "橙色", "橙", "玩具"],
    "lego": ["lego", "积木", "乐高", "白色", "白", "方块"],
}


class MuJoCoPerception:
    """Simulated perception over simulator truth.

    Args:
        objects: returns ``{object: [x, y, z]}`` of the scene's objects, one instant
            (the session's ground truth: the debt this module declares).
        camera: the body's camera (what :meth:`capture` returns).
    """

    def __init__(self, objects: Callable[[], dict[str, list[float]]], camera: CameraProtocol) -> None:
        self._objects = objects
        self._camera = camera
        # Its tracked poses are already in the arm-base frame: the camera->arm
        # calibration it declares is the identity (the Agent needs no calibration.file).
        self.calibration = Calibration()

    def capture(self) -> CameraFrame:
        """One instant of the body's camera: RGB, metric depth, K from its fovy, pose."""
        return self._camera.capture()

    def detect(self, query: str) -> list[Detection]:
        """Match query against scene objects by name/alias.

        Returns ground-truth detections with synthetic bounding boxes.
        """
        query_lower = query.lower().strip()
        objs = self._objects()
        results: list[Detection] = []

        for obj_name, pos in objs.items():
            aliases = _OBJECT_ALIASES.get(obj_name, [obj_name])
            matched = any(alias in query_lower for alias in aliases)

            # Also match if query is generic ("all", "objects", "所有", "物体").
            # Word-boundary match so 'all' does not fire on 'ball'/'small'/'wall'.
            if not matched and (
                any(zh in query_lower for zh in ("所有", "物体"))
                or query_lower in {"all", "objects", "everything", "all objects"}
                or any(
                    re.search(r"\b" + re.escape(kw) + r"\b", query_lower)
                    for kw in ("all", "objects", "everything")
                )
            ):
                matched = True

            if matched:
                # A synthetic bbox (centered, fixed size)
                bbox = (280, 200, 360, 280)
                label = obj_name.replace("_", " ")
                results.append(Detection(label=label, bbox=bbox, confidence=1.0))
                logger.info("[SIM DETECT] Matched '%s' → %s at (%.3f, %.3f, %.3f)",
                            query, obj_name, pos[0], pos[1], pos[2])

        if not results:
            logger.info("[SIM DETECT] No match for query '%s' in objects: %s",
                        query, list(objs.keys()))
        return results

    def caption(self, length: str = "long") -> str:
        """Describe the scene from simulator truth (no VLM needed)."""
        objs = self._objects()
        if not objs:
            return "I don't see any objects on the table."
        return "I can see: " + ", ".join(name.replace("_", " ") for name in objs) + "."

    def visual_query(self, question: str, **kwargs: Any) -> str:
        """Answer a free-form question about the scene from simulator truth."""
        return self.caption()

    def track(self, detections: list[Detection]) -> list[TrackedObject]:
        """TrackedObjects with the simulator-truth 3D position of each detection."""
        objs = self._objects()
        tracked: list[TrackedObject] = []
        for idx, det in enumerate(detections):
            pos = objs.get(det.label.replace(" ", "_"))
            tracked.append(TrackedObject(
                track_id=idx,
                label=det.label,
                bbox_2d=det.bbox,
                pose=None if pos is None else Pose3D(x=pos[0], y=pos[1], z=pos[2]),
                confidence=1.0,
            ))
        return tracked
