# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""MuJoCo-based simulated perception.

Provides ground-truth object detection and tracking using MuJoCo's
internal scene state — no camera, VLM, or tracker needed.

Implements the same interface that DetectSkill and PickSkill expect
from a PerceptionProtocol-compatible pipeline.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from vector_os_nano.core.types import (
    CameraIntrinsics,
    Detection,
    Pose3D,
    TrackedObject,
)
from vector_os_nano.perception.calibration import Calibration

if TYPE_CHECKING:
    from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm

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

# Simulated camera parameters (matches overhead camera in MJCF)
_SIM_WIDTH = 640
_SIM_HEIGHT = 480


class MuJoCoPerception:
    """Simulated perception using MuJoCo ground truth.

    detect(query) matches the query string against known object names
    in the scene and returns their positions directly from MuJoCo state.
    No VLM or tracker needed.

    Args:
        mujoco_arm: A connected MuJoCoArm instance.
    """

    def __init__(self, mujoco_arm: "MuJoCoArm") -> None:
        self._arm = mujoco_arm
        # Its tracked poses are already in the arm-base frame: the camera->arm
        # calibration it declares is the identity (the Agent needs no calibration.file).
        self.calibration = Calibration()
        self._last_detections: list[Detection] = []
        self._last_tracked: list[TrackedObject] = []

    def get_color_frame(self) -> np.ndarray:
        """Render RGB image from MuJoCo overhead camera."""
        img = self._arm.render(camera_name="overhead", width=_SIM_WIDTH, height=_SIM_HEIGHT)
        if img is None:
            return np.zeros((_SIM_HEIGHT, _SIM_WIDTH, 3), dtype=np.uint8)
        return img

    def get_depth_frame(self) -> np.ndarray:
        """Return synthetic depth frame (zeros — not needed for sim)."""
        return np.zeros((_SIM_HEIGHT, _SIM_WIDTH), dtype=np.uint16)

    def get_intrinsics(self) -> CameraIntrinsics:
        """Return simulated camera intrinsics."""
        return CameraIntrinsics(
            fx=500.0, fy=500.0,
            cx=_SIM_WIDTH / 2, cy=_SIM_HEIGHT / 2,
            width=_SIM_WIDTH, height=_SIM_HEIGHT,
        )

    def detect(self, query: str) -> list[Detection]:
        """Match query against scene objects by name/alias.

        Returns ground-truth detections with synthetic bounding boxes.
        """
        if not self._arm._connected:
            return []

        query_lower = query.lower().strip()
        objs = self._arm.get_object_positions()
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
                # Create a synthetic bbox (centered, fixed size)
                cx, cy = _SIM_WIDTH // 2, _SIM_HEIGHT // 2
                half = 40
                bbox = (cx - half, cy - half, cx + half, cy + half)

                # Use a human-readable label
                label = obj_name.replace("_", " ")
                det = Detection(label=label, bbox=bbox, confidence=1.0)
                results.append(det)
                logger.info("[SIM DETECT] Matched '%s' → %s at (%.3f, %.3f, %.3f)",
                            query, obj_name, pos[0], pos[1], pos[2])

        self._last_detections = results
        if not results:
            logger.info("[SIM DETECT] No match for query '%s' in objects: %s",
                        query, list(objs.keys()))
        return results

    def caption(self, length: str = "long") -> str:
        """Describe the visible scene from MuJoCo ground truth (no VLM needed)."""
        if not self._arm._connected:
            return "I can't see anything — the camera is not connected."
        objs = self._arm.get_object_positions()
        if not objs:
            return "I don't see any objects on the table."
        labels = [name.replace("_", " ") for name in objs]
        return "I can see: " + ", ".join(labels) + "."

    def visual_query(self, question: str, **kwargs: Any) -> str:
        """Answer a free-form question about the scene from ground truth."""
        return self.caption()

    def track(self, detections: list[Detection]) -> list[TrackedObject]:
        """Return TrackedObjects with ground-truth 3D positions from MuJoCo.

        Each detection is matched back to the scene object to get its pose.
        """
        if not self._arm._connected:
            return []

        objs = self._arm.get_object_positions()
        tracked: list[TrackedObject] = []

        for idx, det in enumerate(detections):
            # Match detection label back to scene object name
            det_key = det.label.replace(" ", "_")
            pos = objs.get(det_key)
            if pos is None:
                # Try fuzzy match
                for obj_name, obj_pos in objs.items():
                    if obj_name.replace("_", " ") == det.label:
                        pos = obj_pos
                        break

            pose = None
            if pos is not None:
                pose = Pose3D(x=pos[0], y=pos[1], z=pos[2])

            tracked.append(TrackedObject(
                track_id=idx,
                label=det.label,
                bbox_2d=det.bbox,
                pose=pose,
                confidence=1.0,
            ))

        self._last_tracked = tracked
        return tracked

    def get_point_cloud(self, mask: np.ndarray | None = None) -> np.ndarray:
        """Return empty point cloud (not used in sim mode)."""
        return np.zeros((0, 3), dtype=np.float64)

    def connect(self) -> None:
        """No-op (perception is backed by MuJoCoArm)."""
        pass

    def disconnect(self) -> None:
        """No-op."""
        pass

    def stop_continuous_tracking(self) -> None:
        """No-op (no background threads in sim)."""
        pass
