# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""G1HeadPerception — a thin RGB frame source over the G1 humanoid's head camera.

The minimal perception adapter g1 needs so the learned grounding-dino
:class:`DetectorCapability` can run on the SECOND embodiment's sensor (R4). It
exposes exactly the one method the detector's frame resolver looks for —
``get_color_frame()`` — and delegates to ``MuJoCoG1.get_camera_frame()`` (the
``g1_head_rgb`` camera), so the detector stays embodiment-agnostic (it never knows
which body it perceives through).

Deliberately NARROW: g1 has a camera but NO arm, so this adapter carries NO
depth / intrinsics / segmentation / grasp-point machinery (all of which the
CameraPerception gives a body with a depth camera). g1 only LOCALIZES with the
detector — read-only perception. Adding depth/IK is a future round when g1 grows
a manipulator.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class G1HeadPerception:
    """RGB frame source over a MuJoCoG1 head camera (``get_color_frame`` only).

    Args:
        base: A connected MuJoCoG1 (provides ``get_camera_frame(width, height)``
            -> (H, W, 3) uint8 RGB from the ``g1_head_rgb`` camera).
        width/height: Render resolution handed to the camera.
    """

    name = "g1_head"

    def __init__(self, base: Any, *, width: int = 640, height: int = 480) -> None:
        self._base = base
        self._w = int(width)
        self._h = int(height)

    def get_color_frame(self) -> np.ndarray:
        """(H, W, 3) uint8 RGB from g1's head camera — the detector's frame source."""
        return self._base.get_camera_frame(self._w, self._h)

    def get_camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """(cam_xpos, cam_xmat) for g1's head camera (best-effort; for downstream geo)."""
        return self._base.get_camera_pose()
