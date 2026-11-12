# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Test intrinsics: the contract's K (charter 3.1) for a synthetic frame."""
from __future__ import annotations

import math

from vector_os_nano.core.types import CameraIntrinsics


def pinhole(width: int = 320, height: int = 240, fovy_deg: float = 42.0) -> CameraIntrinsics:
    """fy = (H/2)/tan(fovy/2), fx = fy, cx = W/2, cy = H/2, as ``RGBDFrame.K``."""
    fy = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    K = [[fy, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]]
    return CameraIntrinsics.from_matrix(K, width, height)
