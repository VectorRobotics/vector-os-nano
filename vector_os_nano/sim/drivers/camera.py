# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Camera — the ONE camera driver (``interfaces.CameraProtocol``) for every body and backend.

Built from a manifest camera entry and a ``SimRuntime``. :meth:`capture` is one
instant through the runtime's world-side camera path: RGB, depth and the robot's own
pixels (a mask the world side computes from its segmentation, which never crosses),
the intrinsics and the camera pose. Depth outside the manifest's range reads 0
(invalid), like the sensor; a colour-only camera has no depth.
"""
from __future__ import annotations

from vector_os_nano.schemas.robot import CameraSpec
from vector_os_nano.interfaces.devices import CameraFrame
from vector_os_nano.sim.runtime import SimRuntime


class Camera:
    def __init__(self, spec: CameraSpec, runtime: SimRuntime) -> None:
        self._spec = spec
        self._rt = runtime

    def capture(self, width: int | None = None, height: int | None = None) -> CameraFrame:
        """One camera instant at (*width*, *height*), the manifest's size by default."""
        s = self._spec
        f = self._rt.capture(s.name, width or s.width, height or s.height, depth=s.depth is not None)
        if s.depth is None:
            return f
        near, far = s.depth
        f.depth[(f.depth < near) | (f.depth > far)] = 0.0
        return f
