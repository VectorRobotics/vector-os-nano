# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""MuJoCo virtual sensors (rclpy-free; samples are plain data).

* :func:`raycast_lidar` — the ONE shared lidar ray loop (go2 + g1).
* :func:`g1_lidar_scan` — the g1 head lidar over it.
"""
from __future__ import annotations

from vector_os_nano.hardware.sim.sensors.g1_lidar import g1_lidar_scan
from vector_os_nano.hardware.sim.sensors.lidar_raycast import (
    RaycastResult,
    raycast_lidar,
)

__all__ = [
    "RaycastResult",
    "g1_lidar_scan",
    "raycast_lidar",
]
