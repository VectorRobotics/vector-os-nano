# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""MuJoCo virtual ROS2 sensors (v2.4 SysNav simulation integration).

Module-load is rclpy-free: each sensor returns plain dataclass samples
that the bridge subprocess converts to ROS2 messages with rclpy in
scope. This keeps unit tests runnable on a CPU-only Python interpreter
without sourcing a ROS2 workspace.

Public surface:

* :class:`GroundTruthOdomPublisher` — emits :class:`OdomSample` from
  MuJoCo body pose, skipping SLAM in sim.
* :class:`MuJoCoLivox360` — virtual Mid-360 lidar (T1, pending).
* :class:`MuJoCoPano360` — virtual 360-degree RGBD camera (T4, pending).
"""
from __future__ import annotations

from vector_os_nano.hardware.sim.sensors.gt_odom import (
    GroundTruthOdomPublisher,
    OdomSample,
)
from vector_os_nano.hardware.sim.sensors.lidar360 import (
    LidarSample,
    MuJoCoLivox360,
)

__all__ = [
    "GroundTruthOdomPublisher",
    "LidarSample",
    "MuJoCoLivox360",
    "OdomSample",
]
