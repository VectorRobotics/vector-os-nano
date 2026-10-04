# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Perception stack — camera drivers, VLM detection, tracking, pointcloud.

No ROS2 imports anywhere in this subpackage.

Public API
----------
PerceptionProtocol   — structural Protocol all backends satisfy
RealSenseCamera      — Intel RealSense D405 driver (lazy pyrealsense2)
VLMDetector          — Moondream VLM (local / Station / API)
EdgeTAMTracker       — EdgeTAM segmentation tracker (lazy torch)
PerceptionPipeline   — orchestrator: camera -> VLM -> tracker -> 3D
Calibration          — camera-to-arm coordinate transform
"""
from __future__ import annotations

from vector_os_nano.perception.base import PerceptionProtocol
from vector_os_nano.perception.calibration import Calibration
from vector_os_nano.perception.pipeline import PerceptionPipeline
from vector_os_nano.perception.pointcloud import (
    pointcloud_to_bbox3d_fast,
    rgbd_to_pointcloud_fast,
)
from vector_os_nano.perception.realsense import RealSenseCamera
from vector_os_nano.perception.tracker import EdgeTAMTracker
from vector_os_nano.perception.vlm import VLMDetector



__all__ = [
    "PerceptionProtocol",
    "RealSenseCamera",
    "VLMDetector",
    "EdgeTAMTracker",
    "PerceptionPipeline",
    "Calibration",
    # Direct function exports
    "rgbd_to_pointcloud_fast",
    "pointcloud_to_bbox3d_fast",
]
