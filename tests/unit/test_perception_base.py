# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Unit tests for perception.base — PerceptionProtocol."""
from __future__ import annotations

import numpy as np
import pytest

from vector_os_nano.perception.base import PerceptionProtocol
from vector_os_nano.core.types import CameraIntrinsics, TrackedObject
from vector_os_nano.interfaces.perception import Detection


# ---------------------------------------------------------------------------
# test_protocol_defined
# ---------------------------------------------------------------------------

def test_protocol_defined():
    """PerceptionProtocol is a runtime-checkable Protocol with required methods."""
    assert hasattr(PerceptionProtocol, "__protocol_attrs__") or hasattr(
        PerceptionProtocol, "_is_protocol"
    ) or hasattr(PerceptionProtocol, "__abstractmethods__")
    # Confirm protocol is marked runtime_checkable
    # (by verifying isinstance check doesn't raise TypeError)
    class MockPerception:
        def get_color_frame(self) -> np.ndarray:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        def get_depth_frame(self) -> np.ndarray:
            return np.zeros((480, 640), dtype=np.uint16)

        def get_intrinsics(self) -> CameraIntrinsics:
            return CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)

        def caption(self, length: str = "normal") -> str:
            return ""

        def visual_query(self, question: str) -> str:
            return ""

        def detect(self, query: str) -> list[Detection]:
            return []

        def track(self, detections: list[Detection]) -> list[TrackedObject]:
            return []

        def get_point_cloud(self, mask=None) -> np.ndarray:
            return np.zeros((0, 3))

    obj = MockPerception()
    assert isinstance(obj, PerceptionProtocol)


def test_protocol_missing_method_fails():
    """Object missing a required method does NOT satisfy PerceptionProtocol."""
    class IncompletePerception:
        def get_color_frame(self) -> np.ndarray:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        # Missing get_depth_frame, get_intrinsics, detect, track, get_point_cloud

    obj = IncompletePerception()
    assert not isinstance(obj, PerceptionProtocol)


def test_protocol_method_names():
    """PerceptionProtocol exposes the six required method names."""
    required = {
        "get_color_frame",
        "get_depth_frame",
        "get_intrinsics",
        "detect",
        "track",
        "get_point_cloud",
    }
    # Python 3.12+: __protocol_attrs__; Python 3.10: inspect annotations
    import inspect
    members = {name for name, _ in inspect.getmembers(PerceptionProtocol, predicate=inspect.isfunction)}
    assert required <= members, f"Missing methods: {required - members}"
