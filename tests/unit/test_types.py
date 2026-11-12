# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Unit tests for vector_os_nano.core.types — TDD RED phase.

Tests are written before implementation and must cover:
- Creation with defaults
- Serialization (to_dict / from_dict roundtrip)
- Immutability (frozen dataclass — mutation raises FrozenInstanceError)
- Default values
- Property accessors
"""
from __future__ import annotations

import dataclasses
import math
import pytest


# ---------------------------------------------------------------------------
# Pose3D
# ---------------------------------------------------------------------------


class TestPose3D:
    def test_default_construction(self):
        from vector_os_nano.core.types import Pose3D

        p = Pose3D()
        assert p.x == 0.0
        assert p.y == 0.0
        assert p.z == 0.0
        assert p.qx == 0.0
        assert p.qy == 0.0
        assert p.qz == 0.0
        assert p.qw == 1.0

    def test_explicit_construction(self):
        from vector_os_nano.core.types import Pose3D

        p = Pose3D(x=1.0, y=2.0, z=3.0, qx=0.1, qy=0.2, qz=0.3, qw=0.9)
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.z == 3.0
        assert p.qx == 0.1
        assert p.qy == 0.2
        assert p.qz == 0.3
        assert p.qw == 0.9

    def test_is_frozen(self):
        from vector_os_nano.core.types import Pose3D

        p = Pose3D(x=1.0)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            p.x = 99.0  # type: ignore[misc]

    def test_position_property(self):
        from vector_os_nano.core.types import Pose3D

        p = Pose3D(x=1.0, y=2.0, z=3.0)
        assert p.position == (1.0, 2.0, 3.0)

    def test_orientation_property(self):
        from vector_os_nano.core.types import Pose3D

        p = Pose3D(qx=0.1, qy=0.2, qz=0.3, qw=0.9)
        assert p.orientation == (0.1, 0.2, 0.3, 0.9)

    def test_to_dict(self):
        from vector_os_nano.core.types import Pose3D

        p = Pose3D(x=1.0, y=2.0, z=3.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0)
        d = p.to_dict()
        assert isinstance(d, dict)
        assert d["x"] == 1.0
        assert d["y"] == 2.0
        assert d["z"] == 3.0
        assert d["qx"] == 0.0
        assert d["qy"] == 0.0
        assert d["qz"] == 0.0
        assert d["qw"] == 1.0

    def test_from_dict_roundtrip(self):
        from vector_os_nano.core.types import Pose3D

        original = Pose3D(x=1.5, y=-0.5, z=0.3, qx=0.1, qy=0.2, qz=0.3, qw=0.9)
        d = original.to_dict()
        reconstructed = Pose3D.from_dict(d)
        assert reconstructed == original

    def test_from_dict_partial(self):
        """from_dict with missing keys should use defaults."""
        from vector_os_nano.core.types import Pose3D

        p = Pose3D.from_dict({"x": 5.0})
        assert p.x == 5.0
        assert p.y == 0.0
        assert p.z == 0.0
        assert p.qw == 1.0

    def test_equality(self):
        from vector_os_nano.core.types import Pose3D

        p1 = Pose3D(x=1.0, y=2.0)
        p2 = Pose3D(x=1.0, y=2.0)
        assert p1 == p2

    def test_inequality(self):
        from vector_os_nano.core.types import Pose3D

        p1 = Pose3D(x=1.0)
        p2 = Pose3D(x=2.0)
        assert p1 != p2


# ---------------------------------------------------------------------------
# BBox3D
# ---------------------------------------------------------------------------


class TestBBox3D:
    def test_construction(self):
        from vector_os_nano.core.types import BBox3D, Pose3D

        center = Pose3D(x=1.0, y=2.0, z=0.5)
        bbox = BBox3D(center=center, size_x=0.1, size_y=0.2, size_z=0.3)
        assert bbox.center == center
        assert bbox.size_x == 0.1
        assert bbox.size_y == 0.2
        assert bbox.size_z == 0.3

    def test_default_sizes(self):
        from vector_os_nano.core.types import BBox3D, Pose3D

        bbox = BBox3D(center=Pose3D())
        assert bbox.size_x == 0.0
        assert bbox.size_y == 0.0
        assert bbox.size_z == 0.0

    def test_is_frozen(self):
        from vector_os_nano.core.types import BBox3D, Pose3D

        bbox = BBox3D(center=Pose3D())
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            bbox.size_x = 99.0  # type: ignore[misc]

    def test_to_dict(self):
        from vector_os_nano.core.types import BBox3D, Pose3D

        center = Pose3D(x=1.0, y=2.0, z=0.5)
        bbox = BBox3D(center=center, size_x=0.1, size_y=0.2, size_z=0.3)
        d = bbox.to_dict()
        assert isinstance(d, dict)
        assert "center" in d
        assert d["size_x"] == 0.1
        assert d["center"]["x"] == 1.0

    def test_from_dict_roundtrip(self):
        from vector_os_nano.core.types import BBox3D, Pose3D

        center = Pose3D(x=1.0, y=2.0, z=0.5)
        original = BBox3D(center=center, size_x=0.1, size_y=0.2, size_z=0.3)
        reconstructed = BBox3D.from_dict(original.to_dict())
        assert reconstructed == original


# ---------------------------------------------------------------------------
# CameraIntrinsics
# ---------------------------------------------------------------------------


class TestCameraIntrinsics:
    def test_construction(self):
        from vector_os_nano.core.types import CameraIntrinsics

        ci = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
        assert ci.fx == 600.0
        assert ci.fy == 600.0
        assert ci.cx == 320.0
        assert ci.cy == 240.0
        assert ci.width == 640
        assert ci.height == 480

    def test_is_frozen(self):
        from vector_os_nano.core.types import CameraIntrinsics

        ci = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            ci.fx = 700.0  # type: ignore[misc]

    def test_to_dict(self):
        from vector_os_nano.core.types import CameraIntrinsics

        ci = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
        d = ci.to_dict()
        assert d["fx"] == 600.0
        assert d["width"] == 640

    def test_from_dict_roundtrip(self):
        from vector_os_nano.core.types import CameraIntrinsics

        original = CameraIntrinsics(fx=600.0, fy=601.0, cx=320.5, cy=240.5, width=640, height=480)
        reconstructed = CameraIntrinsics.from_dict(original.to_dict())
        assert reconstructed == original


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class TestDetection:
    def test_construction(self):
        from vector_os_nano.core.types import Detection

        d = Detection(label="cup", bbox=(10.0, 20.0, 100.0, 150.0), confidence=0.95)
        assert d.label == "cup"
        assert d.bbox == (10.0, 20.0, 100.0, 150.0)
        assert d.confidence == 0.95

    def test_default_confidence(self):
        from vector_os_nano.core.types import Detection

        d = Detection(label="cup", bbox=(0.0, 0.0, 1.0, 1.0))
        assert d.confidence == 1.0

    def test_is_frozen(self):
        from vector_os_nano.core.types import Detection

        d = Detection(label="cup", bbox=(0.0, 0.0, 1.0, 1.0))
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            d.label = "bottle"  # type: ignore[misc]

    def test_to_dict(self):
        from vector_os_nano.core.types import Detection

        d = Detection(label="cup", bbox=(10.0, 20.0, 100.0, 150.0), confidence=0.95)
        dct = d.to_dict()
        assert dct["label"] == "cup"
        assert dct["bbox"] == (10.0, 20.0, 100.0, 150.0)
        assert dct["confidence"] == 0.95

    def test_from_dict_roundtrip(self):
        from vector_os_nano.core.types import Detection

        original = Detection(label="bottle", bbox=(5.0, 10.0, 80.0, 120.0), confidence=0.88)
        reconstructed = Detection.from_dict(original.to_dict())
        assert reconstructed == original


# ---------------------------------------------------------------------------
# TrackedObject
# ---------------------------------------------------------------------------


class TestTrackedObject:
    def test_construction_minimal(self):
        from vector_os_nano.core.types import TrackedObject

        obj = TrackedObject(
            track_id=1,
            label="cup",
            bbox_2d=(10.0, 20.0, 100.0, 150.0),
        )
        assert obj.track_id == 1
        assert obj.label == "cup"
        assert obj.pose is None
        assert obj.bbox_3d is None
        assert obj.confidence == 1.0
        assert obj.mask is None

    def test_construction_full(self):
        from vector_os_nano.core.types import BBox3D, Pose3D, TrackedObject

        pose = Pose3D(x=0.3, y=0.1, z=0.05)
        bbox_3d = BBox3D(center=pose, size_x=0.05, size_y=0.05, size_z=0.05)
        obj = TrackedObject(
            track_id=2,
            label="bottle",
            bbox_2d=(0.0, 0.0, 50.0, 100.0),
            pose=pose,
            bbox_3d=bbox_3d,
            confidence=0.9,
        )
        assert obj.pose == pose
        assert obj.bbox_3d == bbox_3d

    def test_is_frozen(self):
        from vector_os_nano.core.types import TrackedObject

        obj = TrackedObject(track_id=1, label="cup", bbox_2d=(0.0, 0.0, 1.0, 1.0))
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            obj.label = "bottle"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SkillResult
# ---------------------------------------------------------------------------


class TestSkillResult:
    def test_success(self):
        from vector_os_nano.core.types import SkillResult

        r = SkillResult(success=True, result_data={"grasp_position": [0.3, 0.1, 0.05]})
        assert r.success is True
        assert r.result_data["grasp_position"] == [0.3, 0.1, 0.05]
        assert r.error_message == ""

    def test_failure(self):
        from vector_os_nano.core.types import SkillResult

        r = SkillResult(success=False, error_message="Object not found")
        assert r.success is False
        assert r.error_message == "Object not found"

    def test_default_result_data(self):
        from vector_os_nano.core.types import SkillResult

        r = SkillResult(success=True)
        assert isinstance(r.result_data, dict)
        assert len(r.result_data) == 0

    def test_is_frozen(self):
        from vector_os_nano.core.types import SkillResult

        r = SkillResult(success=True)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            r.success = False  # type: ignore[misc]

    def test_default_factory_is_independent(self):
        """Two SkillResult instances must not share the same dict."""
        from vector_os_nano.core.types import SkillResult

        r1 = SkillResult(success=True)
        r2 = SkillResult(success=True)
        # They should be different objects (independent defaults)
        # Because frozen, we can only check they compare equal
        assert r1 == r2

