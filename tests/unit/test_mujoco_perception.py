"""Tests for MuJoCoPerception — ground-truth simulated perception."""
from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")

from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm
from vector_os_nano.hardware.sim.mujoco_perception import MuJoCoPerception


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def arm():
    a = MuJoCoArm(gui=False)
    a.connect()
    yield a
    a.disconnect()


@pytest.fixture
def perception(arm):
    return MuJoCoPerception(arm)


# ---------------------------------------------------------------------------
# TestMuJoCoPerceptionDetect
# ---------------------------------------------------------------------------

class TestMuJoCoPerceptionDetect:
    def test_detect_banana(self, perception: MuJoCoPerception):
        """detect('banana') should return exactly one detection with label containing 'banana'."""
        detections = perception.detect("banana")
        assert len(detections) == 1
        assert "banana" in detections[0].label.lower()

    def test_detect_all(self, perception: MuJoCoPerception):
        """detect('all') should return at least 6 detections (one per scene object)."""
        detections = perception.detect("all")
        assert len(detections) >= 6

    def test_detect_nonexistent(self, perception: MuJoCoPerception):
        """detect('laptop') should return an empty list — no laptop in the scene."""
        detections = perception.detect("laptop")
        assert detections == []

    def test_detect_chinese_alias(self, perception: MuJoCoPerception):
        """detect('杯子') should match 'mug' via the alias table and return detections."""
        detections = perception.detect("杯子")
        assert len(detections) >= 1
        # Confirm at least one detection is the mug
        labels = [d.label.lower() for d in detections]
        assert any("mug" in lbl for lbl in labels)

    def test_detection_has_bbox(self, perception: MuJoCoPerception):
        """Every detection must carry a .bbox that is a tuple of 4 numeric elements."""
        detections = perception.detect("all")
        assert len(detections) >= 1
        for det in detections:
            assert det.bbox is not None, f"Detection '{det.label}' has no bbox"
            assert len(det.bbox) == 4, (
                f"Detection '{det.label}' bbox has wrong length: {det.bbox}"
            )
            for coord in det.bbox:
                assert isinstance(coord, (int, float)), (
                    f"bbox element is not numeric: {coord!r}"
                )


# ---------------------------------------------------------------------------
# TestMuJoCoPerceptionTrack
# ---------------------------------------------------------------------------

class TestMuJoCoPerceptionTrack:
    def test_track_returns_poses(self, perception: MuJoCoPerception):
        """track(detections) should produce TrackedObjects each with pose not None."""
        detections = perception.detect("all")
        assert len(detections) >= 1, "Need at least one detection to track"
        tracked = perception.track(detections)
        assert len(tracked) == len(detections)
        for obj in tracked:
            assert obj.pose is not None, (
                f"TrackedObject '{obj.label}' has pose=None"
            )

    def test_tracked_position_matches_mujoco(self, perception: MuJoCoPerception, arm: MuJoCoArm):
        """Tracked pose xyz should roughly match arm.get_object_positions()."""
        # Settle objects first
        arm.step(200)
        detections = perception.detect("banana")
        assert len(detections) == 1
        tracked = perception.track(detections)
        assert len(tracked) == 1

        obj = tracked[0]
        assert obj.pose is not None

        mj_pos = arm.get_object_positions()["banana"]
        # Allow 1 cm tolerance (positions are live ground-truth)
        assert abs(obj.pose.x - mj_pos[0]) < 0.01, (
            f"x mismatch: tracked={obj.pose.x:.4f}, mujoco={mj_pos[0]:.4f}"
        )
        assert abs(obj.pose.y - mj_pos[1]) < 0.01, (
            f"y mismatch: tracked={obj.pose.y:.4f}, mujoco={mj_pos[1]:.4f}"
        )
        assert abs(obj.pose.z - mj_pos[2]) < 0.01, (
            f"z mismatch: tracked={obj.pose.z:.4f}, mujoco={mj_pos[2]:.4f}"
        )


# ---------------------------------------------------------------------------
# TestMuJoCoPerceptionCamera
# ---------------------------------------------------------------------------

class TestMuJoCoPerceptionCamera:
    def test_color_frame_shape(self, perception: MuJoCoPerception):
        """get_color_frame() should return an array of shape (480, 640, 3) and dtype uint8."""
        import numpy as np
        frame = perception.get_color_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3), (
            f"Expected (480, 640, 3), got {frame.shape}"
        )
        assert frame.dtype == np.uint8, (
            f"Expected uint8, got {frame.dtype}"
        )

    def test_intrinsics_valid(self, perception: MuJoCoPerception):
        """get_intrinsics() should return focal lengths > 0, width=640, height=480."""
        intrinsics = perception.get_intrinsics()
        assert intrinsics.fx > 0, f"fx should be positive, got {intrinsics.fx}"
        assert intrinsics.fy > 0, f"fy should be positive, got {intrinsics.fy}"
        assert intrinsics.width == 640, f"Expected width=640, got {intrinsics.width}"
        assert intrinsics.height == 480, f"Expected height=480, got {intrinsics.height}"
