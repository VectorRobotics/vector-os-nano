# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Unit tests for PerceptionPipeline continuous background tracking.

Tests the start/stop/loop mechanics added in feat/beta-continuous-tracking.
All tests use synthetic frames and mock tracker — no real hardware required.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from vector_os_nano.core.types import (
    BBox3D,
    CameraIntrinsics,
    Pose3D,
    TrackedObject,
)
from vector_os_nano.perception.pipeline import PerceptionPipeline, _TRACKING_FRAME_STRIDE


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

INTRINSICS = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
COLOR = np.zeros((480, 640, 3), dtype=np.uint8)
DEPTH = np.zeros((480, 640), dtype=np.uint16)

MASK = np.zeros((480, 640), dtype=np.uint8)
MASK[200:300, 200:400] = 1  # non-empty region

RAW_TRACK = {
    "track_id": 1,
    "mask": MASK,
    "bbox": [200, 200, 400, 300],
    "score": 0.95,
}


def _make_tracker(is_tracking: bool = True, process_result=None) -> MagicMock:
    tracker = MagicMock()
    tracker.is_tracking.return_value = is_tracking
    tracker.process_image.return_value = process_result if process_result is not None else [RAW_TRACK]
    tracker.init_track.return_value = [RAW_TRACK]
    tracker.stop.return_value = None
    return tracker


def _make_pipeline(tracker=None, is_tracking: bool = True) -> PerceptionPipeline:
    """Return a PerceptionPipeline with synthetic frames and optional mock tracker."""
    if tracker is None:
        tracker = _make_tracker(is_tracking=is_tracking)
    pipeline = PerceptionPipeline(tracker=tracker)
    pipeline.set_synthetic_frames(COLOR, DEPTH, INTRINSICS)
    return pipeline


# ---------------------------------------------------------------------------
# Thread state initialisation
# ---------------------------------------------------------------------------

class TestThreadStateInit:
    """_tracking_thread and _stop_tracking start as None."""

    def test_tracking_thread_initially_none(self):
        pipeline = _make_pipeline()
        assert pipeline._tracking_thread is None

    def test_stop_tracking_event_initially_none(self):
        pipeline = _make_pipeline()
        assert pipeline._stop_tracking is None


# ---------------------------------------------------------------------------
# start_continuous_tracking
# ---------------------------------------------------------------------------

class TestStartContinuousTracking:
    def test_start_creates_daemon_thread(self):
        pipeline = _make_pipeline()
        try:
            pipeline.start_continuous_tracking()
            assert pipeline._tracking_thread is not None
            assert pipeline._tracking_thread.daemon is True
        finally:
            pipeline.stop_continuous_tracking()

    def test_start_sets_stop_event(self):
        pipeline = _make_pipeline()
        try:
            pipeline.start_continuous_tracking()
            assert pipeline._stop_tracking is not None
        finally:
            pipeline.stop_continuous_tracking()

    def test_thread_is_alive_after_start(self):
        pipeline = _make_pipeline()
        try:
            pipeline.start_continuous_tracking()
            assert pipeline._tracking_thread.is_alive()
        finally:
            pipeline.stop_continuous_tracking()

    def test_double_start_is_noop(self):
        """Calling start twice should not create a second thread."""
        pipeline = _make_pipeline()
        try:
            pipeline.start_continuous_tracking()
            first_thread = pipeline._tracking_thread
            pipeline.start_continuous_tracking()
            assert pipeline._tracking_thread is first_thread
        finally:
            pipeline.stop_continuous_tracking()


# ---------------------------------------------------------------------------
# stop_continuous_tracking
# ---------------------------------------------------------------------------

class TestStopContinuousTracking:
    def test_stop_before_start_is_safe(self):
        pipeline = _make_pipeline()
        pipeline.stop_continuous_tracking()  # must not raise

    def test_stop_clears_thread_reference(self):
        pipeline = _make_pipeline()
        pipeline.start_continuous_tracking()
        pipeline.stop_continuous_tracking()
        assert pipeline._tracking_thread is None

    def test_stop_clears_event_reference(self):
        pipeline = _make_pipeline()
        pipeline.start_continuous_tracking()
        pipeline.stop_continuous_tracking()
        assert pipeline._stop_tracking is None

    def test_thread_no_longer_alive_after_stop(self):
        pipeline = _make_pipeline()
        pipeline.start_continuous_tracking()
        thread = pipeline._tracking_thread
        pipeline.stop_continuous_tracking()
        assert not thread.is_alive()


# ---------------------------------------------------------------------------
# Background loop: _last_tracked updates
# ---------------------------------------------------------------------------

class TestTrackingLoopUpdates:
    def test_loop_updates_last_tracked(self):
        """After starting, _last_tracked should be populated by the background thread."""
        tracker = _make_tracker()
        pipeline = _make_pipeline(tracker=tracker)
        try:
            pipeline.start_continuous_tracking()
            # Give the thread time to run at least one iteration
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                with pipeline._lock:
                    if pipeline._last_tracked:
                        break
                time.sleep(0.05)
            with pipeline._lock:
                assert len(pipeline._last_tracked) > 0
        finally:
            pipeline.stop_continuous_tracking()

    def test_loop_calls_process_image(self):
        """Background thread must call tracker.process_image() repeatedly."""
        tracker = _make_tracker()
        pipeline = _make_pipeline(tracker=tracker)
        try:
            pipeline.start_continuous_tracking()
            # Wait for at least 2 process_image calls
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if tracker.process_image.call_count >= 2:
                    break
                time.sleep(0.05)
            assert tracker.process_image.call_count >= 1
        finally:
            pipeline.stop_continuous_tracking()

    def test_loop_sleeps_when_tracker_not_active(self):
        """When tracker.is_tracking() returns False, loop must not call process_image."""
        tracker = _make_tracker(is_tracking=False)
        pipeline = _make_pipeline(tracker=tracker)
        try:
            pipeline.start_continuous_tracking()
            time.sleep(0.3)
            assert tracker.process_image.call_count == 0
        finally:
            pipeline.stop_continuous_tracking()

    def test_loop_tolerates_empty_process_image(self):
        """When process_image returns [], loop must not crash and must retry."""
        tracker = _make_tracker(process_result=[])
        pipeline = _make_pipeline(tracker=tracker)
        try:
            pipeline.start_continuous_tracking()
            time.sleep(0.3)
            # Thread should still be alive even with empty results
            assert pipeline._tracking_thread.is_alive()
        finally:
            pipeline.stop_continuous_tracking()

    def test_loop_tolerates_exception_in_process_image(self):
        """A transient exception in process_image must not kill the thread."""
        tracker = _make_tracker()
        tracker.process_image.side_effect = RuntimeError("transient GPU error")
        pipeline = _make_pipeline(tracker=tracker)
        try:
            pipeline.start_continuous_tracking()
            time.sleep(0.3)
            assert pipeline._tracking_thread.is_alive()
        finally:
            pipeline.stop_continuous_tracking()

    def test_tracked_objects_under_lock(self):
        """_tracked_objects and _last_tracked must be consistent under _lock."""
        tracker = _make_tracker()
        pipeline = _make_pipeline(tracker=tracker)
        try:
            pipeline.start_continuous_tracking()
            time.sleep(0.3)
            with pipeline._lock:
                # Both lists must reference the same object
                assert pipeline._tracked_objects is pipeline._last_tracked
        finally:
            pipeline.stop_continuous_tracking()


# ---------------------------------------------------------------------------
# track() auto-starts background loop
# ---------------------------------------------------------------------------

class TestTrackAutoStartsLoop:
    def test_track_starts_background_thread(self):
        """Calling track() should automatically start the background thread."""
        from vector_os_nano.interfaces.perception import Detection
        tracker = _make_tracker()
        pipeline = _make_pipeline(tracker=tracker)

        detection = Detection(
            label="cup",
            bbox=(200.0, 200.0, 400.0, 300.0),
            confidence=0.9,
        )
        try:
            pipeline.track([detection])
            assert pipeline._tracking_thread is not None
            assert pipeline._tracking_thread.is_alive()
        finally:
            pipeline.stop_continuous_tracking()

    def test_double_track_does_not_create_second_thread(self):
        """Calling track() twice must reuse the existing background thread."""
        from vector_os_nano.interfaces.perception import Detection
        tracker = _make_tracker()
        pipeline = _make_pipeline(tracker=tracker)

        detection = Detection(
            label="cup",
            bbox=(200.0, 200.0, 400.0, 300.0),
            confidence=0.9,
        )
        try:
            pipeline.track([detection])
            first_thread = pipeline._tracking_thread
            pipeline.track([detection])
            assert pipeline._tracking_thread is first_thread
        finally:
            pipeline.stop_continuous_tracking()


# ---------------------------------------------------------------------------
# 2D-only lightweight update (_build_tracked_objects_2d)
# ---------------------------------------------------------------------------

class TestBuildTrackedObjects2D:
    """Lightweight update reuses existing 3D data and refreshes 2D bbox/mask."""

    def _make_old_obj(self) -> TrackedObject:
        pose = Pose3D(x=0.1, y=0.2, z=0.5)
        bbox_3d = BBox3D(
            center=pose,
            size_x=0.05,
            size_y=0.05,
            size_z=0.05,
        )
        return TrackedObject(
            track_id=1,
            label="cup",
            bbox_2d=(200.0, 200.0, 400.0, 300.0),
            pose=pose,
            bbox_3d=bbox_3d,
            confidence=0.9,
        )

    def test_reuses_existing_pose(self):
        pipeline = _make_pipeline()
        old_obj = self._make_old_obj()
        old_map = {1: old_obj}
        result = pipeline._build_tracked_objects_2d([RAW_TRACK], ["cup"], old_map)
        assert len(result) == 1
        assert result[0].pose is old_obj.pose

    def test_reuses_existing_bbox_3d(self):
        pipeline = _make_pipeline()
        old_obj = self._make_old_obj()
        old_map = {1: old_obj}
        result = pipeline._build_tracked_objects_2d([RAW_TRACK], ["cup"], old_map)
        assert result[0].bbox_3d is old_obj.bbox_3d

    def test_updates_bbox_2d_from_raw(self):
        pipeline = _make_pipeline()
        old_map = {}  # no previous object
        result = pipeline._build_tracked_objects_2d([RAW_TRACK], ["cup"], old_map)
        assert len(result) == 1
        assert result[0].bbox_2d == (200.0, 200.0, 400.0, 300.0)

    def test_pose_none_when_no_prior_track(self):
        pipeline = _make_pipeline()
        old_map = {}
        result = pipeline._build_tracked_objects_2d([RAW_TRACK], ["cup"], old_map)
        assert result[0].pose is None
        assert result[0].bbox_3d is None

    def test_track_id_preserved(self):
        pipeline = _make_pipeline()
        old_map = {}
        result = pipeline._build_tracked_objects_2d([RAW_TRACK], ["cup"], old_map)
        assert result[0].track_id == 1

    def test_confidence_from_raw_track(self):
        pipeline = _make_pipeline()
        old_map = {}
        result = pipeline._build_tracked_objects_2d([RAW_TRACK], ["cup"], old_map)
        assert abs(result[0].confidence - 0.95) < 1e-6


# ---------------------------------------------------------------------------
# Frame stride constant
# ---------------------------------------------------------------------------

class TestFrameStride:
    def test_stride_constant_is_positive_int(self):
        assert isinstance(_TRACKING_FRAME_STRIDE, int)
        assert _TRACKING_FRAME_STRIDE >= 1

    def test_stride_is_4(self):
        """Default stride is 4 (5Hz 3D updates at 20fps camera rate)."""
        assert _TRACKING_FRAME_STRIDE == 4
