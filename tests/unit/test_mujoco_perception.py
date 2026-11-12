# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""MuJoCoPerception — the SO-101 skill family's object source (simulator truth), on
the so101 session: detect / track from the scene's object poses, the image, depth and
intrinsics from the body's generic Camera."""
from __future__ import annotations

import io

import numpy as np
import pytest

from tests.harness.sim_session import started

pytestmark = pytest.mark.sim


@pytest.fixture(scope="module")
def so101():
    with started("so101") as s:
        yield s


@pytest.fixture
def perception(so101):
    return so101.agent._perception


class TestMuJoCoPerceptionDetect:
    def test_detect_banana(self, perception):
        """detect('banana') should return exactly one detection with label containing 'banana'."""
        detections = perception.detect("banana")
        assert len(detections) == 1
        assert "banana" in detections[0].label.lower()

    def test_detect_all(self, perception):
        """detect('all') should return one detection per scene object."""
        assert len(perception.detect("all")) == 6

    def test_detect_nonexistent(self, perception):
        assert perception.detect("laptop") == []

    def test_detect_chinese_alias(self, perception):
        """detect('杯子') should match 'mug' via the alias table."""
        labels = [d.label.lower() for d in perception.detect("杯子")]
        assert any("mug" in lbl for lbl in labels)

    def test_detection_has_bbox(self, perception):
        for det in perception.detect("all"):
            assert len(det.bbox) == 4 and all(isinstance(c, (int, float)) for c in det.bbox), det


class TestMuJoCoPerceptionTrack:
    def test_track_returns_poses(self, perception):
        detections = perception.detect("all")
        tracked = perception.track(detections)
        assert len(tracked) == len(detections) and all(o.pose is not None for o in tracked)

    def test_tracked_position_is_the_scene_truth(self, perception, so101):
        (obj,) = perception.track(perception.detect("banana"))
        truth = so101.gt.object_positions()["banana"]
        assert np.allclose([obj.pose.x, obj.pose.y, obj.pose.z], truth, atol=1e-3)


class TestMuJoCoPerceptionCamera:
    """The capture is the body's camera (R6-V2 / the intrinsics mismatch)."""

    def test_capture_is_the_manifest_camera(self, perception, so101):
        cam = so101.config.camera
        f = perception.capture()
        assert f.rgb.shape == (cam.height, cam.width, 3) and f.rgb.dtype == np.uint8
        fy = (cam.height / 2) / np.tan(np.radians(cam.fovy) / 2)  # the camera's fovy, not 500
        assert np.allclose(f.K, [[fy, 0, cam.width / 2], [0, fy, cam.height / 2], [0, 0, 1]])
        # metric depth along the (vertical) optical axis: a bare table-top point (z 0.02)
        p = f.cam_mat.T @ (np.array([0.35, -0.15, 0.02]) - f.cam_pos)
        row, col = int(f.K[1, 2] - f.K[1, 1] * p[1] / -p[2]), int(f.K[0, 0] * p[0] / -p[2] + f.K[0, 2])
        assert f.depth[row, col] == pytest.approx(f.cam_pos[2] - 0.02, abs=2e-3)

    def test_colour_is_rgb(self, perception, so101):
        """The red mug is red in channel 0 (it was served BGR: red and blue swapped)."""
        _assert_red(perception.capture().rgb, _mug_pixel(perception.capture(), so101))

    def test_the_mcp_camera_frame_is_rgb(self, perception, so101):
        """camera://frame serves what the perception captures, red as red (R6-V2)."""
        pytest.importorskip("mcp")
        from PIL import Image

        from vector_os_nano.mcp.server import _camera_png

        png = np.asarray(Image.open(io.BytesIO(_camera_png(so101.agent))))
        _assert_red(png, _mug_pixel(perception.capture(), so101))


def _mug_pixel(f, so101) -> tuple[int, int]:
    x, y, z = so101.gt.object_positions()["mug"]
    p = f.cam_mat.T @ (np.array([x, y, z + 0.015]) - f.cam_pos)  # the mug's top
    return int(f.K[1, 2] - f.K[1, 1] * p[1] / -p[2]), int(f.K[0, 0] * p[0] / -p[2] + f.K[0, 2])


def _assert_red(rgb, pixel) -> None:
    r, g, b = (int(c) for c in rgb[pixel][:3])
    assert r > g + 40 and r > b + 40, (r, g, b)
