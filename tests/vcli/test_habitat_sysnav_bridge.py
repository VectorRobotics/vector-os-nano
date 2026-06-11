# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""M4 Part A — habitat→SysNav input bridge.

Pure-math unprojection tests pin the world-frame convention; the rclpy
integration test (skipped where ROS2 is absent — the campaign venv has it)
spins the REAL node against a fake pano source and self-subscribes all
three SysNav input topics.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from vector_os_nano.playground.habitat.sysnav_bridge import (
    crop_to_sysnav_image,
    make_pointcloud2_parts,
    unproject_equirect_depth,
)


def _flat_depth(h: int = 64, w: int = 128, value: float = 2.0) -> np.ndarray:
    return np.full((h, w), value, dtype=np.float32)


class TestUnprojection:
    def test_center_pixel_is_straight_ahead(self) -> None:
        pts = unproject_equirect_depth(
            _flat_depth(), (1.0, 2.0, 0.5), heading=0.0, stride=1
        )
        # the point closest to the analytic forward ray end:
        target = np.array([1.0 + 2.0, 2.0, 0.5 + 1.2])
        nearest = pts[np.argmin(np.linalg.norm(pts - target, axis=1))]
        assert np.linalg.norm(nearest - target) < 0.15

    def test_quarter_turn_column_is_to_the_right(self) -> None:
        pts = unproject_equirect_depth(
            _flat_depth(), (0.0, 0.0, 0.0), heading=0.0, stride=1
        )
        target = np.array([0.0, -2.0, 1.2])  # lon=+90° => agent's RIGHT => -y
        nearest = pts[np.argmin(np.linalg.norm(pts - target, axis=1))]
        assert np.linalg.norm(nearest - target) < 0.15

    def test_heading_rotates_the_cloud(self) -> None:
        pts = unproject_equirect_depth(
            _flat_depth(), (0.0, 0.0, 0.0), heading=math.pi / 2, stride=1
        )
        target = np.array([0.0, 2.0, 1.2])  # facing +y now
        nearest = pts[np.argmin(np.linalg.norm(pts - target, axis=1))]
        assert np.linalg.norm(nearest - target) < 0.15

    def test_invalid_and_far_returns_dropped(self) -> None:
        d = _flat_depth(value=2.0)
        d[0, :] = np.inf
        d[1, :] = 0.0
        d[2, :] = 50.0  # beyond max_depth
        pts = unproject_equirect_depth(d, (0, 0, 0), 0.0, stride=1, max_depth=20.0)
        assert np.isfinite(pts).all()
        assert (np.linalg.norm(pts - np.array([0, 0, 1.2]), axis=1) < 21.0).all()

    def test_stride_subsamples(self) -> None:
        full = unproject_equirect_depth(_flat_depth(), (0, 0, 0), 0.0, stride=1)
        sub = unproject_equirect_depth(_flat_depth(), (0, 0, 0), 0.0, stride=4)
        assert 0 < len(sub) < len(full)


class TestPointCloudParts:
    def test_layout_and_step(self) -> None:
        pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        data, layout, step = make_pointcloud2_parts(pts)
        assert step == 16  # x,y,z,intensity float32
        assert [n for n, _, _ in layout] == ["x", "y", "z", "intensity"]
        arr = np.frombuffer(data, dtype=np.float32).reshape(-1, 4)
        assert arr[0].tolist() == [1.0, 2.0, 3.0, 1.0]


def _rclpy_available() -> bool:
    try:
        import rclpy  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _rclpy_available(), reason="rclpy not importable")
class TestBridgeNodeRoundtrip:
    def test_publishes_sysnav_triplet(self) -> None:
        import rclpy

        from vector_os_nano.playground.habitat.sysnav_bridge import (
            HabitatSysnavBridge,
        )

        fake = SimpleNamespace(
            get_pano=lambda: {
                "rgb": np.zeros((64, 128, 3), dtype=np.uint8),
                "depth": _flat_depth(64, 128, 3.0),
                "pos": [1.0, -1.0, 0.2],
                "heading": 0.3,
            }
        )
        bridge = HabitatSysnavBridge(fake, hz=50.0)
        got: dict = {}
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Image, PointCloud2

        sub_node = rclpy.create_node("sysnav_probe")
        sub_node.create_subscription(
            Image, "/camera/image", lambda m: got.setdefault("img", m), 5
        )
        sub_node.create_subscription(
            PointCloud2, "/registered_scan", lambda m: got.setdefault("cloud", m), 5
        )
        sub_node.create_subscription(
            Odometry, "/state_estimation", lambda m: got.setdefault("odom", m), 10
        )
        try:
            import time

            # Phase 1 — DDS discovery: wait until every publisher matched.
            # (Publishing while spinning starves two of the three callbacks:
            # the single-threaded wait-set keeps picking the same always-ready
            # subscription. Match first, publish once, then drain.)
            deadline = time.time() + 5.0
            while time.time() < deadline and not all(
                p.get_subscription_count() == 1
                for p in (bridge._pub_image, bridge._pub_cloud, bridge._pub_odom)
            ):
                rclpy.spin_once(sub_node, timeout_sec=0.02)
                rclpy.spin_once(bridge.node, timeout_sec=0.0)

            # Phase 2 — one publish, then drain the three queued messages.
            bridge.publish_once()
            for _ in range(100):
                if len(got) == 3:
                    break
                rclpy.spin_once(sub_node, timeout_sec=0.05)
            assert set(got) == {"img", "cloud", "odom"}, f"missing: {got.keys()}"

            img = got["img"]
            assert (img.height, img.width, img.encoding) == (64, 128, "rgb8")
            cloud = got["cloud"]
            assert cloud.header.frame_id == "map"
            assert cloud.width > 0 and cloud.point_step == 16
            arr = np.frombuffer(bytes(cloud.data), dtype=np.float32).reshape(-1, 4)
            assert np.isfinite(arr).all()
            odom = got["odom"]
            assert odom.pose.pose.position.x == pytest.approx(1.0)
            yaw = 2.0 * math.atan2(
                odom.pose.pose.orientation.z, odom.pose.pose.orientation.w
            )
            assert yaw == pytest.approx(0.3, abs=1e-6)
        finally:
            sub_node.destroy_node()
            bridge.destroy()


class TestSysnavImageContract:
    def test_full_pano_crops_to_1920x640(self) -> None:
        full = np.zeros((960, 1920, 3), dtype=np.uint8)
        full[160] = 7   # first kept row
        full[799] = 9   # last kept row
        out = crop_to_sysnav_image(full)
        assert out.shape == (640, 1920, 3)
        assert out[0, 0, 0] == 7 and out[-1, 0, 0] == 9

    def test_already_contract_shape_passthrough(self) -> None:
        img = np.zeros((640, 1920, 3), dtype=np.uint8)
        assert crop_to_sysnav_image(img) is img

    def test_incompatible_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="contract"):
            crop_to_sysnav_image(np.zeros((512, 1024, 3), dtype=np.uint8))
