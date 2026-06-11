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
    filter_ceiling,
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

    def test_cubemap_face_depth_converted_to_euclidean(self) -> None:
        # habitat equirect depth is cube-FACE z-depth (tricky-bugs Case 5):
        # a pixel at 45° elevation straddles the down/forward face boundary
        # where |û·n̂| = cos(45°), so face depth 1.0 must unproject to a
        # point √2 from the camera — NOT 1.0.
        h, w = 64, 128
        d = np.full((h, w), 1.0, dtype=np.float32)
        pts = unproject_equirect_depth(d, (0.0, 0.0, 0.0), 0.0, stride=1,
                                       eye_height=0.0)
        # ray: lat=-45°, lon=0 (forward-down) -> direction (√2/2, 0, -√2/2)
        target = np.array([1.0, 0.0, -1.0])  # √2 along that ray
        nearest = pts[np.argmin(np.linalg.norm(pts - target, axis=1))]
        assert np.linalg.norm(nearest - target) < 0.08

    def test_face_center_depth_unchanged(self) -> None:
        # At a face CENTER the ray is the face normal: euclidean == z-depth.
        h, w = 64, 128
        d = np.full((h, w), 2.0, dtype=np.float32)
        pts = unproject_equirect_depth(d, (0.0, 0.0, 0.0), 0.0, stride=1,
                                       eye_height=0.0)
        for target in ([2.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, -2.0, 0.0]):
            t = np.array(target)
            nearest = pts[np.argmin(np.linalg.norm(pts - t, axis=1))]
            assert np.linalg.norm(nearest - t) < 0.15, target


class TestCeilingFilter:
    def test_drops_points_above_band(self) -> None:
        pts = np.array(
            [[0, 0, 0.1], [0, 0, 1.5], [0, 0, 2.4]], dtype=np.float32
        )
        out = filter_ceiling(pts, base_z=0.12, ceiling_m=1.8)
        assert out[:, 2].max() <= 0.12 + 1.8
        assert len(out) == 2

    def test_none_is_passthrough(self) -> None:
        pts = np.array([[0, 0, 9.0]], dtype=np.float32)
        assert filter_ceiling(pts, 0.0, None) is pts

    def test_empty_cloud_safe(self) -> None:
        pts = np.zeros((0, 3), dtype=np.float32)
        assert len(filter_ceiling(pts, 0.0, 1.8)) == 0


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


@pytest.mark.skipif(not _rclpy_available(), reason="rclpy not importable")
class TestN1StateAndCmdVel:
    def _bridge(self, fake):
        from vector_os_nano.playground.habitat.sysnav_bridge import (
            HabitatSysnavBridge,
        )

        # hz tiny so the pano timer never fires during the test window;
        # state_hz=0 so ticks are driven explicitly (deterministic).
        return HabitatSysnavBridge(fake, hz=0.01, state_hz=0.0)

    def test_publish_state_once_carries_twist(self) -> None:
        import rclpy
        import time
        from nav_msgs.msg import Odometry

        fake = SimpleNamespace(
            get_state=lambda: {
                "pos": [2.0, -1.0, 0.1], "heading": 0.5, "vel": [0.3, 0.0, -0.2],
            }
        )
        bridge = self._bridge(fake)
        got: dict = {}
        sub_node = rclpy.create_node("state_probe")
        sub_node.create_subscription(
            Odometry, "/state_estimation", lambda m: got.setdefault("odom", m), 10
        )
        try:
            deadline = time.time() + 5.0
            while (
                time.time() < deadline
                and bridge._pub_odom_fast.get_subscription_count() != 1
            ):
                rclpy.spin_once(sub_node, timeout_sec=0.02)
            bridge.publish_state_once()
            for _ in range(100):
                if got:
                    break
                rclpy.spin_once(sub_node, timeout_sec=0.05)
            odom = got["odom"]
            assert odom.pose.pose.position.x == pytest.approx(2.0)
            assert odom.twist.twist.linear.x == pytest.approx(0.3)
            assert odom.twist.twist.angular.z == pytest.approx(-0.2)
        finally:
            sub_node.destroy_node()
            bridge.destroy()

    def test_cmd_vel_streams_into_the_base(self) -> None:
        import rclpy
        import time
        from geometry_msgs.msg import Twist

        seen: list = []
        fake = SimpleNamespace(
            set_velocity=lambda vx, vy, vyaw: seen.append((vx, vy, vyaw))
        )
        bridge = self._bridge(fake)
        pub_node = rclpy.create_node("cmd_probe")
        pub = pub_node.create_publisher(Twist, "/cmd_vel", 10)
        try:
            deadline = time.time() + 5.0
            while time.time() < deadline and pub.get_subscription_count() != 1:
                rclpy.spin_once(bridge.fast_node, timeout_sec=0.02)
            msg = Twist()
            msg.linear.x = 0.7
            msg.angular.z = -0.4
            pub.publish(msg)
            for _ in range(100):
                if seen:
                    break
                rclpy.spin_once(bridge.fast_node, timeout_sec=0.05)
            assert seen and seen[0] == (0.7, 0.0, -0.4)
        finally:
            pub_node.destroy_node()
            bridge.destroy()

    def test_state_fetch_failure_keeps_ticking(self) -> None:
        def _boom() -> dict:
            raise RuntimeError("stream down")

        bridge = self._bridge(SimpleNamespace(get_state=_boom))
        try:
            bridge.publish_state_once()  # logs a warning, never raises
        finally:
            bridge.destroy()

    def test_odom_is_sensor_pose_with_tf(self) -> None:
        # N2 CMU contract: odom z = base z + eye height (the cloud's capture
        # origin) and every fast tick broadcasts TF map->sensor.
        import rclpy
        import time
        from tf2_msgs.msg import TFMessage

        fake = SimpleNamespace(
            get_state=lambda: {
                "pos": [2.0, -1.0, 0.12], "heading": 0.0, "vel": [0, 0, 0],
            }
        )
        bridge = self._bridge(fake)
        got: dict = {}
        sub_node = rclpy.create_node("tf_probe")
        sub_node.create_subscription(
            TFMessage, "/tf", lambda m: got.setdefault("tf", m), 10
        )
        try:
            deadline = time.time() + 5.0
            while time.time() < deadline and not got:
                bridge.publish_state_once()
                rclpy.spin_once(sub_node, timeout_sec=0.05)
            tf = got["tf"].transforms[0]
            assert tf.header.frame_id == "map"
            assert tf.child_frame_id == "sensor"
            assert tf.transform.translation.z == pytest.approx(0.12 + 1.2)
        finally:
            sub_node.destroy_node()
            bridge.destroy()

    def test_navigation_cmd_vel_stamped_reaches_base(self) -> None:
        # pathFollower publishes TwistStamped /navigation_cmd_vel (N2).
        import rclpy
        import time
        from geometry_msgs.msg import TwistStamped

        seen: list = []
        fake = SimpleNamespace(
            set_velocity=lambda vx, vy, vyaw: seen.append((vx, vy, vyaw))
        )
        bridge = self._bridge(fake)
        pub_node = rclpy.create_node("navcmd_probe")
        pub = pub_node.create_publisher(TwistStamped, "/navigation_cmd_vel", 10)
        try:
            deadline = time.time() + 5.0
            while time.time() < deadline and pub.get_subscription_count() != 1:
                rclpy.spin_once(bridge.fast_node, timeout_sec=0.02)
            msg = TwistStamped()
            msg.twist.linear.x = 0.6
            msg.twist.angular.z = 0.2
            pub.publish(msg)
            for _ in range(100):
                if seen:
                    break
                rclpy.spin_once(bridge.fast_node, timeout_sec=0.05)
            assert seen and seen[0] == (0.6, 0.0, 0.2)
        finally:
            pub_node.destroy_node()
            bridge.destroy()

    def test_speed_keepalive_published(self) -> None:
        import rclpy
        import time
        from std_msgs.msg import Float32

        bridge = self._bridge(SimpleNamespace())
        got: dict = {}
        sub_node = rclpy.create_node("speed_probe")
        sub_node.create_subscription(
            Float32, "/speed", lambda m: got.setdefault("speed", m), 5
        )
        try:
            deadline = time.time() + 5.0
            while time.time() < deadline and not got:
                bridge._publish_speed()
                rclpy.spin_once(sub_node, timeout_sec=0.05)
            assert got["speed"].data == pytest.approx(0.8)
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
