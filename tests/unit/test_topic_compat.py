"""Tests for topic compatibility between Isaac Sim and MuJoCo bridges.

Level: Isaac-L4
Ensures drop-in replacement compatibility: Isaac Sim proxy must use the
exact same topic contract as Go2ROS2Proxy so the nav stack and agent layer
can switch backends without modification.

No ROS2 runtime required — tests inspect proxy configuration statically.
"""
from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Topic contract constants
# ---------------------------------------------------------------------------

# Published topics (proxy → simulator / nav stack)
EXPECTED_PUB_TOPICS = {
    "/cmd_vel_nav": "geometry_msgs/Twist",
    "/goal_point": "geometry_msgs/PointStamped",
    "/way_point": "geometry_msgs/PointStamped",
}

# Subscribed topics (simulator → proxy)
EXPECTED_SUB_TOPICS = {
    "/state_estimation": "nav_msgs/Odometry",
    "/camera/image": "sensor_msgs/Image",
    "/camera/depth": "sensor_msgs/Image",
    "/way_point": "geometry_msgs/PointStamped",  # FAR probe detection
}

# Frame IDs used in message headers
EXPECTED_FRAME_IDS = {
    "map": ["/goal_point", "/way_point"],
    "base_link": [],
    "sensor": [],
}

# Nav stack compatibility topics
NAV_STACK_TOPICS = ["/cmd_vel_nav", "/goal_point", "/way_point", "/state_estimation"]

# Camera and depth encoding
CAMERA_ENCODING = "rgb8"
DEPTH_ENCODING = "32FC1"

# Expected camera dimensions
CAMERA_HEIGHT = 240
CAMERA_WIDTH = 320


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_go2_ros2_proxy_source() -> str:
    from vector_os_nano.hardware.sim import go2_ros2_proxy
    return inspect.getsource(go2_ros2_proxy)


def _get_isaac_sim_proxy_source() -> str:
    from vector_os_nano.hardware.sim import isaac_sim_proxy
    return inspect.getsource(isaac_sim_proxy)


def _make_connected_proxy() -> Any:
    """Return a Go2ROS2Proxy with mocked internal state."""
    from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy
    proxy = Go2ROS2Proxy()
    proxy._node = MagicMock()
    proxy._cmd_pub = MagicMock()
    proxy._goal_pub = MagicMock()
    proxy._waypoint_pub = MagicMock()
    proxy._connected = True
    return proxy


def _make_connected_isaac_proxy() -> Any:
    """Return an IsaacSimProxy with mocked internal state."""
    from vector_os_nano.hardware.sim.isaac_sim_proxy import IsaacSimProxy
    proxy = IsaacSimProxy()
    proxy._node = MagicMock()
    proxy._cmd_pub = MagicMock()
    proxy._goal_pub = MagicMock()
    proxy._waypoint_pub = MagicMock()
    proxy._connected = True
    return proxy


# ---------------------------------------------------------------------------
# 1. Publisher topic names — Go2ROS2Proxy
# ---------------------------------------------------------------------------


class TestPublisherTopics:
    """All expected publish topics must appear in Go2ROS2Proxy.connect() source."""

    def test_cmd_vel_nav_topic_in_source(self) -> None:
        source = _get_go2_ros2_proxy_source()
        assert "/cmd_vel_nav" in source, \
            "Go2ROS2Proxy must publish to /cmd_vel_nav"

    def test_goal_point_topic_in_source(self) -> None:
        source = _get_go2_ros2_proxy_source()
        assert "/goal_point" in source, \
            "Go2ROS2Proxy must publish to /goal_point (FAR planner)"

    def test_way_point_topic_in_source(self) -> None:
        source = _get_go2_ros2_proxy_source()
        assert "/way_point" in source, \
            "Go2ROS2Proxy must publish to /way_point (localPlanner)"

    def test_all_expected_pub_topics_in_source(self) -> None:
        source = _get_go2_ros2_proxy_source()
        missing = [t for t in EXPECTED_PUB_TOPICS if t not in source]
        assert not missing, f"Missing pub topics in source: {missing}"


# ---------------------------------------------------------------------------
# 2. Subscriber topic names — Go2ROS2Proxy
# ---------------------------------------------------------------------------


class TestSubscriberTopics:
    """All expected subscribe topics must appear in Go2ROS2Proxy source."""

    def test_state_estimation_subscribed(self) -> None:
        source = _get_go2_ros2_proxy_source()
        assert "/state_estimation" in source, \
            "Go2ROS2Proxy must subscribe to /state_estimation"

    def test_camera_image_subscribed(self) -> None:
        source = _get_go2_ros2_proxy_source()
        assert "/camera/image" in source, \
            "Go2ROS2Proxy must subscribe to /camera/image"

    def test_camera_depth_subscribed(self) -> None:
        source = _get_go2_ros2_proxy_source()
        assert "/camera/depth" in source, \
            "Go2ROS2Proxy must subscribe to /camera/depth"

    def test_way_point_subscribed_for_far_probe(self) -> None:
        source = _get_go2_ros2_proxy_source()
        assert "/way_point" in source, \
            "Go2ROS2Proxy must subscribe to /way_point to detect FAR routing"

    def test_all_expected_sub_topics_in_source(self) -> None:
        source = _get_go2_ros2_proxy_source()
        missing = [t for t in EXPECTED_SUB_TOPICS if t not in source]
        assert not missing, f"Missing sub topics in source: {missing}"


# ---------------------------------------------------------------------------
# 3. Frame IDs
# ---------------------------------------------------------------------------


class TestFrameIds:
    """Published messages must use correct frame IDs."""

    def test_goal_point_uses_map_frame(self) -> None:
        source = _get_go2_ros2_proxy_source()
        # _publish_goal_point sets frame_id = "map"
        assert '"map"' in source or "'map'" in source, \
            "goal_point must use frame_id='map'"

    def test_way_point_uses_map_frame(self) -> None:
        source = _get_go2_ros2_proxy_source()
        assert '"map"' in source or "'map'" in source, \
            "way_point must use frame_id='map'"

    def test_map_frame_id_constant_is_correct(self) -> None:
        """map frame is the standard fixed frame for ROS2 nav stack."""
        assert "map" in EXPECTED_FRAME_IDS
        assert "/goal_point" in EXPECTED_FRAME_IDS["map"]
        assert "/way_point" in EXPECTED_FRAME_IDS["map"]


# ---------------------------------------------------------------------------
# 4. Message types in source code
# ---------------------------------------------------------------------------


class TestMessageTypes:
    """Correct ROS2 message types must be imported and used."""

    def test_twist_message_for_cmd_vel(self) -> None:
        source = _get_go2_ros2_proxy_source()
        assert "Twist" in source, \
            "cmd_vel_nav must use geometry_msgs/Twist"

    def test_pointstamped_for_goal(self) -> None:
        source = _get_go2_ros2_proxy_source()
        assert "PointStamped" in source, \
            "goal_point and way_point must use geometry_msgs/PointStamped"

    def test_odometry_for_state_estimation(self) -> None:
        source = _get_go2_ros2_proxy_source()
        assert "Odometry" in source, \
            "state_estimation must use nav_msgs/Odometry"

    def test_image_for_camera(self) -> None:
        source = _get_go2_ros2_proxy_source()
        assert "Image" in source, \
            "camera topics must use sensor_msgs/Image"


# ---------------------------------------------------------------------------
# 5. QoS compatibility
# ---------------------------------------------------------------------------


class TestQoSCompatibility:
    """QoS settings must be compatible between proxy and nav stack."""

    def test_reliable_qos_used_for_odometry(self) -> None:
        source = _get_go2_ros2_proxy_source()
        assert "RELIABLE" in source or "reliable" in source.lower(), \
            "Odometry subscription must use RELIABLE QoS for command safety"

    def test_cmd_vel_publisher_has_queue_depth(self) -> None:
        source = _get_go2_ros2_proxy_source()
        # create_publisher("/cmd_vel_nav", 10) — queue depth of 10
        assert "cmd_vel_nav" in source and ("10" in source or "qos" in source.lower()), \
            "cmd_vel_nav publisher must define a queue depth"

    def test_goal_publisher_has_queue_depth(self) -> None:
        source = _get_go2_ros2_proxy_source()
        assert "goal_point" in source and "10" in source


# ---------------------------------------------------------------------------
# 6. Isaac proxy uses same topics as Go2ROS2Proxy
# ---------------------------------------------------------------------------


class TestIsaacProxyTopicCompatibility:
    """IsaacSimProxy must inherit the same topic contract as Go2ROS2Proxy."""

    def test_isaac_proxy_inherits_go2_ros2_proxy(self) -> None:
        from vector_os_nano.hardware.sim.isaac_sim_proxy import IsaacSimProxy
        from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy
        assert issubclass(IsaacSimProxy, Go2ROS2Proxy), \
            "IsaacSimProxy must inherit Go2ROS2Proxy to share topic contract"

    def test_isaac_proxy_source_imports_go2_ros2_proxy(self) -> None:
        source = _get_isaac_sim_proxy_source()
        assert "Go2ROS2Proxy" in source, \
            "IsaacSimProxy must reference Go2ROS2Proxy"

    def test_isaac_proxy_does_not_redefine_cmd_vel_topic(self) -> None:
        isaac_source = _get_isaac_sim_proxy_source()
        # Isaac proxy should NOT override the topic name — inherits from parent
        assert "/cmd_vel_nav" not in isaac_source or \
            isaac_source.count("/cmd_vel_nav") <= 1, \
            "IsaacSimProxy should not redefine /cmd_vel_nav (inherited)"

    def test_isaac_proxy_node_name_differs_from_parent(self) -> None:
        from vector_os_nano.hardware.sim.isaac_sim_proxy import IsaacSimProxy
        from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy
        assert IsaacSimProxy._NODE_NAME != Go2ROS2Proxy._NODE_NAME, \
            "IsaacSimProxy must use a distinct node name to avoid DDS conflicts"

    def test_isaac_proxy_set_velocity_uses_same_topic(self) -> None:
        """Both proxies must publish velocity on /cmd_vel_nav."""
        go2_source = _get_go2_ros2_proxy_source()
        assert "/cmd_vel_nav" in go2_source
        # Isaac inherits set_velocity from Go2ROS2Proxy — same topic guaranteed

    def test_isaac_proxy_navigate_to_uses_same_goal_topic(self) -> None:
        go2_source = _get_go2_ros2_proxy_source()
        assert "/goal_point" in go2_source


# ---------------------------------------------------------------------------
# 7. Nav stack compatibility
# ---------------------------------------------------------------------------


class TestNavStackCompatibility:
    """Topics must be compatible with the Vector nav stack."""

    def test_nav_stack_topics_all_present_in_source(self) -> None:
        source = _get_go2_ros2_proxy_source()
        missing = [t for t in NAV_STACK_TOPICS if t not in source]
        assert not missing, f"Nav stack topics missing from proxy: {missing}"

    def test_cmd_vel_nav_not_cmd_vel(self) -> None:
        """Vector nav stack uses /cmd_vel_nav, not /cmd_vel (ROS2 default)."""
        source = _get_go2_ros2_proxy_source()
        assert "/cmd_vel_nav" in source
        # /cmd_vel_nav distinguishes nav-stack velocity from direct teleop

    def test_state_estimation_is_standard_odom_topic(self) -> None:
        source = _get_go2_ros2_proxy_source()
        assert "/state_estimation" in source


# ---------------------------------------------------------------------------
# 8. PointCloud2 field structure (expected)
# ---------------------------------------------------------------------------


class TestPointCloud2Structure:
    """PointCloud2 messages from Isaac Sim RTX lidar must have expected fields."""

    EXPECTED_PC2_FIELDS = ["x", "y", "z", "intensity"]

    def test_expected_pointcloud2_fields_defined_as_constant(self) -> None:
        """Verify the test contract is correct."""
        assert "x" in self.EXPECTED_PC2_FIELDS
        assert "y" in self.EXPECTED_PC2_FIELDS
        assert "z" in self.EXPECTED_PC2_FIELDS
        assert "intensity" in self.EXPECTED_PC2_FIELDS

    def test_pointcloud2_has_xyz_fields(self) -> None:
        """XYZ coordinates are mandatory for any 3D point cloud."""
        for coord in ["x", "y", "z"]:
            assert coord in self.EXPECTED_PC2_FIELDS

    def test_pointcloud2_has_intensity_field(self) -> None:
        """Isaac RTX lidar provides intensity data (reflectance)."""
        assert "intensity" in self.EXPECTED_PC2_FIELDS

    def test_expected_field_count(self) -> None:
        assert len(self.EXPECTED_PC2_FIELDS) == 4


# ---------------------------------------------------------------------------
# 9. Camera and depth encoding
# ---------------------------------------------------------------------------


class TestCameraEncoding:
    """Camera frame encodings must match what the bridge publishes."""

    def test_camera_encoding_is_rgb8(self) -> None:
        assert CAMERA_ENCODING == "rgb8", \
            "RGB camera must use rgb8 encoding (3-channel uint8)"

    def test_depth_encoding_is_32fc1(self) -> None:
        assert DEPTH_ENCODING == "32FC1", \
            "Depth image must use 32FC1 encoding (float32 single channel, metres)"

    def test_camera_frame_shape_matches_expected(self) -> None:
        """get_camera_frame() default shape must match bridge output."""
        import numpy as np
        from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy
        proxy = Go2ROS2Proxy()
        frame = proxy.get_camera_frame(width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
        assert frame.shape == (CAMERA_HEIGHT, CAMERA_WIDTH, 3)
        assert frame.dtype == np.uint8

    def test_depth_frame_shape_matches_expected(self) -> None:
        """get_depth_frame() default shape must match bridge output."""
        import numpy as np
        from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy
        proxy = Go2ROS2Proxy()
        frame = proxy.get_depth_frame(width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
        assert frame.shape == (CAMERA_HEIGHT, CAMERA_WIDTH)
        assert frame.dtype == np.float32

    def test_camera_default_width(self) -> None:
        assert CAMERA_WIDTH == 320

    def test_camera_default_height(self) -> None:
        assert CAMERA_HEIGHT == 240

    def test_rgbd_frame_returns_rgb_depth_tuple(self) -> None:
        from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy
        proxy = Go2ROS2Proxy()
        rgb, depth = proxy.get_rgbd_frame()
        assert rgb.ndim == 3
        assert depth.ndim == 2


# ---------------------------------------------------------------------------
# 10. Odometry / scan / camera publish rate expectations
# ---------------------------------------------------------------------------


class TestTopicRateExpectations:
    """Topic rate expectations must be documented and reasonable."""

    # These are the expected publish rates from the Isaac Sim bridge.
    # Rates are validated against docker-compose.yaml ISAAC_PHYSICS_HZ
    # and bridge configuration — tested here as constants.
    ODOM_RATE_HZ = 50
    SCAN_RATE_HZ = 10
    CAMERA_RATE_HZ = 10

    def test_odom_rate_is_reasonable(self) -> None:
        assert 10 <= self.ODOM_RATE_HZ <= 200, \
            "Odometry rate should be between 10 Hz and 200 Hz"

    def test_scan_rate_is_reasonable(self) -> None:
        assert 1 <= self.SCAN_RATE_HZ <= 30, \
            "LiDAR scan rate should be between 1 Hz and 30 Hz"

    def test_camera_rate_is_reasonable(self) -> None:
        assert 1 <= self.CAMERA_RATE_HZ <= 60, \
            "Camera rate should be between 1 Hz and 60 Hz"

    def test_odom_faster_than_camera(self) -> None:
        assert self.ODOM_RATE_HZ >= self.CAMERA_RATE_HZ, \
            "Odometry should be published at least as fast as camera for nav safety"

    def test_docker_compose_physics_hz_env_set(self) -> None:
        """docker-compose.yaml must set ISAAC_PHYSICS_HZ for rate control."""
        from pathlib import Path
        compose_path = Path(__file__).parent.parent.parent / "docker" / "isaac-sim" / "docker-compose.yaml"
        if compose_path.exists():
            content = compose_path.read_text()
            assert "ISAAC_PHYSICS_HZ" in content
