"""Gazebo Harmonic Go2 proxy — controls Go2 in Gz Sim via ROS2 topics.

Inherits from Go2ROS2Proxy since Gazebo publishes identical ROS2 topics.
Adds /clock topic detection for Gazebo health checking and LiDAR support flag.

Topic interface (same as Go2ROS2Proxy):
    Publishes:   /cmd_vel_nav       (geometry_msgs/Twist)
                 /goal_point        (geometry_msgs/PointStamped)
                 /way_point         (geometry_msgs/PointStamped)
    Subscribes:  /state_estimation  (nav_msgs/Odometry)
                 /camera/image      (sensor_msgs/Image)
                 /camera/depth      (sensor_msgs/Image)
"""
from __future__ import annotations

import logging
import subprocess

from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy

logger = logging.getLogger(__name__)


class GazeboGo2Proxy(Go2ROS2Proxy):
    """BaseProtocol implementation for Go2 running in Gz Sim Harmonic.

    Same ROS2 topic interface as Go2ROS2Proxy — no topic rewiring needed.
    Gazebo publishes simulated LiDAR data (supports_lidar = True) and
    camera frames at the same topic names.
    """

    _NODE_NAME: str = "gazebo_go2_proxy"

    @property
    def name(self) -> str:
        return "gazebo_go2"

    @property
    def supports_lidar(self) -> bool:
        return True

    def connect(self) -> None:
        """Connect to Gazebo Go2 via ROS2 topics.

        Verifies Gazebo is running first by checking for /clock topic.

        Raises:
            ConnectionError: If Gazebo is not running (no /clock topic found).
        """
        if not self.is_gazebo_running():
            raise ConnectionError(
                "Gazebo is not running — /clock topic not found. "
                "Start with: gz sim <world>.sdf"
            )
        logger.info("Gazebo confirmed running (/clock found), connecting via ROS2...")
        super().connect()
        logger.info("GazeboGo2Proxy connected (node=%s)", self._NODE_NAME)

    @staticmethod
    def is_gazebo_running() -> bool:
        """Check if Gazebo is running by detecting the /clock topic."""
        try:
            result = subprocess.run(
                ["ros2", "topic", "list"],
                capture_output=True, text=True, timeout=5,
            )
            return "/clock" in result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as exc:
            logger.debug("is_gazebo_running check failed: %s", exc)
            return False
