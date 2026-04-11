"""Gazebo Harmonic Go2 proxy — velocity-controlled Go2 in Gz Sim.

Uses Go2ROS2Proxy's /cmd_vel_nav (Twist) interface directly.
Gazebo's VelocityControl plugin accepts Twist and moves the base.
No joint controller needed — legs are fixed in standing pose.

Topic interface (inherited from Go2ROS2Proxy):
    Publishes:   /cmd_vel_nav       (geometry_msgs/Twist)
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
    """BaseProtocol for Go2 in Gazebo Harmonic (velocity-controlled)."""

    _NODE_NAME: str = "gazebo_go2_proxy"

    @property
    def name(self) -> str:
        return "gazebo_go2"

    @property
    def supports_lidar(self) -> bool:
        return True

    def connect(self) -> None:
        if not self.is_gazebo_running():
            raise ConnectionError(
                "Gazebo not running — /clock topic not found. "
                "Start with: bash scripts/launch_gazebo.sh"
            )
        logger.info("Gazebo confirmed running, connecting via ROS2...")
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
