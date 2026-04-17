"""Isaac Sim Go2 proxy — controls Go2 in Isaac Sim Docker via ROS2 topics.

Inherits from Go2ROS2Proxy since Isaac Sim publishes identical ROS2 topics.
Adds Docker container health checking and Isaac-specific capability flags.

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

_ISAAC_CONTAINER_NAME: str = "vector-isaac-sim"


class IsaacSimProxy(Go2ROS2Proxy):
    """BaseProtocol implementation for Go2 running in Isaac Sim Docker.

    Same ROS2 topic interface as Go2ROS2Proxy — no topic rewiring needed.
    Isaac Sim publishes RTX LiDAR data (supports_lidar = True) and
    photorealistic camera frames at the same topic names.
    """

    _NODE_NAME: str = "isaac_sim_proxy"

    @property
    def name(self) -> str:
        return "isaac_go2"

    @property
    def supports_lidar(self) -> bool:
        return True

    def connect(self) -> None:
        """Connect to Isaac Sim Go2 via ROS2 topics.

        Verifies the Docker container is running first.

        Raises:
            ConnectionError: If Isaac Sim Docker container is not running.
        """
        if not self.is_isaac_sim_running():
            raise ConnectionError(
                f"Isaac Sim container '{_ISAAC_CONTAINER_NAME}' not running. "
                "Start with: ./scripts/launch_isaac.sh"
            )
        logger.info("Isaac Sim container confirmed running, connecting via ROS2...")
        super().connect()
        logger.info("IsaacSimProxy connected (node=%s)", self._NODE_NAME)

    @staticmethod
    def is_isaac_sim_running() -> bool:
        """Check if the Isaac Sim Docker container is running."""
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={_ISAAC_CONTAINER_NAME}",
                 "--filter", "status=running", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=5,
            )
            return _ISAAC_CONTAINER_NAME in result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as exc:
            logger.debug("is_isaac_sim_running check failed: %s", exc)
            return False
