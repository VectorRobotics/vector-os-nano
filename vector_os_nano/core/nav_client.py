"""NavStackClient -- wrapper for vector_navigation_stack ROS2 interface.

Provides a simple Python API for publishing navigation goals and receiving
state estimation / goal_reached feedback from the navigation stack.

The navigation stack uses these ROS2 topics:
  /way_point         (geometry_msgs/PointStamped) -- goal position
  /state_estimation  (nav_msgs/Odometry) -- robot pose from SLAM
  /goal_reached      (std_msgs/Bool) -- True when goal reached
  /cancel_goal       (std_msgs/Bool) -- cancel active navigation

This module lazy-imports rclpy. When ROS2 is not available, NavStackClient
still instantiates but is_available returns False and navigate_to returns False.

No ROS2 imports at module level.
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class NavStackClient:
    """Python wrapper for the vector navigation stack.

    Args:
        node: An rclpy.node.Node instance (or None if no ROS2).
        timeout: Max seconds to wait for goal_reached in navigate_to().
    """

    def __init__(self, node: Any = None, timeout: float = 60.0) -> None:
        self._node = node
        self._timeout = timeout
        self._goal_reached: bool = False
        self._last_odom: Any = None  # Odometry | None
        self._waypoint_pub: Any = None
        self._cancel_pub: Any = None

        if node is not None:
            self._setup_ros2(node)

    # ------------------------------------------------------------------
    # ROS2 setup (lazy imports)
    # ------------------------------------------------------------------

    def _setup_ros2(self, node: Any) -> None:
        """Create ROS2 publishers and subscribers. All imports are lazy."""
        try:
            from geometry_msgs.msg import PointStamped  # noqa: F401
            from std_msgs.msg import Bool  # noqa: F401
            from nav_msgs.msg import Odometry as OdomMsg  # noqa: F401

            self._waypoint_pub = node.create_publisher(PointStamped, "/way_point", 10)
            self._cancel_pub = node.create_publisher(Bool, "/cancel_goal", 10)

            node.create_subscription(Bool, "/goal_reached", self._on_goal_reached, 10)
            node.create_subscription(OdomMsg, "/state_estimation", self._on_state_estimation, 10)

            logger.info("NavStackClient: ROS2 publishers/subscribers created")
        except (ImportError, ModuleNotFoundError):
            logger.warning("NavStackClient: ROS2 messages not available — running without ROS2")
            self._node = None

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------

    def _on_goal_reached(self, msg: Any) -> None:
        """Handle /goal_reached Bool message."""
        self._goal_reached = bool(msg.data)

    def _on_state_estimation(self, msg: Any) -> None:
        """Handle /state_estimation Odometry message, convert to internal type."""
        try:
            from vector_os_nano.core.types import Odometry

            p = msg.pose.pose.position
            o = msg.pose.pose.orientation
            t = msg.twist.twist
            self._last_odom = Odometry(
                timestamp=time.time(),
                x=float(p.x),
                y=float(p.y),
                z=float(p.z),
                qx=float(o.x),
                qy=float(o.y),
                qz=float(o.z),
                qw=float(o.w),
                vx=float(t.linear.x),
                vy=float(t.linear.y),
                vz=float(t.linear.z),
                vyaw=float(t.angular.z),
            )
        except Exception as exc:
            logger.warning("NavStackClient: state estimation callback error: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True if ROS2 node is connected and publishers are ready."""
        return self._node is not None and self._waypoint_pub is not None

    def navigate_to(self, x: float, y: float, timeout: float | None = None) -> bool:
        """Publish a waypoint goal and wait for goal_reached.

        Args:
            x: Target X in map frame (meters).
            y: Target Y in map frame (meters).
            timeout: Override default timeout (seconds).

        Returns:
            True if goal reached within timeout, False otherwise.
        """
        if not self.is_available:
            logger.warning("NavStackClient: not available, cannot navigate")
            return False

        try:
            from geometry_msgs.msg import PointStamped
        except (ImportError, ModuleNotFoundError):
            logger.error("NavStackClient: geometry_msgs not importable")
            return False

        self._goal_reached = False

        msg = PointStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.point.x = float(x)
        msg.point.y = float(y)
        msg.point.z = 0.0
        self._waypoint_pub.publish(msg)

        logger.info("NavStackClient: navigating to (%.2f, %.2f)", x, y)

        wait_timeout = timeout if timeout is not None else self._timeout
        start = time.time()
        while not self._goal_reached and (time.time() - start) < wait_timeout:
            executor = getattr(self._node, "executor", None)
            if executor is not None:
                executor.spin_once(timeout_sec=0.1)
            else:
                time.sleep(0.05)

        if self._goal_reached:
            logger.info("NavStackClient: goal reached")
            return True

        logger.warning("NavStackClient: navigation timed out after %.1f s", wait_timeout)
        return False

    def cancel(self) -> None:
        """Cancel the active navigation goal by publishing to /cancel_goal."""
        if not self.is_available:
            return
        try:
            from std_msgs.msg import Bool

            msg = Bool()
            msg.data = True
            self._cancel_pub.publish(msg)
            logger.info("NavStackClient: navigation cancelled")
        except Exception as exc:
            logger.warning("NavStackClient: cancel failed: %s", exc)

    def get_state_estimation(self) -> Any:
        """Return the latest state estimation snapshot as Odometry, or None."""
        return self._last_odom
