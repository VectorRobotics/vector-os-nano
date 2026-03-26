"""NavStackClient -- unified navigation client for CMU nav stack and Nav2.

Supports two navigation back-ends selectable at construction time:

CMU mode ("cmu")
    Original topic-based interface:
      /way_point         (geometry_msgs/PointStamped) -- goal position
      /state_estimation  (nav_msgs/Odometry) -- robot pose from SLAM
      /goal_reached      (std_msgs/Bool) -- True when goal reached
      /cancel_goal       (std_msgs/Bool) -- cancel active navigation

Nav2 mode ("nav2")
    ROS2 action-server interface:
      /navigate_to_pose  (nav2_msgs/action/NavigateToPose) -- goal + feedback + result
      /odom              (nav_msgs/Odometry) -- robot pose

Auto mode ("auto") [default]
    Probes for the Nav2 action server (2-second timeout).  If found, uses Nav2;
    otherwise falls back to the CMU topic interface.

All rclpy / ROS2 message imports are lazy so this module can be imported in
environments that do not have ROS2 installed.  When ROS2 is unavailable,
``is_available`` returns ``False`` and every navigation call returns ``False``.

Thread safety note
------------------
``navigate_to`` blocks the calling thread with a polling loop (0.1 s steps).
ROS2 callbacks (``_on_goal_reached``, ``_on_state_estimation``,
``_on_nav2_feedback``) fire on the executor spin thread.  The only shared
state mutations are simple attribute assignments which are GIL-protected in
CPython.  For production use, run the ROS2 node's executor in a dedicated
thread alongside the blocking ``navigate_to`` call.
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class NavStackClient:
    """Unified navigation client supporting both CMU nav stack and Nav2.

    Modes:
        "cmu":  Original topic-based interface (/way_point, /goal_reached).
        "nav2": Nav2 action server interface (NavigateToPose).
        "auto": Try Nav2 first (2 s timeout), fallback to CMU.

    Args:
        node:    An rclpy.node.Node instance, or None when ROS2 is unavailable.
        mode:    One of "cmu", "nav2", or "auto" (default "auto").
        timeout: Default seconds to wait for navigation to complete.
    """

    def __init__(
        self,
        node: Any = None,
        mode: str = "auto",
        timeout: float = 60.0,
    ) -> None:
        self._node = node
        self._timeout = timeout
        self._requested_mode = mode   # "auto" | "nav2" | "cmu"
        self._active_mode: str | None = None  # resolved after _setup_ros2

        # CMU state (existing, preserved exactly)
        self._goal_reached: bool = False
        self._last_odom: Any = None
        self._waypoint_pub: Any = None
        self._cancel_pub: Any = None

        # Nav2 state (new)
        self._nav2_client: Any = None
        self._nav2_goal_handle: Any = None
        self._nav2_feedback: Any = None

        if node is not None:
            self._setup_ros2(node)

    # ------------------------------------------------------------------
    # ROS2 setup (lazy imports)
    # ------------------------------------------------------------------

    def _setup_ros2(self, node: Any) -> None:
        """Initialise ROS2 interface based on the requested mode.

        Mode resolution:
        - "nav2"  -- create ActionClient; fail gracefully if nav2_msgs missing.
        - "cmu"   -- create topic publishers/subscribers (original behaviour).
        - "auto"  -- probe Nav2 server for 2 s; use nav2 if found, else cmu.

        After this call ``_active_mode`` is set to "nav2" or "cmu".
        """
        mode = self._requested_mode

        if mode in ("nav2", "auto"):
            try:
                from rclpy.action import ActionClient
                from nav2_msgs.action import NavigateToPose

                self._nav2_client = ActionClient(node, NavigateToPose, "navigate_to_pose")

                if mode == "auto":
                    if self._nav2_client.wait_for_server(timeout_sec=2.0):
                        self._active_mode = "nav2"
                        logger.info("NavStackClient: Nav2 action server detected")
                    else:
                        self._active_mode = "cmu"
                        try:
                            self._nav2_client.destroy()
                        except Exception:
                            pass
                        self._nav2_client = None
                        logger.info(
                            "NavStackClient: Nav2 not available, falling back to CMU"
                        )
                else:
                    # Explicit nav2 mode -- do not wait, just register client
                    self._active_mode = "nav2"

            except (ImportError, ModuleNotFoundError):
                if mode == "nav2":
                    logger.error("NavStackClient: nav2_msgs not available")
                    self._node = None
                    return
                # auto fallthrough to CMU
                self._active_mode = "cmu"
                self._nav2_client = None
            except Exception as exc:
                # Catches AttributeError/TypeError that can occur when nav2_msgs types
                # are mocked and the real rclpy ActionClient tries type-support checks.
                logger.warning("NavStackClient: Nav2 setup failed (%s), using CMU", exc)
                self._active_mode = "cmu"
                self._nav2_client = None

        if self._active_mode != "nav2":
            self._setup_cmu(node)
            if self._active_mode is None:
                self._active_mode = "cmu"

        # Subscribe to odometry -- topic differs by mode
        try:
            from nav_msgs.msg import Odometry as OdomMsg  # noqa: F401

            odom_topic = "/odom" if self._active_mode == "nav2" else "/state_estimation"
            node.create_subscription(OdomMsg, odom_topic, self._on_state_estimation, 10)
        except (ImportError, ModuleNotFoundError):
            pass

    def _setup_cmu(self, node: Any) -> None:
        """Create CMU nav-stack publishers and subscribers (original logic)."""
        try:
            from geometry_msgs.msg import PointStamped  # noqa: F401
            from std_msgs.msg import Bool  # noqa: F401

            self._waypoint_pub = node.create_publisher(PointStamped, "/way_point", 10)
            self._cancel_pub = node.create_publisher(Bool, "/cancel_goal", 10)
            node.create_subscription(Bool, "/goal_reached", self._on_goal_reached, 10)

            logger.info("NavStackClient: CMU publishers/subscribers created")
        except (ImportError, ModuleNotFoundError):
            logger.warning(
                "NavStackClient: ROS2 messages not available — running without ROS2"
            )
            self._node = None

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------

    def _on_goal_reached(self, msg: Any) -> None:
        """Handle /goal_reached Bool message (CMU mode)."""
        self._goal_reached = bool(msg.data)

    def _on_state_estimation(self, msg: Any) -> None:
        """Handle odometry message, convert to internal Odometry type.

        Topic is /state_estimation (CMU) or /odom (Nav2) depending on mode.
        """
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

    def _on_nav2_feedback(self, feedback_msg: Any) -> None:
        """Store the latest Nav2 NavigateToPose feedback."""
        self._nav2_feedback = feedback_msg.feedback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True if ROS2 node is connected and the active back-end is ready."""
        if self._node is None:
            return False
        if self._active_mode == "nav2":
            return self._nav2_client is not None
        return self._waypoint_pub is not None

    @property
    def mode(self) -> str | None:
        """Return the active navigation mode ('nav2', 'cmu', or None)."""
        return self._active_mode

    @property
    def nav2_feedback(self) -> Any:
        """Return the latest Nav2 feedback object, or None.

        The feedback type is ``nav2_msgs.action.NavigateToPose.Feedback`` and
        contains fields such as ``current_pose``, ``distance_remaining``,
        ``navigation_time``, and ``estimated_time_remaining``.
        """
        return self._nav2_feedback

    def navigate_to(self, x: float, y: float, timeout: float | None = None) -> bool:
        """Navigate to (x, y) using whichever back-end is active.

        Blocks until the goal is reached, rejected, or timeout expires.

        Args:
            x:       Target X in the map frame (metres).
            y:       Target Y in the map frame (metres).
            timeout: Override the instance default timeout (seconds).

        Returns:
            True if goal reached; False on failure, rejection, or timeout.
        """
        if not self.is_available:
            logger.warning("NavStackClient: not available, cannot navigate")
            return False

        if self._active_mode == "nav2":
            return self._nav2_navigate(x, y, timeout)
        return self._cmu_navigate(x, y, timeout)

    # ------------------------------------------------------------------
    # CMU back-end (original logic, preserved exactly)
    # ------------------------------------------------------------------

    def _cmu_navigate(self, x: float, y: float, timeout: float | None = None) -> bool:
        """Navigate via CMU nav stack topic interface."""
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
            time.sleep(0.1)

        if self._goal_reached:
            logger.info("NavStackClient: goal reached")
            return True

        logger.warning("NavStackClient: navigation timed out after %.1f s", wait_timeout)
        return False

    # ------------------------------------------------------------------
    # Nav2 back-end
    # ------------------------------------------------------------------

    def _nav2_navigate(self, x: float, y: float, timeout: float | None = None) -> bool:
        """Navigate using the Nav2 NavigateToPose action server."""
        try:
            from nav2_msgs.action import NavigateToPose
            from geometry_msgs.msg import PoseStamped
            from action_msgs.msg import GoalStatus
        except (ImportError, ModuleNotFoundError):
            logger.error("NavStackClient: nav2_msgs not importable")
            return False

        wait_timeout = timeout if timeout is not None else self._timeout

        # Build goal message
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self._node.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.position.z = 0.0
        goal.pose.pose.orientation.w = 1.0  # face forward

        self._nav2_feedback = None
        logger.info("NavStackClient [nav2]: navigating to (%.2f, %.2f)", x, y)

        send_future = self._nav2_client.send_goal_async(
            goal,
            feedback_callback=self._on_nav2_feedback,
        )

        start = time.time()

        # Wait for goal acceptance
        while not send_future.done() and (time.time() - start) < wait_timeout:
            time.sleep(0.1)

        if not send_future.done():
            logger.warning("NavStackClient [nav2]: goal send timed out")
            return False

        self._nav2_goal_handle = send_future.result()
        if not self._nav2_goal_handle.accepted:
            logger.warning("NavStackClient [nav2]: goal rejected")
            self._nav2_goal_handle = None
            return False

        logger.info("NavStackClient [nav2]: goal accepted")

        result_future = self._nav2_goal_handle.get_result_async()

        while not result_future.done() and (time.time() - start) < wait_timeout:
            time.sleep(0.1)

        if not result_future.done():
            logger.warning("NavStackClient [nav2]: timeout, cancelling goal")
            self._nav2_goal_handle.cancel_goal_async()
            self._nav2_goal_handle = None
            return False

        result = result_future.result()
        self._nav2_goal_handle = None

        if result.status == GoalStatus.STATUS_SUCCEEDED:
            logger.info("NavStackClient [nav2]: goal reached")
            return True

        logger.warning(
            "NavStackClient [nav2]: navigation failed (status=%d)", result.status
        )
        return False

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Cancel the active navigation goal.

        Nav2 mode: cancels via the action goal handle.
        CMU mode:  publishes True to /cancel_goal.
        """
        if not self.is_available:
            return

        if self._active_mode == "nav2" and self._nav2_goal_handle is not None:
            try:
                self._nav2_goal_handle.cancel_goal_async()
                logger.info("NavStackClient [nav2]: goal cancelled")
            except Exception as exc:
                logger.warning("NavStackClient [nav2]: cancel failed: %s", exc)
            self._nav2_goal_handle = None
        elif self._active_mode != "nav2":
            try:
                from std_msgs.msg import Bool

                msg = Bool()
                msg.data = True
                self._cancel_pub.publish(msg)
                logger.info("NavStackClient: navigation cancelled")
            except Exception as exc:
                logger.warning("NavStackClient: cancel failed: %s", exc)

    # ------------------------------------------------------------------
    # State estimation
    # ------------------------------------------------------------------

    def get_state_estimation(self) -> Any:
        """Return the latest state estimation snapshot as an Odometry object, or None."""
        return self._last_odom
