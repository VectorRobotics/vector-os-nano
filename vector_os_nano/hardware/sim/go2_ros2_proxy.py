"""Go2 ROS2 Proxy — controls Go2 via ROS2 topics instead of direct MuJoCo.

Used when the MuJoCo simulation is managed by an external process
(e.g., launch_explore.sh) and we need to send commands via ROS2.
"""
from __future__ import annotations

import math
import os
import time
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nav config loader (lazy, module-level cache)
# ---------------------------------------------------------------------------

_NAV_CFG: dict | None = None


def _load_nav_config() -> dict:
    """Load nav.yaml with defaults. Searches relative paths then falls back."""
    global _NAV_CFG
    if _NAV_CFG is not None:
        return _NAV_CFG

    import yaml

    _search = [
        "config/nav.yaml",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "nav.yaml"),
    ]
    for path in _search:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = yaml.safe_load(f) or {}
                _NAV_CFG = data
                return _NAV_CFG
            except Exception as exc:
                logger.warning("nav.yaml load failed (%s), using defaults", exc)
    _NAV_CFG = {}
    return _NAV_CFG


def _nav(key: str, default: float) -> float:
    """Look up a navigation parameter by key, return default if absent."""
    cfg = _load_nav_config()
    nav_section = cfg.get("navigation", {})
    return float(nav_section.get(key, default))


class Go2ROS2Proxy:
    """Proxy that implements the same interface as MuJoCoGo2 but via ROS2 topics.

    Publishes: /cmd_vel_nav (Twist) for velocity commands
    Subscribes: /state_estimation (Odometry) for position/heading
                /camera/image (Image) for VLM perception
    """

    _NODE_NAME: str = "go2_agent_proxy"

    def __init__(self) -> None:
        self._node: Any = None
        self._cmd_pub: Any = None
        self._position: tuple[float, float, float] = (0.0, 0.0, 0.28)
        self._heading: float = 0.0
        self._connected: bool = False
        self._last_odom: Any = None
        self._last_camera_frame: Any = None  # numpy (H, W, 3) uint8
        self._last_depth_frame: Any = None   # numpy (H, W) float32 metres
        self._last_camera_ts: float = 0.0    # monotonic time of last /camera/image
        self._last_depth_ts: float = 0.0     # monotonic time of last /camera/depth
        # Tracks the last time a /path message arrived from localPlanner.
        # Used by navigate_to() FAR probe phase to detect whether FAR has a
        # routing graph. Value 0.0 means no_path_received yet.
        self._last_path_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Initialise rclpy node, publisher, and odometry subscriber."""
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, ReliabilityPolicy
            from nav_msgs.msg import Odometry
            import threading

            if not rclpy.ok():
                rclpy.init()

            self._node = Node(self._NODE_NAME)

            reliable_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                depth=5,
            )

            from geometry_msgs.msg import Twist, PointStamped  # noqa: F401
            from sensor_msgs.msg import Image

            self._cmd_pub = self._node.create_publisher(Twist, "/cmd_vel_nav", 10)
            # /goal_point → FAR planner (global route planning)
            self._goal_pub = self._node.create_publisher(
                PointStamped, "/goal_point", 10
            )
            # /way_point → localPlanner (direct goal, overrides TARE at 2Hz)
            self._waypoint_pub = self._node.create_publisher(
                PointStamped, "/way_point", 10
            )
            self._node.create_subscription(
                Odometry, "/state_estimation", self._odom_cb, reliable_qos
            )
            self._node.create_subscription(
                Image, "/camera/image", self._camera_cb, reliable_qos
            )
            self._node.create_subscription(
                Image, "/camera/depth", self._depth_cb, reliable_qos
            )
            # Subscribe to /way_point from FAR planner — used by navigate_to()
            # to detect whether FAR has a routing graph and is actively routing.
            # /way_point is ONLY published by FAR (or TARE during explore).
            # /path from localPlanner is unreliable — it publishes even without
            # a valid goal, causing false-positive FAR detection.
            self._last_waypoint_time: float = 0.0
            self._node.create_subscription(
                PointStamped, "/way_point", self._waypoint_cb, 10
            )

            # Scene graph marker publisher (agent sets self._scene_graph)
            self._scene_graph = None
            self._nav_goal: tuple[float, float] | None = None
            self._trajectory: list[tuple[float, float]] = []
            self._last_marker_hash: int | None = None
            self._last_marker_publish_time: float = 0.0
            try:
                from visualization_msgs.msg import MarkerArray
                self._marker_pub = self._node.create_publisher(
                    MarkerArray, "/scene_graph_markers", 5
                )
                self._node.create_timer(3.0, self._publish_markers)
            except ImportError:
                self._marker_pub = None

            # Spin in a background daemon thread so the caller is not blocked.
            self._spin_thread = threading.Thread(
                target=lambda: rclpy.spin(self._node), daemon=True
            )
            self._spin_thread.start()
            self._connected = True

            # Wait up to 5 s for the first odometry message.
            for _ in range(50):
                if self._position != (0.0, 0.0, 0.28):
                    break
                time.sleep(0.1)

            logger.info("Go2ROS2Proxy connected")
        except Exception as exc:
            logger.error("Failed to connect Go2ROS2Proxy: %s", exc)
            self._connected = False

    def disconnect(self) -> None:
        """Destroy the rclpy node and mark proxy as disconnected."""
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
        self._connected = False

    # ------------------------------------------------------------------
    # Internal callback
    # ------------------------------------------------------------------

    def _odom_cb(self, msg: Any) -> None:
        """Update cached position and heading from incoming Odometry message."""
        self._last_odom = msg
        self._position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        )
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._heading = math.atan2(siny_cosp, cosy_cosp)

    def _camera_cb(self, msg: Any) -> None:
        """Cache latest camera frame from /camera/image (RGB8 240x320)."""
        try:
            import numpy as np
            frame = np.frombuffer(msg.data, dtype=np.uint8)
            frame = frame.reshape((msg.height, msg.width, 3))
            self._last_camera_frame = frame
            self._last_camera_ts = time.monotonic()
        except Exception:
            pass

    def _depth_cb(self, msg: Any) -> None:
        """Cache latest depth frame from /camera/depth (32FC1 240x320).

        Bridge publishes depth as 32FC1 (float32, single channel) in metres.
        """
        try:
            import numpy as np
            frame = np.frombuffer(msg.data, dtype=np.float32)
            frame = frame.reshape((msg.height, msg.width))
            self._last_depth_frame = frame
            self._last_depth_ts = time.monotonic()
        except Exception:
            pass

    def _waypoint_cb(self, msg: Any) -> None:
        """Record timestamp and position when FAR publishes a /way_point.

        This is the ONLY reliable signal that FAR has a routing graph and is
        actively routing to our /goal_point. localPlanner's /path is unreliable
        because it publishes even without a valid goal from FAR.
        """
        self._last_waypoint_time = time.time()
        self._last_waypoint_pos = (msg.point.x, msg.point.y)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_position(self) -> tuple[float, float, float]:
        """Return last known (x, y, z) position in metres."""
        return self._position

    def get_heading(self) -> float:
        """Return last known heading in radians (yaw from odometry)."""
        return self._heading

    def get_camera_frame(self, width: int = 320, height: int = 240) -> Any:
        """Return latest RGB camera frame as (H, W, 3) uint8 numpy array.

        Received from /camera/image topic published by Go2VNavBridge.
        Returns a black frame if no image has been received yet.
        Logs a warning if the cached frame is more than 1 second old.
        """
        import numpy as np
        if self._last_camera_ts > 0:
            age = time.monotonic() - self._last_camera_ts
            if age > 1.0:
                logger.warning("[GO2] Camera frame is %.1fs old", age)
        if self._last_camera_frame is not None:
            return self._last_camera_frame.copy()
        return np.zeros((height, width, 3), dtype=np.uint8)

    def get_depth_frame(self, width: int = 320, height: int = 240) -> Any:
        """Return latest depth frame as (H, W) float32 array in metres.

        Received from /camera/depth topic (32FC1) published by Go2VNavBridge.
        Returns a zero frame if no depth has been received yet.
        Logs a warning if the cached frame is more than 1 second old.
        """
        import numpy as np
        if self._last_depth_ts > 0:
            age = time.monotonic() - self._last_depth_ts
            if age > 1.0:
                logger.warning("[GO2] Depth frame is %.1fs old", age)
        if self._last_depth_frame is not None:
            return self._last_depth_frame.copy()
        return np.zeros((height, width), dtype=np.float32)

    def get_rgbd_frame(self, width: int = 320, height: int = 240) -> Any:
        """Return aligned (rgb, depth) tuple.

        Sim-to-real compatible: same interface as MuJoCoGo2.get_rgbd_frame().
        """
        return self.get_camera_frame(width, height), self.get_depth_frame(width, height)

    def get_camera_pose(self) -> tuple:
        """Compute D435 camera world pose from robot odometry + mount config.

        Returns (cam_xpos, cam_xmat) matching MuJoCoGo2.get_camera_pose().
        Camera mounted at: 0.3m forward, 0.05m up, -5deg pitch on base_link.
        """
        import numpy as np

        pos = self._position
        heading = self._heading

        # Mount offset in body frame
        mount_fwd, mount_up = 0.3, 0.05
        pitch = math.radians(-5.0)  # -5 deg downward tilt

        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        cos_p = math.cos(pitch)
        sin_p = math.sin(pitch)

        # Camera world position
        cam_x = pos[0] + cos_h * mount_fwd
        cam_y = pos[1] + sin_h * mount_fwd
        cam_z = pos[2] + mount_up
        cam_xpos = np.array([cam_x, cam_y, cam_z])

        # Camera rotation: MuJoCo convention columns = [right, up, -forward]
        # Body frame: forward=(cos_h, sin_h, 0), right=(-sin_h, cos_h, 0)
        # With pitch: forward rotated by pitch around right axis
        fwd = np.array([cos_h * cos_p, sin_h * cos_p, sin_p])
        right = np.array([-sin_h, cos_h, 0.0])
        up = np.cross(right, fwd)  # ensure orthogonal

        # MuJoCo xmat: columns = [right, up, -forward]
        cam_xmat = np.column_stack([right, up, -fwd]).flatten()

        return (cam_xpos, cam_xmat)

    def get_odometry(self) -> Any:
        """Return latest Odometry data as a types.Odometry dataclass."""
        from vector_os_nano.core.types import Odometry
        pos = self._position
        return Odometry(
            timestamp=time.time(),
            x=pos[0], y=pos[1], z=pos[2],
            qx=0.0, qy=0.0, qz=0.0, qw=1.0,
            vx=0.0, vy=0.0, vz=0.0, vyaw=0.0,
        )

    @property
    def name(self) -> str:
        return "go2_ros2_proxy"

    @property
    def supports_lidar(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Motion interface (matches MuJoCoGo2 public API)
    # ------------------------------------------------------------------

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        """Publish a Twist command on /cmd_vel_nav (non-blocking)."""
        if self._node is None:
            return
        from geometry_msgs.msg import Twist

        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(vyaw)
        self._cmd_pub.publish(msg)

    def walk(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        vyaw: float = 0.0,
        duration: float = 1.0,
    ) -> bool:
        """Walk at the given velocity for *duration* seconds, then stop.

        Publishes velocity at 4Hz to keep overriding the bridge path follower
        (which reclaims control 0.5s after last /cmd_vel_nav message).
        """
        deadline = time.time() + duration
        while time.time() < deadline:
            self.set_velocity(vx, vy, vyaw)
            time.sleep(0.25)  # 4Hz — keeps _teleop_until fresh
        self.set_velocity(0.0, 0.0, 0.0)
        return True

    def stop(self) -> None:
        """Emergency stop — immediately halt all motion."""
        self.set_velocity(0.0, 0.0, 0.0)

    def stand(self, duration: float = 1.0) -> bool:
        """Stop motion and hold position for *duration* seconds."""
        self.set_velocity(0.0, 0.0, 0.0)
        time.sleep(duration)
        return True

    def sit(self, duration: float = 1.0) -> bool:
        """Best-effort sit: stop motion (cannot command sit via ROS2 velocity)."""
        self.set_velocity(0.0, 0.0, 0.0)
        time.sleep(duration)
        return True

    # ------------------------------------------------------------------
    # Navigation via FAR planner / nav stack
    # ------------------------------------------------------------------

    def navigate_to(
        self,
        x: float,
        y: float,
        timeout: float = 60.0,
        on_progress: Callable[[float, float], None] | None = None,
    ) -> bool:
        """Navigate to (x, y) via FAR planner global route planning.

        Two-phase strategy:

        Phase 1 — FAR probe (3 s, /goal_point only):
          Publish /goal_point and monitor /path messages from localPlanner.
          If FAR has a routing graph it will begin publishing /way_point, which
          causes localPlanner to emit /path messages.  If no_path_received
          within 3 s we perform an early_fallback: return False immediately so
          NavigateSkill can fall through to dead-reckoning (door-aware routing).

        Phase 2 — Full navigation (FAR responded):
          Publish /goal_point only at 2 Hz. FAR routes through V-Graph and
          publishes intermediate /way_point to localPlanner. We do NOT publish
          /way_point directly — that would override FAR's door routing.

        FAR's 5 Hz /way_point naturally overrides TARE's 1 Hz exploration
        waypoints during navigation.

        Returns True when within 0.8 m of goal, False on timeout or when
        FAR graph is unavailable.
        """
        if self._node is None:
            logger.warning("[NAV] navigate_to called but node not connected")
            return False

        # Enable bridge path follower
        try:
            with open("/tmp/vector_nav_active", "w") as fh:
                fh.write("1")
        except OSError as exc:
            logger.warning("[NAV] Could not create nav flag: %s", exc)

        self._nav_goal = (float(x), float(y))
        logger.info("[NAV] navigate_to(%.2f, %.2f) timeout=%.0fs", x, y, timeout)

        start_time = time.time()
        _ARRIVAL_DIST: float = _nav("arrival_radius", 0.8)
        _FAR_PROBE_S: float = _nav("far_probe_timeout", 3.0)  # shorter probe — door-chain is reliable backup
        _MIN_PROBE_S: float = 1.5    # minimum wait

        # Reset waypoint timestamp to ignore stale /way_point from TARE.
        # /way_point is published by FAR (routing) or TARE (exploring).
        # We only care about FAR's response to our /goal_point.
        self._last_waypoint_time = 0.0

        # Phase 1: probe FAR — send /goal_point, wait for /way_point response
        probe_deadline = start_time + _FAR_PROBE_S
        while time.time() < probe_deadline:
            if not os.path.exists("/tmp/vector_nav_active"):
                logger.info("[NAV] Cancelled by stop command")
                self._nav_goal = None
                return False
            self._publish_goal_point(x, y)
            time.sleep(0.5)
            elapsed = time.time() - start_time
            if self._last_waypoint_time > start_time and elapsed >= _MIN_PROBE_S:
                logger.info("[NAV] FAR responded with /way_point after %.1fs", elapsed)
                break

        far_available = self._last_waypoint_time > start_time
        if not far_available:
            # FAR has no V-Graph — use door-chain fallback via localPlanner.
            # Publish door waypoints to /way_point one by one. localPlanner
            # handles obstacle avoidance for each segment.
            logger.warning(
                "[NAV] No FAR response after %.0fs — using door-chain fallback",
                _FAR_PROBE_S,
            )
            remaining = timeout - (time.time() - start_time)
            return self._navigate_via_doors(x, y, remaining)

        # Phase 2: full navigation loop (FAR is routing)
        # ONLY publish /goal_point — let FAR handle /way_point routing.
        # FAR routes through doors via V-Graph and publishes intermediate
        # /way_point at 5Hz.
        # Phase 2: full navigation loop (FAR is routing)
        deadline = start_time + timeout
        _last_diag = 0.0
        _last_dist = float("inf")  # for stall detection
        _stall_time = 0.0
        _last_progress = time.time()
        while time.time() < deadline:
            # --- Cancel check: stop command removes nav flag ---
            if not os.path.exists("/tmp/vector_nav_active"):
                logger.info("[NAV] Cancelled by stop command")
                self._nav_goal = None
                self.set_velocity(0.0, 0.0, 0.0)
                return False

            # --- Abort check (cognitive layer) ---
            try:
                from vector_os_nano.vcli.cognitive.abort import is_abort_requested
                if is_abort_requested():
                    logger.info("[NAV] Abort requested — cancelling navigate_to")
                    self._nav_goal = None
                    return False
            except ImportError:
                pass

            # --- Stall flag written by bridge _stuck_detector ---
            if os.path.exists("/tmp/vector_nav_stalled"):
                os.remove("/tmp/vector_nav_stalled")
                logger.warning("[NAV] Stall flag detected — aborting navigate_to")
                self._nav_goal = None
                return False

            self._publish_goal_point(x, y)
            time.sleep(0.5)

            pos = self.get_position()
            dist = math.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)

            # --- Progress callback every 2s ---
            now = time.time()
            if on_progress is not None and now - _last_progress >= 2.0:
                _last_progress = now
                on_progress(dist, now - start_time)

            # --- Stall detection: if not getting closer, fall back to door-chain ---
            if dist < _last_dist - 0.1:
                _stall_time = 0.0  # making progress
            else:
                _stall_time += 0.5
            _last_dist = dist

            if _stall_time > 10.0:
                logger.warning(
                    "[NAV] Stalled %.0fs (dist=%.1fm not decreasing) — switching to door-chain",
                    _stall_time, dist,
                )
                remaining = timeout - (time.time() - start_time)
                return self._navigate_via_doors(x, y, remaining)

            elapsed = time.time() - start_time
            if elapsed - _last_diag >= 5.0:
                _last_diag = elapsed
                wp_age = time.time() - self._last_waypoint_time if self._last_waypoint_time > 0 else -1
                wp_pos = getattr(self, "_last_waypoint_pos", None)
                wp_str = f"wp=({wp_pos[0]:.1f},{wp_pos[1]:.1f})" if wp_pos else "wp=?"
                logger.info(
                    "[NAV] t=%.0fs pos=(%.1f,%.1f) goal=(%.1f,%.1f) "
                    "dist=%.1fm %s age=%.1fs",
                    elapsed, pos[0], pos[1], x, y, dist, wp_str, wp_age,
                )

            if dist < _ARRIVAL_DIST:
                logger.info(
                    "[NAV] Arrived at (%.2f, %.2f) — distance=%.2fm", x, y, dist
                )
                self._nav_goal = None
                return True

        logger.warning(
            "[NAV] navigate_to(%.2f, %.2f) far_timeout after %.0fs", x, y, timeout
        )
        return False

    def _publish_waypoint(self, x: float, y: float) -> None:
        """Publish PointStamped to /way_point (localPlanner goal topic)."""
        if self._node is None:
            return
        try:
            from geometry_msgs.msg import PointStamped

            msg = PointStamped()
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.header.frame_id = "map"
            msg.point.x = float(x)
            msg.point.y = float(y)
            msg.point.z = 0.0
            self._waypoint_pub.publish(msg)
        except Exception as exc:
            logger.warning("[NAV] Failed to publish waypoint: %s", exc)

    def go_to_waypoint(
        self,
        x: float,
        y: float,
        timeout: float = 30.0,
        on_progress: Callable[[float, float], None] | None = None,
    ) -> bool:
        """Navigate to (x, y) by publishing /way_point directly to localPlanner.

        Unlike navigate_to(), this does NOT probe FAR and does NOT fall back
        to door-chain.  It simply publishes /way_point at 2 Hz and waits for
        localPlanner + path follower to reach the goal.

        Used by dead_reckoning (NavigateSkill) to avoid recursive cascades
        where navigate_to → door-chain → navigate_to → door-chain → ...

        Returns True when within arrival radius, False on timeout or stall.
        """
        if self._node is None:
            return False

        _ARRIVAL = _nav("arrival_radius", 0.8)
        _STALL_LIMIT = _nav("stall_timeout", 30.0)

        start = time.time()
        deadline = start + timeout
        last_dist = float("inf")
        stall_time = 0.0
        last_progress_time = start

        logger.info("[NAV] go_to_waypoint(%.1f, %.1f) timeout=%.0fs", x, y, timeout)

        while time.time() < deadline:
            # Cancel check
            if not os.path.exists("/tmp/vector_nav_active"):
                self.set_velocity(0.0, 0.0, 0.0)
                return False

            # Abort check
            try:
                from vector_os_nano.vcli.cognitive.abort import is_abort_requested
                if is_abort_requested():
                    return False
            except ImportError:
                pass

            self._publish_waypoint(x, y)
            time.sleep(0.5)

            pos = self.get_position()
            dist = math.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)

            # Progress callback
            now = time.time()
            if on_progress is not None and now - last_progress_time >= 2.0:
                last_progress_time = now
                on_progress(dist, now - start)

            # Arrival check
            if dist < _ARRIVAL:
                logger.info("[NAV] go_to_waypoint arrived (dist=%.1fm)", dist)
                return True

            # Stall detection — no recursive fallback, just return False
            if dist < last_dist - 0.1:
                stall_time = 0.0
            else:
                stall_time += 0.5
            last_dist = dist

            if stall_time >= _STALL_LIMIT:
                logger.warning(
                    "[NAV] go_to_waypoint stalled %.0fs at dist=%.1fm",
                    stall_time, dist,
                )
                return False

        logger.warning("[NAV] go_to_waypoint timeout after %.0fs", timeout)
        return False

    def _navigate_via_doors(
        self, x: float, y: float, timeout: float,
    ) -> bool:
        """Navigate to (x,y) by publishing door waypoints to /way_point.

        Uses SceneGraph door chain instead of hardcoded room map.
        localPlanner provides obstacle avoidance for each segment.
        """
        sg = self._scene_graph
        if sg is None or not hasattr(sg, "get_door_chain"):
            logger.warning("[NAV] No SceneGraph available for door-chain navigation")
            return False

        pos = self.get_position()
        src_room = sg.nearest_room(float(pos[0]), float(pos[1]))
        dst_room = sg.nearest_room(float(x), float(y))

        if src_room is None or dst_room is None:
            logger.warning(
                "[NAV] Cannot determine rooms for door-chain: src=%s dst=%s",
                src_room, dst_room,
            )
            return False

        # Get waypoint chain from SceneGraph BFS
        chain = sg.get_door_chain(src_room, dst_room)
        if not chain:
            logger.warning("[NAV] No door chain found: %s -> %s", src_room, dst_room)
            return False

        # Replace final waypoint with exact goal coordinates
        # (door chain ends at dst room center, but actual goal may differ)
        chain[-1] = (x, y, chain[-1][2])

        logger.info(
            "[NAV] Door-chain: %s -> %s (%d waypoints)",
            src_room, dst_room, len(chain),
        )

        start = time.time()
        deadline = start + timeout

        for i, (wx, wy, label) in enumerate(chain):
            remaining = deadline - time.time()
            if remaining <= 0:
                logger.warning("[NAV] Door-chain global timeout")
                return False
            n_remaining = len(chain) - i
            per_wp = max(remaining / n_remaining, 5.0)

            logger.info("[NAV] Door-chain -> %s (%.1f, %.1f) budget=%.0fs",
                        label, wx, wy, per_wp)

            ok = self.go_to_waypoint(wx, wy, timeout=per_wp)
            if not ok:
                logger.warning("[NAV] Door-chain: failed to reach %s", label)
                return False

        # Final arrival check
        pos = self.get_position()
        final_dist = math.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)
        arrived = final_dist < 2.0
        logger.info(
            "[NAV] Door-chain %s -- final dist=%.1fm",
            "arrived" if arrived else "failed", final_dist,
        )
        return arrived

    def _publish_goal_point(self, x: float, y: float) -> None:
        """Publish PointStamped to /goal_point (FAR planner input for routing)."""
        if self._node is None:
            return
        try:
            from geometry_msgs.msg import PointStamped

            msg = PointStamped()
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.header.frame_id = "map"
            msg.point.x = float(x)
            msg.point.y = float(y)
            msg.point.z = 0.0
            self._goal_pub.publish(msg)
        except Exception as exc:
            logger.warning("[NAV] Failed to publish goal point: %s", exc)

    def cancel_navigation(self) -> None:
        """Cancel active navigation: publish zero velocity and clear goal.

        The /tmp/vector_nav_active flag is intentionally kept so the bridge
        path follower remains armed — call stop_navigation() to fully
        disarm.
        """
        self.set_velocity(0.0, 0.0, 0.0)
        self._nav_goal = None
        logger.info("[NAV] Navigation cancelled (nav flag retained)")

    def stop_navigation(self) -> None:
        """Fully stop navigation: remove nav flag, zero velocity, clear goal.

        Use this when navigation is complete AND no further nav-stack
        motion is expected (e.g., exploration resumes via its own flag
        logic).
        """
        try:
            os.remove("/tmp/vector_nav_active")
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("[NAV] Could not remove nav flag: %s", exc)

        self.set_velocity(0.0, 0.0, 0.0)
        self._nav_goal = None
        logger.info("[NAV] Navigation stopped, nav flag removed")

    def _scene_graph_hash(self) -> int:
        """Compute a lightweight hash of the current scene graph state.

        Combines rooms count, viewpoints count, objects count, and robot
        position rounded to 0.5 m grid so minor drift does not trigger
        a re-publish.  Returns 0 when no scene graph is available.
        """
        sg = self._scene_graph
        rooms_count = 0
        vp_count = 0
        obj_count = 0
        if sg is not None:
            try:
                rooms = sg.get_all_rooms()
                rooms_count = len(list(rooms))
                for room in sg.get_all_rooms():
                    vp_count += len(sg.get_viewpoints_in_room(room.room_id))
                    obj_count += len(sg.find_objects_in_room(room.room_id))
            except Exception:
                pass
        pos = self._position
        rx = round(pos[0] / 0.5)
        ry = round(pos[1] / 0.5)
        return hash((rooms_count, vp_count, obj_count, rx, ry))

    def _publish_markers(self) -> None:
        """Publish scene graph visualization as MarkerArray at 3 Hz.

        Records current position into trajectory history on every call.
        Caps trajectory at 200 entries to avoid unbounded memory growth.

        Only rebuilds and publishes the MarkerArray when the scene graph
        state hash changes, or every 10 seconds as a keep-alive fallback.
        This prevents unnecessary RViz re-renders that cause flickering.
        """
        if self._marker_pub is None:
            return
        try:
            from vector_os_nano.ros2.nodes.scene_graph_viz import (
                build_scene_graph_markers,
                _TRAJECTORY_MAX_POINTS,
            )
            pos = self._position
            # Always record trajectory (position history should be continuous)
            self._trajectory.append((pos[0], pos[1]))
            if len(self._trajectory) > _TRAJECTORY_MAX_POINTS:
                del self._trajectory[: len(self._trajectory) - _TRAJECTORY_MAX_POINTS]

            # Decide whether to publish: state changed OR 10 s fallback
            now = time.time()
            current_hash = self._scene_graph_hash()
            elapsed = now - self._last_marker_publish_time
            state_changed = current_hash != self._last_marker_hash
            fallback_due = elapsed >= 10.0

            if not (state_changed or fallback_due):
                return

            ma = build_scene_graph_markers(
                scene_graph=self._scene_graph,
                robot_x=pos[0],
                robot_y=pos[1],
                robot_heading=self._heading,
                nav_goal=self._nav_goal,
                trajectory=self._trajectory,
            )
            if ma is not None:
                self._marker_pub.publish(ma)
                self._last_marker_hash = current_hash
                self._last_marker_publish_time = now
        except Exception:
            pass
