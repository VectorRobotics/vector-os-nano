# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

"""ROS2 sensor/odometry bridge owned by the optional navigation session."""
import math
import os
import struct
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# ROS2
# ---------------------------------------------------------------------------
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry as OdometryMsg, Path
from sensor_msgs.msg import PointCloud2, PointField, Joy, LaserScan as LaserScanMsg, Image, CompressedImage
from geometry_msgs.msg import TwistStamped, Twist, TransformStamped, PointStamped, PolygonStamped, Point32
from std_msgs.msg import Float32, Header
from tf2_ros import TransformBroadcaster
import numpy as np

from vector_os_nano.core.run_dir import run_path

from .motion import NavigationMotion
from .terrain import TerrainAccumulator
from .parameters import _PF_DT, _SENSOR_X, _SENSOR_Y, _SENSOR_Z, _CEILING_FILTER_HEIGHT

class NavigationBridge(NavigationMotion, Node):
    """ROS2 node bridging a legged sim base (interfaces.BaseProtocol) to the Vector
    Navigation Stack: its camera and lidar drivers, and the session's ``reset`` (stand
    the body up again) for the reset flag file."""

    def __init__(self, go2: Any, camera: Any, lidar: Any, reset: Any, *, terrain_map: os.PathLike,
                 quiet: bool = False) -> None:
        super().__init__("go2_vnav_bridge")
        self._go2 = go2
        self._camera = camera  # the head camera (sim/drivers/camera.Camera)
        self._lidar = lidar    # the head lidar (sim/drivers/lidar.Lidar)
        self._reset = reset
        self._last_cmd_time = time.time()
        # Suppress verbose logging when embedded in CLI (floods terminal)
        if quiet or os.environ.get("VECTOR_BRIDGE_QUIET"):
            import rclpy.logging
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.WARN)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Publishers — topics the Vector Nav Stack expects
        # /state_estimation must be RELIABLE — terrainAnalysis and sensorScanGeneration require it
        self._odom_pub = self.create_publisher(
            OdometryMsg, "/state_estimation", reliable_qos
        )
        self._pc_pub = self.create_publisher(
            PointCloud2, "/registered_scan", reliable_qos
        )
        self._scan_pub = self.create_publisher(
            LaserScanMsg, "/scan", reliable_qos
        )
        self._joy_pub = self.create_publisher(Joy, "/joy", 5)
        self._speed_pub = self.create_publisher(Float32, "/speed", 5)
        self._img_pub = self.create_publisher(Image, "/camera/image", reliable_qos)
        self._depth_pub = self.create_publisher(Image, "/camera/depth", reliable_qos)
        # Real d435_rgb world pose (D36): the bridge owns the live MjData, so it
        # is the single source of truth for the camera pose the depth pixels were
        # rendered from. Published as a flat Float64MultiArray [xpos(3), xmat(9)]
        # so Go2ROS2Proxy.get_camera_pose() can back-project /camera/depth with
        # the EXACT pose (not a hand-approximated mount), which is what makes
        # grasp_point_from_rgbd land on the object instead of high+far.
        from std_msgs.msg import Float64MultiArray
        self._cam_pose_pub = self.create_publisher(
            Float64MultiArray, "/camera/pose", reliable_qos
        )

        self._tf_broadcaster = TransformBroadcaster(self)
        # NOTE: static TF sensor→base_link is published by local_planner.launch.py
        # (vehicleTransPublisher node) — do NOT duplicate here

        # Subscribe to /path from localPlanner — we follow it with our own Python follower
        # (C++ pathFollower oscillates with MPC gait, bypassed)
        self.create_subscription(
            Path, "/path", self._path_cb,
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=1)
        )
        self._current_path: list = []

        # Subscribe to /exploration_finish from TARE — stop path following when done
        from std_msgs.msg import Bool as BoolMsg
        self.create_subscription(BoolMsg, "/exploration_finish", self._exploration_finish_cb, 5)
        self._exploration_finished = False
        self._path_time = 0.0

        # Manual control: teleop /joy → direct velocity
        self.create_subscription(Joy, "/joy", self._joy_cb, 10)
        # Nav stack path follower output
        self.create_subscription(TwistStamped, "/navigation_cmd_vel", self._cmd_vel_stamped_cb, 10)
        # Manual CLI control
        self.create_subscription(Twist, "/cmd_vel_nav", self._cmd_vel_cb, 10)

        # Python path follower — primary. C++ pathFollower sends zeros between
        # TARE waypoints (pathSize<=1), so Python follower is needed to keep
        # the dog moving on stale paths until TARE replans.
        self.create_timer(_PF_DT, self._follow_path)
        # Camera rendering (5 Hz)
        self.create_timer(1.0 / 5.0, self._publish_camera)
        self._cmd_count = 0
        self._teleop_until = 0.0  # teleop priority timestamp

        # Timers
        self.create_timer(1.0 / 200.0, self._publish_odom)       # 200 Hz
        self.create_timer(1.0 / 5.0, self._publish_pointcloud)   # 5 Hz — matches MuJoCo lidar
        self.create_timer(1.0 / 5.0, self._publish_scan)          # 5 Hz — matches MuJoCo lidar
        self.create_timer(0.5, self._publish_joy_speed)           # 2 Hz
        self.create_timer(1.0, self._safety_check)                # 1 Hz
        self.create_timer(2.0, self._stuck_detector)             # 0.5 Hz
        self.create_timer(2.0, self._publish_nav_boundary)       # 0.5 Hz — TARE needs periodic boundary

        # Stuck detector state — triggers /reset_waypoint when no progress
        self._stuck_pos = None       # (x, y) at last check
        self._stuck_count = 0        # consecutive checks with < 0.3m progress
        self._stuck_history: list = []  # (x, y) positions where /reset_waypoint was sent
        self._stuck_wall_clock: float = 0.0  # wall-clock time when first stuck detected
        self._reset_waypoint_pub = self.create_publisher(
            PointStamped, "/reset_waypoint", 5
        )

        # Wall escape state — two-phase: pure reverse then strafe+turn.
        # All in SIM time (like the ramps): the maneuver must travel the same
        # plant distance at any sim/wall ratio, and freeze if the sim pauses.
        self._wall_contact_time: float = 0.0     # sim-seconds pinned (front_d<0.30 AND slow)
        self._wall_escape_elapsed: float = 0.0   # sim-seconds since escape trigger
        self._wall_escape_total: float = 0.0     # escape duration (sim-s); elapsed>=total => inactive
        self._wall_escape_phase2_at: float = 0.0 # sim-s offset where phase 2 (strafe) starts

        # Sim-clock sample from the previous _follow_path tick (None until the
        # first tick). All follower ramps integrate against sim-dt, not wall
        # ticks — see _sim_tick_dt.
        self._pf_last_sim_t: float | None = None

        # Scene graph visualization (1 Hz MarkerArray)
        self._scene_graph = None  # set externally by agent
        try:
            from visualization_msgs.msg import MarkerArray
            self._marker_pub = self.create_publisher(MarkerArray, "/scene_graph_markers", 5)
            self.create_timer(1.0, self._publish_scene_graph_markers)
        except ImportError:
            self._marker_pub = None

        # Scene graph JSON topic (0.5 Hz) — structured data for Foxglove Raw Messages
        try:
            from std_msgs.msg import String
            self._sg_json_pub = self.create_publisher(String, "/vector_os/scene_graph", 5)
            self.create_timer(2.0, self._publish_scene_graph_json)
        except ImportError:
            self._sg_json_pub = None

        # Diagnostic counters — track message flow for TARE data-starvation debugging.
        # Logged every 10s at INFO level via _log_diagnostics timer.
        self._diag_odom_count: int = 0
        self._diag_scan_count: int = 0
        self._diag_path_count: int = 0
        self.create_timer(10.0, self._log_diagnostics)  # 0.1 Hz

        # Terrain persistence — accumulates pointcloud voxels during explore.
        # Auto-saved every 30s while nav is active.
        self._terrain_acc = TerrainAccumulator(voxel_size=0.1)
        self._terrain_map_path = str(terrain_map)  # $VECTOR_HOME/memory/<world>/terrain_map.npz
        self.create_timer(30.0, self._auto_save_terrain)

        # Reset pose check (1 Hz) — triggered by /tmp/vector_reset_pose file flag
        self.create_timer(1.0, self._check_reset_flag)

        # Front obstacle detection from cached pointcloud
        self._cached_points: list = []

        # Navigation gate: path follower is DISABLED until explicitly enabled.
        # Uses a file flag (/tmp/vector_nav_active) — 100% reliable, no ROS2
        # message race conditions. explore.py creates this file when starting.
        self._nav_enabled = False
        self._goal_pub = self.create_publisher(
            PointStamped, "/goal_point", 5
        )
        self.create_timer(1.0, self._check_nav_flag)

        # Terrain replay: load saved map and publish to FAR on startup.
        # CRITICAL TIMING: replay must wait for terrainAnalysis + FAR to start.
        # launch_explore.sh startup order:
        #   Bridge (0s) → localPlanner (7s) → sensorScan (11s) →
        #   terrainAnalysis (12s) → FAR (15s) → TARE (17s) → RViz (19s)
        # Replay before terrainAnalysis = data gets dropped = FAR has no graph.
        # _REPLAY_DELAY: wait this many seconds after bridge init before replaying.
        _REPLAY_DELAY = 20.0  # seconds — all nodes should be up by then
        self._terrain_replay_points: list = []
        self._terrain_replay_count: int = 0
        self._terrain_replay_max: int = 50  # 5Hz × 10s = 50 frames (longer burst)
        self._terrain_replay_start: float = time.time() + _REPLAY_DELAY
        terrain_path = self._terrain_map_path
        if os.path.isfile(terrain_path):
            acc = TerrainAccumulator()
            if acc.load(terrain_path):
                self._terrain_replay_points = acc.to_pointcloud()
                self.get_logger().info(
                    f"Loaded terrain map: {acc.size} voxels from {terrain_path} "
                    f"— replay starts in {_REPLAY_DELAY:.0f}s"
                )
                # Delayed replay timer — checks if it's time to start
                self._terrain_replay_timer = self.create_timer(1.0, self._terrain_replay_gate)
            else:
                self.get_logger().warn(f"Failed to load terrain map from {terrain_path}")

        self.get_logger().info(
            "Go2VNavBridge started — /state_estimation, /registered_scan, /joy, /speed"
        )

    def _check_reset_flag(self) -> None:
        """Check for reset pose request (1 Hz). Triggered by /tmp/vector_reset_pose."""
        if os.path.exists(run_path("vector_reset_pose")):
            try:
                os.remove(run_path("vector_reset_pose"))
            except OSError:
                pass
            self.get_logger().warn("RESET POSE — standing up at current position")
            self._reset()
            self._current_path = []
            self._pf_speed = 0.0
            self._pf_lat = 0.0
            self._pf_yawrate = 0.0

    def _check_nav_flag(self) -> None:
        """Check file flag to enable/disable path following (1 Hz)."""
        flag = os.path.exists(run_path("vector_nav_active"))
        if flag and not self._nav_enabled:
            self._nav_enabled = True
            self._exploration_finished = False  # reset for new explore/navigate
            # Cancel startup terrain replay — new exploration generates fresh data.
            # Without this, old terrain_map.npz makes TARE think area is already explored.
            if self._terrain_replay_points:
                self._terrain_replay_points = []
                self.get_logger().info("Cancelled startup terrain replay for fresh exploration")
            self.get_logger().info("Navigation ENABLED (flag file detected)")
        elif not flag and self._nav_enabled:
            self._nav_enabled = False
            self._current_path = []
            self._go2.set_velocity(0.0, 0.0, 0.0)
            self.get_logger().info("Navigation DISABLED (flag file removed)")

        # Check for terrain replay trigger (set by explore.py after exploration)
        replay_flag = run_path("vector_terrain_replay")
        if os.path.exists(replay_flag):
            try:
                os.remove(replay_flag)
            except OSError:
                pass
            self.save_terrain()
            self._terrain_replay_points = self._terrain_acc.to_pointcloud()
            self._terrain_replay_count = 0
            if self._terrain_replay_points:
                self._terrain_replay_timer = self.create_timer(0.2, self._replay_terrain)
                self.get_logger().info(
                    f"Terrain replay triggered: {len(self._terrain_replay_points)} points"
                )

    def _terrain_replay_gate(self) -> None:
        """Wait for all nav stack nodes to start, then switch to fast replay."""
        if time.time() < self._terrain_replay_start:
            return  # not yet — wait for terrainAnalysis + FAR to start
        # Time to replay — switch from 1Hz gate to 5Hz replay
        self._terrain_replay_timer.cancel()
        self.get_logger().info(
            f"Starting terrain replay: {len(self._terrain_replay_points)} points, "
            f"{self._terrain_replay_max} frames at 5Hz"
        )
        self._terrain_replay_timer = self.create_timer(0.2, self._replay_terrain)

    def _build_terrain_pc2(self, points: list) -> PointCloud2:
        """Build a PointCloud2 message from (x, y, z, intensity) tuples."""
        now = self.get_clock().now().to_msg()
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        point_step = 16
        data = bytearray()
        for x, y, z, intensity in points:
            data.extend(struct.pack("ffff", x, y, z, intensity))
        msg = PointCloud2()
        msg.header.stamp = now
        msg.header.frame_id = "map"
        msg.height = 1
        msg.width = len(points)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = point_step
        msg.row_step = point_step * len(points)
        msg.data = bytes(data)
        msg.is_dense = True
        return msg

    def _replay_terrain(self) -> None:
        """Publish saved terrain as /registered_scan to seed FAR planner (5Hz burst)."""
        if self._terrain_replay_count >= self._terrain_replay_max:
            self._terrain_replay_timer.cancel()
            self.get_logger().info(
                f"Terrain replay complete: {self._terrain_replay_count} frames published"
            )
            return

        if not self._terrain_replay_points:
            self._terrain_replay_timer.cancel()
            return

        msg = self._build_terrain_pc2(self._terrain_replay_points)
        self._pc_pub.publish(msg)

        self._terrain_replay_count += 1
        if self._terrain_replay_count == 1:
            self.get_logger().info(
                f"Terrain replay: publishing {len(self._terrain_replay_points)} points "
                f"({self._terrain_replay_max} frames at 5Hz)"
            )

    def _joy_cb(self, msg: Joy) -> None:
        """Direct teleop: /joy axes → velocity (bypasses pathFollower).

        Sets _teleop_active to prevent path follower from overriding.
        """
        if len(msg.axes) < 5:
            return
        linear = msg.axes[4]    # right stick Y → forward/back
        angular = msg.axes[3]   # right stick X → yaw
        if abs(linear) > 0.05 or abs(angular) > 0.05:
            vx = float(np.clip(linear * 0.8, -0.6, 0.8))
            vyaw = float(np.clip(angular * 2.0, -2.0, 2.0))
            self._go2.set_velocity(vx, 0.0, vyaw)
            self._last_cmd_time = time.time()
            self._teleop_until = time.time() + 0.5
            # Clear path so path follower doesn't resume after teleop
            self._current_path = []
            self._cmd_count += 1
            if self._cmd_count <= 3 or self._cmd_count % 50 == 0:
                self.get_logger().info(f"teleop: vx={vx:.2f} vyaw={vyaw:.2f}")

    def _cmd_vel_stamped_cb(self, msg: TwistStamped) -> None:
        """DISABLED — C++ pathFollower frame convention (sensor/base_link)
        doesn't match MuJoCo Go2 body frame. Needs proper frame transform
        analysis before enabling. Using Python _follow_path instead."""
        pass

    def _cmd_vel_cb(self, msg: Twist) -> None:
        self._go2.set_velocity(msg.linear.x, msg.linear.y, msg.angular.z)
        self._last_cmd_time = time.time()
        # Skill-level velocity command — protect from path-follower override
        # for 0.5s. Go2ROS2Proxy.walk() republishes at 4Hz to keep this fresh.
        self._teleop_until = time.time() + 0.5
        # Clear any stale path so follow_path doesn't resume after teleop
        self._current_path = []

    def _publish_odom(self) -> None:
        """Publish /state_estimation at 200 Hz with frame map→sensor."""
        odom = self._go2.get_odometry()
        now = self.get_clock().now().to_msg()

        # Apply sensor offset: sensor frame is offset from body center
        # In the nav stack convention, state_estimation is in map→sensor frame
        heading = self._go2.get_heading()
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        # Sensor position = body position + rotated offset
        sx = odom.x + cos_h * _SENSOR_X - sin_h * _SENSOR_Y
        sy = odom.y + sin_h * _SENSOR_X + cos_h * _SENSOR_Y
        sz = odom.z + _SENSOR_Z

        msg = OdometryMsg()
        msg.header.stamp = now
        msg.header.frame_id = "map"
        msg.child_frame_id = "sensor"
        msg.pose.pose.position.x = sx
        msg.pose.pose.position.y = sy
        msg.pose.pose.position.z = sz
        msg.pose.pose.orientation.x = odom.qx
        msg.pose.pose.orientation.y = odom.qy
        msg.pose.pose.orientation.z = odom.qz
        msg.pose.pose.orientation.w = odom.qw
        msg.twist.twist.linear.x = odom.vx
        msg.twist.twist.linear.y = odom.vy
        msg.twist.twist.linear.z = odom.vz
        msg.twist.twist.angular.z = odom.vyaw
        self._odom_pub.publish(msg)
        self._diag_odom_count += 1

        # TF: map → sensor
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "map"
        t.child_frame_id = "sensor"
        t.transform.translation.x = sx
        t.transform.translation.y = sy
        t.transform.translation.z = sz
        t.transform.rotation.x = odom.qx
        t.transform.rotation.y = odom.qy
        t.transform.rotation.z = odom.qz
        t.transform.rotation.w = odom.qw
        self._tf_broadcaster.sendTransform(t)

        # TF: map → vehicle (body center, for visualization)
        tv = TransformStamped()
        tv.header.stamp = now
        tv.header.frame_id = "map"
        tv.child_frame_id = "vehicle"
        tv.transform.translation.x = odom.x
        tv.transform.translation.y = odom.y
        tv.transform.translation.z = odom.z
        tv.transform.rotation.x = odom.qx
        tv.transform.rotation.y = odom.qy
        tv.transform.rotation.z = odom.qz
        tv.transform.rotation.w = odom.qw
        self._tf_broadcaster.sendTransform(tv)

    def _publish_pointcloud(self) -> None:
        """Publish /registered_scan (PointCloud2 in map frame, live local lidar only).

        Ceiling-filtered: points above _CEILING_FILTER_HEIGHT are excluded.
        TARE subscribes to this topic for frontier detection — must be LOCAL ONLY.
        """
        points = self._lidar.points()
        if not points:
            return

        odom = self._go2.get_odometry()
        ground_z = odom.z - 0.28

        now = self.get_clock().now().to_msg()
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        point_step = 16
        data = bytearray()
        filtered_count = 0

        # 1. Live local lidar (with ceiling filter)
        for x, y, z, _ in points:
            intensity = z - ground_z
            if intensity > _CEILING_FILTER_HEIGHT:
                continue
            data.extend(struct.pack("ffff", x, y, z, intensity))
            filtered_count += 1

        msg = PointCloud2()
        msg.header.stamp = now
        msg.header.frame_id = "map"
        msg.height = 1
        msg.width = filtered_count
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = point_step
        msg.row_step = point_step * filtered_count
        msg.data = bytes(data)
        msg.is_dense = True
        self._pc_pub.publish(msg)
        self._diag_scan_count += 1
        self._cached_points = points  # cache for obstacle check (live only)

        # Accumulate terrain for persistence (only during active navigation)
        if self._nav_enabled:
            self._terrain_acc.add(points)

    def _publish_scan(self) -> None:
        """Publish /scan (LaserScan) for compatibility."""
        scan = self._lidar.scan()
        now = self.get_clock().now().to_msg()

        msg = LaserScanMsg()
        msg.header.stamp = now
        msg.header.frame_id = "sensor"
        msg.angle_min = scan.angle_min
        msg.angle_max = scan.angle_max
        msg.angle_increment = scan.angle_increment
        msg.range_min = scan.range_min
        msg.range_max = scan.range_max
        msg.ranges = list(scan.ranges)
        msg.time_increment = 0.0
        msg.scan_time = 0.1
        self._scan_pub.publish(msg)

    def _publish_joy_speed(self) -> None:
        """Publish /speed for pathFollower velocity control.

        NOTE: We do NOT publish /joy — pathFollower's joystickHandler
        zeros joySpeed when axes[4]==0, which defeats autonomyMode.
        Without /joy, autonomyMode initialization keeps joySpeed=1.0
        and /speed handler scales it to desired speed.
        """
        speed = Float32()
        speed.data = 0.8
        self._speed_pub.publish(speed)

    def _publish_camera(self) -> None:
        """Publish ONE head-camera instant: RGB, depth and camera pose, one stamp.

        The frame comes from the head Camera (rendered by the world from a copy of
        one instant), so pixels and the published
        d435 pose belong to the same sim time (D36: grasp_point_from_rgbd
        back-projects with exactly this pose).
        """
        try:
            frame = self._camera.capture(320, 240)
            now = self.get_clock().now().to_msg()

            from std_msgs.msg import Float64MultiArray
            pose_msg = Float64MultiArray()
            pose_msg.data = [float(v) for v in frame.cam_pos] + [float(v) for v in frame.cam_mat.ravel()]
            self._cam_pose_pub.publish(pose_msg)

            img_msg = Image()
            img_msg.header.stamp = now
            img_msg.header.frame_id = "camera_link"
            img_msg.height = 240
            img_msg.width = 320
            img_msg.encoding = "rgb8"
            img_msg.step = 320 * 3
            img_msg.data = bytes(frame.rgb)
            self._img_pub.publish(img_msg)

            depth_msg = Image()
            depth_msg.header.stamp = now
            depth_msg.header.frame_id = "camera_link"
            depth_msg.height = 240
            depth_msg.width = 320
            depth_msg.encoding = "32FC1"
            depth_msg.step = 320 * 4
            depth_msg.data = bytes(frame.depth.astype(np.float32))
            self._depth_pub.publish(depth_msg)
        except Exception as e:
            if not hasattr(self, '_cam_err_logged'):
                self.get_logger().warn(f"Camera render failed: {e}")
                self._cam_err_logged = True

    def _publish_nav_boundary(self) -> None:
        """Publish /navigation_boundary polygon from worlds/apartment/world.yaml bounding box.

        TARE only searches for frontiers WITHIN this boundary. Without it,
        TARE's grid extends far beyond the walls and never finishes.
        """
        if not hasattr(self, "_boundary_pub"):
            self._boundary_pub = self.create_publisher(
                PolygonStamped, "/navigation_boundary", 5
            )
        if not hasattr(self, "_boundary_points"):
            self._boundary_points = self._load_boundary_from_layout()
        msg = PolygonStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        for x, y in self._boundary_points:
            p = Point32()
            p.x, p.y, p.z = float(x), float(y), 0.0
            msg.polygon.points.append(p)
        self._boundary_pub.publish(msg)

    def _load_boundary_from_layout(self) -> list[tuple[float, float]]:
        """Compute boundary rectangle from the apartment scene's wall positions.

        Apartment walls: X=[0, 20], Y=[0, 14]. Boundary 0.3m inside walls.
        """
        x_min, y_min = 0.0, 0.0
        x_max, y_max = 20.0, 14.0
        self.get_logger().info(f"Nav boundary: [{x_min},{y_min}]-[{x_max},{y_max}]")
        return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)]

    def save_terrain(self) -> bool:
        """Save accumulated terrain map for next session."""
        if self._terrain_acc.size == 0:
            return False
        ok = self._terrain_acc.save(self._terrain_map_path)
        if ok:
            self.get_logger().info(
                f"Terrain map saved: {self._terrain_acc.size} voxels -> {self._terrain_map_path}"
            )
        return ok

    def _auto_save_terrain(self) -> None:
        """Auto-save terrain every 30s when nav is active and terrain has data."""
        if self._nav_enabled and self._terrain_acc.size > 0:
            self.save_terrain()

    def _publish_scene_graph_markers(self) -> None:
        """Publish scene graph visualization as MarkerArray (1 Hz)."""
        if self._marker_pub is None:
            return
        try:
            from vector_os_nano.sim.nav.ros2.viz import (
                build_scene_graph_markers,
            )
            odom = self._go2.get_odometry()
            heading = self._go2.get_heading()
            stamp = self.get_clock().now().to_msg()
            ma = build_scene_graph_markers(
                scene_graph=self._scene_graph,
                stamp=stamp,
                robot_x=odom.x,
                robot_y=odom.y,
                robot_heading=heading,
            )
            if ma is not None:
                self._marker_pub.publish(ma)
        except Exception:
            pass  # visualization is best-effort

    def _publish_scene_graph_json(self) -> None:
        """Publish scene graph as JSON string (0.5 Hz) for Foxglove Raw Messages."""
        if self._sg_json_pub is None or self._scene_graph is None:
            return
        try:
            import json
            from std_msgs.msg import String

            sg = self._scene_graph
            rooms = sg.get_all_rooms()
            doors = sg.get_all_doors()
            objects = []
            for room in rooms:
                objects.extend(sg.find_objects_in_room(room.room_id))

            data = {
                "rooms": [
                    {
                        "id": r.room_id,
                        "center": [round(r.center_x, 2), round(r.center_y, 2)],
                        "area": round(r.area, 1),
                        "visits": r.visit_count,
                        "description": r.representative_description[:80] if r.representative_description else "",
                        "connected": list(r.connected_rooms),
                    }
                    for r in rooms
                ],
                "doors": [
                    {
                        "rooms": list(k),
                        "position": [round(v[0], 2), round(v[1], 2)],
                    }
                    for k, v in doors.items()
                ],
                "objects": [
                    {
                        "category": o.category,
                        "room": o.room_id,
                        "position": [round(o.x, 2), round(o.y, 2)],
                        "confidence": round(o.confidence, 2),
                    }
                    for o in objects
                ],
                "stats": sg.stats(),
            }

            msg = String()
            msg.data = json.dumps(data, ensure_ascii=False)
            self._sg_json_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"scene_graph_json publish error: {e}")
