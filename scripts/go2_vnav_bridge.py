#!/usr/bin/env python3
"""Go2 MuJoCo ↔ Vector Navigation Stack bridge.

Publishes the topics the CMU/Ji Zhang nav stack expects:
  - /state_estimation (Odometry, 200 Hz, frame: map→sensor)
  - /registered_scan (PointCloud2, 5 Hz, frame: map)
  - /joy (Joy, 2 Hz, fake LT trigger for autonomyMode)
  - /speed (Float32, 2 Hz, desired speed)
  - TF: map→sensor, map→vehicle
  - Subscribes: /cmd_vel (TwistStamped) → go2.set_velocity()

Usage:
    source /opt/ros/jazzy/setup.bash
    cd ~/Desktop/vector_os_nano
    ./scripts/launch_vnav.sh
"""
from __future__ import annotations

import math
import os
import struct
import sys
import time
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo))

pkg = types.ModuleType("vector_os_nano")
pkg.__path__ = [str(_repo / "vector_os_nano")]
pkg.__package__ = "vector_os_nano"
sys.modules.setdefault("vector_os_nano", pkg)

core = types.ModuleType("vector_os_nano.core")
core.__path__ = [str(_repo / "vector_os_nano" / "core")]
core.__package__ = "vector_os_nano.core"
sys.modules.setdefault("vector_os_nano.core", core)

hw = types.ModuleType("vector_os_nano.hardware")
hw.__path__ = [str(_repo / "vector_os_nano" / "hardware")]
sys.modules.setdefault("vector_os_nano.hardware", hw)

sim_mod = types.ModuleType("vector_os_nano.hardware.sim")
sim_mod.__path__ = [str(_repo / "vector_os_nano" / "hardware" / "sim")]
sys.modules.setdefault("vector_os_nano.hardware.sim", sim_mod)

import importlib.util
_types_path = _repo / "vector_os_nano" / "core" / "types.py"
_ts = importlib.util.spec_from_file_location("vector_os_nano.core.types", str(_types_path))
_tm = importlib.util.module_from_spec(_ts)
sys.modules.setdefault("vector_os_nano.core.types", _tm)
_ts.loader.exec_module(_tm)

# ---------------------------------------------------------------------------
# ROS2
# ---------------------------------------------------------------------------
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry as OdometryMsg, Path
from sensor_msgs.msg import PointCloud2, PointField, Joy, LaserScan as LaserScanMsg, Image, CompressedImage
from geometry_msgs.msg import TwistStamped, Twist, TransformStamped, PointStamped
from std_msgs.msg import Float32, Header
from tf2_ros import TransformBroadcaster
import numpy as np

from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2

# Sensor mounting offset — on top of Go2 head, above all leg geoms.
# Must match mujoco_go2.py _LIDAR_OFFSET and nav stack sensorOffset.
_SENSOR_X: float = 0.3
_SENSOR_Y: float = 0.0
_SENSOR_Z: float = 0.2


class TerrainAccumulator:
    """Accumulates pointcloud data into a 2D voxel grid for terrain persistence."""

    def __init__(self, voxel_size: float = 0.1, z_min: float = -0.5, z_max: float = 2.0):
        self._voxel_size = voxel_size
        self._grid: dict[tuple[int, int], float] = {}  # (ix, iy) → max_z
        self._z_min = z_min
        self._z_max = z_max
        self._count = 0  # total points added

    def add(self, points: list[tuple[float, float, float, float]]) -> None:
        """Add pointcloud (x, y, z, intensity) to grid. Keep max z per voxel."""
        for x, y, z, _ in points:
            if z < self._z_min or z > self._z_max:
                continue
            ix = int(x / self._voxel_size)
            iy = int(y / self._voxel_size)
            key = (ix, iy)
            if key not in self._grid or z > self._grid[key]:
                self._grid[key] = z
        self._count += len(points)

    def save(self, path: str) -> bool:
        """Save grid as numpy npz. Returns True on success."""
        if not self._grid:
            return False
        import numpy as np
        keys = list(self._grid.keys())
        xs = np.array([k[0] for k in keys], dtype=np.int32)
        ys = np.array([k[1] for k in keys], dtype=np.int32)
        zs = np.array([self._grid[k] for k in keys], dtype=np.float32)
        try:
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(path, ix=xs, iy=ys, z=zs, voxel_size=np.float32(self._voxel_size))
            return True
        except Exception:
            return False

    def load(self, path: str) -> bool:
        """Load grid from npz. Returns True on success."""
        try:
            import numpy as np
            data = np.load(path)
            self._voxel_size = float(data['voxel_size'])
            self._grid = {}
            for ix, iy, z in zip(data['ix'], data['iy'], data['z']):
                self._grid[(int(ix), int(iy))] = float(z)
            return True
        except Exception:
            return False

    def to_pointcloud(self) -> list[tuple[float, float, float, float]]:
        """Convert grid back to pointcloud for publishing as /registered_scan."""
        points = []
        half = self._voxel_size / 2
        for (ix, iy), z in self._grid.items():
            x = ix * self._voxel_size + half
            y = iy * self._voxel_size + half
            # intensity = height above ground (same as bridge convention)
            intensity = z
            points.append((x, y, z, intensity))
        return points

    @property
    def size(self) -> int:
        return len(self._grid)

    @property
    def point_count(self) -> int:
        return self._count


class Go2VNavBridge(Node):
    """ROS2 node bridging MuJoCoGo2 to Vector Navigation Stack."""

    def __init__(self, go2: MuJoCoGo2, quiet: bool = False) -> None:
        super().__init__("go2_vnav_bridge")
        self._go2 = go2
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
        self._terrain_map_pub = self.create_publisher(
            PointCloud2, "/terrain_map", reliable_qos
        )
        self._terrain_map_ext_pub = self.create_publisher(
            PointCloud2, "/terrain_map_ext", reliable_qos
        )
        self._scan_pub = self.create_publisher(
            LaserScanMsg, "/scan", reliable_qos
        )
        self._joy_pub = self.create_publisher(Joy, "/joy", 5)
        self._speed_pub = self.create_publisher(Float32, "/speed", 5)
        self._img_pub = self.create_publisher(Image, "/camera/image", reliable_qos)
        self._depth_pub = self.create_publisher(Image, "/camera/depth", reliable_qos)

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
        self.create_timer(1.0 / 20.0, self._follow_path)
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

        # Stuck detector state — triggers /reset_waypoint when no progress
        self._stuck_pos = None       # (x, y) at last check
        self._stuck_count = 0        # consecutive checks with < 0.3m progress
        self._stuck_history: list = []  # (x, y) positions where /reset_waypoint was sent
        self._reset_waypoint_pub = self.create_publisher(
            PointStamped, "/reset_waypoint", 5
        )

        # Wall escape state — two-phase: pure reverse then strafe+turn
        self._wall_contact_time: float = 0.0    # seconds spent pinned (front_d<0.25 AND slow)
        self._wall_escape_until: float = 0.0    # timestamp when escape maneuver ends
        self._wall_escape_phase2: float = 0.0   # timestamp when phase 2 (strafe) starts

        # Scene graph visualization (1 Hz MarkerArray)
        self._scene_graph = None  # set externally by agent
        try:
            from visualization_msgs.msg import MarkerArray
            self._marker_pub = self.create_publisher(MarkerArray, "/scene_graph_markers", 5)
            self.create_timer(1.0, self._publish_scene_graph_markers)
        except ImportError:
            self._marker_pub = None

        # Diagnostic counters — track message flow for TARE data-starvation debugging.
        # Logged every 10s at INFO level via _log_diagnostics timer.
        self._diag_odom_count: int = 0
        self._diag_scan_count: int = 0
        self._diag_path_count: int = 0
        self.create_timer(10.0, self._log_diagnostics)  # 0.1 Hz

        # Terrain persistence — accumulates pointcloud voxels during explore.
        # Auto-saved every 30s while nav is active.
        self._terrain_acc = TerrainAccumulator(voxel_size=0.1)
        self._terrain_map_path = os.path.expanduser("~/.vector_os_nano/terrain_map.npz")
        self.create_timer(30.0, self._auto_save_terrain)

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
        terrain_path = os.path.expanduser("~/.vector_os_nano/terrain_map.npz")
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

    def _check_nav_flag(self) -> None:
        """Check file flag to enable/disable path following (1 Hz)."""
        flag = os.path.exists("/tmp/vector_nav_active")
        if flag and not self._nav_enabled:
            self._nav_enabled = True
            self.get_logger().info("Navigation ENABLED (flag file detected)")
        elif not flag and self._nav_enabled:
            self._nav_enabled = False
            self._current_path = []
            self._go2.set_velocity(0.0, 0.0, 0.0)
            self.get_logger().info("Navigation DISABLED (flag file removed)")

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

        # Build PointCloud2 from saved points (same format as _publish_pointcloud)
        now = self.get_clock().now().to_msg()
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        point_step = 16
        data = bytearray()
        for x, y, z, intensity in self._terrain_replay_points:
            data.extend(struct.pack("ffff", x, y, z, intensity))

        msg = PointCloud2()
        msg.header.stamp = now
        msg.header.frame_id = "map"
        msg.height = 1
        msg.width = len(self._terrain_replay_points)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = point_step
        msg.row_step = point_step * len(self._terrain_replay_points)
        msg.data = bytes(data)
        msg.is_dense = True
        self._pc_pub.publish(msg)
        self._terrain_map_pub.publish(msg)
        self._terrain_map_ext_pub.publish(msg)

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
        self._teleop_until = time.time() + 0.5
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
        """Publish /registered_scan (PointCloud2 in map frame)."""
        points = self._go2.get_3d_pointcloud()
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
        for x, y, z, _ in points:
            intensity = z - ground_z
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
        self._pc_pub.publish(msg)
        self._diag_scan_count += 1
        self._cached_points = points  # cache for obstacle check

        # Accumulate terrain for persistence (only during active navigation)
        if self._nav_enabled:
            self._terrain_acc.add(points)

    def _publish_scan(self) -> None:
        """Publish /scan (LaserScan) for compatibility."""
        scan = self._go2.get_lidar_scan()
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
        """Render first-person camera from Go2 head (free camera).

        Uses a free camera positioned at the sensor mount looking forward.
        The named d435_rgb camera produces a rotated image in this context,
        so we keep the manually positioned free camera for ROS /camera/image.

        NOTE: The VLM pipeline (get_camera_frame) uses the named d435
        camera directly in mujoco_go2.py — that is separate from this bridge
        camera and produces correct images for VLM scene description.
        """
        try:
            import mujoco
            if not hasattr(self, '_renderer'):
                self._renderer = mujoco.Renderer(
                    self._go2._mj.model, 240, 320
                )
                self._cam = mujoco.MjvCamera()
                self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE

            model = self._go2._mj.model
            data = self._go2._mj.data

            odom = self._go2.get_odometry()
            heading = self._go2.get_heading()
            cos_h = math.cos(heading)
            sin_h = math.sin(heading)

            cam_x = odom.x + cos_h * (_SENSOR_X + 0.1)
            cam_y = odom.y + sin_h * (_SENSOR_X + 0.1)
            cam_z = odom.z + _SENSOR_Z - 0.04

            self._cam.lookat[:] = [
                cam_x + cos_h * 1.0,
                cam_y + sin_h * 1.0,
                cam_z - 0.1,
            ]
            self._cam.distance = 1.0
            self._cam.azimuth = math.degrees(heading) + 180
            self._cam.elevation = -10

            self._renderer.update_scene(data, camera=self._cam)
            rgb = self._renderer.render().copy()
            now = self.get_clock().now().to_msg()

            img_msg = Image()
            img_msg.header.stamp = now
            img_msg.header.frame_id = "camera_link"
            img_msg.height = 240
            img_msg.width = 320
            img_msg.encoding = "rgb8"
            img_msg.step = 320 * 3
            img_msg.data = bytes(rgb)
            self._img_pub.publish(img_msg)

            # Depth via separate renderer
            if not hasattr(self, '_depth_renderer'):
                try:
                    self._depth_renderer = mujoco.Renderer(
                        self._go2._mj.model, 240, 320
                    )
                    self._depth_renderer.enable_depth_rendering()
                except Exception:
                    self._depth_renderer = None

            if self._depth_renderer is not None:
                try:
                    self._depth_renderer.update_scene(data, camera=self._cam)
                    depth = self._depth_renderer.render().copy()

                    depth_msg = Image()
                    depth_msg.header.stamp = now
                    depth_msg.header.frame_id = "camera_link"
                    depth_msg.height = 240
                    depth_msg.width = 320
                    depth_msg.encoding = "32FC1"
                    depth_msg.step = 320 * 4
                    depth_msg.data = bytes(depth.astype(np.float32))
                    self._depth_pub.publish(depth_msg)
                except Exception:
                    pass
        except Exception as e:
            if not hasattr(self, '_cam_err_logged'):
                self.get_logger().warn(f"Camera render failed: {e}")
                self._cam_err_logged = True

    def _path_cb(self, msg: Path) -> None:
        """Store path for Python follower + log."""
        if not self._nav_enabled:
            return

        odom = self._go2.get_odometry()
        heading = self._go2.get_heading()
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        sx = odom.x + cos_h * _SENSOR_X - sin_h * _SENSOR_Y
        sy = odom.y + sin_h * _SENSOR_X + cos_h * _SENSOR_Y

        new_path = []
        for p in msg.poses:
            lx, ly = p.pose.position.x, p.pose.position.y
            mx = sx + lx * cos_h - ly * sin_h
            my = sy + lx * sin_h + ly * cos_h
            new_path.append((mx, my))

        self._current_path = new_path
        self._path_time = time.time()
        self._pf_point_id = 0  # reset progress on new path (like C++)

        self._diag_path_count += 1
        now_mono = time.monotonic()
        if new_path and now_mono - getattr(self, '_last_path_log', 0) > 3.0:
            self._last_path_log = now_mono
            pe = new_path[-1]
            self.get_logger().info(
                f"Path: {len(new_path)} pts, "
                f"end=({pe[0]:.1f},{pe[1]:.1f}) "
                f"robot=({odom.x:.1f},{odom.y:.1f})"
            )

    def _check_front_obstacle(self) -> float:
        """Check for obstacles in front cone. Returns min distance (meters).

        Uses cached pointcloud (10 Hz). Checks 60-degree cone in front,
        within 1.0m range, above ground level.
        """
        if not self._cached_points:
            return float('inf')

        odom = self._go2.get_odometry()
        heading = self._go2.get_heading()
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        ground_z = odom.z - 0.28  # approximate ground level

        min_dist = float('inf')
        for x, y, z, _ in self._cached_points:
            # Skip ground points
            if z - ground_z < 0.10:
                continue
            # Skip high points (above robot)
            if z - ground_z > 0.5:
                continue
            # Transform to robot frame
            dx = x - odom.x
            dy = y - odom.y
            # Project onto heading direction
            forward = dx * cos_h + dy * sin_h
            lateral = -dx * sin_h + dy * cos_h

            # Front cone: forward > 0, |lateral| < forward * tan(30deg)
            if forward > 0.1 and forward < 1.0 and abs(lateral) < forward * 0.577:
                dist = math.sqrt(forward * forward + lateral * lateral)
                if dist < min_dist:
                    min_dist = dist

        return min_dist

    def _scan_surroundings(self) -> tuple[float, float, float]:
        """Scan cached pointcloud for nearby obstacles in 3 zones.

        Returns (front_dist, left_dist, right_dist) in meters.
        Each is the min distance to an obstacle in that zone, or inf.
        Uses body-frame projection: front = ±30deg, left/right = 30-120deg.
        """
        if not self._cached_points:
            return (float('inf'), float('inf'), float('inf'))

        odom = self._go2.get_odometry()
        heading = self._go2.get_heading()
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        ground_z = odom.z - 0.28

        front_min = float('inf')
        left_min = float('inf')
        right_min = float('inf')

        for x, y, z, _ in self._cached_points:
            dz = z - ground_z
            if dz < 0.10 or dz > 0.5:
                continue
            dx = x - odom.x
            dy = y - odom.y
            fwd = dx * cos_h + dy * sin_h
            lat = -dx * sin_h + dy * cos_h
            d = math.sqrt(fwd * fwd + lat * lat)
            if d > 1.0 or d < 0.05:
                continue

            angle = math.atan2(abs(lat), fwd)
            if fwd > 0 and angle < 0.52:  # front ±30deg
                front_min = min(front_min, d)
            elif lat > 0 and angle < 2.1:  # left 30-120deg
                left_min = min(left_min, d)
            elif lat < 0 and angle < 2.1:  # right 30-120deg
                right_min = min(right_min, d)

        return (front_min, left_min, right_min)

    def _follow_path(self) -> None:
        """Omnidirectional quadruped path follower (20 Hz).

        Go2 can walk forward, backward, and sideways. This follower:
        1. ALWAYS prefers forward motion (vx > 0)
        2. Uses lateral velocity (vy) to track path when direction error is large
        3. Reverses ONLY when target is directly behind AND very close (dead end)
        4. Applies reactive wall avoidance overlay from cached pointcloud
        """
        if time.time() < self._teleop_until:
            return

        # --- Wall escape mode — two-phase: reverse first, then strafe ---
        # Phase 1 (1s): pure reverse to clear wall contact
        # Phase 2 (1.5s): strafe toward open side + turn away
        # Head jammed against wall prevents lateral movement, so must back up first.
        now = time.time()
        if now < self._wall_escape_until:
            if now < self._wall_escape_phase2:
                # Phase 1: pure reverse — clear the wall
                self._go2.set_velocity(-0.4, 0.0, 0.0)
            else:
                # Phase 2: strafe + turn toward open side
                front_d, left_d, right_d = self._scan_surroundings()
                escape_vy = 0.35 if right_d > left_d else -0.35
                escape_vyaw = 0.5 if right_d > left_d else -0.5
                self._go2.set_velocity(-0.15, escape_vy, escape_vyaw)
            self._last_cmd_time = now
            return

        # --- Wall contact detection (accumulate time when stuck against wall) ---
        front_d_check = self._check_front_obstacle()
        if front_d_check < 0.25 and abs(self._pf_speed if hasattr(self, '_pf_speed') else 0.0) < 0.05:
            self._wall_contact_time += 1.0 / 20.0  # 20 Hz tick
        else:
            self._wall_contact_time = 0.0

        if self._wall_contact_time > 0.5:  # 0.5s — trigger before dog jams
            front_d, left_d, right_d = self._scan_surroundings()
            open_side = "right" if right_d > left_d else "left"
            self.get_logger().warn(
                f"Wall escape: reverse 1s then strafe {open_side}, front_d={front_d_check:.2f}"
            )
            self._wall_escape_phase2 = now + 1.0   # phase 1 = 1s pure reverse
            self._wall_escape_until = now + 2.5     # total = 2.5s (1s reverse + 1.5s strafe)
            self._wall_contact_time = 0.0
            self._current_path = []
            self._stuck_count = 0
            return

        if not hasattr(self, '_pf_speed'):
            self._pf_speed = 0.0       # current forward speed
            self._pf_lat = 0.0         # current lateral speed
            self._pf_yawrate = 0.0     # current yaw rate
            self._pf_point_id = 0      # path progress index

        # Constants — adapted from C++ pathFollower for quadruped omni-walk.
        # Key difference from wheeled: Go2 can strafe, so heading gate is wide.
        # cos/sin decomposition handles all heading errors up to the gate.
        _MAX_SPEED = 0.8               # m/s forward cruise
        _MAX_LAT = 0.4                 # m/s max lateral speed
        _MAX_YAW_RATE = 1.0            # rad/s max yaw rate
        _YAW_GAIN = 7.5                # P-gain for yaw correction
        _LOOK_AHEAD = 0.8              # m lookahead (wider than C++ to smooth curves)
        _STOP_DIS = 0.2                # m — stop within this
        _SLOW_DWN_DIS = 1.0            # m — start decelerating
        _ACCEL = 0.05                  # m/s per step @ 20Hz
        _PATH_TIMEOUT = 8.0            # seconds before path considered stale

        has_path = (self._current_path
                    and time.time() - self._path_time < _PATH_TIMEOUT)

        if not has_path:
            if not self._nav_enabled:
                self._go2.set_velocity(0.0, 0.0, 0.0)
                self._last_cmd_time = time.time()
                return
            # Idle wander: no path but nav enabled (waiting for TARE/FAR replan)
            front_dist = self._check_front_obstacle()
            if front_dist < 0.4:
                # Front blocked — back away + turn to find open space
                self._go2.set_velocity(-0.2, 0.0, 0.4)
            else:
                self._go2.set_velocity(0.15, 0.0, 0.10)
            self._last_cmd_time = time.time()
            self._pf_point_id = 0
            return

        odom = self._go2.get_odometry()
        rx, ry = odom.x, odom.y
        heading = self._go2.get_heading()
        path = self._current_path
        path_size = len(path)

        # --- Endpoint distance ---
        ex, ey = path[-1]
        end_dis = math.sqrt((ex - rx)**2 + (ey - ry)**2)

        # --- Progressive lookahead ---
        while self._pf_point_id < path_size - 1:
            px, py = path[self._pf_point_id]
            d = math.sqrt((px - rx)**2 + (py - ry)**2)
            if d < _LOOK_AHEAD:
                self._pf_point_id += 1
            else:
                break
        self._pf_point_id = min(self._pf_point_id, path_size - 1)

        # --- Direction to lookahead point ---
        tx, ty = path[self._pf_point_id]
        dx, dy = tx - rx, ty - ry
        path_dir = math.atan2(dy, dx)

        dir_diff = path_dir - heading
        while dir_diff > math.pi: dir_diff -= 2 * math.pi
        while dir_diff < -math.pi: dir_diff += 2 * math.pi

        abs_err = abs(dir_diff)

        # --- Target speed based on endpoint distance (matches C++) ---
        target_speed = _MAX_SPEED
        if end_dis < _SLOW_DWN_DIS:
            target_speed = _MAX_SPEED * (end_dis / _SLOW_DWN_DIS)
        if end_dis < _STOP_DIS or path_size <= 1:
            target_speed = 0.0

        # --- Full omnidirectional velocity decomposition ---
        # No heading gate. cos/sin naturally maps ANY heading error:
        #   0°: full forward  |  90°: pure strafe  |  180°: reverse
        # Quadruped can walk in all directions — no need to stop and turn.
        vyaw = _YAW_GAIN * dir_diff  # always correcting heading

        if end_dis > _STOP_DIS:
            vx = target_speed * math.cos(dir_diff)
            vy = -target_speed * math.sin(dir_diff)
            # Cap reverse speed (can't see behind, be cautious)
            vx = max(-0.3, vx)
            vy = max(-_MAX_LAT, min(_MAX_LAT, vy))
        else:
            vx = 0.0
            vy = 0.0

        # --- Cylinder body safety boundary (Go2 MJCF collision) ---
        # The dog is NOT a point — it's a cylinder with radius ~0.19m.
        # The path follower tracks the centerline, but the body extends around it.
        # Obstacle distances are from the SENSOR (center), so we must subtract
        # the body radius to get the actual gap between body surface and wall.
        #
        # MJCF collision: front 0.34m (head), side 0.19m (hip+thigh swing)
        # Strategy: 3 zones based on gap between body surface and obstacle
        #   Zone 1 (gap > 0.15m): normal — path follower tracks freely
        #   Zone 2 (gap 0.05-0.15m): strong push away + slow down
        #   Zone 3 (gap < 0.05m): hard stop/reverse — body about to contact
        _BODY_FRONT = 0.34
        _BODY_SIDE = 0.19
        _COMFORT = 0.15      # desired gap between body surface and wall
        _DANGER = 0.05       # gap below this = imminent contact

        front_d, left_d, right_d = self._scan_surroundings()

        # Compute gaps (obstacle distance minus body extent)
        front_gap = front_d - _BODY_FRONT
        left_gap = left_d - _BODY_SIDE
        right_gap = right_d - _BODY_SIDE

        # Front: brake based on gap, not raw distance
        if front_gap < _COMFORT and vx > 0:
            if front_gap <= _DANGER:
                vx = 0.0  # body surface ~0.05m from wall — full stop
            else:
                # Scale: full speed at _COMFORT gap, zero at _DANGER gap
                vx *= (front_gap - _DANGER) / (_COMFORT - _DANGER)
            # Slow down when BOTH sides are tight (corridor/gap), but not
            # for single-side obstacles (table legs, door frames)
            if left_gap < _DANGER and right_gap < _DANGER:
                vx *= 0.3  # crawl when squeezed on BOTH sides

        # Sides: push HARD away from wall when gap < comfort zone
        if left_gap < _COMFORT:
            if left_gap <= _DANGER:
                # Imminent contact — maximum push right
                vy = -_MAX_LAT
            else:
                # Proportional push right
                push_strength = (_COMFORT - left_gap) / (_COMFORT - _DANGER)
                vy = max(-_MAX_LAT, vy - push_strength * 0.4)
            # Block any leftward motion
            if vy > 0:
                vy = 0.0
        if right_gap < _COMFORT:
            if right_gap <= _DANGER:
                vy = _MAX_LAT
            else:
                push_strength = (_COMFORT - right_gap) / (_COMFORT - _DANGER)
                vy = min(_MAX_LAT, vy + push_strength * 0.4)
            if vy < 0:
                vy = 0.0

        # --- Smooth acceleration ---
        if self._pf_speed < vx:
            self._pf_speed = min(vx, self._pf_speed + _ACCEL)
        elif self._pf_speed > vx:
            self._pf_speed = max(vx, self._pf_speed - _ACCEL)
        self._pf_lat = vy  # lateral can change instantly (no inertia issue)
        self._pf_yawrate = float(np.clip(vyaw, -_MAX_YAW_RATE, _MAX_YAW_RATE))

        # Clamp final speeds
        self._pf_speed = float(np.clip(self._pf_speed, -0.3, _MAX_SPEED))
        self._pf_lat = float(np.clip(self._pf_lat, -_MAX_LAT, _MAX_LAT))

        self._go2.set_velocity(self._pf_speed, self._pf_lat, self._pf_yawrate)
        self._last_cmd_time = time.time()

        if not hasattr(self, '_follow_count'):
            self._follow_count = 0
        self._follow_count += 1
        if self._follow_count % 20 == 0:
            self.get_logger().info(
                f"PyPF: vx={self._pf_speed:.2f} vy={self._pf_lat:.2f} "
                f"yr={self._pf_yawrate:.2f} endD={end_dis:.1f} "
                f"err={math.degrees(dir_diff):.0f}° "
                f"pt={self._pf_point_id}/{path_size} "
                f"wall=F{front_d:.1f}/L{left_d:.1f}/R{right_d:.1f}"
            )

    def _log_diagnostics(self) -> None:
        """Log bridge data flow stats every 10s for TARE starvation debugging."""
        self.get_logger().info(
            f"Diag: odom={self._diag_odom_count} "
            f"scan={self._diag_scan_count} "
            f"path={self._diag_path_count} "
            f"nav={'ON' if self._nav_enabled else 'OFF'}"
        )

    def _safety_check(self) -> None:
        # 5s timeout — must be long enough for TARE to initialize and
        # start publishing waypoints after exploration begins.
        # Also skip if exploration background thread is actively running.
        if time.time() - self._last_cmd_time > 5.0:
            try:
                from vector_os_nano.skills.go2.explore import is_exploring
                if is_exploring():
                    return  # don't zero velocity during exploration startup
            except ImportError:
                pass
            self._go2.set_velocity(0.0, 0.0, 0.0)

    def _stuck_detector(self) -> None:
        """Detect when robot is stuck and take recovery action.

        Runs at 0.5 Hz. Two-stage recovery:
        1. After 4s no progress: send /reset_waypoint to TARE
        2. After 8s still stuck: back up 0.3m to escape tight spot
        """
        if not self._nav_enabled:
            self._stuck_count = 0
            self._stuck_pos = None
            return

        odom = self._go2.get_odometry()
        cur = (odom.x, odom.y)

        if self._stuck_pos is not None:
            dx = cur[0] - self._stuck_pos[0]
            dy = cur[1] - self._stuck_pos[1]
            moved = math.sqrt(dx * dx + dy * dy)

            if moved < 0.1:
                self._stuck_count += 1
            else:
                self._stuck_count = 0

            if self._stuck_count == 2:  # 4s — request TARE replan
                msg = PointStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "map"
                msg.point.x = odom.x
                msg.point.y = odom.y
                msg.point.z = odom.z
                self._reset_waypoint_pub.publish(msg)
                self._stuck_history.append((odom.x, odom.y))
                self.get_logger().warn(
                    f"Stuck 4s at ({odom.x:.1f},{odom.y:.1f}) — "
                    f"sent /reset_waypoint to TARE"
                )
            elif self._stuck_count == 4:  # 8s — escape: back up
                self.get_logger().warn(
                    f"Stuck 8s — backing up to escape"
                )
                # Back up for 1 second (will be overridden by next _follow_path)
                self._go2.set_velocity(-0.4, 0.0, 0.0)
                self._last_cmd_time = time.time()
                self._current_path = []  # clear stale path
                self._stuck_count = 0  # reset for fresh detection

                # Stuck loop detection: if last 3 resets all within 0.5m → aggressive escape
                if len(self._stuck_history) >= 3:
                    recent = self._stuck_history[-3:]
                    cx = sum(p[0] for p in recent) / 3
                    cy = sum(p[1] for p in recent) / 3
                    if all(
                        math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2) < 0.5
                        for p in recent
                    ):
                        self.get_logger().warn(
                            "Stuck loop detected — aggressive escape: lateral strafe"
                        )
                        self._go2.set_velocity(-0.3, 0.3, 0.5)
                        self._last_cmd_time = time.time()
                        self._stuck_history.clear()
                        self._current_path = []

        self._stuck_pos = cur

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
            from vector_os_nano.ros2.nodes.scene_graph_viz import (
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Go2 Vector Nav Bridge")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--sinusoidal", action="store_true")
    args = parser.parse_args()

    backend = "sinusoidal" if args.sinusoidal else "auto"
    gui = not args.no_gui

    print(f"Starting MuJoCoGo2 (gui={gui}, backend={backend})...")
    go2 = MuJoCoGo2(gui=gui, room=True, backend=backend)
    go2.connect()
    print("Standing up...")
    go2.stand(duration=2.0)
    pos = go2.get_position()
    print(f"Go2 at ({pos[0]:.1f}, {pos[1]:.1f}), z={pos[2]:.3f}m")

    print("Starting Vector Nav Bridge...")
    rclpy.init()
    node = Go2VNavBridge(go2)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        go2.disconnect()
        print("Bridge stopped.")


if __name__ == "__main__":
    main()
