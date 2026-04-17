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
from geometry_msgs.msg import TwistStamped, Twist, TransformStamped, PointStamped, PolygonStamped, Point32
from std_msgs.msg import Float32, Header
from tf2_ros import TransformBroadcaster
import numpy as np

from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2

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
        str(_repo / "config" / "nav.yaml"),
        "config/nav.yaml",
    ]
    for path in _search:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = yaml.safe_load(f) or {}
                _NAV_CFG = data
                return _NAV_CFG
            except Exception:
                pass
    _NAV_CFG = {}
    return _NAV_CFG


def _nav(key: str, default: float) -> float:
    """Look up a navigation parameter by key, return default if absent."""
    cfg = _load_nav_config()
    nav_section = cfg.get("navigation", {})
    return float(nav_section.get(key, default))


# Sensor mounting offset — on top of Go2 head, above all leg geoms.
# Must match mujoco_go2.py _LIDAR_OFFSET and nav stack sensorOffset.
_SENSOR_X: float = 0.3
_SENSOR_Y: float = 0.0
_SENSOR_Z: float = 0.2

# Ceiling filter: points with height above ground exceeding this threshold are
# excluded from /registered_scan. Matches FAR's vehicle_height (indoor.yaml: 1.0).
# Debug harness (test_vgraph_debug.py) confirmed doors are CLEAR at 1.0m.
# Loaded from config/nav.yaml at startup; fallback 1.8 keeps original behaviour.
_CEILING_FILTER_HEIGHT: float = _nav("ceiling_filter_height", 1.8)


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
        self.create_timer(2.0, self._publish_nav_boundary)       # 0.5 Hz — TARE needs periodic boundary

        # Stuck detector state — triggers /reset_waypoint when no progress
        self._stuck_pos = None       # (x, y) at last check
        self._stuck_count = 0        # consecutive checks with < 0.3m progress
        self._stuck_history: list = []  # (x, y) positions where /reset_waypoint was sent
        self._stuck_wall_clock: float = 0.0  # wall-clock time when first stuck detected
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
        self._terrain_map_path = os.path.expanduser("~/.vector_os_nano/terrain_map.npz")
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

    def _check_reset_flag(self) -> None:
        """Check for reset pose request (1 Hz). Triggered by /tmp/vector_reset_pose."""
        if os.path.exists("/tmp/vector_reset_pose"):
            try:
                os.remove("/tmp/vector_reset_pose")
            except OSError:
                pass
            self.get_logger().warn("RESET POSE — standing up at current position")
            self._go2.reset_pose()
            self._current_path = []
            self._pf_speed = 0.0
            self._pf_lat = 0.0
            self._pf_yawrate = 0.0

    def _check_nav_flag(self) -> None:
        """Check file flag to enable/disable path following (1 Hz)."""
        flag = os.path.exists("/tmp/vector_nav_active")
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
        replay_flag = "/tmp/vector_terrain_replay"
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
        """Publish /registered_scan (PointCloud2 in map frame, live local lidar only).

        Ceiling-filtered: points above _CEILING_FILTER_HEIGHT are excluded.
        TARE subscribes to this topic for frontier detection — must be LOCAL ONLY.
        """
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

    def _publish_nav_boundary(self) -> None:
        """Publish /navigation_boundary polygon from room_layout.yaml bounding box.

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
        """Compute boundary rectangle from go2_room.xml wall positions.

        Apartment walls: X=[0, 20], Y=[0, 14]. Boundary 0.3m inside walls.
        """
        x_min, y_min = 0.0, 0.0
        x_max, y_max = 20.0, 14.0
        self.get_logger().info(f"Nav boundary: [{x_min},{y_min}]-[{x_max},{y_max}]")
        return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)]

    def _exploration_finish_cb(self, msg) -> None:
        """TARE says exploration is complete — replay terrain, then stop.

        Sequence: terrain replay (seeds FAR V-Graph) → stop robot → notify explore skill.
        """
        if msg.data and not self._exploration_finished:
            self._exploration_finished = True
            self.get_logger().warn("TARE exploration complete — replaying terrain for FAR")
            # Trigger terrain replay so FAR gets complete V-Graph data
            try:
                with open("/tmp/vector_terrain_replay", "w") as f:
                    f.write("1")
            except OSError:
                pass
            # Stop robot after short delay for terrain replay to start
            self._go2.set_velocity(0.0, 0.0, 0.0)
            self._current_path = []
            # Signal explore skill
            try:
                with open("/tmp/vector_explore_finished", "w") as f:
                    f.write("1")
            except OSError:
                pass

    def _path_cb(self, msg: Path) -> None:
        """Store path for Python follower + log."""
        if not self._nav_enabled or self._exploration_finished:
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
        # Don't reset _pf_point_id to 0 — the progressive lookahead in
        # _follow_path naturally finds the closest point. Resetting on every
        # path update (5Hz) prevents forward progress at low speeds.

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
            # Skip high points (above obstacle relevance for Go2 at 0.45m standing)
            if z - ground_z > 0.8:
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

    def _scan_surroundings(self) -> tuple[float, float, float, float]:
        """Scan cached pointcloud for nearby obstacles in 4 zones.

        Returns (front_dist, left_dist, right_dist, back_dist) in meters.
        Each is the min distance to an obstacle in that zone, or inf.
        Uses body-frame projection: front/back = ±30deg, left/right = 30-120deg.
        """
        if not self._cached_points:
            return (float('inf'), float('inf'), float('inf'), float('inf'))

        odom = self._go2.get_odometry()
        heading = self._go2.get_heading()
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        ground_z = odom.z - 0.28

        front_min = float('inf')
        left_min = float('inf')
        right_min = float('inf')
        back_min = float('inf')

        for x, y, z, _ in self._cached_points:
            dz = z - ground_z
            if dz < 0.10 or dz > 0.8:
                continue
            dx = x - odom.x
            dy = y - odom.y
            fwd = dx * cos_h + dy * sin_h
            lat = -dx * sin_h + dy * cos_h
            d = math.sqrt(fwd * fwd + lat * lat)
            if d > 1.5 or d < 0.05:
                continue

            angle = math.atan2(abs(lat), fwd)
            if fwd > 0 and angle < 0.52:  # front ±30deg
                front_min = min(front_min, d)
            elif fwd < 0 and angle > 2.62:  # back ±30deg (pi-0.52)
                back_min = min(back_min, d)
            elif lat > 0 and angle < 2.1:  # left 30-120deg
                left_min = min(left_min, d)
            elif lat < 0 and angle < 2.1:  # right 30-120deg
                right_min = min(right_min, d)

        return (front_min, left_min, right_min, back_min)

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

        # --- Wall escape mode: reactive, direction-aware ---
        # Re-scans surroundings EVERY tick to adapt to changing obstacles.
        # Lidar is mounted -20° tilt on head (0.3m forward) — rear coverage
        # is sparser, so back_d uses a conservative clearance threshold.
        now = time.time()
        if now < self._wall_escape_until:
            front_d, left_d, right_d, back_d = self._scan_surroundings()
            # Lidar rear coverage is sparse due to -20° tilt — treat back_d
            # as less reliable. Use conservative threshold (0.40m) vs front (0.30m).
            back_clear = back_d > 0.40
            left_clear = left_d > 0.25
            right_clear = right_d > 0.25

            if now < self._wall_escape_phase2:
                # Phase 1: try to reverse — but only if back is clear
                if back_clear:
                    tgt_vx, tgt_vy, tgt_yaw = -0.35, 0.0, 0.0
                else:
                    # Back blocked — turn in place toward most open side
                    if right_d > left_d:
                        tgt_vx, tgt_vy, tgt_yaw = 0.0, 0.0, -0.4
                    else:
                        tgt_vx, tgt_vy, tgt_yaw = 0.0, 0.0, 0.4
            else:
                # Phase 2: strafe toward open side + slight reverse
                if right_clear and right_d > left_d:
                    tgt_vy, tgt_yaw = 0.15, -0.35
                elif left_clear:
                    tgt_vy, tgt_yaw = -0.15, 0.35
                elif right_clear:
                    tgt_vy, tgt_yaw = 0.15, -0.35
                else:
                    tgt_vy = 0.0
                    tgt_yaw = 0.4 if left_d > right_d else -0.4
                tgt_vx = -0.10 if back_clear else 0.0

            # Moderate ramp — responsive but not violent
            _A = 0.04
            if self._pf_speed < tgt_vx: self._pf_speed = min(tgt_vx, self._pf_speed + _A)
            else: self._pf_speed = max(tgt_vx, self._pf_speed - _A)
            if self._pf_lat < tgt_vy: self._pf_lat = min(tgt_vy, self._pf_lat + 0.03)
            else: self._pf_lat = max(tgt_vy, self._pf_lat - 0.03)
            if self._pf_yawrate < tgt_yaw: self._pf_yawrate = min(tgt_yaw, self._pf_yawrate + 0.06)
            else: self._pf_yawrate = max(tgt_yaw, self._pf_yawrate - 0.06)
            self._go2.set_velocity(self._pf_speed, self._pf_lat, self._pf_yawrate)
            self._last_cmd_time = now
            return

        # --- Wall contact detection (accumulate time when stuck against wall) ---
        front_d_check = self._check_front_obstacle()
        cur_speed = self._pf_speed if hasattr(self, '_pf_speed') else 0.0
        if front_d_check < 0.30 and cur_speed < 0.15:
            self._wall_contact_time += 1.0 / 20.0  # 20 Hz tick
        else:
            self._wall_contact_time = 0.0

        if self._wall_contact_time > 0.4:
            front_d, left_d, right_d, back_d = self._scan_surroundings()
            self.get_logger().warn(
                f"Wall escape triggered: F={front_d:.2f} L={left_d:.2f} "
                f"R={right_d:.2f} B={back_d:.2f}"
            )
            all_tight = front_d < 0.35 and left_d < 0.30 and right_d < 0.30
            if all_tight and back_d > 0.40:
                # Boxed in with only rear open — sustained reverse
                self._wall_escape_phase2 = now + 3.0
                self._wall_escape_until = now + 3.0
            elif back_d > 0.40:
                # Front blocked, back clear — reverse then strafe
                self._wall_escape_phase2 = now + 0.8
                self._wall_escape_until = now + 2.5
            else:
                # Front AND back blocked — turn in place to find opening
                self._wall_escape_phase2 = now  # skip reverse, go straight to strafe/turn
                self._wall_escape_until = now + 2.0
            self._wall_contact_time = 0.0
            self._current_path = []
            self._stuck_count = 0

        if not hasattr(self, '_pf_speed'):
            self._pf_speed = 0.0       # current forward speed
            self._pf_lat = 0.0         # current lateral speed
            self._pf_yawrate = 0.0     # current yaw rate
            self._pf_point_id = 0      # path progress index
            self._pf_turning = False   # True = turn-in-place mode

        # --- Two-mode quadruped path follower ---
        # Mode 1 (TRACK): heading error < 60° → cos/sin omni-walk, full speed
        # Mode 2 (TURN):  heading error > 60° → stop, turn in place, then go
        # Hysteresis: TRACK→TURN at 60°, TURN→TRACK at 30° (prevents oscillation)
        _MAX_SPEED = 0.6               # m/s forward cruise (balance: speed vs stability)
        _MAX_LAT = 0.20                # m/s max lateral speed (indoor dodging)
        _MAX_YAW_RATE = 1.0            # rad/s max yaw rate
        _YAW_GAIN_TRACK = 4.0          # P-gain for yaw in tracking mode (gentle)
        _YAW_GAIN_TURN = 6.0           # P-gain for yaw in turn mode (snappy)
        _TRACK_THRE = 1.05             # rad (60°) — enter turn mode above this
        _TRACK_RESUME = 0.52           # rad (30°) — resume tracking below this (hysteresis)
        _LOOK_AHEAD = 0.8              # m lookahead
        _STOP_DIS = 0.2                # m — stop within this
        _SLOW_DWN_DIS = 1.0            # m — start decelerating
        _ACCEL = 0.03                  # m/s per step @ 20Hz (0→0.8 takes 1.3s)
        _PATH_TIMEOUT = 8.0            # seconds before path considered stale

        # Stop immediately if exploration is done
        if self._exploration_finished:
            self._pf_speed *= 0.9
            self._pf_lat *= 0.9
            self._pf_yawrate *= 0.9
            if abs(self._pf_speed) < 0.01: self._pf_speed = 0.0
            if abs(self._pf_lat) < 0.01: self._pf_lat = 0.0
            if abs(self._pf_yawrate) < 0.01: self._pf_yawrate = 0.0
            self._go2.set_velocity(self._pf_speed, self._pf_lat, self._pf_yawrate)
            return

        has_path = (self._current_path
                    and time.time() - self._path_time < _PATH_TIMEOUT)

        if not has_path:
            if not self._nav_enabled:
                # Decel to stop (smoothed)
                self._pf_speed *= 0.9
                self._pf_lat *= 0.9
                self._pf_yawrate *= 0.9
                if abs(self._pf_speed) < 0.01: self._pf_speed = 0.0
                if abs(self._pf_lat) < 0.01: self._pf_lat = 0.0
                if abs(self._pf_yawrate) < 0.01: self._pf_yawrate = 0.0
            else:
                front_dist = self._check_front_obstacle()
                if front_dist < 0.30:
                    tgt_vx, tgt_yaw = -0.20, 0.4  # back away + turn harder
                elif front_dist < 0.60:
                    tgt_vx, tgt_yaw = 0.0, 0.3    # stop, turn to find new path
                else:
                    tgt_vx, tgt_yaw = 0.05, 0.0   # gentle creep (was 0.15 — too aggressive)
                _A = 0.03
                if self._pf_speed < tgt_vx: self._pf_speed = min(tgt_vx, self._pf_speed + _A)
                else: self._pf_speed = max(tgt_vx, self._pf_speed - _A)
                if self._pf_yawrate < tgt_yaw: self._pf_yawrate = min(tgt_yaw, self._pf_yawrate + 0.03)
                else: self._pf_yawrate = max(tgt_yaw, self._pf_yawrate - 0.03)
                self._pf_lat *= 0.9  # decay lateral
            self._go2.set_velocity(self._pf_speed, self._pf_lat, self._pf_yawrate)
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

        # --- Space-aware speed: slow in tight spaces, fast in open ---
        front_d, left_d, right_d, _back_d = self._scan_surroundings()
        front_gap = front_d - 0.20   # lidar-to-front clearance (lidar on head, ~20cm from nose)
        left_gap = left_d - 0.19     # body side extent
        right_gap = right_d - 0.19

        # Speed based on FRONT gap only — side obstacles don't slow forward motion
        if front_gap > 0.5:
            space_speed = _MAX_SPEED
        elif front_gap > 0.1:
            space_speed = _MAX_SPEED * (front_gap / 0.5)
        else:
            space_speed = 0.10
        # Only crawl when squeezed on BOTH sides AND front tight
        min_side = min(left_gap, right_gap)
        if min_side < 0.05 and front_gap < 0.2:
            space_speed = min(space_speed, 0.10)

        # --- Path curvature: slow down BEFORE turns, not during ---
        # Look 3-5 points ahead on path. If direction changes significantly,
        # reduce speed now so the robot can make the turn without overshooting.
        curve_speed = _MAX_SPEED
        if path_size > 3:
            look_pts = min(self._pf_point_id + 6, path_size - 1)
            fx, fy = path[look_pts]
            future_dir = math.atan2(fy - ry, fx - rx)
            curve = future_dir - path_dir
            while curve > math.pi: curve -= 2 * math.pi
            while curve < -math.pi: curve += 2 * math.pi
            abs_curve = abs(curve)
            if abs_curve > 0.3:  # > 17° turn ahead
                # Scale: 0.3 rad → full speed, 1.5 rad → 30% speed
                curve_speed = _MAX_SPEED * max(0.3, 1.0 - abs_curve / 1.5)

        target_speed = min(space_speed, curve_speed)
        if end_dis < _SLOW_DWN_DIS:
            target_speed = min(target_speed, _MAX_SPEED * (end_dis / _SLOW_DWN_DIS))
        if end_dis < _STOP_DIS or path_size <= 1:
            target_speed = 0.0

        # --- Narrow passage detection ---
        # When both sides are tight but front is clear, robot is in a doorway
        # or corridor. Reduce yaw aggressiveness to prevent spinning.
        _in_narrow = left_gap < 0.20 and right_gap < 0.20 and front_gap > 0.15
        if _in_narrow:
            _eff_yaw_gain_turn = 3.0    # half normal — gentler turning in tight space
            _eff_track_thre = 1.30      # 75° — stay in TRACK mode longer near doors
            _eff_track_resume = 0.40    # 23° — need tighter alignment before resuming
        else:
            _eff_yaw_gain_turn = _YAW_GAIN_TURN   # 6.0
            _eff_track_thre = _TRACK_THRE          # 60°
            _eff_track_resume = _TRACK_RESUME      # 30°

        # --- Two-mode controller ---
        # Mode transition with hysteresis (prevents oscillation at boundary)
        if self._pf_turning:
            if abs_err < _eff_track_resume:
                self._pf_turning = False  # heading aligned → resume tracking
        else:
            if abs_err > _eff_track_thre:
                self._pf_turning = True   # heading way off → stop and turn

        if end_dis <= _STOP_DIS:
            # At goal — stop
            vx = 0.0
            vy = 0.0
            vyaw = _YAW_GAIN_TRACK * dir_diff

        elif self._pf_turning:
            # MODE 2: TURN — heading error large
            if abs_err > 2.1:  # >120° — target is behind
                vx = -0.15     # reverse toward target
                vy = 0.0
                vyaw = _eff_yaw_gain_turn * dir_diff * 0.5
            elif _in_narrow:
                # In doorway/corridor: keep moving forward while turning
                # to avoid spinning in place
                vx = 0.15
                vy = 0.0
                vyaw = _eff_yaw_gain_turn * dir_diff
            else:
                vx = 0.05      # minimal forward creep for gait engagement
                vy = 0.0
                vyaw = _eff_yaw_gain_turn * dir_diff

        else:
            # MODE 1: TRACKING — heading error small
            err_scale = max(0.4, math.cos(abs_err))
            track_speed = target_speed * err_scale
            vx = track_speed * math.cos(dir_diff)
            vy = -track_speed * math.sin(dir_diff)
            vy = max(-_MAX_LAT, min(_MAX_LAT, vy))
            vyaw = _YAW_GAIN_TRACK * dir_diff

        # --- Cylinder body safety (gaps already computed above) ---
        _COMFORT = 0.15      # desired gap between body surface and wall
        _DANGER = 0.05       # gap below this = imminent contact

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

        # Sides: proportional push away from wall (never slam to max)
        # Scale push by (1 - forward_speed_ratio) so fast-moving robot
        # gets gentler lateral push (prevents tipping at speed).
        _speed_ratio = min(1.0, abs(self._pf_speed) / _MAX_SPEED)
        _push_scale = 0.15 * (1.0 - 0.5 * _speed_ratio)  # 0.15 at stop, 0.075 at max speed

        if left_gap < _COMFORT:
            push_strength = (_COMFORT - max(0, left_gap)) / _COMFORT
            vy = max(-_MAX_LAT, vy - push_strength * _push_scale)
            if vy > 0:
                vy = 0.0
        if right_gap < _COMFORT:
            push_strength = (_COMFORT - max(0, right_gap)) / _COMFORT
            vy = min(_MAX_LAT, vy + push_strength * _push_scale)
            if vy < 0:
                vy = 0.0

        # --- Smooth acceleration (prevents MPC gait destabilization) ---
        _ACCEL_LAT = 0.01    # m/s per tick lateral (0→0.15 takes 0.75s)
        _ACCEL_YAW_TRACK = 0.04   # rad/s per tick in tracking (gentle)
        _ACCEL_YAW_TURN = 0.08    # rad/s per tick in turn mode (faster)

        # Forward/reverse
        if self._pf_speed < vx:
            self._pf_speed = min(vx, self._pf_speed + _ACCEL)
        elif self._pf_speed > vx:
            self._pf_speed = max(vx, self._pf_speed - _ACCEL * 2)  # decel 2x faster

        # Lateral
        target_lat = float(np.clip(vy, -_MAX_LAT, _MAX_LAT))
        if self._pf_lat < target_lat:
            self._pf_lat = min(target_lat, self._pf_lat + _ACCEL_LAT)
        elif self._pf_lat > target_lat:
            self._pf_lat = max(target_lat, self._pf_lat - _ACCEL_LAT)

        # Yaw — scale max yaw rate inversely with forward speed (prevent centripetal tip)
        # Fast forward (0.8) → max yaw 0.5 rad/s. Stopped → max yaw 1.2 rad/s.
        _speed_frac = min(1.0, abs(self._pf_speed) / _MAX_SPEED)
        _dynamic_yaw_max = _MAX_YAW_RATE * (1.0 - 0.6 * _speed_frac)

        accel_yaw = _ACCEL_YAW_TURN if self._pf_turning else _ACCEL_YAW_TRACK
        target_yaw = float(np.clip(vyaw, -_dynamic_yaw_max, _dynamic_yaw_max))
        if self._pf_yawrate < target_yaw:
            self._pf_yawrate = min(target_yaw, self._pf_yawrate + accel_yaw)
        elif self._pf_yawrate > target_yaw:
            self._pf_yawrate = max(target_yaw, self._pf_yawrate - accel_yaw)

        # Clamp final speeds
        self._pf_speed = float(np.clip(self._pf_speed, -0.3, _MAX_SPEED))
        self._pf_lat = float(np.clip(self._pf_lat, -_MAX_LAT, _MAX_LAT))

        self._go2.set_velocity(self._pf_speed, self._pf_lat, self._pf_yawrate)
        self._last_cmd_time = time.time()

        if not hasattr(self, '_follow_count'):
            self._follow_count = 0
        self._follow_count += 1
        if self._follow_count % 20 == 0:
            mode = "TURN" if self._pf_turning else "TRACK"
            self.get_logger().info(
                f"PyPF[{mode}]: vx={self._pf_speed:.2f} vy={self._pf_lat:.2f} "
                f"yr={self._pf_yawrate:.2f} endD={end_dis:.1f} "
                f"err={math.degrees(dir_diff):.0f}° "
                f"spd={space_speed:.1f} gap=F{front_gap:.2f}/L{left_gap:.2f}/R{right_gap:.2f}"
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
            self._stuck_wall_clock = 0.0
            return

        odom = self._go2.get_odometry()
        cur = (odom.x, odom.y)

        if self._stuck_pos is not None:
            dx = cur[0] - self._stuck_pos[0]
            dy = cur[1] - self._stuck_pos[1]
            moved = math.sqrt(dx * dx + dy * dy)

            if moved < 0.1:
                if self._stuck_count == 0:
                    # First stuck tick — record wall-clock start
                    self._stuck_wall_clock = time.time()
                self._stuck_count += 1

                # Wall-clock ceiling: abort navigation if stalled too long
                _stall_timeout: float = _nav("stall_timeout", 30.0)
                if self._stuck_wall_clock > 0.0 and (time.time() - self._stuck_wall_clock) > _stall_timeout:
                    self.get_logger().error(
                        "[NAV] Stalled for %.0fs, aborting navigation", _stall_timeout
                    )
                    self._go2.set_velocity(0.0, 0.0, 0.0)
                    try:
                        with open("/tmp/vector_nav_stalled", "w") as fh:
                            fh.write("1")
                    except OSError as exc:
                        self.get_logger().warn(f"[NAV] Could not write stall flag: {exc}")
                    # Reset stuck state so we don't fire again immediately
                    self._stuck_count = 0
                    self._stuck_pos = None
                    self._stuck_wall_clock = 0.0
                    return
            else:
                self._stuck_count = 0
                self._stuck_wall_clock = 0.0

            if self._stuck_count == 1:  # 2s — request TARE replan early
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
            elif self._stuck_count == 2:  # 4s — direction-aware escape
                now = time.time()
                front_d, left_d, right_d, back_d = self._scan_surroundings()
                _escape_dur = _nav("wall_escape_duration", 2.5)
                self.get_logger().warn(
                    f"Stuck 8s: F={front_d:.2f} L={left_d:.2f} "
                    f"R={right_d:.2f} B={back_d:.2f}"
                )
                all_tight = front_d < 0.35 and left_d < 0.30 and right_d < 0.30
                # Conservative rear threshold — lidar rear coverage sparse at -20° tilt
                back_clear = back_d > 0.40
                if all_tight and back_clear:
                    # Boxed in — sustained reverse
                    self._wall_escape_phase2 = now + 4.0
                    self._wall_escape_until = now + 4.0
                elif back_clear:
                    # Front blocked, back clear — reverse then strafe
                    self._wall_escape_phase2 = now + 0.8
                    self._wall_escape_until = now + _escape_dur
                else:
                    # Front AND back blocked — turn in place
                    self._wall_escape_phase2 = now
                    self._wall_escape_until = now + _escape_dur
                self._current_path = []
                self._stuck_count = 0
                self._stuck_wall_clock = 0.0
                self._stuck_history.append((odom.x, odom.y))

                # Stuck loop: 3+ escapes at same spot → longer escape
                if len(self._stuck_history) >= 3:
                    recent = self._stuck_history[-3:]
                    cx = sum(p[0] for p in recent) / 3
                    cy = sum(p[1] for p in recent) / 3
                    if all(
                        math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2) < 0.5
                        for p in recent
                    ):
                        self.get_logger().warn("Stuck loop — extended escape 4s")
                        self._wall_escape_phase2 = now + 2.0
                        self._wall_escape_until = now + 4.0
                        self._stuck_history.clear()

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
