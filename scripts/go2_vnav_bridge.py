#!/usr/bin/env python3
"""Go2 MuJoCo ↔ Vector Navigation Stack bridge.

Publishes the topics the CMU/Ji Zhang nav stack expects:
  - /state_estimation (Odometry, 200 Hz, frame: map→sensor)
  - /registered_scan (PointCloud2, 10 Hz, frame: map)
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

# Sensor mounting offset (from unitree_go2.yaml)
_SENSOR_X: float = 0.2
_SENSOR_Y: float = 0.0
_SENSOR_Z: float = 0.1


class Go2VNavBridge(Node):
    """ROS2 node bridging MuJoCoGo2 to Vector Navigation Stack."""

    def __init__(self, go2: MuJoCoGo2) -> None:
        super().__init__("go2_vnav_bridge")
        self._go2 = go2
        self._last_cmd_time = time.time()

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
        self._path_time = 0.0

        # Manual control: teleop /joy → direct velocity
        self.create_subscription(Joy, "/joy", self._joy_cb, 10)
        # Nav stack path follower output
        self.create_subscription(TwistStamped, "/navigation_cmd_vel", self._cmd_vel_stamped_cb, 10)
        # Manual CLI control
        self.create_subscription(Twist, "/cmd_vel_nav", self._cmd_vel_cb, 10)

        # Path follower timer (20 Hz)
        self.create_timer(1.0 / 20.0, self._follow_path)
        # Camera rendering (5 Hz)
        self.create_timer(1.0 / 5.0, self._publish_camera)
        self._cmd_count = 0
        self._teleop_until = 0.0  # teleop priority timestamp

        # Timers
        self.create_timer(1.0 / 200.0, self._publish_odom)       # 200 Hz
        self.create_timer(1.0 / 10.0, self._publish_pointcloud)  # 10 Hz
        self.create_timer(1.0 / 10.0, self._publish_scan)        # 10 Hz
        self.create_timer(0.5, self._publish_joy_speed)           # 2 Hz
        self.create_timer(1.0, self._safety_check)                # 1 Hz

        self.get_logger().info(
            "Go2VNavBridge started — /state_estimation, /registered_scan, /joy, /speed"
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
        vx = msg.twist.linear.x
        vy = msg.twist.linear.y
        vyaw = msg.twist.angular.z
        self._go2.set_velocity(vx, vy, vyaw)
        self._last_cmd_time = time.time()
        self._cmd_count += 1
        if self._cmd_count <= 3 or self._cmd_count % 100 == 0:
            self.get_logger().info(f"cmd_vel: vx={vx:.3f} vy={vy:.3f} vyaw={vyaw:.3f}")

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

        # Compute ground Z from robot height (~0.28m standing, ground at z≈0)
        odom = self._go2.get_odometry()
        ground_z = odom.z - 0.28  # robot CoM is ~0.28m above ground

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
            # intensity = height above ground (terrain_analysis uses this
            # for ground/obstacle classification: <0.1m = ground)
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
        """Render camera from MuJoCo using a free camera at camera_link position."""
        try:
            import mujoco
            if not hasattr(self, '_renderer'):
                self._renderer = mujoco.Renderer(
                    self._go2._mj.model, 240, 320
                )

            model = self._go2._mj.model
            data = self._go2._mj.data

            # Camera at sensor + camera offset, looking forward
            odom = self._go2.get_odometry()
            heading = self._go2.get_heading()
            cos_h = math.cos(heading)
            sin_h = math.sin(heading)

            # Camera position (sensor + camera offset from go2 config)
            cam_x = odom.x + cos_h * (_SENSOR_X + 0.1)
            cam_y = odom.y + sin_h * (_SENSOR_X + 0.1)
            cam_z = odom.z + _SENSOR_Z - 0.04

            scene = mujoco.MjvScene(model, maxgeom=1000)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [cam_x + cos_h * 2, cam_y + sin_h * 2, cam_z]
            cam.distance = 2.0
            cam.azimuth = math.degrees(heading) + 180
            cam.elevation = -15

            opt = mujoco.MjvOption()
            mujoco.mjv_updateScene(model, data, opt, None, cam,
                                   mujoco.mjtCatBit.mjCAT_ALL, scene)

            # RGB render
            self._renderer.update_scene(data)
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

            # Depth via separate renderer instance
            if not hasattr(self, '_depth_renderer'):
                try:
                    self._depth_renderer = mujoco.Renderer(
                        self._go2._mj.model, 240, 320
                    )
                    self._depth_renderer.enable_depth_rendering(True)
                except Exception:
                    self._depth_renderer = None

            if self._depth_renderer is not None:
                try:
                    self._depth_renderer.update_scene(data)
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
        """Receive planned path from localPlanner and transform to map frame.

        localPlanner publishes paths in the sensor/body frame (relative to robot).
        We transform to map frame using current pose for the path follower.
        """
        odom = self._go2.get_odometry()
        heading = self._go2.get_heading()
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        # Sensor position in map frame
        sx = odom.x + cos_h * _SENSOR_X - sin_h * _SENSOR_Y
        sy = odom.y + sin_h * _SENSOR_X + cos_h * _SENSOR_Y

        self._current_path = []
        for p in msg.poses:
            # Rotate from sensor frame to map frame
            lx, ly = p.pose.position.x, p.pose.position.y
            mx = sx + lx * cos_h - ly * sin_h
            my = sy + lx * sin_h + ly * cos_h
            self._current_path.append((mx, my))

        self._path_time = time.time()
        # Throttle path log to once per second (localPlanner publishes at ~10Hz)
        now_mono = time.monotonic()
        if self._current_path and now_mono - getattr(self, '_last_path_log', 0) > 1.0:
            self._last_path_log = now_mono
            self.get_logger().info(
                f"Path: {len(self._current_path)} pts, "
                f"target=({self._current_path[-1][0]:.1f},{self._current_path[-1][1]:.1f}) "
                f"robot=({odom.x:.1f},{odom.y:.1f})"
            )

    def _follow_path(self) -> None:
        """Simple pure-pursuit path follower at 20 Hz.

        Finds the lookahead point on the path and generates smooth velocity.
        Much more stable than C++ pathFollower with MPC gait.
        """
        # Don't override teleop or stale path
        if time.time() < self._teleop_until:
            return
        if not self._current_path or time.time() - self._path_time > 5.0:
            return

        odom = self._go2.get_odometry()
        rx, ry = odom.x, odom.y
        heading = self._go2.get_heading()

        # Find lookahead point (0.8m ahead on path)
        lookahead = 0.8
        target = None
        for px, py in self._current_path:
            d = math.sqrt((px - rx) ** 2 + (py - ry) ** 2)
            if d >= lookahead:
                target = (px, py)
                break

        # If no point far enough, use last point
        if target is None and self._current_path:
            target = self._current_path[-1]

        if target is None:
            return

        tx, ty = target
        dx = tx - rx
        dy = ty - ry
        dist = math.sqrt(dx * dx + dy * dy)

        # Close enough — stop
        if dist < 0.3:
            self._go2.set_velocity(0.0, 0.0, 0.0)
            self.get_logger().info(f"Path follower: reached target (dist={dist:.2f})")
            self._current_path = []
            return

        # Heading to target
        desired_heading = math.atan2(dy, dx)
        heading_err = desired_heading - heading
        while heading_err > math.pi:
            heading_err -= 2 * math.pi
        while heading_err < -math.pi:
            heading_err += 2 * math.pi

        # Pure pursuit: proportional yaw + forward speed based on alignment
        vyaw = float(np.clip(heading_err * 2.5, -1.5, 1.5))

        # Only drive forward when roughly aligned (< 45 degrees)
        if abs(heading_err) < 0.8:
            vx = float(np.clip(dist * 0.8, 0.15, 0.7))
        else:
            vx = 0.0  # rotate in place first

        self._go2.set_velocity(vx, 0.0, vyaw)
        self._last_cmd_time = time.time()
        # Log every 1 second (20Hz timer, so every 20th call)
        if not hasattr(self, '_follow_count'):
            self._follow_count = 0
        self._follow_count += 1
        if self._follow_count % 20 == 0:
            self.get_logger().info(
                f"Following: vx={vx:.2f} vyaw={vyaw:.2f} dist={dist:.2f} "
                f"heading_err={math.degrees(heading_err):.1f}deg target=({tx:.1f},{ty:.1f})"
            )

    def _safety_check(self) -> None:
        if time.time() - self._last_cmd_time > 0.5:
            self._go2.set_velocity(0.0, 0.0, 0.0)


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
