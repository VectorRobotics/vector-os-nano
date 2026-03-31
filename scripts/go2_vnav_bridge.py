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
from nav_msgs.msg import Odometry as OdometryMsg
from sensor_msgs.msg import PointCloud2, PointField, Joy, LaserScan as LaserScanMsg
from geometry_msgs.msg import TwistStamped, Twist, TransformStamped, PointStamped
from std_msgs.msg import Float32, Header
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
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

        self._tf_broadcaster = TransformBroadcaster(self)
        self._static_tf = StaticTransformBroadcaster(self)

        # Publish static TF: sensor → base_link (sensor mounting offset)
        self._publish_static_tf()

        # Subscribe to /cmd_vel (TwistStamped from nav stack)
        # Also accept plain Twist for manual control
        self.create_subscription(TwistStamped, "/cmd_vel", self._cmd_vel_stamped_cb, 10)
        self.create_subscription(Twist, "/cmd_vel_nav", self._cmd_vel_cb, 10)
        self._cmd_count = 0

        # Timers
        self.create_timer(1.0 / 200.0, self._publish_odom)       # 200 Hz
        self.create_timer(1.0 / 10.0, self._publish_pointcloud)  # 10 Hz
        self.create_timer(1.0 / 10.0, self._publish_scan)        # 10 Hz
        self.create_timer(0.5, self._publish_joy_speed)           # 2 Hz
        self.create_timer(1.0, self._safety_check)                # 1 Hz

        self.get_logger().info(
            "Go2VNavBridge started — /state_estimation, /registered_scan, /joy, /speed"
        )

    def _publish_static_tf(self) -> None:
        """Static TF: sensor → base_link offset."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "sensor"
        t.child_frame_id = "base_link"
        # base_link is behind and below sensor
        t.transform.translation.x = -_SENSOR_X
        t.transform.translation.y = -_SENSOR_Y
        t.transform.translation.z = -_SENSOR_Z
        t.transform.rotation.w = 1.0
        self._static_tf.sendTransform(t)

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
        """Fake joystick for pathFollower autonomyMode + desired speed."""
        joy = Joy()
        joy.header.stamp = self.get_clock().now().to_msg()
        joy.axes = [0.0] * 8
        joy.buttons = [0] * 11
        # LT trigger pressed → autonomyMode = true
        joy.axes[2] = -1.0
        # RT trigger pressed → manualMode = false
        joy.axes[5] = -1.0
        self._joy_pub.publish(joy)

        speed = Float32()
        speed.data = 0.5
        self._speed_pub.publish(speed)

    def _safety_check(self) -> None:
        if time.time() - self._last_cmd_time > 2.0:
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
