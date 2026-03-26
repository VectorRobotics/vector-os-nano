#!/usr/bin/env python3
"""Go2 MuJoCo ↔ ROS2 Navigation Stack Bridge.

Three functions in one node:
  1. Subscribe /cmd_vel → base.set_velocity()
  2. Publish /state_estimation (Odometry, 50Hz) + /tf (odom→base_link)
  3. Publish /registered_scan (PointCloud2, 10Hz) from simulated lidar

This replaces both Unity + vehicle_simulator for MuJoCo-based simulation.
The navigation stack (terrain_analysis, local_planner, FAR planner) runs unchanged.

Usage:
    # In a ROS2-sourced terminal (system python3.10, NOT conda):
    python3 go2_bridge.py
"""
from __future__ import annotations

import math
import struct
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import TwistStamped, TransformStamped
from nav_msgs.msg import Odometry as OdomMsg
from sensor_msgs.msg import Joy, PointCloud2, PointField
from std_msgs.msg import Float32, Header
from tf2_ros import TransformBroadcaster


class Go2MuJoCoBridge(Node):
    """Bridge between MuJoCoGo2 physics and the vector_navigation_stack."""

    def __init__(self, base):
        super().__init__("go2_mujoco_bridge")
        self._base = base
        self.get_logger().info("Go2MuJoCoBridge starting...")

        # QoS for sensor data (match nav stack expectations)
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # --- Subscriber: /cmd_vel → base.set_velocity() ---
        self._cmd_vel_sub = self.create_subscription(
            TwistStamped,
            "/cmd_vel",
            self._on_cmd_vel,
            10,
        )

        # --- Publisher: /state_estimation (nav_msgs/Odometry) ---
        self._odom_pub = self.create_publisher(OdomMsg, "/state_estimation", 5)

        # --- Publisher: /registered_scan (PointCloud2) ---
        # Use RELIABLE QoS (nav stack terrain_analysis expects RELIABLE)
        self._scan_pub = self.create_publisher(PointCloud2, "/registered_scan", 5)

        # --- TF broadcaster: map → sensor ---
        self._tf_broadcaster = TransformBroadcaster(self)

        # --- Publisher: fake joystick to enable autonomy mode in pathFollower ---
        self._joy_pub = self.create_publisher(Joy, "/joy", 5)
        self._speed_pub = self.create_publisher(Float32, "/speed", 5)

        # --- Timers ---
        self._odom_timer = self.create_timer(0.02, self._publish_odom)    # 50 Hz
        self._scan_timer = self.create_timer(0.1, self._publish_scan)     # 10 Hz
        self._joy_timer = self.create_timer(0.5, self._publish_autonomy)  # 2 Hz

        self._last_cmd_time = time.time()
        self.get_logger().info("Go2MuJoCoBridge ready. Publishing /state_estimation + /registered_scan")

    def _publish_autonomy(self) -> None:
        """Publish fake joystick + speed to keep pathFollower in autonomy mode.

        pathFollower requires:
          1. joy.axes[2] < -0.1 (LT trigger) → sets autonomyMode=true
          2. joy.axes[4] == 0 (right stick Y neutral) → joySpeed=0
          3. /speed topic → sets actual desired speed
        Without this, pathFollower outputs zero velocity even with valid paths.
        """
        joy_msg = Joy()
        joy_msg.header.stamp = self.get_clock().now().to_msg()
        # axes[2]=-1.0: LT pressed → autonomyMode=true
        # axes[4]=0.0: no manual forward → joySpeed=0 (overridden by /speed)
        # axes[5]=-1.0: RT pressed → manualMode=false
        joy_msg.axes = [0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0]
        joy_msg.buttons = [0] * 11
        self._joy_pub.publish(joy_msg)

        # Set desired speed for autonomy
        speed_msg = Float32()
        speed_msg.data = 0.5  # 0.5 m/s
        self._speed_pub.publish(speed_msg)

    def _on_cmd_vel(self, msg: TwistStamped) -> None:
        """Receive velocity command from nav stack pathFollower."""
        vx = msg.twist.linear.x
        vy = msg.twist.linear.y
        vyaw = msg.twist.angular.z
        self._base.set_velocity(vx, vy, vyaw)
        self._last_cmd_time = time.time()

    def _publish_odom(self) -> None:
        """Publish odometry as /state_estimation + TF."""
        odom = self._base.get_odometry()
        if odom is None:
            return

        now = self.get_clock().now().to_msg()

        # Odometry message
        msg = OdomMsg()
        msg.header.stamp = now
        msg.header.frame_id = "map"
        msg.child_frame_id = "sensor"  # nav stack expects "sensor" frame

        msg.pose.pose.position.x = odom.x
        msg.pose.pose.position.y = odom.y
        msg.pose.pose.position.z = odom.z
        msg.pose.pose.orientation.x = odom.qx
        msg.pose.pose.orientation.y = odom.qy
        msg.pose.pose.orientation.z = odom.qz
        msg.pose.pose.orientation.w = odom.qw

        msg.twist.twist.linear.x = odom.vx
        msg.twist.twist.linear.y = odom.vy
        msg.twist.twist.linear.z = odom.vz
        msg.twist.twist.angular.z = odom.vyaw

        self._odom_pub.publish(msg)

        # TF: map → sensor (nav stack uses "sensor" as the lidar frame)
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "map"
        t.child_frame_id = "sensor"
        t.transform.translation.x = odom.x
        t.transform.translation.y = odom.y
        t.transform.translation.z = odom.z + 0.15  # lidar height offset
        t.transform.rotation.x = odom.qx
        t.transform.rotation.y = odom.qy
        t.transform.rotation.z = odom.qz
        t.transform.rotation.w = odom.qw
        self._tf_broadcaster.sendTransform(t)

        # Also publish map → vehicle (nav stack local_planner uses "vehicle")
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

    def _publish_scan(self) -> None:
        """Publish 3D lidar as /registered_scan (PointCloud2 in map frame).

        Uses get_3d_pointcloud() which provides multi-ring 3D data
        matching what terrain_analysis expects from a Livox MID360.
        """
        points = self._base.get_3d_pointcloud()
        if not points:
            return

        now = self.get_clock().now().to_msg()

        # Build PointCloud2 (XYZI format)
        msg = PointCloud2()
        msg.header.stamp = now
        msg.header.frame_id = "map"
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * len(points)
        msg.is_dense = True

        data = bytearray()
        for x, y, z, intensity in points:
            data.extend(struct.pack("ffff", x, y, z, intensity))
        msg.data = bytes(data)

        self._scan_pub.publish(msg)

    def _safety_timeout(self) -> None:
        """Stop robot if no cmd_vel received for 1 second."""
        if time.time() - self._last_cmd_time > 1.0:
            self._base.set_velocity(0, 0, 0)


def main():
    """Launch Go2 MuJoCo bridge standalone.

    This script starts MuJoCoGo2 in the CURRENT process (with viewer)
    and bridges it to the nav stack via ROS2 topics.
    """
    # Must use system python, not conda
    import sys
    if sys.version_info[:2] != (3, 10):
        print(f"WARNING: ROS2 Humble requires Python 3.10, got {sys.version}")

    # Add vector_os_nano to path if not installed
    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2

    print("Starting MuJoCoGo2 simulation...")
    base = MuJoCoGo2(gui=True, room=True)
    base.connect()
    base.stand()
    pos = base.get_position()
    print(f"Go2 standing at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    print("Initializing ROS2 bridge...")
    rclpy.init()
    bridge = Go2MuJoCoBridge(base)

    try:
        print("Bridge running. Nav stack can send /cmd_vel and read /state_estimation + /registered_scan")
        print("Press Ctrl+C to stop.")
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        bridge.destroy_node()
        rclpy.shutdown()
        base.disconnect()


if __name__ == "__main__":
    main()
