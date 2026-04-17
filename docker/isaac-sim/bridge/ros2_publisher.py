#!/usr/bin/env python3
"""ROS2 publisher process — runs under system Python 3.12.

Reads Isaac Sim state from shared files and publishes ROS2 topics.
Subscribes to /cmd_vel and writes commands back for the physics process.

Topic names match the MuJoCo bridge exactly for drop-in compatibility.
"""
import os
import sys
import math
import time
import struct
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("ros2_pub")

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField, Image, Joy, JointState
from geometry_msgs.msg import Twist, TwistStamped, TransformStamped, Quaternion, Vector3
from std_msgs.msg import Float32, Header
from tf2_msgs.msg import TFMessage
from builtin_interfaces.msg import Time

_STATE_DIR = os.environ.get("ISAAC_STATE_DIR", "/tmp/isaac_state")

# Go2 joint names (12 DOF)
_GO2_JOINTS = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]


def _reliable_qos(depth: int = 5) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


class IsaacROS2Bridge(Node):
    """ROS2 node that reads Isaac Sim state files and publishes topics."""

    def __init__(self) -> None:
        super().__init__("isaac_sim_bridge")

        reliable = _reliable_qos()

        # Publishers
        self._odom_pub = self.create_publisher(Odometry, "/state_estimation", reliable)
        self._tf_pub = self.create_publisher(TFMessage, "/tf", 10)
        self._joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self._joy_pub = self.create_publisher(Joy, "/joy", reliable)
        self._speed_pub = self.create_publisher(Float32, "/speed", 10)
        self._scan_pub = self.create_publisher(PointCloud2, "/registered_scan", 10)
        self._rgb_pub = self.create_publisher(Image, "/camera/image", 10)
        self._depth_pub = self.create_publisher(Image, "/camera/depth", 10)

        # Subscribers
        self.create_subscription(Twist, "/cmd_vel_nav", self._cmd_vel_nav_cb, 10)
        self.create_subscription(TwistStamped, "/cmd_vel", self._cmd_vel_cb, 10)

        # Timers
        self.create_timer(1.0 / 50, self._publish_odom)
        self.create_timer(1.0 / 50, self._publish_tf)
        self.create_timer(1.0 / 50, self._publish_joints)
        self.create_timer(0.5, self._publish_joy)
        self.create_timer(0.5, self._publish_speed)

        self._last_odom = None
        self.get_logger().info("IsaacROS2Bridge started")

    def _make_header(self, frame_id: str = "map") -> Header:
        now = self.get_clock().now().to_msg()
        h = Header()
        h.stamp = now
        h.frame_id = frame_id
        return h

    def _read_odom(self) -> tuple | None:
        """Read odom state: (x,y,z, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz)."""
        path = os.path.join(_STATE_DIR, "odom.bin")
        try:
            with open(path, "rb") as f:
                data = f.read(52)  # 13 * 4 bytes
            if len(data) == 52:
                return struct.unpack("13f", data)
        except (FileNotFoundError, OSError):
            pass
        return None

    def _read_joints(self) -> list[float] | None:
        """Read joint positions."""
        path = os.path.join(_STATE_DIR, "joints.bin")
        try:
            with open(path, "rb") as f:
                data = f.read()
            n = len(data) // 4
            if n >= 12:
                return list(struct.unpack(f"{n}f", data))
        except (FileNotFoundError, OSError):
            pass
        return None

    def _write_cmd_vel(self, vx: float, vy: float, vyaw: float) -> None:
        """Write velocity command for physics process."""
        path = os.path.join(_STATE_DIR, "cmd_vel.bin")
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(struct.pack("fff", vx, vy, vyaw))
        os.replace(tmp, path)

    # -- Subscribers --

    def _cmd_vel_nav_cb(self, msg: Twist) -> None:
        self._write_cmd_vel(msg.linear.x, msg.linear.y, msg.angular.z)
        self.get_logger().info("cmd_vel_nav: vx=%.2f vy=%.2f vyaw=%.2f" % (msg.linear.x, msg.linear.y, msg.angular.z), throttle_duration_sec=2.0)

    def _cmd_vel_cb(self, msg: TwistStamped) -> None:
        self._write_cmd_vel(msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z)

    # -- Publishers --

    def _publish_odom(self) -> None:
        odom = self._read_odom()
        if odom is None:
            return
        self._last_odom = odom
        x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz = odom

        msg = Odometry()
        msg.header = self._make_header("map")
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = z
        msg.pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        msg.twist.twist.linear = Vector3(x=vx, y=vy, z=vz)
        msg.twist.twist.angular = Vector3(x=wx, y=wy, z=wz)
        self._odom_pub.publish(msg)

    def _publish_tf(self) -> None:
        if self._last_odom is None:
            return
        x, y, z, qx, qy, qz, qw = self._last_odom[:7]

        # map -> sensor (nav stack convention)
        tf = TransformStamped()
        tf.header = self._make_header("map")
        tf.child_frame_id = "sensor"
        tf.transform.translation = Vector3(x=x + 0.3, y=y, z=z + 0.2)  # sensor offset
        tf.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        # map -> vehicle
        tf2 = TransformStamped()
        tf2.header = self._make_header("map")
        tf2.child_frame_id = "vehicle"
        tf2.transform.translation = Vector3(x=x, y=y, z=z)
        tf2.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        msg = TFMessage()
        msg.transforms = [tf, tf2]
        self._tf_pub.publish(msg)

    def _publish_joints(self) -> None:
        joints = self._read_joints()
        if joints is None:
            return
        msg = JointState()
        msg.header = self._make_header("base_link")
        msg.name = _GO2_JOINTS[:len(joints)]
        msg.position = [float(j) for j in joints[:12]]
        self._joint_pub.publish(msg)

    def _publish_joy(self) -> None:
        msg = Joy()
        msg.header = self._make_header("base_link")
        msg.axes = [0.0] * 8
        msg.buttons = [0] * 11
        msg.buttons[4] = 1  # autonomous mode for pathFollower
        self._joy_pub.publish(msg)

    def _publish_speed(self) -> None:
        if self._last_odom is None:
            return
        vx, vy = self._last_odom[7], self._last_odom[8]
        msg = Float32()
        msg.data = math.sqrt(vx * vx + vy * vy)
        self._speed_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = IsaacROS2Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
