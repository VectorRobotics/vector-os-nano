#!/usr/bin/env python3
"""Go2 MuJoCo ↔ ROS2 Nav2 bridge.

Runs MuJoCoGo2 (MPC backend) and publishes ROS2 topics for Nav2:
  - /odom (nav_msgs/Odometry, 50 Hz)
  - /scan (sensor_msgs/LaserScan, 10 Hz)
  - TF: odom → base (50 Hz)
  - Subscribes: /cmd_vel → go2.set_velocity()

Usage:
    source /opt/ros/jazzy/setup.bash
    cd ~/Desktop/vector_os_nano
    PYTHONPATH=".venv-nano/lib/python3.12/site-packages:\
.venv-nano/lib/python3.12/site-packages/cmeel.prefix/lib/python3.12/site-packages:\
/home/yusen/Desktop/go2-convex-mpc/src:\
.:$PYTHONPATH" python3 scripts/go2_nav_bridge.py
"""
from __future__ import annotations

import math
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: stub vector_os_nano packages to avoid httpx cascade
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

# Load core.types properly (needed by mujoco_go2)
import importlib.util

_types_path = _repo / "vector_os_nano" / "core" / "types.py"
_ts = importlib.util.spec_from_file_location("vector_os_nano.core.types", str(_types_path))
_tm = importlib.util.module_from_spec(_ts)
sys.modules.setdefault("vector_os_nano.core.types", _tm)
_ts.loader.exec_module(_tm)

# ---------------------------------------------------------------------------
# ROS2 imports
# ---------------------------------------------------------------------------

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry as OdometryMsg
from sensor_msgs.msg import LaserScan as LaserScanMsg
from geometry_msgs.msg import Twist, TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np

# ---------------------------------------------------------------------------
# MuJoCoGo2 import
# ---------------------------------------------------------------------------

from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2


class Go2NavBridge(Node):
    """ROS2 node bridging MuJoCoGo2 to Nav2."""

    def __init__(self, go2: MuJoCoGo2) -> None:
        super().__init__("go2_nav_bridge")
        self._go2 = go2
        self._last_cmd_time = time.time()

        # QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        # /scan must be RELIABLE — Nav2 costmap and AMCL require it
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Publishers
        self._odom_pub = self.create_publisher(OdometryMsg, "/odom", sensor_qos)
        self._scan_pub = self.create_publisher(LaserScanMsg, "/scan", reliable_qos)
        self._tf_broadcaster = TransformBroadcaster(self)

        # Subscribe to all cmd_vel variants — Nav2 Jazzy pipeline:
        # controller → cmd_vel_nav → smoother → cmd_vel_smoothed → monitor → cmd_vel
        for topic in ["/cmd_vel", "/cmd_vel_nav", "/cmd_vel_smoothed"]:
            self.create_subscription(Twist, topic, self._make_cmd_cb(topic), 10)
        self._cmd_count = 0

        # Timers
        self.create_timer(1.0 / 50.0, self._publish_odom)   # 50 Hz
        self.create_timer(1.0 / 10.0, self._publish_scan)   # 10 Hz
        self.create_timer(1.0, self._safety_check)           # 1 Hz

        self.get_logger().info("Go2NavBridge started — publishing /odom, /scan, TF")

    def _make_cmd_cb(self, topic: str):
        def cb(msg: Twist) -> None:
            vx = msg.linear.x
            vy = msg.linear.y
            vyaw = msg.angular.z
            self._go2.set_velocity(vx, vy, vyaw)
            self._last_cmd_time = time.time()
            self._cmd_count += 1
            if self._cmd_count <= 3 or self._cmd_count % 50 == 0:
                self.get_logger().info(
                    f"cmd_vel from {topic}: vx={vx:.3f} vy={vy:.3f} vyaw={vyaw:.3f}"
                )
        return cb

    def _publish_odom(self) -> None:
        odom = self._go2.get_odometry()
        now = self.get_clock().now().to_msg()

        # Odometry message
        msg = OdometryMsg()
        msg.header.stamp = now
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base"
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

        # TF: odom → base
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "odom"
        t.child_frame_id = "base"
        t.transform.translation.x = odom.x
        t.transform.translation.y = odom.y
        t.transform.translation.z = odom.z
        t.transform.rotation.x = odom.qx
        t.transform.rotation.y = odom.qy
        t.transform.rotation.z = odom.qz
        t.transform.rotation.w = odom.qw
        self._tf_broadcaster.sendTransform(t)

    def _publish_scan(self) -> None:
        scan = self._go2.get_lidar_scan()
        now = self.get_clock().now().to_msg()

        msg = LaserScanMsg()
        msg.header.stamp = now
        msg.header.frame_id = "base"
        msg.angle_min = scan.angle_min
        msg.angle_max = scan.angle_max
        msg.angle_increment = scan.angle_increment
        msg.range_min = scan.range_min
        msg.range_max = scan.range_max
        msg.ranges = list(scan.ranges)
        # time_increment and scan_time for 360 rays at 10 Hz
        msg.time_increment = 0.0
        msg.scan_time = 0.1
        self._scan_pub.publish(msg)

    def _safety_check(self) -> None:
        # Stop if no cmd_vel for 2 seconds
        if time.time() - self._last_cmd_time > 2.0:
            self._go2.set_velocity(0.0, 0.0, 0.0)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Go2 Nav2 Bridge")
    parser.add_argument("--no-gui", action="store_true", help="Run headless")
    parser.add_argument("--sinusoidal", action="store_true", help="Force sinusoidal backend")
    args = parser.parse_args()

    backend = "sinusoidal" if args.sinusoidal else "auto"
    gui = not args.no_gui

    print(f"Starting MuJoCoGo2 (gui={gui}, backend={backend})...")
    go2 = MuJoCoGo2(gui=gui, room=True, backend=backend)
    go2.connect()
    print("Standing up...")
    go2.stand(duration=2.0)
    pos = go2.get_position()
    print(f"Go2 standing at ({pos[0]:.1f}, {pos[1]:.1f}), z={pos[2]:.3f}m")

    print("Starting ROS2 bridge...")
    rclpy.init()
    node = Go2NavBridge(go2)

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
