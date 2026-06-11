# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""HabitatSysnavBridge — feed SysNav from the photoreal world (M4, Part A).

Publishes the THREE input topics the SysNav sibling workspace consumes
(archived contract, `git show 2f83690:docs/archive/sysnav_integration.md`):

- ``/camera/image``      sensor_msgs/Image — equirect RGB pano (Theta-Z1 role)
- ``/registered_scan``   sensor_msgs/PointCloud2 — WORLD-frame cloud,
                         unprojected from the SAME pose-synced equirect depth
- ``/state_estimation``  nav_msgs/Odometry — exact ground-truth pose

N1 makes this ALSO the locomotion boundary: a second timer publishes
``/state_estimation`` at nav-stack rate (50 Hz default, with twist) off the
bridge's STREAM channel — never queued behind the pano render — and a
``/cmd_vel`` (geometry_msgs/Twist) subscription streams velocity commands
into the base. The 2 Hz pano tick keeps publishing the pose-synced triplet
(M4 contract unchanged).

The v2.4 MuJoCo sensors stalled on semantically EMPTY renders; this bridge
finally delivers the wiring on real visuals. rclpy imports are lazy (module
imports clean without ROS2); the unprojection is a PURE function tested
without rclpy. PointCloud2 byte layout reuses the tested
``hardware/sim/sensors/lidar360.LidarSample`` serialiser.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_EYE_HEIGHT_M = 1.2  # pano sensors mount (mirrors server.py specs)

# SysNav camera contract (cloud_image_fusion CAMERA_PARA): a 1920x640 equirect
# with 120° vertical FOV — i.e. a full 960-row equirect cropped 30° (160 rows)
# top and bottom. The server renders 960x1920; we publish rows [160:800].
_SYSNAV_IMG_WIDTH = 1920
_SYSNAV_IMG_HEIGHT = 640
_SYSNAV_CROP_TOP = 160


def crop_to_sysnav_image(rgb: np.ndarray) -> np.ndarray:
    """Crop a full equirect pano to the SysNav 1920x640 ±60° contract."""
    h, w = rgb.shape[0], rgb.shape[1]
    if (h, w) == (_SYSNAV_IMG_HEIGHT, _SYSNAV_IMG_WIDTH):
        return rgb
    if w != _SYSNAV_IMG_WIDTH or h < _SYSNAV_CROP_TOP + _SYSNAV_IMG_HEIGHT:
        raise ValueError(
            f"pano {h}x{w} cannot satisfy the SysNav "
            f"{_SYSNAV_IMG_HEIGHT}x{_SYSNAV_IMG_WIDTH} contract"
        )
    return rgb[_SYSNAV_CROP_TOP:_SYSNAV_CROP_TOP + _SYSNAV_IMG_HEIGHT]


def unproject_equirect_depth(
    depth: np.ndarray,
    pos_world: "tuple[float, float, float] | list[float]",
    heading: float,
    *,
    eye_height: float = _EYE_HEIGHT_M,
    stride: int = 4,
    max_depth: float = 20.0,
) -> np.ndarray:
    """Equirect depth pano -> (N, 3) float32 WORLD-frame points. Pure math.

    Conventions (pinned by tests): pixel column W/2 looks along the agent
    heading; lon grows to the agent's RIGHT (lon=+pi/2 is heading - 90°);
    row H/2 is the horizon; the sensor sits ``eye_height`` above the agent's
    world z. Invalid/far returns are dropped.
    """
    h, w = depth.shape
    d = depth[::stride, ::stride].astype(np.float32)
    rows = np.arange(0, h, stride, dtype=np.float32)
    cols = np.arange(0, w, stride, dtype=np.float32)
    lat = (0.5 - rows / float(h)) * math.pi          # +pi/2 (up) .. -pi/2
    lon = (cols / float(w) - 0.5) * 2.0 * math.pi    # -pi .. +pi, 0 = ahead
    lon_g, lat_g = np.meshgrid(lon, lat)

    # Bearing in the WORLD frame: heading at lon=0, turning RIGHT as lon grows.
    bearing = heading - lon_g
    cos_lat = np.cos(lat_g)
    dirs = np.stack(
        [
            cos_lat * np.cos(bearing),   # world x
            cos_lat * np.sin(bearing),   # world y
            np.sin(lat_g),               # world z (up)
        ],
        axis=-1,
    )
    valid = np.isfinite(d) & (d > 0.05) & (d < max_depth)
    pts = dirs[valid] * d[valid][..., None]
    origin = np.array(
        [pos_world[0], pos_world[1], pos_world[2] + eye_height], dtype=np.float32
    )
    return (pts + origin).astype(np.float32)


def make_pointcloud2_parts(points_xyz: np.ndarray) -> "tuple[bytes, list, int]":
    """(data_bytes, field_layout, point_step) via the tested LidarSample path.

    LidarSample carries (N, 4) ``[x, y, z, intensity]`` — pad a unit
    intensity channel onto the unprojected xyz cloud.
    """
    from vector_os_nano.hardware.sim.sensors.lidar360 import LidarSample

    xyzi = np.concatenate(
        [
            points_xyz.astype(np.float32),
            np.ones((len(points_xyz), 1), dtype=np.float32),
        ],
        axis=1,
    )
    sample = LidarSample(stamp_seconds=0.0, frame_id="map", points=xyzi)
    return sample.pointcloud2_data_bytes(), sample.field_layout, sample.point_step


class HabitatSysnavBridge:
    """rclpy node publishing the SysNav input triplet from a HabitatBase.

    Lazy ROS2: constructing requires rclpy; importing this module does not.
    """

    def __init__(
        self,
        base: Any,
        hz: float = 2.0,
        *,
        state_hz: float = 50.0,
        cmd_vel_topic: str = "/cmd_vel",
        node_name: str = "habitat_sysnav_bridge",
    ) -> None:
        import rclpy  # noqa: PLC0415 — lazy by design
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from sensor_msgs.msg import Image, PointCloud2, PointField

        self._base = base
        self._Image = Image
        self._PointCloud2 = PointCloud2
        self._PointField = PointField
        self._Odometry = Odometry

        if not rclpy.ok():
            rclpy.init()
        self.node: "Node" = rclpy.create_node(node_name)
        self._pub_image = self.node.create_publisher(Image, "/camera/image", 5)
        self._pub_cloud = self.node.create_publisher(PointCloud2, "/registered_scan", 5)
        self._pub_odom = self.node.create_publisher(Odometry, "/state_estimation", 10)
        self._timer = self.node.create_timer(1.0 / hz, self.publish_once)
        # N1: fast odom off the stream channel + streaming velocity input,
        # on a SEPARATE node. Live-measured: rclpy's MultiThreadedExecutor
        # caps a 50 Hz timer at ~29 Hz (with or without pano load) while a
        # SingleThreadedExecutor hits 50.0 — so the fast path gets its own
        # node and spin_in_background() gives each node a dedicated
        # single-threaded executor.
        self.fast_node: "Node" = rclpy.create_node(node_name + "_fast")
        self._pub_odom_fast = self.fast_node.create_publisher(
            Odometry, "/state_estimation", 10
        )
        self._state_timer = None
        if state_hz and state_hz > 0:
            self._state_timer = self.fast_node.create_timer(
                1.0 / state_hz, self.publish_state_once
            )
        self._sub_cmd = None
        if cmd_vel_topic:
            self._sub_cmd = self.fast_node.create_subscription(
                Twist, cmd_vel_topic, self._on_cmd_vel, 10
            )
        self._executors: list = []
        logger.info(
            "HabitatSysnavBridge up (pano %.1f Hz, state %.1f Hz, cmd %s)",
            hz, state_hz or 0.0, cmd_vel_topic or "<off>",
        )

    def spin_in_background(self) -> None:
        """Spin both nodes, each on its OWN single-threaded executor thread
        (the only configuration measured to sustain the 50 Hz fast path —
        see the class docstring note on MultiThreadedExecutor). Idempotent."""
        import threading

        from rclpy.executors import SingleThreadedExecutor

        if self._executors:
            return
        for node in (self.node, self.fast_node):
            ex = SingleThreadedExecutor()
            ex.add_node(node)
            threading.Thread(
                target=ex.spin, daemon=True, name=f"sysnav-{node.get_name()}"
            ).start()
            self._executors.append(ex)

    # one tick — also directly callable from tests (no timer needed)
    def publish_once(self) -> None:
        try:
            pano = self._base.get_pano()
        except Exception as exc:  # noqa: BLE001 — keep ticking, log loud
            logger.warning("sysnav bridge: pano fetch failed: %s", exc)
            return
        now = self.node.get_clock().now().to_msg()
        try:
            img = crop_to_sysnav_image(pano["rgb"])
        except ValueError as exc:
            logger.warning("sysnav bridge: %s — publishing uncropped", exc)
            img = pano["rgb"]
        self._pub_image.publish(self._image_msg(img, now))
        # The cloud keeps the FULL sphere (richer for voxel clustering); the
        # fusion projects it onto the cropped image with its own model.
        pts = unproject_equirect_depth(pano["depth"], pano["pos"], pano["heading"])
        self._pub_cloud.publish(self._cloud_msg(pts, now))
        self._pub_odom.publish(self._odom_msg(pano["pos"], pano["heading"], now))

    def publish_state_once(self) -> None:
        """Fast odom tick (N1) — one stream-channel roundtrip, with twist."""
        try:
            st = self._base.get_state()
        except Exception as exc:  # noqa: BLE001 — keep ticking, log loud
            logger.warning("sysnav bridge: state fetch failed: %s", exc)
            return
        now = self.fast_node.get_clock().now().to_msg()
        self._pub_odom_fast.publish(
            self._odom_msg(st["pos"], st["heading"], now, vel=st.get("vel"))
        )

    def _on_cmd_vel(self, msg: Any) -> None:
        """Stream a /cmd_vel Twist into the base (non-blocking, deadman'd
        server-side — a dead publisher stops the robot, never a runaway)."""
        try:
            self._base.set_velocity(
                float(msg.linear.x), float(msg.linear.y), float(msg.angular.z)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("sysnav bridge: cmd_vel apply failed: %s", exc)

    # -- message builders ---------------------------------------------------
    def _image_msg(self, rgb: np.ndarray, stamp: Any) -> Any:
        msg = self._Image()
        msg.header.stamp = stamp
        msg.header.frame_id = "camera"
        msg.height, msg.width = int(rgb.shape[0]), int(rgb.shape[1])
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = rgb.tobytes()
        return msg

    def _cloud_msg(self, points: np.ndarray, stamp: Any) -> Any:
        data, layout, point_step = make_pointcloud2_parts(points)
        msg = self._PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            self._PointField(
                name=name, offset=off, datatype=self._PointField.FLOAT32, count=1
            )
            for name, off, _ in layout
        ]
        msg.is_bigendian = False
        msg.point_step = point_step
        msg.row_step = point_step * len(points)
        msg.data = data
        msg.is_dense = True
        return msg

    def _odom_msg(
        self,
        pos: "list[float]",
        heading: float,
        stamp: Any,
        vel: "list[float] | None" = None,
    ) -> Any:
        msg = self._Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.child_frame_id = "sensor"
        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])
        msg.pose.pose.orientation.z = math.sin(heading / 2.0)
        msg.pose.pose.orientation.w = math.cos(heading / 2.0)
        if vel:
            msg.twist.twist.linear.x = float(vel[0])
            msg.twist.twist.angular.z = float(vel[2])
        return msg

    def destroy(self) -> None:
        for ex in self._executors:
            try:
                ex.shutdown(timeout_sec=2.0)
            except Exception:  # noqa: BLE001
                pass
        self._executors = []
        for node in (self.node, self.fast_node):
            try:
                node.destroy_node()
            except Exception:  # noqa: BLE001
                pass


def main(argv: "list[str] | None" = None) -> int:
    """Run the habitat→SysNav feed standalone:

        python -m vector_os_nano.playground.habitat.sysnav_bridge \\
            --scene <path|scenario-ref> [--hz 2] [--wander]

    Boots the habitat conda subprocess, publishes the SysNav input triplet,
    and (with --wander) keeps navigating to random navmesh points so the
    semantic mapper sees varied viewpoints. Ctrl-C to stop; the watchdog
    sweeps strays by VECTOR_RUN_ID on abnormal exit.
    """
    import argparse
    import random
    import time

    import rclpy

    from vector_os_nano.playground.habitat.base import HabitatBase
    from vector_os_nano.playground.habitat.scenes import resolve_scene_ref

    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--hz", type=float, default=2.0)
    ap.add_argument("--wander", action="store_true",
                    help="keep navigating to random navmesh points")
    args = ap.parse_args(argv)

    base = HabitatBase(scene=resolve_scene_ref(args.scene))
    base.connect()
    bridge = HabitatSysnavBridge(base, hz=args.hz)
    bridge.spin_in_background()  # dedicated per-node executors (50 Hz fast path)
    logger.info("habitat sysnav feed up — Ctrl-C to stop")
    next_wander = time.time() + 5.0
    try:
        while rclpy.ok():
            time.sleep(0.1)
            if args.wander and time.time() >= next_wander:
                here = base.get_position()
                dx, dy = random.uniform(-4, 4), random.uniform(-4, 4)
                tgt = base.snap_point([here[0] + dx, here[1] + dy, here[2]])
                base.navigate_to(tgt[0], tgt[1])
                next_wander = time.time() + 5.0
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy()
        base.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
