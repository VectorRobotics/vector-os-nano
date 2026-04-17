#!/usr/bin/env python3
"""Go2 sensor configuration for Isaac Sim 5.1 — Livox MID-360 + RealSense D435.

Sensor mounting matches the MuJoCo configuration:
  Lidar:  base_link + (0.3, 0.0, 0.2)m, -20 deg pitch
  Camera: base_link + (0.3, 0.0, 0.05)m, -5 deg pitch
"""
from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger("go2_sensors")

# Sensor mounting — must match MuJoCo go2.xml + mujoco_go2.py
LIDAR_MOUNT_POS = (0.3, 0.0, 0.2)
LIDAR_MOUNT_PITCH = -20.0  # degrees

CAMERA_MOUNT_POS = (0.3, 0.0, 0.05)
CAMERA_MOUNT_PITCH = -5.0  # degrees
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30


def attach_lidar(robot_prim_path: str) -> str:
    """Attach RTX lidar to Go2 using omni.kit.commands (Isaac Sim 5.1 API)."""
    import omni.kit.commands
    from pxr import Gf

    lidar_path = f"{robot_prim_path}/Lidar"

    try:
        success, prim = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path="/Lidar",
            parent=robot_prim_path,
            config="Example_Rotary",
            translation=Gf.Vec3d(*LIDAR_MOUNT_POS),
            orientation=Gf.Quatd(
                math.cos(math.radians(LIDAR_MOUNT_PITCH / 2)),
                0.0,
                math.sin(math.radians(LIDAR_MOUNT_PITCH / 2)),
                0.0,
            ),
        )
        if success and prim:
            lidar_path = str(prim.GetPath())
            logger.info("Livox MID-360 lidar attached at %s", lidar_path)
            return lidar_path
        else:
            logger.warning("LidarRtx command returned success=%s", success)
            return ""
    except Exception as exc:
        logger.warning("Failed to attach lidar: %s", exc)
        return ""


def attach_camera(robot_prim_path: str) -> tuple[str, str]:
    """Attach RealSense D435 camera (RGB + depth) to Go2."""
    from omni.isaac.sensor import Camera

    rgb_path = f"{robot_prim_path}/D435_RGB"
    depth_path = f"{robot_prim_path}/D435_Depth"

    # Compute orientation quaternion for pitch
    pitch_rad = math.radians(CAMERA_MOUNT_PITCH)
    qw = math.cos(pitch_rad / 2)
    qy = math.sin(pitch_rad / 2)
    orient = np.array([qw, 0.0, qy, 0.0])  # w, x, y, z

    try:
        # RGB camera
        Camera(
            prim_path=rgb_path,
            name="d435_rgb",
            resolution=(CAMERA_WIDTH, CAMERA_HEIGHT),
            frequency=CAMERA_FPS,
            position=np.array(CAMERA_MOUNT_POS),
            orientation=orient,
        )
        logger.info("D435 RGB camera at %s (%dx%d)", rgb_path, CAMERA_WIDTH, CAMERA_HEIGHT)

        # Depth camera (co-located)
        Camera(
            prim_path=depth_path,
            name="d435_depth",
            resolution=(CAMERA_WIDTH, CAMERA_HEIGHT),
            frequency=CAMERA_FPS,
            position=np.array(CAMERA_MOUNT_POS),
            orientation=orient,
        )
        logger.info("D435 Depth camera at %s", depth_path)

        return rgb_path, depth_path
    except Exception as exc:
        logger.warning("Failed to attach camera: %s", exc)
        return "", ""


def attach_all_sensors(robot_prim_path: str) -> dict[str, str]:
    """Attach all sensors to Go2. Returns dict of sensor paths."""
    sensors: dict[str, str] = {}

    lidar_path = attach_lidar(robot_prim_path)
    if lidar_path:
        sensors["lidar"] = lidar_path

    rgb_path, depth_path = attach_camera(robot_prim_path)
    if rgb_path:
        sensors["camera_rgb"] = rgb_path
        sensors["camera_depth"] = depth_path

    logger.info("Sensors attached: %s", list(sensors.keys()))
    return sensors
