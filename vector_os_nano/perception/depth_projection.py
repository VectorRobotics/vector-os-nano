"""Depth-to-world projection for RealSense D435 (sim and real).

Converts pixel coordinates + depth value to world (x, y, z) using
camera intrinsics and robot pose. Sim-to-real compatible: same code
runs against MuJoCo depth or real RealSense frames.

RealSense D435 specs (640x480 mode):
    Depth FOV: 87° x 58° (H x V)
    RGB FOV:   69° x 42° (H x V)
    Range:     0.1m – 10m
    We use aligned_depth_to_color, so depth uses the RGB FOV.

Camera mounting on Go2: forward-facing on head, 0.3m forward of body
center, 0.15m above CoM, slight downward tilt.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics (no distortion for simulation).

    For real D435, distortion coefficients can be read from
    rs2_intrinsics and applied separately.
    """

    width: int
    height: int
    fx: float  # focal length x (pixels)
    fy: float  # focal length y (pixels)
    cx: float  # principal point x (pixels)
    cy: float  # principal point y (pixels)


def d435_intrinsics(width: int = 320, height: int = 240) -> CameraIntrinsics:
    """Return RealSense D435 intrinsics for the given resolution.

    Based on D435 RGB FOV of ~69° horizontal.
    Use this for REAL robot only. For MuJoCo sim, use mujoco_intrinsics().
    """
    hfov_rad = math.radians(69.0)
    fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
    fy = fx
    return CameraIntrinsics(
        width=width, height=height, fx=fx, fy=fy,
        cx=width / 2.0, cy=height / 2.0,
    )


def mujoco_intrinsics(width: int = 320, height: int = 240, vfov_deg: float = 42.0) -> CameraIntrinsics:
    """Return intrinsics for the MuJoCo d435_rgb/d435_depth named camera.

    The d435 cameras are defined in go2.xml with fovy=42° (matching real
    D435 vertical FOV). Horizontal FOV derives from aspect ratio.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        vfov_deg: Vertical FOV in degrees (from MJCF camera fovy attribute).
    """
    vfov_rad = math.radians(vfov_deg)
    fy = (height / 2.0) / math.tan(vfov_rad / 2.0)
    fx = fy  # square pixels
    return CameraIntrinsics(
        width=width, height=height, fx=fx, fy=fy,
        cx=width / 2.0, cy=height / 2.0,
    )


def get_intrinsics(width: int = 320, height: int = 240, sim: bool = True) -> CameraIntrinsics:
    """Return camera intrinsics for sim or real.

    In simulation, uses the MJCF d435 camera's fovy=42° (same as real D435).
    In practice, sim and real should give very similar intrinsics.
    """
    if sim:
        return mujoco_intrinsics(width, height)
    return d435_intrinsics(width, height)


# ---------------------------------------------------------------------------
# Projection functions
# ---------------------------------------------------------------------------


def pixel_to_camera(
    u: float,
    v: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float, float]:
    """Project a pixel (u, v) at depth_m to camera-frame 3D point.

    Camera frame: Z forward, X right, Y down (OpenCV convention).

    Returns (x_cam, y_cam, z_cam) in metres.
    """
    z = depth_m
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    return (x, y, z)


def camera_to_world(
    x_cam: float,
    y_cam: float,
    z_cam: float,
    robot_x: float,
    robot_y: float,
    robot_z: float,
    robot_heading: float,
    cam_xpos: Any = None,
    cam_xmat: Any = None,
) -> tuple[float, float, float]:
    """Transform camera-frame point to world frame.

    Two modes:
    1. If cam_xpos and cam_xmat are provided (from MuJoCo data.cam_xpos/xmat),
       uses the EXACT camera world pose. Most accurate for simulation.
    2. Otherwise, approximates from robot pose + mount offset + heading.

    Camera pixel convention: x_cam=right, y_cam=down, z_cam=depth (OpenCV).
    MuJoCo camera convention: col0=right, col1=up, col2=-forward (OpenGL).

    Returns (world_x, world_y, world_z).
    """
    if cam_xpos is not None and cam_xmat is not None:
        # Use exact MuJoCo camera transform
        import numpy as np
        xmat = np.array(cam_xmat, dtype=np.float64).reshape(3, 3)
        pos = np.array(cam_xpos, dtype=np.float64)
        # MuJoCo xmat columns: right, up, -forward
        cam_right = xmat[:, 0]
        cam_up = xmat[:, 1]
        cam_forward = -xmat[:, 2]
        # pixel_to_camera gives (x=right, y=down, z=forward)
        world_pt = pos + x_cam * cam_right + (-y_cam) * cam_up + z_cam * cam_forward
        return (float(world_pt[0]), float(world_pt[1]), float(world_pt[2]))

    # Fallback: approximate from robot heading + mount offset
    cos_h = math.cos(robot_heading)
    sin_h = math.sin(robot_heading)
    mount_forward = 0.3
    mount_up = 0.05

    body_forward = z_cam
    body_right = x_cam
    body_up = -y_cam

    world_x = robot_x + cos_h * (mount_forward + body_forward) - sin_h * body_right
    world_y = robot_y + sin_h * (mount_forward + body_forward) + cos_h * body_right
    world_z = robot_z + mount_up + body_up

    return (world_x, world_y, world_z)


def depth_to_world(
    depth_frame: np.ndarray,
    u: float,
    v: float,
    intrinsics: CameraIntrinsics,
    robot_x: float,
    robot_y: float,
    robot_z: float,
    robot_heading: float,
) -> tuple[float, float, float] | None:
    """Project pixel (u, v) from depth frame to world coordinates.

    Returns (world_x, world_y, world_z) or None if depth is invalid.
    """
    h, w = depth_frame.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if ui < 0 or ui >= w or vi < 0 or vi >= h:
        return None

    d = float(depth_frame[vi, ui])
    if d <= 0.0 or d > 10.0:
        return None

    x_cam, y_cam, z_cam = pixel_to_camera(u, v, d, intrinsics)
    return camera_to_world(
        x_cam, y_cam, z_cam,
        robot_x, robot_y, robot_z, robot_heading,
    )


def center_depth(depth_frame: np.ndarray, region_frac: float = 0.2) -> float:
    """Sample median depth in the center region of the frame.

    Args:
        depth_frame: (H, W) float32 depth in metres.
        region_frac: Fraction of width/height for the center crop (0.2 = 20%).

    Returns:
        Median depth in metres, or 0.0 if no valid depth pixels.
    """
    h, w = depth_frame.shape[:2]
    rw = int(w * region_frac / 2)
    rh = int(h * region_frac / 2)
    cx, cy = w // 2, h // 2

    crop = depth_frame[
        max(0, cy - rh): min(h, cy + rh),
        max(0, cx - rw): min(w, cx + rw),
    ]

    valid = crop[(crop > 0.1) & (crop < 10.0)]
    if len(valid) == 0:
        return 0.0
    return float(np.median(valid))


def project_center_to_world(
    depth_frame: np.ndarray,
    intrinsics: CameraIntrinsics,
    robot_x: float,
    robot_y: float,
    robot_z: float,
    robot_heading: float,
) -> tuple[float, float, float] | None:
    """Project the center-of-view depth to a world point.

    Uses median depth in the center 20% of the frame for robustness.
    This is the "what is the robot looking at" world position.

    Returns (world_x, world_y, world_z) or None if no valid depth.
    """
    d = center_depth(depth_frame)
    if d <= 0.0:
        return None

    u = intrinsics.cx
    v = intrinsics.cy
    x_cam, y_cam, z_cam = pixel_to_camera(u, v, d, intrinsics)
    return camera_to_world(
        x_cam, y_cam, z_cam,
        robot_x, robot_y, robot_z, robot_heading,
    )
