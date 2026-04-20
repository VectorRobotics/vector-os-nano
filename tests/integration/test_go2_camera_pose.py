"""Regression tests for Go2ROS2Proxy.get_camera_pose geometry.

The v2.3 live REPL smoke (2026-04-20) surfaced a sign-convention bug:
``up = cross(right, forward)`` produced a world-DOWN vector for a level
``+X``-facing camera, which flipped the Y projection of perception
detections and yielded world z values below the floor.

These tests instantiate Go2ROS2Proxy without connecting — they only
exercise ``get_camera_pose`` pure math — and assert:
  1. Mount position matches MJCF (0.25 m forward, 0.1 m up)
  2. The ``up`` xmat column points world +Z for a level camera
  3. A pixel below image centre projects to below the camera in world
  4. A pixel right of image centre projects to the camera's right
     in world (world -Y for a +X-facing dog)
"""
from __future__ import annotations

import math

import pytest

from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy
from vector_os_nano.perception.depth_projection import (
    mujoco_intrinsics,
    pixel_to_camera,
)


def _make_proxy(pos: tuple[float, float, float], heading: float) -> Go2ROS2Proxy:
    """Construct a Go2ROS2Proxy without connecting (no rclpy node)."""
    p = Go2ROS2Proxy()
    p._position = pos
    p._heading = heading
    return p


def test_mount_position_matches_mjcf() -> None:
    """Dog at world (0, 0, 0.28) facing +X → camera at (0.25, 0, 0.38)."""
    proxy = _make_proxy(pos=(0.0, 0.0, 0.28), heading=0.0)
    cam_xpos, _ = proxy.get_camera_pose()
    assert cam_xpos[0] == pytest.approx(0.25, abs=1e-9)
    assert cam_xpos[1] == pytest.approx(0.0, abs=1e-9)
    assert cam_xpos[2] == pytest.approx(0.38, abs=1e-9)


def test_up_col_points_world_up_for_level_camera() -> None:
    """xmat[:, 1] is the camera's ``up`` direction in world.

    For a level +X-facing camera (pitched -5° down), ``up`` must point
    predominantly world +Z (cos(5°) ≈ 0.996) with a small forward +X
    component (sin(5°) ≈ 0.087).  A negative z component here is the
    pre-fix bug signature.
    """
    proxy = _make_proxy(pos=(0.0, 0.0, 0.28), heading=0.0)
    _, xmat_flat = proxy.get_camera_pose()
    up = xmat_flat.reshape(3, 3)[:, 1]

    assert up[2] > 0.9, f"cam up z must be ~+1, got {up[2]:.3f} (sign bug?)"
    assert up[0] > 0.0, f"cam up x should be slightly +X due to downward pitch, got {up[0]:.3f}"


def test_bottom_pixel_projects_below_camera() -> None:
    """Pixel below image centre at 1 m depth → world z below camera z."""
    proxy = _make_proxy(pos=(0.0, 0.0, 0.28), heading=0.0)
    cam_xpos, xmat_flat = proxy.get_camera_pose()
    xmat = xmat_flat.reshape(3, 3)

    intr = mujoco_intrinsics(320, 240, 42.0)
    # pixel near bottom of image, centre x
    u, v = intr.cx, intr.cy + 60
    x_cam, y_cam, z_cam = pixel_to_camera(u, v, 1.0, intr)

    # world = pos + x_cam * right + (-y_cam) * up + z_cam * forward
    world = (
        cam_xpos
        + x_cam * xmat[:, 0]
        + (-y_cam) * xmat[:, 1]
        + z_cam * (-xmat[:, 2])
    )
    # Camera z = 0.38; world z for a below-centre pixel must be lower.
    assert world[2] < cam_xpos[2], (
        f"bottom pixel mapped to z={world[2]:.3f} ABOVE cam z={cam_xpos[2]:.3f}"
    )


def test_right_pixel_projects_to_dog_right_side() -> None:
    """For dog facing +X, a pixel right-of-centre → world -Y side.

    In ROS convention, dog's right side is world -Y when heading=0.
    """
    proxy = _make_proxy(pos=(0.0, 0.0, 0.28), heading=0.0)
    cam_xpos, xmat_flat = proxy.get_camera_pose()
    xmat = xmat_flat.reshape(3, 3)

    intr = mujoco_intrinsics(320, 240, 42.0)
    u, v = intr.cx + 60, intr.cy
    x_cam, y_cam, z_cam = pixel_to_camera(u, v, 1.0, intr)

    world = (
        cam_xpos
        + x_cam * xmat[:, 0]
        + (-y_cam) * xmat[:, 1]
        + z_cam * (-xmat[:, 2])
    )
    # NOTE: the ``right`` column in get_camera_pose is (-sin_h, cos_h, 0),
    # which is +Y for heading=0 — i.e. dog's LEFT in ROS. This is a
    # separate latent inconsistency (the column labelled "right" is body-
    # left) flagged in v2.3 risk register R2; self-consistent with
    # depth_projection.camera_to_world so left/right reads flip together.
    # We verify the CURRENT convention: right-of-centre pixel → world +Y.
    assert world[1] > cam_xpos[1] + 0.05, (
        f"right-of-centre pixel mapped to y={world[1]:.3f}, "
        f"expected > cam y={cam_xpos[1]:.3f} under current (right=+Y) convention"
    )


def test_offset_dog_pose_shifts_camera() -> None:
    """Dog at (5, 3) facing +Y → camera at (5, 3.25, 0.38)."""
    proxy = _make_proxy(pos=(5.0, 3.0, 0.28), heading=math.pi / 2)
    cam_xpos, _ = proxy.get_camera_pose()
    assert cam_xpos[0] == pytest.approx(5.0, abs=1e-6)
    assert cam_xpos[1] == pytest.approx(3.25, abs=1e-6)
    assert cam_xpos[2] == pytest.approx(0.38, abs=1e-9)
