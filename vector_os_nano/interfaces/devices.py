# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""What the actor (skills, perception) may touch of a body: base, arm, gripper, camera.

One generic driver per morphology class implements each protocol (``sim/drivers``);
a body is a manifest, never a class. No method returns the pose of anything but the
robot itself, and no frame carries segmentation: ground truth stays on the world side.
Positions are world-frame metres, joints radians in ``joint_names`` order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    """One camera instant as the actor gets it: an ``RGBDFrame`` without segmentation.

    ``rgb`` (H, W, 3) uint8; ``depth`` (H, W) float32 metres along the optical axis,
    0 = invalid (None for a colour-only camera); ``self_mask`` (H, W) bool marks the
    robot's own pixels (computed on the world side; only the mask crosses). ``K``,
    ``cam_pos``, ``cam_mat``, ``sim_time`` and ``camera`` as in ``RGBDFrame``.
    """

    rgb: np.ndarray
    depth: np.ndarray | None
    self_mask: np.ndarray | None
    K: np.ndarray
    cam_pos: np.ndarray
    cam_mat: np.ndarray
    sim_time: float
    camera: str


@runtime_checkable
class ArmProtocol(Protocol):
    """A position-controlled arm. Motions are timed in sim seconds and return whether
    the arm settled at the target; ``stop`` (and an abort) preempt them."""

    @property
    def joint_names(self) -> list[str]: ...

    @property
    def dof(self) -> int: ...

    def get_joint_positions(self) -> list[float]: ...

    def move_joints(self, positions: list[float], duration: float = 3.0) -> bool:
        """Interpolate to *positions*; True iff every joint settled within tolerance."""

    def move_cartesian(self, target_xyz: tuple[float, float, float], duration: float = 3.0) -> bool:
        """``ik_top_down`` + ``move_joints``; False when unreachable."""

    def fk(self, joint_positions: list[float]) -> tuple[list[float], list[list[float]]]:
        """End-effector (position, 3x3 rotation) in the world for *joint_positions*."""

    def ik(self, target_xyz: tuple[float, float, float],
           current_joints: list[float] | None = None, *,
           target_rotation: list[list[float]] | None = None) -> list[float] | None:
        """Joints reaching a world position and optional 3x3 tool rotation, or None.
        A driver without oriented IK must reject that request explicitly."""

    def ik_top_down(self, target_xyz: tuple[float, float, float],
                    current_joints: list[float] | None = None) -> list[float] | None:
        """Joints that put the end effector at *target_xyz* pointing straight down, or None."""

    def home(self, duration: float = 3.0) -> bool: ...

    def stop(self) -> None:
        """Preempt any motion and hold the current joint positions."""

    def ctrl_motion(self) -> float:
        """Cumulative commanded joint displacement of the actor (causation grading)."""


@runtime_checkable
class GripperProtocol(Protocol):
    """A jaw. Whether it holds something is the world's decision, never the driver's."""

    def open(self) -> bool: ...

    def close(self) -> bool: ...

    def is_holding(self, name: str | None = None) -> bool: ...

    def get_position(self) -> float:
        """0.0 closed, 1.0 fully open."""


@dataclass(frozen=True)
class FingerLoad:
    """Summed force ON the fingertips in world axes (N), at one sensor instant.

    Contains no external body identity, position or support state. A simulated
    ideal contact sensor is not a calibrated physical force sensor.
    """
    sim_time: float
    force: tuple[float, float, float]


@runtime_checkable
class FingerLoadProtocol(Protocol):
    def get_finger_load(self) -> FingerLoad | None:
        """Current measured fingertip force; None when no sensor is declared."""


@runtime_checkable
class CameraProtocol(Protocol):
    def capture(self, width: int, height: int) -> CameraFrame: ...


@runtime_checkable
class BaseProtocol(Protocol):
    """A mobile base driven by body-frame velocity commands. Velocities are vx, vy
    (m/s, body frame, +y left) and vyaw (rad/s, counter-clockwise); timed calls count
    SIM seconds. Postures (``stand`` and the manifest's others) are held until a non-zero
    velocity command."""

    @property
    def limits(self) -> tuple[float, float, float]:
        """The largest |vx|, |vy|, |vyaw| a command may ask; larger ones are clipped."""

    @property
    def holonomic(self) -> bool:
        """True when it can strafe (vy)."""

    def connect(self) -> None: ...

    def disconnect(self) -> None:
        """Idempotent."""

    def stop(self) -> None:
        """Zero the command now, from any thread; idempotent and never raises."""

    def walk(self, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0, duration: float = 1.0) -> bool:
        """Hold the command for *duration* sim seconds, then stop; True iff it ran its full
        time and the body was up (its root above the fall height) at the start and the end."""

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        """Set the command now (non-blocking); it holds until replaced."""

    def set_posture(self, name: str, duration: float = 2.0) -> bool:
        """Move the joints to the posture *name* over *duration* sim seconds and hold it;
        True iff it was reached. An unknown name is a ``KeyError`` naming the valid ones."""

    def navigate_to(self, x: float, y: float, tol: float | None = None, speed: float | None = None,
                    timeout: float | None = None, cancel: Any = None, on_progress: Any = None) -> Any:
        """Walk to world (*x*, *y*) around the scene's obstacles with the body's own planner.
        Returns a mapping whose truth is arrival and whose ``reason`` is one of arrived,
        unreachable, timeout, aborted, stalled, fell; ``timeout`` is sim seconds; *cancel*
        (``is_set()``) or a ``stop`` ends it 'aborted'; ``on_progress(remaining_m,
        elapsed_s)`` is called while it walks."""

    def get_position(self) -> list[float]:
        """[x, y, z] of the base in the world (m)."""

    def get_heading(self) -> float:
        """Yaw in the world (rad, counter-clockwise from +x)."""

    def get_velocity(self) -> list[float]:
        """[vx, vy, vz] of the base in the WORLD frame (m/s)."""

    def get_odometry(self) -> Any:
        """``core.types.Odometry`` of one instant."""

    def get_lidar_scan(self) -> Any:
        """The latest ``core.types.LaserScan``, or None without a lidar."""

    def get_sim_time(self) -> float: ...

    def wait_sim(self, seconds: float, cancel: Any = None) -> bool:
        """Block until *seconds* of sim time pass; False when *cancel* (anything with
        ``is_set()``) fires or the clock stops advancing."""

    def cmd_motion(self) -> float:
        """Cumulative commanded |vx| + |vy| + |vyaw| of the actor (causation grading)."""
