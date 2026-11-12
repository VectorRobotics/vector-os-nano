# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""What the actor (skills, perception) may touch of a body: arm, gripper, camera.

One generic driver per morphology class implements each protocol (``sim/drivers``);
a body is a manifest, never a class. No method returns the pose of anything but the
robot itself, and no frame carries segmentation: ground truth stays on the world side.
Positions are world-frame metres, joints radians in ``joint_names`` order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

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
           current_joints: list[float] | None = None) -> list[float] | None:
        """Joints that put the end effector at *target_xyz* (any orientation), or None."""

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


@runtime_checkable
class CameraProtocol(Protocol):
    def capture(self, width: int, height: int) -> CameraFrame: ...
