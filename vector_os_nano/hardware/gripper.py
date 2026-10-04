# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""GripperProtocol — abstract interface for any gripper.

All gripper implementations (SO101Gripper, MuJoCoGripper, etc.) must
satisfy this Protocol.

No hardware imports. No ROS2 imports.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class GripperProtocol(Protocol):
    """Abstract interface for any end-effector gripper.

    Implementations must be thread-safe (gripper may be controlled
    concurrently with arm motion in future skill versions).
    """

    def open(self) -> bool:
        """Open the gripper fully.

        Returns:
            True if the open command was sent successfully.
        """
        ...

    def close(self) -> bool:
        """Close the gripper.

        Returns:
            True if the close command was sent and an object was grasped,
            or True if close completed without object (implementation-dependent).
            Returns False on communication failure.
        """
        ...

    def is_holding(self) -> bool:
        """Return True if the gripper is closed on an object.

        Uses encoder position, current sensing, or force feedback depending
        on hardware capability. Falls back to position heuristic if no
        force sensor is available.

        Returns:
            True if an object is detected in the gripper.
        """
        ...

    def get_position(self) -> float:
        """Return normalized gripper position.

        Returns:
            0.0 = fully closed, 1.0 = fully open.
        """
        ...

    def get_force(self) -> float | None:
        """Return current grip force in Newtons.

        Returns:
            Force in Newtons, or None if force sensing is not available.
        """
        ...
