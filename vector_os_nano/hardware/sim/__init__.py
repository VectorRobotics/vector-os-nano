# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""vector_os_nano.hardware.sim — the MuJoCo simulation backend.

Drop-in replacements for the hardware arm, gripper, perception and base, so
skills and the Agent run in an in-process simulation without hardware.
"""
from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm
from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2
from vector_os_nano.hardware.sim.mujoco_gripper import MuJoCoGripper
from vector_os_nano.hardware.sim.mujoco_perception import MuJoCoPerception

__all__ = [
    "MuJoCoArm",
    "MuJoCoGo2",
    "MuJoCoGripper",
    "MuJoCoPerception",
]
