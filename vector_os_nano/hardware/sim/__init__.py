# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""vector_os_nano.hardware.sim — the MuJoCo simulation backend.

Drop-in replacements for the hardware base and the SO-101's object source, so
skills and the Agent run in an in-process simulation without hardware.
"""
from vector_os_nano.hardware.sim.mujoco_perception import MuJoCoPerception

__all__ = [
    "MuJoCoPerception",
]
