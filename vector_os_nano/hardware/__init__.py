# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Hardware abstraction layer.

Contains the BaseProtocol definition plus
concrete implementations (SO-101, MuJoCo sim, etc.).
"""
from vector_os_nano.hardware.base import BaseProtocol

__all__ = ["BaseProtocol"]
