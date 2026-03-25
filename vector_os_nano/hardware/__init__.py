"""Hardware abstraction layer.

Contains ArmProtocol, GripperProtocol, and BaseProtocol definitions plus
concrete implementations (SO-101, MuJoCo sim, etc.).
"""
from vector_os_nano.hardware.base import BaseProtocol

__all__ = ["BaseProtocol"]
