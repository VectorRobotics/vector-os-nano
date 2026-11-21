# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The public contracts of Vector OS Nano.

Everything a plug-in implements or consumes is declared here and nowhere else:

* :mod:`.physics` — ``PhysicsWorld`` / ``SimBackend`` (the only thing a simulator
  implements), ``CompiledScene`` (the canonical model), ``RGBDFrame``, ``GraspSpec``.
* :mod:`.devices` — ``BaseProtocol`` / ``ArmProtocol`` / ``GripperProtocol`` /
  ``CameraProtocol`` and ``CameraFrame``: what skills and perception may touch of a body.
* :mod:`.policy` — ``ControlPolicy`` / ``PolicyObs`` (a locomotion controller).

This package imports the standard library, numpy and ``vector_os_nano.schemas``
only (tests/unit/test_layering.py). The binding prose is
docs/plan-release-1.0-interfaces.md.
"""
from vector_os_nano.interfaces.devices import (
    ArmProtocol,
    BaseProtocol,
    CameraFrame,
    CameraProtocol,
    GripperProtocol,
    FingerLoad,
    FingerLoadProtocol,
)
from vector_os_nano.interfaces.physics import (
    CONTACT_DISTANCE_M,
    ContactLoad,
    CompiledScene,
    GraspSpec,
    PhysicsWorld,
    RGBDFrame,
    SimBackend,
)
from vector_os_nano.interfaces.policy import ControlPolicy, PolicyObs

__all__ = [
    "CONTACT_DISTANCE_M",
    "ArmProtocol",
    "BaseProtocol",
    "CameraFrame",
    "CameraProtocol",
    "CompiledScene",
    "ContactLoad",
    "ControlPolicy",
    "GraspSpec",
    "GripperProtocol",
    "FingerLoad",
    "FingerLoadProtocol",
    "PhysicsWorld",
    "PolicyObs",
    "RGBDFrame",
    "SimBackend",
]
