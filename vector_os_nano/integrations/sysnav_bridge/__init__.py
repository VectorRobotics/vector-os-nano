# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""SysNav bridge — consumes the SysNav scene-graph topic contract.

SysNav (https://github.com/zwandering/SysNav, PolyForm-Noncommercial)
runs as a sibling ROS2 workspace. This package adapts its published
``tare_planner/ObjectNodeList`` and ``tare_planner/RoomNodeList``
messages onto Vector OS Nano's internal ``WorldModel`` / ``SceneGraph``
without copying or redistributing any SysNav source code.

See ``docs/sysnav_integration.md`` for the full bringup procedure,
field mapping, and license boundary discussion.

Public surface:

* :class:`ObjectNodeShadow` / :class:`RoomNodeShadow` — typed dataclass
  shadows of the SysNav messages, used in tests where the upstream
  ``tare_planner.msg`` package may not be importable.
* :class:`SysnavBridge` — the live adapter (deferred ROS2 imports).
* :func:`object_node_to_state` — pure-function field mapping used by
  both the live adapter and tests.
"""
from __future__ import annotations

from vector_os_nano.integrations.sysnav_bridge.topic_interfaces import (
    ObjectNodeShadow,
    ObjectNodeListShadow,
    RoomNodeShadow,
    object_node_to_state,
)

__all__ = [
    "ObjectNodeShadow",
    "ObjectNodeListShadow",
    "RoomNodeShadow",
    "object_node_to_state",
]
