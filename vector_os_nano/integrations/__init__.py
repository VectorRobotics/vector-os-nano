# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""External system integrations.

Each subpackage adapts a sibling project's ROS2 / IPC contract to
Vector OS Nano's internal types (``world_model``, ``scene_graph``,
``skill`` context, etc.) without copying the upstream sources.

Subpackages here MUST NOT import upstream Python sources at module-load
time. ROS2 message imports are deferred so that ``import
vector_os_nano.integrations.<name>`` works in environments where the
upstream workspace is not sourced — tests can substitute typed
shadow dataclasses defined alongside each adapter.
"""
