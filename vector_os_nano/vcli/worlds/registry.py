# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""World resolution — the kernel selects a world from the connected agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vector_os_nano.vcli.worlds.base import World


def resolve_world(agent: Any = None) -> "World":
    """The robot world when a robot *agent* is connected, else the default dev world."""
    from vector_os_nano.vcli.worlds.dev import DevWorld
    from vector_os_nano.vcli.worlds.robot import RobotWorld

    return RobotWorld() if agent is not None else DevWorld()
