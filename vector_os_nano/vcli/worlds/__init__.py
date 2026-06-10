# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""World plugins for the agent kernel.

A *world* adapts the domain-general kernel to a domain (dev/code, robot, ...).
The default ``DevWorld`` ships with the kernel; ``RobotWorld`` is selected when
a robot agent is connected.
"""

from __future__ import annotations

from vector_os_nano.vcli.worlds.base import DecomposeVocab, World
from vector_os_nano.vcli.worlds.blueprint import (
    BlueprintWorld,
    WorldBlueprint,
    blueprint_of,
)
from vector_os_nano.vcli.worlds.dev import DevWorld, dev_verify_namespace
from vector_os_nano.vcli.worlds.registry import (
    WorldRegistry,
    get_world_registry,
    resolve_world,
    resolve_world_named,
)
from vector_os_nano.vcli.worlds.robot import RobotWorld

__all__ = [
    "World",
    "DecomposeVocab",
    "WorldBlueprint",
    "BlueprintWorld",
    "blueprint_of",
    "DevWorld",
    "RobotWorld",
    "dev_verify_namespace",
    "resolve_world",
    "resolve_world_named",
    "WorldRegistry",
    "get_world_registry",
]
