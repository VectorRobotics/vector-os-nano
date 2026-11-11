# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""MuJoCo backend: the canonical compiler's own engine runs the scene as compiled."""
from __future__ import annotations

from typing import Any

from vector_os_nano.interfaces.physics import CompiledScene, PhysicsWorld


class MuJoCoBackend:
    name = "mujoco"

    def supports(self, robot: Any, world: Any) -> str | None:
        return None  # every body and world is authored for MuJoCo first

    def load(self, scene: CompiledScene, *, gui: bool = False) -> PhysicsWorld:
        from vector_os_nano.sim.backends.mujoco.world import MuJoCoWorld

        return MuJoCoWorld(scene, gui=gui)


BACKEND = MuJoCoBackend()
