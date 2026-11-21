# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""MuJoCo backend: the canonical compiler's own engine runs the scene as compiled."""
from __future__ import annotations

from vector_os_nano.schemas.robot import EmbodimentConfig
from vector_os_nano.schemas.world import WorldManifest

from vector_os_nano.interfaces.physics import CompiledScene, PhysicsWorld


class MuJoCoBackend:
    name = "mujoco"

    def supports(self, robot: EmbodimentConfig, world: WorldManifest) -> str | None:
        if world.scene is None or not world.scene.is_file():
            return f"{world.path}.scene: world scene file is missing"
        return None  # every body and world is authored for MuJoCo first

    def load(self, scene: CompiledScene, *, gui: bool = False) -> PhysicsWorld:
        from vector_os_nano.sim.backends.mujoco.world import MuJoCoWorld

        return MuJoCoWorld(scene, gui=gui)


BACKEND = MuJoCoBackend()
