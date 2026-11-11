# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""PyBullet backend: the compiled MuJoCo model rebuilt as Bullet multibodies.

``pybullet`` is an optional extra (``.[pybullet]``) and is imported only inside
this package, and only when a world is loaded.
"""
from __future__ import annotations

import importlib.util
from typing import Any

from vector_os_nano.interfaces.physics import CompiledScene, PhysicsWorld

INSTALL_HINT = "pybullet is not installed: uv pip install -e '.[pybullet]' (builds from source; needs g++ and python3-dev)"


class PyBulletBackend:
    name = "pybullet"

    def supports(self, robot: Any, world: Any) -> str | None:
        if importlib.util.find_spec("pybullet") is None:
            return INSTALL_HINT
        # ponytail: today's manifests name a TorchScript policy by its file (g1: motion.pt);
        # the schema-1 robot.yaml (C1) names it `torchscript`.
        ref = str(getattr(getattr(robot, "policy", None), "ref", "") or "")
        if ref == "torchscript" or ref.endswith(".pt"):
            return "a TorchScript (RL) policy is not run on PyBullet in 1.0: it was trained for MuJoCo"
        return None

    def load(self, scene: CompiledScene, *, gui: bool = False) -> PhysicsWorld:
        if importlib.util.find_spec("pybullet") is None:
            raise ImportError(INSTALL_HINT)
        from vector_os_nano.sim.backends.pybullet.world import PyBulletWorld

        return PyBulletWorld(scene, gui=gui)


BACKEND = PyBulletBackend()
