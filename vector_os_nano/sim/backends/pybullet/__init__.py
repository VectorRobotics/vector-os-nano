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
# the policies measured on PyBullet (charter section 7); anything else is a no, with the reason
CERTIFIED = ("position_actuators", "sinusoidal", "torchscript")


class PyBulletBackend:
    name = "pybullet"

    def supports(self, robot: Any, world: Any) -> str | None:
        if importlib.util.find_spec("pybullet") is None:
            return INSTALL_HINT
        # Fixed arms use the scene's position actuators; a base names its policy.
        base = getattr(robot, "base", None)
        ref = base.policy.on(self.name)[0] if base is not None else "position_actuators"
        if ref == "convex_mpc":
            return "convex_mpc falls on PyBullet (measured: within 2 s); use the sinusoidal gait there"
        if ref not in CERTIFIED:
            return f"policy {ref!r} is not certified on PyBullet in 1.0; certified: {list(CERTIFIED)}"
        return None

    def load(self, scene: CompiledScene, *, gui: bool = False) -> PhysicsWorld:
        if importlib.util.find_spec("pybullet") is None:
            raise ImportError(INSTALL_HINT)
        from vector_os_nano.sim.backends.pybullet.world import PyBulletWorld

        return PyBulletWorld(scene, gui=gui)


BACKEND = PyBulletBackend()
