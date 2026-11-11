# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The canonical model (charter section 2): MuJoCo compiles every scene once.

* :mod:`.compiled` — ``compiled_scene(xml)`` -> ``interfaces.CompiledScene``.
* :mod:`.reader` — read-only views of the compiled model, for backend authors.
* :mod:`.canonical` — body / site poses and shape AABBs from a state.

``mujoco`` is imported here and under ``sim/backends/mujoco`` only.
"""
from vector_os_nano.sim.scene.compiled import compiled_scene

__all__ = ["compiled_scene"]
