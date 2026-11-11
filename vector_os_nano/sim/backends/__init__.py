# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Simulator backends: one row each, imported lazily (an engine that is not
installed costs nothing until it is asked for).

A backend module exposes ``BACKEND``, an ``interfaces.SimBackend``; its world
implements ``interfaces.PhysicsWorld`` and passes ``tests/conformance``.
"""
from __future__ import annotations

import importlib

from vector_os_nano.interfaces.physics import SimBackend

BACKENDS: dict[str, str] = {
    "mujoco": "vector_os_nano.sim.backends.mujoco",
    "pybullet": "vector_os_nano.sim.backends.pybullet",
}


def get(name: str) -> SimBackend:
    """The backend row *name*; a ``KeyError`` names the valid ones."""
    try:
        module = BACKENDS[name]
    except KeyError:
        raise KeyError(f"no simulator backend {name!r}; backends: {sorted(BACKENDS)}") from None
    return importlib.import_module(module).BACKEND
