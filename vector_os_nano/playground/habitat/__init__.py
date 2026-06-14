# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Habitat third-world backend (ADR-009) — playground track.

The repo venv never imports habitat: ``server.py`` runs under the pinned
conda interpreter as a subprocess; ``HabitatBridge`` is the socket client;
``HabitatBase`` is the kinematic ``BaseProtocol`` over it. Lazy imports
only — this package must stay importable with zero heavy deps.
"""
from __future__ import annotations

from vector_os_nano.playground.habitat.base import HabitatBase
from vector_os_nano.playground.habitat.bridge import (
    HabitatBridge,
    HabitatBridgeError,
    habitat_available,
    habitat_python,
)

__all__ = [
    "HabitatBase",
    "HabitatBridge",
    "HabitatBridgeError",
    "habitat_available",
    "habitat_python",
]
