# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""CapabilityRegistry — per-run registry of routable capabilities.

Worlds populate it via ``World.register_capabilities`` (the robot world registers
its detector); the engine binds it and the native loop offers each capability as
a tool.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CapabilityRegistry:
    """Collects Capability instances by name."""

    def __init__(self) -> None:
        self._caps: dict[str, Any] = {}

    def register(self, capability: Any) -> None:
        """Register *capability* (keyed by ``capability.name``); replaces by name."""
        name = getattr(capability, "name", "")
        if not name:
            logger.warning("CapabilityRegistry: ignoring capability with no name")
            return
        self._caps[name] = capability

    def get(self, name: str) -> Any | None:
        """Return the capability registered under *name*, or None."""
        return self._caps.get(name)

    def names(self) -> frozenset[str]:
        """Return the set of registered capability names."""
        return frozenset(self._caps.keys())

    def __contains__(self, name: object) -> bool:
        return name in self._caps

    def __len__(self) -> int:
        return len(self._caps)
