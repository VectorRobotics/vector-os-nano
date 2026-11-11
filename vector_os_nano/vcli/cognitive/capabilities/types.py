# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Capability protocol — the routable-capability contract (Phase C).

A *capability* is anything a sub-goal can be routed to with a typed
``(input -> output)`` contract and measured stats: a chat LLM, a specialized
detector/segmenter, a planner, a VLA policy, a classical skill, an atomic
action. The kernel routes a sub-goal to the best-fitting capability by measured
success rate (the existing StrategyStats bandit) and verifies the result with
the same deterministic predicate — the capability never self-certifies.

Phase C.1 establishes the seam with a single read-only capability (chat). The
``side_effecting`` flag is part of the contract now so C.3 can gate VLA/tool
capabilities through ``PermissionContext`` without changing this protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class CapabilityResult:
    """Outcome of a capability invocation.

    ``success`` reports only that the capability *ran* and produced output; the
    step's deterministic ``verify`` predicate, evaluated separately by the native
    loop's verifier, decides whether the step actually succeeded.
    """

    success: bool
    output: dict[str, Any] = field(default_factory=dict)  # typed per capability
    error: str = ""
    cost_usd: float = 0.0
    latency_sec: float = 0.0


@runtime_checkable
class Capability(Protocol):
    """Structural contract every routable capability satisfies."""

    name: str            # registry key + stats strategy_name (e.g. "chat", "detect")
    kind: str            # "chat" | "detector" | "planner" | "vla" | "skill" | "atomic"
    side_effecting: bool  # True -> invocation must route through a permission gate
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]

    def estimate(self, payload: dict[str, Any]) -> tuple[float, float]:
        """Return a cheap ``(cost_usd, latency_sec)`` estimate (no I/O).

        Used as a cold-start prior / tiebreak among unmeasured capabilities.
        """
        ...

    def invoke(self, payload: dict[str, Any], context: Any) -> CapabilityResult:
        """Run the capability on *payload* (validated against ``input_schema``)."""
        ...
