# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""World plugin protocol — the kernel/world seam.

The agent kernel (VectorEngine + the native loop + general tools + backends +
session + permissions) is domain-general. A *world* adapts the kernel to a domain
by contributing:

1. a persona (role prompt + tool instructions),
2. a verify namespace (the callables GoalVerifier evaluates predicates against;
   the ``@oracle``-tagged ones are evidence),
3. routable capabilities (detectors, planners, VLA policies).

The default ``DevWorld`` ships with the kernel; ``RobotWorld`` is the robot
adapter. The kernel never imports a concrete world at module load.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class World(Protocol):
    """A domain adapter for the agent kernel."""

    name: str

    def is_robot(self) -> bool:
        """True if this world drives physical/simulated robot hardware."""
        ...

    def persona_blocks(self, agent: Any = None) -> tuple[str, str]:
        """Return (role_prompt, tool_instructions) persona text.

        With the connected *agent*, a world may compose its task policy from that
        body's manifests (the robot world does); the text stays static per body.
        """
        ...

    def build_verify_namespace(self, agent: Any) -> dict[str, Any]:
        """Return the verify callables for GoalVerifier.

        Side-effect-free predicates a verify expression may call, merged into the
        engine's verifier namespace. The first paragraph of each oracle's docstring
        is what the model reads in the verify tool's description.
        """
        ...

    def register_capabilities(self, registry: Any, agent: Any, backend: Any) -> None:
        """Register this world's routable capabilities.

        A *capability* is a model or stack with a typed ``(input -> output)``
        contract (a detector, a planner, a VLA policy); the native loop offers each
        as a tool. Must not import heavy deps at module load.
        """
        ...
