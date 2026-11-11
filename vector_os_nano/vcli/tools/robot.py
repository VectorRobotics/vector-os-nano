# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Robot tools for the Vector CLI agentic harness.

RobotStatusTool — read-only snapshot of connected hardware (arm, gripper, base, perception).
It is safe to call concurrently; it never mutates state. (Objects: scene_graph_query.)
"""
from __future__ import annotations

from typing import Any

from vector_os_nano.vcli.tools.base import ToolContext, ToolResult, live_agent, tool


# ---------------------------------------------------------------------------
# RobotStatusTool
# ---------------------------------------------------------------------------


@tool(
    name="robot_status",
    description=(
        "Get current hardware status: arm, gripper, base, and perception pipeline"
    ),
    read_only=True,
    permission="allow",
)
class RobotStatusTool:
    """Return a snapshot of all connected hardware components."""

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def is_concurrency_safe(self, params: dict[str, Any]) -> bool:
        return True

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        agent = live_agent(context)
        if agent is None:
            return ToolResult(content="No agent available", is_error=True)

        lines: list[str] = []

        # Arm
        arm = getattr(agent, "_arm", None)
        if arm is not None:
            arm_name: str = getattr(arm, "name", type(arm).__name__)
            try:
                joints = arm.get_joint_positions()
                lines.append(
                    f"Arm: {arm_name}, joints={[round(j, 3) for j in joints]}"
                )
            except Exception:
                lines.append(f"Arm: {arm_name} (connected, joints unavailable)")
        else:
            lines.append("Arm: not connected")

        # Gripper
        gripper = getattr(agent, "_gripper", None)
        if gripper is not None:
            gripper_name: str = getattr(gripper, "name", type(gripper).__name__)
            try:
                pos = gripper.get_position()
                holding = gripper.is_holding()
                lines.append(
                    f"Gripper: {gripper_name}, pos={pos:.3f}, holding={holding}"
                )
            except Exception:
                lines.append(f"Gripper: {gripper_name} (connected)")
        else:
            lines.append("Gripper: not connected")

        # Base
        base = getattr(agent, "_base", None)
        if base is not None:
            base_name: str = getattr(base, "name", type(base).__name__)
            try:
                position = base.get_position()
                heading = base.get_heading()
                lines.append(
                    f"Base: {base_name}, pos={[round(p, 3) for p in position]}, "
                    f"heading={round(heading, 3)}"
                )
            except Exception:
                lines.append(f"Base: {base_name} (connected)")
        else:
            lines.append("Base: not connected")

        # Perception
        perception = getattr(agent, "_perception", None)
        lines.append(f"Perception: {'active' if perception is not None else 'not available'}")

        return ToolResult(content="\n".join(lines))
