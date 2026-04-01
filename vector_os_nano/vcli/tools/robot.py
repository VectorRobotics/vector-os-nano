"""Robot tools for the Vector CLI agentic harness.

WorldQueryTool  — read-only view into the agent's world model (objects + robot pose).
RobotStatusTool — read-only snapshot of connected hardware (arm, gripper, base, perception).

Both tools are safe to call concurrently; they never mutate state.
"""
from __future__ import annotations

from typing import Any

from vector_os_nano.vcli.tools.base import ToolContext, ToolResult, tool


# ---------------------------------------------------------------------------
# WorldQueryTool
# ---------------------------------------------------------------------------


@tool(
    name="world_query",
    description="Query the robot's world model for detected objects and robot pose",
    read_only=True,
    permission="allow",
)
class WorldQueryTool:
    """Return the current world model: robot state + list of detected objects."""

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional label filter (case-insensitive substring match)",
                "default": "",
            },
        },
        "required": [],
    }

    def is_concurrency_safe(self, params: dict[str, Any]) -> bool:
        return True

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        agent = context.agent
        if agent is None:
            return ToolResult(content="No agent available", is_error=True)

        wm = agent._world_model
        lines: list[str] = []

        # Robot state — WorldModel exposes get_robot(); mocks may use get_robot_state().
        robot = None
        if hasattr(wm, "get_robot"):
            robot = wm.get_robot()
        elif hasattr(wm, "get_robot_state"):
            robot = wm.get_robot_state()

        if robot is not None:
            pos = getattr(robot, "position_xy", None)
            heading = getattr(robot, "heading", None)
            ee = getattr(robot, "ee_position", None)
            parts: list[str] = []
            if pos is not None:
                parts.append(f"pos={[round(v, 3) for v in pos]}")
            if heading is not None:
                parts.append(f"heading={round(heading, 3)}")
            if ee is not None:
                parts.append(f"ee={[round(v, 3) for v in ee]}")
            if parts:
                lines.append("Robot: " + ", ".join(parts))

        # Objects
        objects: list[Any] = []
        if hasattr(wm, "get_objects"):
            objects = wm.get_objects()

        query: str = params.get("query", "").lower()
        for obj in objects:
            label: str = getattr(obj, "label", str(obj))
            if query and query not in label.lower():
                continue
            x = getattr(obj, "x", "?")
            y = getattr(obj, "y", "?")
            z = getattr(obj, "z", "?")
            lines.append(
                f"  {label}: ({round(x, 3) if isinstance(x, float) else x}, "
                f"{round(y, 3) if isinstance(y, float) else y}, "
                f"{round(z, 3) if isinstance(z, float) else z})"
            )

        if not objects:
            lines.append("No objects detected.")

        if not lines:
            return ToolResult(content="No objects detected. World model is empty.")

        return ToolResult(content="\n".join(lines))


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
        agent = context.agent
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
