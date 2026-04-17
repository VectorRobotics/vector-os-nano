"""ROS2 diagnostic tools for Vector CLI.

Three tools for LLM-driven robot diagnostics:
- ros2_topics: list topics, check hz, echo messages
- ros2_nodes: list nodes, get node info
- ros2_log: read robot log files from /tmp/vector_*.log
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from vector_os_nano.vcli.tools.base import ToolContext, ToolResult, tool

_TIMEOUT = 10  # seconds for ros2 CLI subprocess


def _run_ros2(args: list[str]) -> str:
    """Run ros2 CLI command, return stdout. Returns error text on non-zero exit."""
    result = subprocess.run(
        ["ros2"] + args,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
    )
    if result.returncode != 0:
        return (
            result.stderr.strip()
            or f"ros2 {' '.join(args)} failed (code {result.returncode})"
        )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# ros2_topics
# ---------------------------------------------------------------------------


@tool(
    name="ros2_topics",
    description=(
        "Query ROS2 topics: list all topics, check publishing rate (hz), "
        "or echo recent messages."
    ),
    read_only=True,
    permission="allow",
)
class Ros2TopicsTool:
    """Read-only ROS2 topic diagnostics."""

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "hz", "echo"],
                "description": "Operation to perform.",
            },
            "topic": {
                "type": "string",
                "description": "Topic name (required for hz/echo).",
            },
            "count": {
                "type": "integer",
                "description": "Number of messages for echo (default 1).",
                "default": 1,
            },
        },
        "required": ["action"],
    }

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        action = params["action"]
        topic = params.get("topic", "")
        try:
            if action == "list":
                output = _run_ros2(["topic", "list", "-t"])
                return ToolResult(content=output)

            elif action == "hz":
                if not topic:
                    return ToolResult(
                        content="topic parameter required for hz", is_error=True
                    )
                output = _run_ros2(["topic", "hz", topic, "--window", "5"])
                return ToolResult(content=output)

            elif action == "echo":
                if not topic:
                    return ToolResult(
                        content="topic parameter required for echo", is_error=True
                    )
                count = params.get("count", 1)
                cmd = (
                    ["ros2", "topic", "echo", topic, "--once"]
                    if count <= 1
                    else ["ros2", "topic", "echo", topic, "--max-wait-time=5"]
                )
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=_TIMEOUT,
                )
                output = proc.stdout.strip()
                if len(output) > 2000:
                    output = output[:2000] + "\n... (truncated)"
                return ToolResult(content=output or "(no messages received)")

            else:
                return ToolResult(content=f"Unknown action: {action}", is_error=True)

        except FileNotFoundError:
            return ToolResult(
                content="ros2 CLI not available. Source ROS2 first.", is_error=True
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                content=f"ros2 topic {action} timed out after {_TIMEOUT}s",
                is_error=True,
            )

    def is_concurrency_safe(self, params: dict[str, Any]) -> bool:
        return True


# ---------------------------------------------------------------------------
# ros2_nodes
# ---------------------------------------------------------------------------


@tool(
    name="ros2_nodes",
    description=(
        "Query ROS2 nodes: list all active nodes or get detailed info "
        "for a specific node."
    ),
    read_only=True,
    permission="allow",
)
class Ros2NodesTool:
    """Read-only ROS2 node diagnostics."""

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "info"],
                "description": "Operation to perform.",
            },
            "node": {
                "type": "string",
                "description": "Node name (required for info).",
            },
        },
        "required": ["action"],
    }

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        action = params["action"]
        try:
            if action == "list":
                output = _run_ros2(["node", "list"])
                return ToolResult(content=output)

            elif action == "info":
                node = params.get("node", "")
                if not node:
                    return ToolResult(
                        content="node parameter required for info", is_error=True
                    )
                output = _run_ros2(["node", "info", node])
                return ToolResult(content=output)

            else:
                return ToolResult(content=f"Unknown action: {action}", is_error=True)

        except FileNotFoundError:
            return ToolResult(
                content="ros2 CLI not available. Source ROS2 first.", is_error=True
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                content=f"ros2 node {action} timed out after {_TIMEOUT}s",
                is_error=True,
            )

    def is_concurrency_safe(self, params: dict[str, Any]) -> bool:
        return True


# ---------------------------------------------------------------------------
# ros2_log
# ---------------------------------------------------------------------------


@tool(
    name="ros2_log",
    description=(
        "Read robot log files. Available logs: bridge, tare, nav, nav_explore."
    ),
    read_only=True,
    permission="allow",
)
class Ros2LogTool:
    """Read tail of Vector robot log files."""

    _LOG_MAP: dict[str, str] = {
        "bridge": "/tmp/vector_vnav_bridge.log",
        "tare": "/tmp/vector_tare.log",
        "nav": "/tmp/vector_nav_only.log",
        "nav_explore": "/tmp/vector_nav_explore.log",
    }

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "log_name": {
                "type": "string",
                "enum": ["bridge", "tare", "nav", "nav_explore"],
                "description": "Which log to read.",
            },
            "lines": {
                "type": "integer",
                "description": "Number of lines from end (default 50).",
                "default": 50,
            },
        },
        "required": ["log_name"],
    }

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        log_name: str = params["log_name"]
        lines: int = params.get("lines", 50)

        path = self._LOG_MAP.get(log_name)
        if path is None:
            available = ", ".join(sorted(self._LOG_MAP))
            return ToolResult(
                content=f"Unknown log: {log_name}. Available: {available}",
                is_error=True,
            )

        log_path = Path(path)
        if not log_path.exists():
            return ToolResult(
                content=f"Log file not found: {path}", is_error=True
            )

        try:
            all_lines = log_path.read_text(errors="replace").splitlines()
            tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
            return ToolResult(content="\n".join(tail))
        except Exception as exc:
            return ToolResult(content=f"Failed to read {path}: {exc}", is_error=True)

    def is_concurrency_safe(self, params: dict[str, Any]) -> bool:
        return True
