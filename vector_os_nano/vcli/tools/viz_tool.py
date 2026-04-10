"""FoxgloveTool — start/stop Foxglove Bridge for web visualization."""
from __future__ import annotations

import shutil
import subprocess
from typing import Any

from vector_os_nano.vcli.tools.base import ToolContext, ToolResult, tool

# Module-level process tracking (survives across tool calls)
_foxglove_proc: subprocess.Popen | None = None


def _is_bridge_running() -> bool:
    """Check if foxglove_bridge is already listening on port 8765."""
    try:
        result = subprocess.run(
            ["ss", "-tlnp"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return ":8765" in result.stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


@tool(
    name="open_foxglove",
    description=(
        "打开/关闭可视化 (Open/close visualization). "
        "Start or stop Foxglove Bridge for real-time 3D visualization (可视化/foxglove/viz). "
        "Shows point clouds, navigation paths, camera feed, and scene graph markers. "
        "Call this tool when user says: 打开可视化, open visualization, start foxglove, show viz. "
        "Use action='start' to launch, action='stop' to shut down, action='status' to check."
    ),
    read_only=False,
    permission="allow",
)
class FoxgloveTool:
    """Manage the Foxglove Bridge lifecycle from the CLI."""

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "stop", "status"],
                "description": "start: launch bridge, stop: kill bridge, status: check if running",
                "default": "start",
            },
        },
    }

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        global _foxglove_proc
        action = params.get("action", "start")

        if action == "status":
            running = _is_bridge_running()
            return ToolResult(
                content=f"Foxglove Bridge: {'running on ws://localhost:8765' if running else 'not running'}"
            )

        if action == "stop":
            return self._stop()

        # action == "start"
        return self._start()

    def _start(self) -> ToolResult:
        global _foxglove_proc

        # Already running?
        if _is_bridge_running():
            return ToolResult(
                content=(
                    "Foxglove Bridge already running on ws://localhost:8765\n\n"
                    "Connect: app.foxglove.dev → Open connection → "
                    "Foxglove WebSocket → ws://localhost:8765\n"
                    "Dashboard: foxglove/vector-os-dashboard.json"
                )
            )

        # Check foxglove_bridge is installed
        foxglove_exec = shutil.which("ros2")
        if foxglove_exec is None:
            return ToolResult(
                content="ros2 not found in PATH. Source /opt/ros/jazzy/setup.bash first.",
                is_error=True,
            )

        # Start foxglove_bridge as background process
        try:
            _foxglove_proc = subprocess.Popen(
                [
                    "ros2", "launch", "foxglove_bridge",
                    "foxglove_bridge_launch.xml", "port:=8765",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as e:
            return ToolResult(
                content=f"Failed to start foxglove_bridge: {e}",
                is_error=True,
            )

        # Brief wait and verify
        try:
            _foxglove_proc.wait(timeout=2)
            # If it exited quickly, something went wrong
            return ToolResult(
                content=(
                    "foxglove_bridge exited immediately. "
                    "Check: sudo apt install ros-jazzy-foxglove-bridge"
                ),
                is_error=True,
            )
        except subprocess.TimeoutExpired:
            pass  # Still running — good

        return ToolResult(
            content=(
                "Foxglove Bridge started on ws://localhost:8765\n\n"
                "Connect:\n"
                "  1. Open app.foxglove.dev in Chrome\n"
                "  2. Open connection → Foxglove WebSocket → ws://localhost:8765\n"
                "  3. Import layout: foxglove/vector-os-dashboard.json\n\n"
                "Use open_foxglove(action='stop') to shut down."
            )
        )

    def _stop(self) -> ToolResult:
        global _foxglove_proc

        if _foxglove_proc is not None:
            _foxglove_proc.terminate()
            try:
                _foxglove_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _foxglove_proc.kill()
            _foxglove_proc = None
            return ToolResult(content="Foxglove Bridge stopped.")

        # Try to kill by port even if we don't have the process handle
        if _is_bridge_running():
            try:
                subprocess.run(
                    ["fuser", "-k", "8765/tcp"],
                    capture_output=True,
                    timeout=5,
                )
                return ToolResult(content="Foxglove Bridge stopped (via port kill).")
            except (subprocess.SubprocessError, FileNotFoundError):
                return ToolResult(
                    content="Could not stop foxglove_bridge. Kill manually: fuser -k 8765/tcp",
                    is_error=True,
                )

        return ToolResult(content="Foxglove Bridge is not running.")
