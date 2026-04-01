"""BashTool — execute shell commands and return captured output.

Safety:
- DENY_PATTERNS blocks known destructive commands unconditionally.
- All other commands are "ask" (require explicit user confirmation).
- Output is truncated at MAX_OUTPUT_BYTES to prevent memory exhaustion.
- Execution is bounded by timeout_ms (default 2 minutes).
"""
from __future__ import annotations

import subprocess
from typing import Any

from vector_os_nano.vcli.tools.base import (
    PermissionResult,
    ToolContext,
    ToolResult,
    tool,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_OUTPUT_BYTES: int = 100_000  # 100 KB truncation limit
DEFAULT_TIMEOUT_MS: int = 120_000  # 2 minutes

# Patterns matched as substrings against the raw command string.
# Any match results in an unconditional DENY.
DENY_PATTERNS: list[str] = [
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=",
    ":(){ :|:& };:",
    "> /dev/sd",
    "chmod -R 777 /",
]


# ---------------------------------------------------------------------------
# BashTool
# ---------------------------------------------------------------------------


@tool(
    name="bash",
    description="Execute a shell command and return captured stdout/stderr",
    read_only=False,
    permission="ask",
)
class BashTool:
    """Run an arbitrary bash command inside a subprocess and capture output."""

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute",
            },
            "timeout_ms": {
                "type": "integer",
                "description": "Timeout in milliseconds",
                "default": DEFAULT_TIMEOUT_MS,
            },
        },
        "required": ["command"],
    }

    # ------------------------------------------------------------------
    # Permission gate
    # ------------------------------------------------------------------

    def check_permissions(
        self, params: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        """Deny known destructive commands; otherwise ask the user."""
        command: str = params.get("command", "")
        for pattern in DENY_PATTERNS:
            if pattern in command:
                return PermissionResult(
                    behavior="deny",
                    reason=f"Dangerous command blocked: contains '{pattern}'",
                )
        return PermissionResult(behavior="ask", reason=f"Execute: {command}")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        """Run *command* in bash, capturing stdout + stderr."""
        command: str = params["command"]
        timeout_ms: int = params.get("timeout_ms", DEFAULT_TIMEOUT_MS)
        timeout_sec: float = timeout_ms / 1000.0

        try:
            proc = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                timeout=timeout_sec,
                cwd=str(context.cwd),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                content=f"Command timed out after {timeout_sec}s",
                is_error=True,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(content=f"Error: {exc}", is_error=True)

        stdout: str = proc.stdout.decode("utf-8", errors="replace")
        stderr: str = proc.stderr.decode("utf-8", errors="replace")

        output: str = stdout
        if stderr:
            output += f"\n[stderr]\n{stderr}"

        # Truncate to avoid overwhelming callers with huge payloads
        if len(output.encode("utf-8")) > MAX_OUTPUT_BYTES:
            # Slice bytes, then decode safely to avoid mid-character cuts
            truncated_bytes = output.encode("utf-8")[:MAX_OUTPUT_BYTES]
            output = truncated_bytes.decode("utf-8", errors="replace") + "\n... (truncated)"

        if proc.returncode != 0:
            return ToolResult(
                content=f"Exit code {proc.returncode}\n{output}",
                is_error=True,
            )
        return ToolResult(content=output)
