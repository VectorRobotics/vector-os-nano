# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""BashTool — execute shell commands and return captured output.

Safety:
- Every command is "ask" (explicit user confirmation): THAT is the boundary.
- DENY_PATTERNS is only a substring tripwire for a few well-known destructive
  spellings; it is trivially bypassed by re-spelling and is not a sandbox.
- The command runs in its own process group with stdin closed; on timeout or
  abort the WHOLE group is killed (no orphaned grandchildren).
- Output goes to temp files, so a backgrounded job (``cmd &``) that keeps
  writing cannot hang the call; it is read back capped at MAX_OUTPUT_BYTES.
- Execution is bounded by timeout_ms (default 2 minutes, at most MAX_TIMEOUT_MS).
"""
from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
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
MAX_TIMEOUT_MS: int = 600_000  # 10 minutes

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
        timeout_ms = min(int(params.get("timeout_ms", DEFAULT_TIMEOUT_MS)), MAX_TIMEOUT_MS)
        timeout_sec: float = timeout_ms / 1000.0
        abort = getattr(context, "abort", None)

        try:
            with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
                proc = subprocess.Popen(
                    ["bash", "-c", command],
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    cwd=str(context.cwd),
                    start_new_session=True,
                )
                stopped = _wait(proc, timeout_sec, abort)
                if stopped:
                    return ToolResult(content=stopped, is_error=True)
                out.seek(0)
                err.seek(0)
                stdout_b = out.read(MAX_OUTPUT_BYTES + 1)
                stderr_b = err.read(MAX_OUTPUT_BYTES + 1)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(content=f"Error: {exc}", is_error=True)

        stdout: str = stdout_b.decode("utf-8", errors="replace")
        stderr: str = stderr_b.decode("utf-8", errors="replace")

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


def _wait(proc: subprocess.Popen, timeout_sec: float, abort: Any) -> str:
    """Wait for *proc*; on timeout or abort kill its whole process group.

    Returns "" when the command finished, else the reason it was stopped.
    """
    deadline = time.monotonic() + timeout_sec
    while True:
        try:
            proc.wait(timeout=min(0.1, max(deadline - time.monotonic(), 0.0)))
            return ""
        except subprocess.TimeoutExpired:
            pass
        reason = ""
        if abort is not None and abort.is_set():
            reason = "Command aborted"
        elif time.monotonic() >= deadline:
            reason = f"Command timed out after {timeout_sec}s"
        if reason:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            return reason
