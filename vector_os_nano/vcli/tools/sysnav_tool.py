# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""SysnavPerceptionTool — NL lifecycle for SysNav semantic perception.

"启动sysnav" / "start sysnav" / "开启语义感知" → action="start":
wires the in-process pano/cloud/odom feed + the /object_nodes_list consumer
into the live habitat agent, then launches the heavy perception pair
(detection + semantic mapping, sibling SysNav workspace) as a process group.
Detected objects flow into the agent's world model and become navigable
("走到沙发那里"). Everything fails loud — a missing workspace, an unsourced
shell, or a non-habitat base each return the exact fix, never a silent no-op.
"""
from __future__ import annotations

import logging
from typing import Any

from vector_os_nano.vcli.tools.base import (
    PermissionResult,
    ToolContext,
    ToolResult,
    tool,
)

logger = logging.getLogger(__name__)


@tool(
    name="sysnav_perception",
    description=(
        "Start/stop/inspect SysNav semantic perception for the habitat world "
        "(detection + semantic mapping; detected objects flow into the world "
        "model). Call with action='start' when the user says '启动sysnav' / "
        "'start sysnav' / '开启语义感知' / 'start semantic perception'. "
        "Requires a running habitat simulation (start_simulation "
        "sim_type='habitat')."
    ),
    read_only=False,
    permission="ask",
)
class SysnavPerceptionTool:
    """Lifecycle for the SysNav semantic-perception pipeline."""

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "stop", "status"],
                "description": "start the pipeline, stop it, or report status",
            },
        },
        "required": ["action"],
    }

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        from vector_os_nano.vcli import habitat_runtime

        action: str = params.get("action", "status")
        agent = context.agent
        if agent is None and context.app_state is not None:
            agent = context.app_state.get("agent")

        if agent is None or getattr(agent, "_base", None) is None:
            return ToolResult(
                content=(
                    "No robot base connected — SysNav perception needs the "
                    "habitat world. Start it first: start_simulation("
                    "sim_type=\"habitat\", scenario=\"apartment\")."
                ),
                is_error=True,
            )

        if action == "status":
            return ToolResult(content="\n".join(habitat_runtime.sysnav_status_lines(agent)))

        if action == "stop":
            lines = habitat_runtime.shutdown_sysnav(agent)
            return ToolResult(
                content="; ".join(lines) if lines else "SysNav was not running."
            )

        # action == "start"
        base = agent._base
        # Capability probe (campaign #11 M4, rule 3/7): SysNav runs on ANY base
        # exposing get_pano — habitat, MuJoCo G1, or MuJoCo Go2 — never gated by
        # embodiment/world name. wire_sysnav_feed picks the feed source the same way.
        if not hasattr(base, "get_pano"):
            return ToolResult(
                content=(
                    f"The connected base ({getattr(base, 'name', type(base).__name__)}) "
                    "has no panorama capability (get_pano) — SysNav perception "
                    "needs a pano-capable sim (habitat, mujoco g1, or mujoco go2)."
                ),
                is_error=True,
            )

        proc = getattr(agent, "_sysnav_proc", None)
        if (
            getattr(agent, "_sysnav_feed", None) is not None
            and proc is not None
            and proc.poll() is None
        ):
            return ToolResult(
                content=f"SysNav perception already running (nodes pid {proc.pid})."
            )

        status: list[str] = []
        try:
            habitat_runtime.wire_sysnav_feed(agent, on_status=status.append)
        except Exception as exc:  # noqa: BLE001 — fail loud with the exact fix
            return ToolResult(content=f"SysNav feed wiring failed: {exc}", is_error=True)

        if proc is None or proc.poll() is not None:
            try:
                new_proc, log_fh = habitat_runtime.launch_sysnav_nodes(
                    on_status=status.append
                )
            except Exception as exc:  # noqa: BLE001
                return ToolResult(
                    content=f"SysNav nodes launch failed: {exc}", is_error=True
                )
            agent._sysnav_proc = new_proc
            agent._sysnav_log_fh = log_fh

        status.append(
            "SysNav perception starting — GPU model loading takes about a "
            "minute; after that detected objects (sofa, table, ...) appear in "
            "the world model and become navigation targets."
        )
        return ToolResult(content="\n".join(status))

    def check_permissions(
        self, params: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        if params.get("action") == "status":
            return PermissionResult(behavior="allow", reason="Read-only status")
        return PermissionResult(behavior="allow", reason="SysNav lifecycle")
