# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""SysnavSimTool — one-shot bringup for MuJoCo + SysNav.

A thin orchestrator over :class:`SimStartTool` that, after the Go2 sim
subprocess is up, wires :class:`LiveSysnavBridge` so that incoming
``/object_nodes_list`` messages from the SysNav workspace populate the
agent's ``WorldModel`` directly.

Pre-flight detects whether ``tare_planner.msg`` is importable. When it
is not (SysNav workspace not sourced) the tool still completes — the
sim runs, but the bridge stays inactive. This lets a developer verify
the MuJoCo virtual-sensor side without depending on SysNav being live.

See ``docs/sysnav_simulation.md`` for the full bringup procedure and
``docs/sysnav_integration.md`` for the topic contract.
"""
from __future__ import annotations

import logging
from typing import Any

from vector_os_nano.vcli.tools.base import (
    ToolContext,
    ToolResult,
    tool,
)

logger = logging.getLogger(__name__)


_REQUIRED_INPUT_TOPICS = (
    "/registered_scan",
    "/state_estimation",
    "/camera/image",
)
_REQUIRED_OUTPUT_TOPICS = (
    "/object_nodes_list",
    "/target_object_instruction",
)


@tool(
    name="start_sysnav_sim",
    description=(
        "Start MuJoCo Go2+arm simulation with the SysNav scene-graph bridge "
        "wired in. SysNav workspace must already be running in another "
        "terminal — this tool only handles the MuJoCo + bridge side. "
        f"Bridge subscribes to {_REQUIRED_OUTPUT_TOPICS[0]}; sim publishes "
        f"{', '.join(_REQUIRED_INPUT_TOPICS)}."
    ),
    read_only=False,
    permission="ask",
)
class SysnavSimTool:
    """Run :class:`SimStartTool` then attach :class:`LiveSysnavBridge`."""

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "gui": {
                "type": "boolean",
                "description": "Open MuJoCo viewer (default: true).",
                "default": True,
            },
        },
        "required": [],
    }

    def __init__(self, *, sim_tool_factory: Any = None) -> None:
        # Test seam: dependency injection for SimStartTool. Production
        # path lazy-imports the real one.
        self._sim_tool_factory = sim_tool_factory
        self._bridge_factory: Any = None
        self._bridge: Any = None
        self._sim_tool: Any = None
        self._started: bool = False

    # ------------------------------------------------------------------
    # Public ToolResult API
    # ------------------------------------------------------------------

    def execute(
        self, params: dict[str, Any], context: ToolContext,
    ) -> ToolResult:
        if self._started:
            return ToolResult(
                content="sysnav-sim already running — call stop_sysnav_sim first",
            )

        # 1. Pre-flight — surface dependency gaps to the user before sim boot.
        sysnav_ok = self.preflight_sysnav_workspace()
        if not sysnav_ok:
            logger.warning(
                "[sysnav_sim] tare_planner.msg not importable — bridge will "
                "stay inactive but sim will still start.",
            )

        # 2. Boot the underlying go2 + Piper sim via SimStartTool.
        sim_tool = self._build_sim_tool()
        sim_params = {
            "embodiment": "go2_piper",
            "gui": bool(params.get("gui", True)),
        }
        sim_result = sim_tool.execute(sim_params, context)
        if sim_result.is_error:
            return ToolResult(content=f"sysnav-sim not started: {sim_result.content}",
                              is_error=True)
        self._sim_tool = sim_tool

        # 3. Attach the LiveSysnavBridge to the agent's world_model.
        bridge_active = False
        if context.agent is not None and getattr(context.agent, "_world_model", None):
            self._bridge = self._build_bridge(context.agent._world_model)
            bridge_active = self._bridge.start()
        else:
            logger.warning(
                "[sysnav_sim] context.agent has no _world_model — bridge skipped",
            )

        self._started = True

        msg = "sysnav-sim started: "
        if bridge_active:
            msg += "MuJoCo + bridge live, listening to /object_nodes_list."
        elif sysnav_ok:
            msg += "MuJoCo up; bridge could not start (check rclpy logs)."
        else:
            msg += (
                "MuJoCo up; bridge inactive (source SysNav workspace + "
                "rebuild tare_planner.msg). World model will not populate "
                "from SysNav until the bridge is restarted."
            )
        return ToolResult(content=msg, metadata={
            "sim_result": getattr(sim_result, "metadata", {}),
            "bridge_active": bridge_active,
            "sysnav_workspace_ok": sysnav_ok,
        })

    def stop(self, context: ToolContext | None = None) -> None:
        """Tear down bridge then sim — reverse of execute order.

        ``context`` is optional; if not supplied a minimal ToolContext
        is built so SimStopTool's contract is satisfied.
        """
        import threading
        from pathlib import Path
        from types import SimpleNamespace

        if self._bridge is not None:
            try:
                self._bridge.stop()
            except Exception:
                pass
            self._bridge = None
        if self._sim_tool is not None:
            try:
                stop_tool = self._build_sim_stop_tool()
                if context is None:
                    context = ToolContext(
                        agent=None,
                        cwd=Path.cwd(),
                        session=SimpleNamespace(),
                        permissions=SimpleNamespace(),
                        abort=threading.Event(),
                    )
                stop_tool.execute({}, context)
            except Exception:
                pass
            self._sim_tool = None
        self._started = False

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------

    @staticmethod
    def preflight_sysnav_workspace() -> bool:
        """True iff ``tare_planner.msg`` is importable.

        The pre-flight is intentionally cheap (single try/except) so it
        can be called from places that just want a status read.
        """
        try:
            import tare_planner.msg     # noqa: F401
        except ImportError:
            return False
        return True

    # ------------------------------------------------------------------
    # Test seams
    # ------------------------------------------------------------------

    def _build_sim_tool(self) -> Any:
        if self._sim_tool_factory is not None:
            return self._sim_tool_factory()
        from vector_os_nano.vcli.tools.sim_tool import SimStartTool
        return SimStartTool()

    def _build_sim_stop_tool(self) -> Any:
        from vector_os_nano.vcli.tools.sim_tool import SimStopTool
        return SimStopTool()

    def _build_bridge(self, world_model: Any) -> Any:
        if self._bridge_factory is not None:
            return self._bridge_factory(world_model)
        from vector_os_nano.integrations.sysnav_bridge import LiveSysnavBridge
        return LiveSysnavBridge(world_model)

    # Optional — allow tests to inject the bridge factory.
    def set_bridge_factory(self, factory: Any) -> None:
        self._bridge_factory = factory


__all__ = ["SysnavSimTool"]
