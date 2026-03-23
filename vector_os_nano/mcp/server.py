"""MCP Server for Vector OS Nano.

Exposes robot skills as MCP tools and world/camera state as MCP resources.
Primary transport: stdio (for Claude Code integration).

Usage:
    python -m vector_os_nano.mcp --sim          # MuJoCo simulation with viewer
    python -m vector_os_nano.mcp --sim-headless  # Headless simulation (default for Claude Code)
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from typing import Any

from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
import mcp.types as types

from vector_os_nano.core.agent import Agent

logger = logging.getLogger(__name__)


class VectorMCPServer:
    """MCP server backed by a Vector OS Nano Agent instance.

    Registers all skills as tools (via mcp/tools.py) and world state +
    camera renders as resources (via mcp/resources.py).

    Args:
        agent: A fully initialised Agent instance.
    """

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        self._server = Server("vector-os-nano")
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register MCP tool and resource handlers on the underlying Server."""
        from vector_os_nano.mcp.tools import skills_to_mcp_tools, handle_tool_call
        from vector_os_nano.mcp.resources import get_resource_definitions, read_resource

        server = self._server
        agent = self._agent

        @server.list_tools()
        async def list_tools() -> list[types.Tool]:
            tool_defs = skills_to_mcp_tools(agent._skill_registry)
            return [
                types.Tool(
                    name=t["name"],
                    description=t["description"],
                    inputSchema=t["inputSchema"],
                )
                for t in tool_defs
            ]

        @server.call_tool()
        async def call_tool(
            name: str, arguments: dict
        ) -> list[types.TextContent | types.ImageContent]:
            result_text = await handle_tool_call(agent, name, arguments or {})
            return [types.TextContent(type="text", text=result_text)]

        @server.list_resources()
        async def list_resources() -> list[types.Resource]:
            defs = get_resource_definitions()
            return [
                types.Resource(
                    uri=d["uri"],  # type: ignore[arg-type]
                    name=d["name"],
                    description=d["description"],
                    mimeType=d["mimeType"],
                )
                for d in defs
            ]

        @server.read_resource()
        async def read_resource_handler(uri: Any) -> list[ReadResourceContents]:
            """Read a resource by URI.

            Converts from our internal dict format to the ReadResourceContents
            iterable that the MCP server framework expects.
            """
            uri_str = str(uri)
            result = await read_resource(agent, uri_str)
            contents_raw = result["contents"][0]

            if "text" in contents_raw:
                return [
                    ReadResourceContents(
                        content=contents_raw["text"],
                        mime_type=contents_raw.get("mimeType", "application/json"),
                    )
                ]
            elif "blob" in contents_raw:
                # blob is base64-encoded; decode to bytes for BlobResourceContents
                import base64  # noqa: PLC0415

                raw_bytes = base64.b64decode(contents_raw["blob"])
                return [
                    ReadResourceContents(
                        content=raw_bytes,
                        mime_type=contents_raw.get("mimeType", "image/png"),
                    )
                ]
            else:
                raise ValueError(
                    f"Resource {uri_str!r} returned content without 'text' or 'blob'"
                )

    async def run_stdio(self) -> None:
        """Run the server with stdio transport (for Claude Code)."""
        from mcp.server.stdio import stdio_server  # noqa: PLC0415

        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )


# ---------------------------------------------------------------------------
# Simulation agent factory
# ---------------------------------------------------------------------------


def create_sim_agent(headless: bool = True) -> Agent:
    """Create an Agent with MuJoCo simulation backend.

    Mirrors the initialisation in run.py _init_sim() but simplified for MCP
    use (no calibration YAML loading, direct Agent construction).

    Args:
        headless: If True (default), no MuJoCo viewer window.
                  If False, open an interactive viewer.

    Returns:
        A fully connected Agent ready for skill execution.
    """
    from vector_os_nano.core.config import load_config  # noqa: PLC0415
    from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm  # noqa: PLC0415
    from vector_os_nano.hardware.sim.mujoco_gripper import MuJoCoGripper  # noqa: PLC0415
    from vector_os_nano.hardware.sim.mujoco_perception import MuJoCoPerception  # noqa: PLC0415

    logger.info("Starting MuJoCo simulation (headless=%s)", headless)

    # Load config — try user.yaml first, fall back to defaults
    cfg = _load_config_with_fallback()

    # Apply sim-specific overrides (mirrors run.py _init_sim)
    cfg.setdefault("skills", {}).setdefault("pick", {}).update(
        {
            "z_offset": 0.0,
            "x_offset": 0.0,
            "pre_grasp_height": 0.04,
            "hardware_offsets": False,
            "wrist_roll_offset": math.pi / 2,
        }
    )
    cfg.setdefault("skills", {}).setdefault("home", {}).setdefault(
        "joint_values", [0.0, 0.0, 0.0, 0.0, 0.0]
    )
    cfg["sim_move_duration"] = 3.0

    # Create sim hardware
    arm = MuJoCoArm(gui=not headless)
    arm.connect()

    gripper = MuJoCoGripper(arm)
    gripper.close()

    perception = MuJoCoPerception(arm)

    # API key from config or environment
    api_key = cfg.get("llm", {}).get("api_key") or os.environ.get("OPENROUTER_API_KEY")

    agent = Agent(
        arm=arm,
        gripper=gripper,
        perception=perception,
        llm_api_key=api_key,
        config=cfg,
    )

    logger.info("Sim agent created. Skills: %s", agent.skills)
    return agent


def _load_config_with_fallback() -> dict:
    """Load config, trying user.yaml then defaults."""
    from vector_os_nano.core.config import load_config  # noqa: PLC0415

    for candidate in [
        "config/user.yaml",
        os.path.join(os.path.dirname(__file__), "..", "..", "config", "user.yaml"),
    ]:
        if os.path.exists(candidate):
            try:
                return load_config(candidate)
            except Exception as exc:
                logger.warning("Could not load %s: %s", candidate, exc)
    return load_config(None)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


async def main() -> None:
    """Async entry point for the MCP server."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Vector OS Nano MCP Server")
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Use MuJoCo simulation with viewer window",
    )
    parser.add_argument(
        "--sim-headless",
        action="store_true",
        help="Use MuJoCo simulation without viewer (default)",
    )
    args = parser.parse_args()

    # --sim-headless takes priority; --sim opens viewer; no flag defaults to headless
    headless = True
    if args.sim:
        headless = False
    if args.sim_headless:
        headless = True

    agent = create_sim_agent(headless=headless)

    server = VectorMCPServer(agent)
    try:
        await server.run_stdio()
    finally:
        agent.disconnect()


def main_sync() -> None:
    """Synchronous entry point — used by the vector-os-mcp console script."""
    asyncio.run(main())
