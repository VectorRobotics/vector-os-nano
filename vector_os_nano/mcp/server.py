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
    """MCP server backed by a Vector OS Nano Agent + VectorEngine.

    Registers all skills as tools (via mcp/tools.py) and world state +
    camera renders as resources (via mcp/resources.py).

    Args:
        agent: A fully initialised Agent instance (hardware container).
        engine: A fully initialised VectorEngine instance.
        session: A Session instance for conversation history.
    """

    def __init__(
        self,
        agent: Agent,
        engine: Any,
        session: Any,
    ) -> None:
        self._agent = agent
        self._engine = engine
        self._session = session
        self._server = Server("vector-os-nano")
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register MCP tool and resource handlers on the underlying Server."""
        from vector_os_nano.mcp.tools import skills_to_mcp_tools, handle_tool_call
        from vector_os_nano.mcp.resources import get_resource_definitions, read_resource

        server = self._server
        agent = self._agent
        engine = self._engine
        session = self._session

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
            result_text = await handle_tool_call(agent, engine, session, name, arguments or {})
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
        """Run the server with stdio transport."""
        from mcp.server.stdio import stdio_server  # noqa: PLC0415

        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )

    async def run_sse(self, host: str = "0.0.0.0", port: int = 8100) -> None:
        """Run the server with SSE transport over HTTP.

        Start manually in a terminal, then connect Claude Code to it:
            python -m vector_os_nano.mcp --sim --port 8100
            # In Claude Code: /mcp add url http://localhost:8100/sse
        """
        from mcp.server.sse import SseServerTransport  # noqa: PLC0415
        from starlette.applications import Starlette  # noqa: PLC0415
        from starlette.routing import Mount, Route  # noqa: PLC0415
        import uvicorn  # noqa: PLC0415

        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as (read_stream, write_stream):
                await self._server.run(
                    read_stream,
                    write_stream,
                    self._server.create_initialization_options(),
                )

        app = Starlette(
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        _log(f"[MCP] SSE server listening on http://{host}:{port}/sse")
        _log(f"[MCP] In Claude Code run:  /mcp add url http://localhost:{port}/sse")

        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()


# ---------------------------------------------------------------------------
# Engine builder
# ---------------------------------------------------------------------------


def _build_engine(agent: Agent) -> tuple[Any, Any]:
    """Build a VectorEngine + Session from an Agent (hardware container).

    Mirrors the CLI engine creation flow in vcli/cli.py.

    Args:
        agent: A fully initialised Agent (hardware container).

    Returns:
        (engine, session) tuple ready for VectorMCPServer.
    """
    from vector_os_nano.vcli.config import resolve_credentials  # noqa: PLC0415
    from vector_os_nano.vcli.backends import create_backend  # noqa: PLC0415
    from vector_os_nano.vcli.tools import CategorizedToolRegistry, discover_categorized_tools  # noqa: PLC0415
    from vector_os_nano.vcli.tools.skill_wrapper import wrap_skills  # noqa: PLC0415
    from vector_os_nano.vcli.prompt import build_system_prompt  # noqa: PLC0415
    from vector_os_nano.vcli.engine import VectorEngine  # noqa: PLC0415
    from vector_os_nano.vcli.session import create_session  # noqa: PLC0415

    # 1. Resolve credentials (Claude OAuth > env > config)
    api_key, provider, model, base_url = resolve_credentials()

    # 2. LLM backend
    backend = create_backend(provider=provider, api_key=api_key, model=model, base_url=base_url)

    # 3. Tool registry: built-in tools + skill wrappers
    registry: CategorizedToolRegistry = CategorizedToolRegistry()
    tools_list, cat_map = discover_categorized_tools()
    for t in tools_list:
        cat = "default"
        for c, names in cat_map.items():
            if t.name in names:
                cat = c
                break
        registry.register(t, category=cat)
    for skill_tool in wrap_skills(agent):
        registry.register(skill_tool, category="robot")

    # 4. System prompt
    system_prompt = build_system_prompt(agent=agent)

    # 5. Engine
    engine = VectorEngine(backend=backend, registry=registry, system_prompt=system_prompt)

    # 6. VGG cognitive layer
    engine.init_vgg(agent=agent, skill_registry=agent._skill_registry)

    # 7. Session
    session = create_session(metadata={"source": "mcp", "model": model})

    return engine, session


# ---------------------------------------------------------------------------
# Agent / stack factories
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    """Log to stderr (stdout is reserved for MCP stdio transport)."""
    import sys
    print(msg, file=sys.stderr, flush=True)


def _start_camera_viewer(perception: Any) -> None:
    """Start a background OpenCV viewer for hardware mode.

    Shows RGB + depth side-by-side with tracking annotations:
    - Left: RGB with bounding boxes, labels, and 3D coordinates
    - Right: Depth colormap with mask contours and pointcloud centroids

    Mirrors the viewer in run.py _run_cli().
    """
    if perception is None or not hasattr(perception, "get_color_frame"):
        return

    import threading  # noqa: PLC0415

    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    cam = perception._camera if hasattr(perception, "_camera") else perception

    def _viewer_loop() -> None:
        cv2.namedWindow("Vector OS MCP", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Vector OS MCP", 1280, 480)

        while True:
            try:
                color = cam.get_color_frame() if hasattr(cam, "get_color_frame") else perception.get_color_frame()
                depth = cam.get_depth_frame() if hasattr(cam, "get_depth_frame") else None
                if color is None:
                    continue

                # --- Left: RGB + tracking overlay ---
                rgb_display = cv2.cvtColor(color.copy(), cv2.COLOR_RGB2BGR)

                if hasattr(perception, "_last_tracked") and perception._last_tracked:
                    for obj in perception._last_tracked:
                        if obj.bbox_2d:
                            x1, y1, x2, y2 = [int(v) for v in obj.bbox_2d]
                            cv2.rectangle(rgb_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            lbl = obj.label
                            if obj.pose:
                                lbl += f" ({obj.pose.x:.3f},{obj.pose.y:.3f},{obj.pose.z:.3f})"
                            cv2.putText(rgb_display, lbl, (x1, max(y1 - 8, 12)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                elif hasattr(perception, "_last_detections") and perception._last_detections:
                    for det in perception._last_detections:
                        x1, y1, x2, y2 = [int(v) for v in det.bbox]
                        cv2.rectangle(rgb_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(rgb_display, det.label, (x1, max(y1 - 8, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                # --- Right: Depth colormap + mask + centroid ---
                if depth is not None:
                    depth_f = np.clip(depth.astype(np.float32), 0, 5000)
                    depth_u8 = (depth_f / 5000.0 * 255).astype(np.uint8)
                    depth_colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

                    if hasattr(perception, "_last_tracked") and perception._last_tracked:
                        for obj in perception._last_tracked:
                            # Draw mask contours
                            if obj.mask is not None and obj.mask.shape == depth_colored.shape[:2]:
                                contours, _ = cv2.findContours(
                                    obj.mask.astype(np.uint8),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE,
                                )
                                cv2.drawContours(depth_colored, contours, -1, (255, 255, 255), 2)
                            # Draw pointcloud centroid
                            if obj.pose and obj.bbox_2d:
                                cx = int((obj.bbox_2d[0] + obj.bbox_2d[2]) / 2)
                                cy = int((obj.bbox_2d[1] + obj.bbox_2d[3]) / 2)
                                cv2.circle(depth_colored, (cx, cy), 6, (0, 255, 0), -1)
                                cv2.circle(depth_colored, (cx, cy), 8, (255, 255, 255), 2)
                                info = f"{obj.pose.x:.3f},{obj.pose.y:.3f},{obj.pose.z:.3f}"
                                cv2.putText(depth_colored, info, (cx + 10, cy),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    combined = np.hstack([rgb_display, depth_colored])
                else:
                    combined = rgb_display

                cv2.imshow("Vector OS MCP", combined)
                if cv2.waitKey(33) == 27:
                    break
            except Exception:
                pass

        cv2.destroyAllWindows()

    thread = threading.Thread(target=_viewer_loop, daemon=True)
    thread.start()
    _log("[MCP] Camera viewer started — RGB + depth with annotations (ESC to close)")


def create_sim_stack(headless: bool = True) -> Agent:
    """Create an Agent with MuJoCo simulation backend.

    Mirrors run.py _init_sim() for MCP use.

    Args:
        headless: If True (default), no MuJoCo viewer window.
                  If False, open an interactive viewer.

    Returns:
        A fully connected Agent ready for VectorEngine use.
    """
    from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm  # noqa: PLC0415
    from vector_os_nano.hardware.sim.mujoco_gripper import MuJoCoGripper  # noqa: PLC0415
    from vector_os_nano.hardware.sim.mujoco_perception import MuJoCoPerception  # noqa: PLC0415
    from vector_os_nano.perception.calibration import Calibration  # noqa: PLC0415

    _log(f"[MCP] Starting MuJoCo simulation (headless={headless})...")

    cfg = _load_config_with_fallback()

    # Sim-specific overrides (mirrors run.py _init_sim)
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

    arm = MuJoCoArm(gui=not headless)
    arm.connect()
    _log(f"[MCP] Sim arm connected. Joints: {[round(j, 2) for j in arm.get_joint_positions()]}")

    gripper = MuJoCoGripper(arm)
    gripper.close()

    objs = arm.get_object_positions()
    if objs:
        _log(f"[MCP] Scene objects: {', '.join(objs.keys())}")

    perception = MuJoCoPerception(arm)
    _log("[MCP] Sim perception ready (ground-truth mode).")

    calibration = Calibration()  # identity — sim positions are world-frame

    api_key = cfg.get("llm", {}).get("api_key") or os.environ.get("OPENROUTER_API_KEY")

    agent = Agent(
        arm=arm,
        gripper=gripper,
        perception=perception,
        llm_api_key=api_key,
        config=cfg,
    )
    agent._calibration = calibration

    _log(f"[MCP] Sim stack ready. Skills: {agent.skills}")
    return agent


def create_hardware_stack() -> Agent:
    """Create an Agent with real SO-101 hardware.

    Mirrors run.py _init_hardware() for MCP use.
    Starts: SO-101 arm + RealSense D405 + Moondream VLM + EdgeTAM tracker.

    Returns:
        A fully connected Agent ready for VectorEngine use.
    """
    cfg = _load_config_with_fallback()
    api_key = cfg.get("llm", {}).get("api_key") or os.environ.get("OPENROUTER_API_KEY")

    # --- Arm ---
    arm = None
    gripper = None
    try:
        from vector_os_nano.hardware.so101 import SO101Arm, SO101Gripper  # noqa: PLC0415
        port = cfg.get("arm", {}).get("port", "/dev/ttyACM0")
        _log(f"[MCP] Connecting arm on {port}...")
        arm = SO101Arm(port=port)
        arm.connect()
        gripper = SO101Gripper(arm._bus)
        joints = [round(j, 2) for j in arm.get_joint_positions()]
        _log(f"[MCP] Arm connected. Joints: {joints}")
    except Exception as exc:
        _log(f"[MCP] Arm not available: {exc}")

    # --- Perception (camera + VLM + tracker) ---
    perception = None
    try:
        from vector_os_nano.perception.realsense import RealSenseCamera  # noqa: PLC0415
        from vector_os_nano.perception.vlm import VLMDetector  # noqa: PLC0415
        from vector_os_nano.perception.tracker import EdgeTAMTracker  # noqa: PLC0415
        from vector_os_nano.perception.pipeline import PerceptionPipeline  # noqa: PLC0415

        _log("[MCP] Connecting camera (RealSense D405)...")
        camera = RealSenseCamera()
        camera.connect()
        _log("[MCP] Camera connected.")

        vlm_model = (
            cfg.get("perception", {}).get("vlm_model")
            or os.environ.get("MOONDREAM_MODEL", "vikhyatk/moondream2")
        )
        os.environ.setdefault("MOONDREAM_MODEL", vlm_model)
        _log(f"[MCP] Loading VLM ({vlm_model})...")
        vlm = VLMDetector()
        _log("[MCP] VLM loaded.")

        _log("[MCP] Loading tracker (EdgeTAM)...")
        tracker = EdgeTAMTracker()
        _log("[MCP] Tracker loaded.")

        perception = PerceptionPipeline(camera=camera, vlm=vlm, tracker=tracker)
        _log("[MCP] Perception pipeline ready.")
    except Exception as exc:
        _log(f"[MCP] Perception not available: {exc}")

    # --- Calibration ---
    calibration = None
    try:
        cal_file = cfg.get("calibration", {}).get(
            "file", "config/workspace_calibration.yaml"
        )
        if cal_file and os.path.exists(cal_file):
            if cal_file.endswith((".yaml", ".yml")):
                # Use the same YAML loader from run.py
                import yaml  # noqa: PLC0415
                import numpy as np  # noqa: PLC0415
                from vector_os_nano.perception.calibration import Calibration  # noqa: PLC0415
                with open(cal_file, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                if isinstance(data, dict):
                    calibration = Calibration()
                    raw_matrix = data.get("transform_matrix")
                    if raw_matrix is not None:
                        matrix = np.array(raw_matrix, dtype=np.float64)
                        if matrix.shape == (4, 4):
                            calibration._matrix = matrix
                    _log(f"[MCP] Calibration loaded from {cal_file}")
            else:
                from vector_os_nano.perception.calibration import Calibration  # noqa: PLC0415
                calibration = Calibration.load(cal_file)
                _log(f"[MCP] Calibration loaded from {cal_file}")
        else:
            _log(f"[MCP] No calibration file at {cal_file!r}")
    except Exception as exc:
        _log(f"[MCP] Calibration load failed: {exc}")

    agent = Agent(
        arm=arm,
        gripper=gripper,
        perception=perception,
        llm_api_key=api_key,
        config=cfg,
    )
    if calibration is not None:
        agent._calibration = calibration

    # Start live camera feed (hardware mode only, daemon thread).
    _start_camera_viewer(perception)

    _log(f"[MCP] Hardware stack ready. Skills: {agent.skills}")
    _log(f"[MCP] Arm: {'connected' if arm else 'NOT available'}")
    _log(f"[MCP] Perception: {'ready' if perception else 'NOT available'}")
    _log(f"[MCP] Calibration: {'loaded' if calibration else 'NOT loaded'}")
    return agent


# Keep old names as aliases for backward compatibility with any external callers
def create_sim_agent(headless: bool = True) -> Agent:
    """Deprecated alias for create_sim_stack(). Use create_sim_stack() instead."""
    return create_sim_stack(headless=headless)


def create_hardware_agent() -> Agent:
    """Deprecated alias for create_hardware_stack(). Use create_hardware_stack() instead."""
    return create_hardware_stack()


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

    parser = argparse.ArgumentParser(
        description="Vector OS Nano MCP Server",
        epilog=(
            "examples:\n"
            "  python -m vector_os_nano.mcp --sim            # sim + viewer + SSE on :8100\n"
            "  python -m vector_os_nano.mcp --sim-headless   # sim headless + SSE\n"
            "  python -m vector_os_nano.mcp --hardware       # real arm + SSE\n"
            "  python -m vector_os_nano.mcp --sim --stdio    # sim + stdio (for .mcp.json)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    hw_mode = parser.add_mutually_exclusive_group()
    hw_mode.add_argument(
        "--sim",
        action="store_true",
        help="MuJoCo simulation with viewer window",
    )
    hw_mode.add_argument(
        "--sim-headless",
        action="store_true",
        help="MuJoCo simulation without viewer",
    )
    hw_mode.add_argument(
        "--hardware",
        action="store_true",
        help="Real SO-101 arm + RealSense + VLM",
    )
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Use stdio transport (for .mcp.json auto-start). Default is SSE.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8100,
        help="SSE server port (default: 8100)",
    )
    args = parser.parse_args()

    # Create agent (hardware container)
    if args.hardware:
        agent = create_hardware_stack()
    elif args.sim:
        agent = create_sim_stack(headless=False)
    elif args.sim_headless:
        agent = create_sim_stack(headless=True)
    else:
        # Default: sim with viewer
        agent = create_sim_stack(headless=False)

    # Build engine + session
    engine, session = _build_engine(agent)

    server = VectorMCPServer(agent, engine, session)
    try:
        if args.stdio:
            await server.run_stdio()
        else:
            await server.run_sse(port=args.port)
    finally:
        agent.disconnect()


def main_sync() -> None:
    """Synchronous entry point — used by the vector-os-mcp console script."""
    asyncio.run(main())
