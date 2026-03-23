"""MCP resources: world state and simulation camera renders."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vector_os_nano.core.agent import Agent


# ---------------------------------------------------------------------------
# Resource URI definitions
# ---------------------------------------------------------------------------

RESOURCE_DEFINITIONS: list[dict] = [
    {
        "uri": "world://state",
        "name": "World State",
        "description": "Complete world model: objects, robot state, spatial relations",
        "mimeType": "application/json",
    },
    {
        "uri": "world://objects",
        "name": "Objects",
        "description": "List of detected objects with positions and confidence",
        "mimeType": "application/json",
    },
    {
        "uri": "world://robot",
        "name": "Robot State",
        "description": "Current robot state: joint positions, gripper, held object",
        "mimeType": "application/json",
    },
    {
        "uri": "camera://overhead",
        "name": "Overhead Camera",
        "description": "Top-down view of the workspace (640x480 PNG)",
        "mimeType": "image/png",
    },
    {
        "uri": "camera://front",
        "name": "Front Camera",
        "description": "Front view of the workspace (640x480 PNG)",
        "mimeType": "image/png",
    },
    {
        "uri": "camera://side",
        "name": "Side Camera",
        "description": "Side view of the workspace (640x480 PNG)",
        "mimeType": "image/png",
    },
    {
        "uri": "camera://live",
        "name": "Live Camera",
        "description": "Live RealSense D405 RGB feed (hardware mode)",
        "mimeType": "image/png",
    },
]


def get_resource_definitions() -> list[dict]:
    """Return all available MCP resource definitions."""
    return list(RESOURCE_DEFINITIONS)


# ---------------------------------------------------------------------------
# Resource dispatcher
# ---------------------------------------------------------------------------


async def read_resource(agent: Agent, uri: str) -> dict:
    """Read an MCP resource by URI.

    Args:
        agent: The Agent instance (provides world and arm access).
        uri: Resource URI (e.g., "world://state", "camera://overhead").

    Returns:
        Dict with "contents" key matching MCP resource read response format.
        JSON resources include a "text" field; image resources use "blob"
        (base64-encoded PNG bytes).

    Raises:
        ValueError: If the URI is not recognised.
    """
    if uri == "world://state":
        return _read_world_state(agent)
    if uri == "world://objects":
        return _read_objects(agent)
    if uri == "world://robot":
        return _read_robot_state(agent)
    if uri.startswith("camera://"):
        camera_name = uri.split("://", 1)[1]
        return await _read_camera(agent, camera_name)
    raise ValueError(f"Unknown resource URI: {uri!r}")


# ---------------------------------------------------------------------------
# JSON resource handlers
# ---------------------------------------------------------------------------


def _read_world_state(agent: Agent) -> dict:
    """Read complete world state as JSON."""
    world = agent.world
    data: dict[str, Any] = world.to_dict() if world is not None else {"objects": [], "robot": {}}
    return {
        "contents": [
            {
                "uri": "world://state",
                "mimeType": "application/json",
                "text": json.dumps(data, indent=2),
            }
        ]
    }


def _read_objects(agent: Agent) -> dict:
    """Read just the objects list."""
    world = agent.world
    if world is None:
        objects: list = []
    else:
        objects = world.to_dict().get("objects", [])
    return {
        "contents": [
            {
                "uri": "world://objects",
                "mimeType": "application/json",
                "text": json.dumps(objects, indent=2),
            }
        ]
    }


def _read_robot_state(agent: Agent) -> dict:
    """Read just the robot state."""
    world = agent.world
    if world is None:
        robot: dict = {}
    else:
        robot = world.to_dict().get("robot", {})
    return {
        "contents": [
            {
                "uri": "world://robot",
                "mimeType": "application/json",
                "text": json.dumps(robot, indent=2),
            }
        ]
    }


# ---------------------------------------------------------------------------
# Camera resource handler
# ---------------------------------------------------------------------------


async def _read_camera(agent: Agent, camera_name: str) -> dict:
    """Render a camera view and return as base64 PNG.

    Supports two paths:
    - camera://live  → RealSense D405 via perception.get_color_frame() (hardware mode)
    - camera://overhead|front|side → MuJoCo arm.render() (sim mode)

    Raises:
        ValueError: If the camera source is not available.
    """
    if camera_name == "live":
        return await _read_live_camera(agent)
    return await _read_sim_camera(agent, camera_name)


async def _read_live_camera(agent: Agent) -> dict:
    """Read a frame from the RealSense D405 camera (hardware mode)."""
    perception = agent._perception
    if perception is None or not hasattr(perception, "get_color_frame"):
        raise ValueError("Live camera not available (no perception pipeline)")

    rgb_array = await asyncio.to_thread(perception.get_color_frame)

    if rgb_array is None:
        raise ValueError("Live camera returned no frame")

    png_bytes = _numpy_to_png(rgb_array)
    b64 = base64.b64encode(png_bytes).decode("ascii")

    return {
        "contents": [
            {
                "uri": "camera://live",
                "mimeType": "image/png",
                "blob": b64,
            }
        ]
    }


async def _read_sim_camera(agent: Agent, camera_name: str) -> dict:
    """Render a MuJoCo camera view (sim mode)."""
    arm = agent._arm
    if arm is None or not hasattr(arm, "render"):
        raise ValueError(
            f"Camera render not available (arm={arm!r} does not support render)"
        )

    bgr_array = await asyncio.to_thread(arm.render, camera_name=camera_name)

    if bgr_array is None:
        raise ValueError(f"Camera {camera_name!r} returned no image")

    rgb_array = _bgr_to_rgb(bgr_array)
    png_bytes = _numpy_to_png(rgb_array)
    b64 = base64.b64encode(png_bytes).decode("ascii")

    return {
        "contents": [
            {
                "uri": f"camera://{camera_name}",
                "mimeType": "image/png",
                "blob": b64,
            }
        ]
    }


# ---------------------------------------------------------------------------
# Image conversion helpers
# ---------------------------------------------------------------------------


def _bgr_to_rgb(bgr_array: Any) -> Any:
    """Reverse the last axis to convert BGR -> RGB (or vice-versa)."""
    import numpy as np  # noqa: PLC0415

    return np.ascontiguousarray(bgr_array[:, :, ::-1])


def _numpy_to_png(rgb_array: Any) -> bytes:
    """Convert a numpy RGB array (H, W, 3) uint8 to PNG bytes.

    Tries PIL first; falls back to OpenCV if PIL is not available.

    Raises:
        RuntimeError: If neither PIL nor cv2 can encode the image.
    """
    try:
        import io  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415

        img = Image.fromarray(rgb_array)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        pass

    try:
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        # cv2 expects BGR — our input is already RGB, so re-convert
        bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        success, buf = cv2.imencode(".png", bgr)
        if success:
            return buf.tobytes()
    except ImportError:
        pass

    raise RuntimeError("Failed to encode PNG: neither PIL nor cv2 is available")
