"""FastAPI web dashboard for Vector OS Nano.

Serves a single-page web dashboard with:
- Real-time AI chat via WebSocket
- Live robot status updates
- Command execution through the Agent

Usage:
    from vector_os_nano.web.app import create_app
    app = create_app(agent, config)
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from vector_os_nano.web.chat import ChatManager

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"

# Connected WebSocket clients for status broadcast
_status_clients: set[WebSocket] = set()
_chat_clients: set[WebSocket] = set()

# Global references (set by create_app)
_agent: Any = None
_chat_manager: ChatManager | None = None
_command_lock = asyncio.Lock()
_command_history: list[dict] = []


def _get_status() -> dict:
    """Build current robot status dict."""
    if _agent is None:
        return {"mode": "disconnected", "arm": None, "objects": []}

    status: dict[str, Any] = {
        "mode": "sim" if hasattr(_agent._arm, "_model") else "real",
        "arm": {
            "connected": _agent._arm is not None,
            "joints": [],
            "joint_names": [],
            "gripper": "unknown",
        },
        "objects": [],
        "last_command": _command_history[-1] if _command_history else None,
    }

    if _agent._arm is not None:
        try:
            status["arm"]["joints"] = [
                round(j, 3) for j in _agent._arm.get_joint_positions()
            ]
            status["arm"]["joint_names"] = list(_agent._arm.joint_names)
        except Exception:
            pass

    if _agent._gripper is not None:
        try:
            pos = _agent._gripper.get_position()
            status["arm"]["gripper"] = "open" if pos > 0.5 else "closed"
        except Exception:
            pass

    # Objects from perception or sim
    if hasattr(_agent._arm, "get_object_positions"):
        try:
            objs = _agent._arm.get_object_positions()
            status["objects"] = [
                {"name": name, "position": [round(p, 3) for p in pos]}
                for name, pos in objs.items()
            ]
        except Exception:
            pass

    return status


def _get_state_info() -> str:
    """Build state info string for LLM context."""
    s = _get_status()
    parts = [f"Mode: {s['mode']}"]
    if s["arm"]["connected"]:
        parts.append(f"Gripper: {s['arm']['gripper']}")
    return ", ".join(parts)


def _get_objects_info() -> str:
    """Build objects info string for LLM context."""
    s = _get_status()
    if not s["objects"]:
        return "No objects detected"
    return ", ".join(
        f"{o['name']} at ({o['position'][0]:.2f}, {o['position'][1]:.2f})"
        for o in s["objects"]
    )


async def _status_broadcast_loop():
    """Background task: push status to all connected clients every 500ms."""
    while True:
        if _status_clients:
            status = _get_status()
            msg = json.dumps({"type": "status", **status})
            disconnected = set()
            for ws in _status_clients:
                try:
                    await ws.send_text(msg)
                except Exception:
                    disconnected.add(ws)
            _status_clients -= disconnected
        await asyncio.sleep(0.5)


async def _execute_command(text: str) -> dict:
    """Execute a robot command via the Agent. Runs in thread pool."""
    global _command_history

    start = time.time()
    result = {"text": text, "status": "running", "duration": 0}

    try:
        loop = asyncio.get_event_loop()
        exec_result = await loop.run_in_executor(None, _agent.execute, text)
        elapsed = time.time() - start
        result.update({
            "status": "success" if exec_result.success else "failed",
            "duration": round(elapsed, 1),
            "details": exec_result.result_data if hasattr(exec_result, 'result_data') else None,
        })
    except Exception as exc:
        elapsed = time.time() - start
        result.update({
            "status": "error",
            "duration": round(elapsed, 1),
            "error": str(exc),
        })

    _command_history.append(result)
    if len(_command_history) > 50:
        _command_history = _command_history[-50:]

    return result


def create_app(agent: Any, config: dict) -> FastAPI:
    """Create the FastAPI application.

    Args:
        agent: Vector OS Nano Agent instance.
        config: Application config dict.
    """
    global _agent, _chat_manager

    _agent = agent

    # Create chat manager
    api_key = config.get("llm", {}).get("api_key", "")
    model = config.get("llm", {}).get("model", "anthropic/claude-haiku-4-5")
    api_base = config.get("llm", {}).get("api_base", "https://openrouter.ai/api/v1")

    if api_key:
        _chat_manager = ChatManager(
            api_key=api_key, model=model, api_base=api_base
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Start status broadcaster
        task = asyncio.create_task(_status_broadcast_loop())
        logger.info("Web dashboard started at http://localhost:8000")
        yield
        task.cancel()
        if _chat_manager:
            await _chat_manager.close()

    app = FastAPI(title="Vector OS Nano", lifespan=lifespan)

    # Serve static files
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = _STATIC_DIR / "index.html"
        return HTMLResponse(html_path.read_text())

    @app.get("/api/status")
    async def api_status():
        return _get_status()

    @app.get("/api/history")
    async def api_history():
        return _command_history

    @app.get("/api/skills")
    async def api_skills():
        """List available skills with JSON schemas."""
        if _agent is None:
            return {"skills": []}
        return {"skills": _agent._skill_registry.to_schemas()}

    @app.get("/api/world")
    async def api_world():
        """Current world model state."""
        if _agent is None:
            return {"objects": [], "robot": {}}
        return _agent.world.to_dict()

    @app.post("/api/execute")
    async def api_execute(body: dict):
        """Execute a natural language instruction.

        Body: {"instruction": "pick the banana"}
        Returns: ExecutionResult as JSON.
        """
        if _agent is None:
            return {"success": False, "error": "No agent configured"}
        instruction = body.get("instruction", "")
        if not instruction:
            return {"success": False, "error": "Missing 'instruction' field"}
        async with _command_lock:
            result = await asyncio.get_event_loop().run_in_executor(
                None, _agent.execute, instruction
            )
        return result.to_dict()

    @app.post("/api/skill/{skill_name}")
    async def api_skill(skill_name: str, body: dict = None):
        """Execute a skill directly with structured params.

        POST /api/skill/pick {"object_label": "banana", "mode": "drop"}
        Returns: ExecutionResult as JSON.
        """
        if _agent is None:
            return {"success": False, "error": "No agent configured"}
        params = body or {}
        async with _command_lock:
            result = await asyncio.get_event_loop().run_in_executor(
                None, _agent.execute_skill, skill_name, params
            )
        return result.to_dict()

    @app.post("/api/run_goal")
    async def api_run_goal(body: dict):
        """Execute an iterative goal via agent loop.

        Body: {"goal": "clean the table", "max_iterations": 10, "verify": true}
        Returns: GoalResult as JSON.
        """
        if _agent is None:
            return {"success": False, "error": "No agent configured"}
        goal = body.get("goal", "")
        if not goal:
            return {"success": False, "error": "Missing 'goal' field"}
        max_iter = body.get("max_iterations", 10)
        verify = body.get("verify", True)
        async with _command_lock:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: _agent.run_goal(goal, max_iterations=max_iter, verify=verify)
            )
        return result.to_dict()

    @app.get("/api/camera")
    async def api_camera():
        """Get current camera frame as JPEG."""
        from fastapi.responses import Response
        if _agent is None or _agent._perception is None:
            return Response(content=b"", media_type="text/plain", status_code=404)
        try:
            frame = await asyncio.get_event_loop().run_in_executor(
                None, _agent._perception.get_color_frame
            )
            if frame is None:
                return Response(content=b"", media_type="text/plain", status_code=404)
            # Encode as JPEG
            import cv2
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return Response(content=buf.tobytes(), media_type="image/jpeg")
        except Exception as exc:
            return Response(content=str(exc).encode(), media_type="text/plain", status_code=500)

    @app.websocket("/ws/chat")
    async def ws_chat(ws: WebSocket):
        await ws.accept()
        _chat_clients.add(ws)
        logger.info("Chat client connected")

        # Send welcome
        await ws.send_json({
            "type": "response",
            "role": "assistant",
            "content": "Welcome to Vector OS Nano! I can control the robot arm for you. Try saying '抓杯子' or 'pick up the mug'.",
        })

        try:
            while True:
                data = await ws.receive_json()
                user_text = data.get("content", "").strip()
                if not user_text:
                    continue

                # Check if it's a robot command
                is_cmd = _chat_manager.is_command(user_text) if _chat_manager else False

                if is_cmd and _agent is not None:
                    # Robot command flow
                    async with _command_lock:
                        # Get AI acknowledgment first
                        if _chat_manager:
                            ai_ack = await _chat_manager.chat(
                                user_text,
                                state_info=_get_state_info(),
                                objects_info=_get_objects_info(),
                            )
                            await ws.send_json({
                                "type": "response",
                                "role": "assistant",
                                "content": ai_ack,
                            })

                        # Execute command
                        await ws.send_json({
                            "type": "executing",
                            "content": f"Executing: {user_text}",
                        })

                        result = await _execute_command(user_text)

                        # Report result
                        status_emoji = "OK" if result["status"] == "success" else "FAILED"
                        result_msg = f"{status_emoji} ({result['duration']}s)"
                        if _chat_manager:
                            _chat_manager.add_system_message(
                                f"Command '{user_text}' completed: {result['status']} in {result['duration']}s"
                            )

                        await ws.send_json({
                            "type": "command_result",
                            "result": result,
                        })

                elif _chat_manager:
                    # General chat
                    response = await _chat_manager.chat(
                        user_text,
                        state_info=_get_state_info(),
                        objects_info=_get_objects_info(),
                    )
                    await ws.send_json({
                        "type": "response",
                        "role": "assistant",
                        "content": response,
                    })
                else:
                    await ws.send_json({
                        "type": "response",
                        "role": "assistant",
                        "content": "No LLM configured. Set OPENROUTER_API_KEY.",
                    })

        except WebSocketDisconnect:
            logger.info("Chat client disconnected")
        except Exception as exc:
            logger.warning("Chat WebSocket error: %s", exc)
        finally:
            _chat_clients.discard(ws)

    @app.websocket("/ws/status")
    async def ws_status(ws: WebSocket):
        await ws.accept()
        _status_clients.add(ws)
        logger.info("Status client connected")
        try:
            while True:
                await ws.receive_text()  # keep alive
        except WebSocketDisconnect:
            pass
        finally:
            _status_clients.discard(ws)

    return app
