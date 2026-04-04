"""SimStartTool — start/stop robot simulations at runtime.

Allows V to spin up MuJoCo simulations mid-conversation without
requiring --sim or --sim-go2 flags at startup.

Supported simulations:
  arm   — SO-101 6-DOF arm (MuJoCoArm)
  go2   — Unitree Go2 quadruped (MuJoCoGo2)
"""
from __future__ import annotations

from typing import Any

from vector_os_nano.vcli.tools.base import (
    PermissionResult,
    ToolContext,
    ToolResult,
    tool,
)


@tool(
    name="start_simulation",
    description="Start a robot simulation (arm or go2 quadruped). No restart needed.",
    read_only=False,
    permission="ask",
)
class SimStartTool:
    """Start a MuJoCo simulation and register its skills into the tool registry."""

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "sim_type": {
                "type": "string",
                "enum": ["arm", "go2"],
                "description": "Which simulation to start: 'arm' (SO-101) or 'go2' (Unitree Go2)",
            },
            "gui": {
                "type": "boolean",
                "description": "Open the MuJoCo viewer window (default: true)",
                "default": True,
            },
        },
        "required": ["sim_type"],
    }

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        sim_type: str = params["sim_type"]
        gui: bool = params.get("gui", True)
        app = context.app_state
        if app is None:
            return ToolResult(content="No app state available", is_error=True)

        # Check if already running
        current_agent = app.get("agent")
        if current_agent is not None:
            current_arm = getattr(current_agent, "_arm", None)
            current_base = getattr(current_agent, "_base", None)
            if sim_type == "arm" and current_arm is not None:
                return ToolResult(content=f"Arm sim already running: {type(current_arm).__name__}")
            if sim_type == "go2" and current_base is not None:
                return ToolResult(content=f"Go2 sim already running: {type(current_base).__name__}")

        try:
            if sim_type == "arm":
                agent = self._start_arm(gui=gui)
            elif sim_type == "go2":
                agent = self._start_go2(gui=gui)
            else:
                return ToolResult(content=f"Unknown sim type: {sim_type}", is_error=True)
        except Exception as exc:
            return ToolResult(content=f"Failed to start {sim_type} sim: {exc}", is_error=True)

        # Update app state
        app["agent"] = agent

        # Register skill tools
        registry = app.get("registry")
        if registry is not None:
            from vector_os_nano.vcli.tools.skill_wrapper import wrap_skills
            for skill_tool in wrap_skills(agent):
                registry.register(skill_tool)

        # Rebuild system prompt
        engine = app.get("engine")
        if engine is not None:
            from vector_os_nano.vcli.prompt import build_system_prompt
            engine._system_prompt = build_system_prompt(agent=agent, cwd=context.cwd)

        hw_name = type(getattr(agent, "_arm", None) or getattr(agent, "_base", None)).__name__
        skill_count = len(agent._skill_registry.list_skills()) if hasattr(agent, "_skill_registry") else 0
        return ToolResult(
            content=f"Started {sim_type} simulation: {hw_name}, {skill_count} skills registered."
        )

    @staticmethod
    def _start_arm(gui: bool = True) -> Any:
        from vector_os_nano.core.agent import Agent  # type: ignore[import]
        from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm  # type: ignore[import]
        arm = MuJoCoArm(gui=gui)
        arm.connect()
        return Agent(arm=arm)

    @staticmethod
    def _start_go2(gui: bool = True) -> Any:
        import os
        import signal
        import subprocess
        import atexit
        import time as _time
        from vector_os_nano.core.agent import Agent  # type: ignore[import]
        from vector_os_nano.core.config import load_config

        # Launch full stack as SEPARATE PROCESS (stable gait — no GIL contention)
        # This is the same architecture as run.py --sim-go2 --explore
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))))
        # Use launch_explore.sh — all nodes (bridge + nav stack + TARE) must be
        # in ONE process group for reliable DDS communication. The nav flag
        # (/tmp/vector_nav_active) is NOT created here — dog stays still.
        # explore.py creates the flag to start movement.
        vnav_script = os.path.join(repo, "scripts", "launch_explore.sh")
        gui_flag = [] if gui else ["--no-gui"]

        log_fh = open("/tmp/vector_vnav.log", "w")
        vnav_proc = subprocess.Popen(
            ["bash", vnav_script] + gui_flag,
            stdout=log_fh, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        def _cleanup():
            try:
                os.killpg(os.getpgid(vnav_proc.pid), signal.SIGTERM)
                vnav_proc.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(vnav_proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            log_fh.close()

        atexit.register(_cleanup)

        # Wait for MuJoCo + bridge + nav stack to initialize
        _time.sleep(20)

        # Connect via ROS2 proxy (same as run.py --explore)
        from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy
        base = Go2ROS2Proxy()
        base.connect()

        pos = base.get_position()
        if pos == (0.0, 0.0, 0.28):
            # Default position — wait more for odom
            _time.sleep(5)
            pos = base.get_position()

        # Load config for API key
        cfg_path = os.path.join(repo, "config", "user.yaml")
        cfg = load_config(cfg_path) if os.path.exists(cfg_path) else {}
        api_key = cfg.get("llm", {}).get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")

        agent = Agent(base=base, llm_api_key=api_key, config=cfg)

        # Go2 skills
        from vector_os_nano.skills.go2 import get_go2_skills  # type: ignore[import]
        for skill in get_go2_skills():
            agent._skill_registry.register(skill)

        # VLM perception (GPT-4o via OpenRouter)
        if api_key:
            try:
                from vector_os_nano.perception.vlm_go2 import Go2VLMPerception
                agent._vlm = Go2VLMPerception(config={"api_key": api_key})
            except Exception:
                agent._vlm = None

        # Scene graph — persistent, also attach to proxy for RViz marker publishing
        import os as _os
        from vector_os_nano.core.scene_graph import SceneGraph
        _persist_path = _os.path.expanduser("~/.vector_os_nano/scene_graph.yaml")
        _os.makedirs(_os.path.dirname(_persist_path), exist_ok=True)
        sg = SceneGraph(persist_path=_persist_path)
        sg.load()
        _stats = sg.stats()
        if _stats["rooms"] > 0:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "[SceneGraph] restored %d rooms, %d objects from %s",
                _stats["rooms"], _stats["objects"], _persist_path,
            )
        agent._spatial_memory = sg
        base._scene_graph = agent._spatial_memory

        return agent

    def check_permissions(
        self, params: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        return PermissionResult(behavior="allow", reason="Simulation startup")
