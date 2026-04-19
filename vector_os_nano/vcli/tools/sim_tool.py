"""SimStartTool — start/stop robot simulations at runtime.

Allows V to spin up simulations mid-conversation without requiring
--sim or --sim-go2 flags at startup.

Supported simulations:
  arm   — SO-101 6-DOF arm (MuJoCoArm)
  go2   — Unitree Go2 quadruped (MuJoCoGo2 / IsaacSimProxy / GazeboGo2Proxy)

Supported backends:
  mujoco — MuJoCo (default, physics + textured rendering)
  mujoco — MuJoCo 3.x (lightweight fallback)
  gazebo — Gz Sim Harmonic (ROS2-native, open-source)
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
    name="start_simulation",
    description="Start a robot simulation (arm or go2 quadruped) with isaac, mujoco, or gazebo backend. No restart needed.",
    read_only=False,
    permission="ask",
)
class SimStartTool:
    """Start a simulation and register its skills into the tool registry.

    Backends: mujoco (default), gazebo (Gz Sim Harmonic), isaac (Docker, archived).
    """

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
                "description": "Open viewer/GUI window (default: true)",
                "default": True,
            },
            "backend": {
                "type": "string",
                "enum": ["isaac", "mujoco", "gazebo"],
                "default": "mujoco",
                "description": (
                    "Simulation backend: 'mujoco' (default, physics + textured rendering), "
                    "'gazebo' (Gz Sim Harmonic), or 'isaac' (Docker, archived)"
                ),
            },
            "with_arm": {
                "type": "boolean",
                "description": (
                    "ONLY for sim_type='go2'. True = mount Piper 6-DoF arm on "
                    "Go2's back (enables pick/place; forces sinusoidal gait "
                    "because convex_mpc is 12-DoF-only). False = pure Go2 "
                    "(smoother MPC gait, no manipulation). BEFORE calling this "
                    "tool, ASK the user which mode they want — both have real "
                    "tradeoffs. If the user gives an ambiguous command like "
                    "'go2sim' or '启动仿真', ask before calling."
                ),
            },
        },
        "required": ["sim_type"],
    }

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        sim_type: str = params["sim_type"]
        gui: bool = params.get("gui", True)
        backend: str = params.get("backend", "mujoco")
        with_arm: bool = bool(params.get("with_arm", False))
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
            if backend == "isaac":
                if sim_type == "go2":
                    agent = self._start_isaac_go2()
                elif sim_type == "arm":
                    agent = self._start_isaac_arm()
                else:
                    return ToolResult(content=f"Unknown sim type: {sim_type}", is_error=True)
            elif backend == "gazebo":
                if sim_type == "go2":
                    agent = self._start_gazebo_go2()
                else:
                    return ToolResult(
                        content="Gazebo backend only supports go2",
                        is_error=True,
                    )
            else:
                # Default: mujoco backend (existing paths unchanged)
                if sim_type == "arm":
                    agent = self._start_arm(gui=gui)
                elif sim_type == "go2":
                    agent = self._start_go2(gui=gui, with_arm=with_arm)
                else:
                    return ToolResult(content=f"Unknown sim type: {sim_type}", is_error=True)
        except Exception as exc:
            return ToolResult(content=f"Failed to start {sim_type} sim: {exc}", is_error=True)

        # Update app state
        app["agent"] = agent
        app["scene_graph"] = getattr(agent, "_spatial_memory", None)
        app["skill_registry"] = getattr(agent, "_skill_registry", None)

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
            # Reinit VGG with new agent so verifier has live robot state
            try:
                engine.init_vgg(
                    agent=agent,
                    skill_registry=getattr(agent, "_skill_registry", None),
                    on_vgg_step=getattr(engine, "_vgg_step_callback", None),
                )
            except Exception:
                pass

        # Report SceneGraph status
        sg = getattr(agent, "_spatial_memory", None)
        sg_stats = sg.stats() if sg else {}
        sg_info = ""
        if sg_stats.get("rooms", 0) > 0:
            sg_info = f" SceneGraph restored: {sg_stats['rooms']} rooms."

        hw_name = type(getattr(agent, "_arm", None) or getattr(agent, "_base", None)).__name__
        skill_count = len(agent._skill_registry.list_skills()) if hasattr(agent, "_skill_registry") else 0
        return ToolResult(
            content=f"Started {sim_type} simulation: {hw_name}, {skill_count} skills registered.{sg_info}"
        )

    @staticmethod
    def _shutdown_agent(agent: Any) -> str:
        """Tear down a running sim agent: kill subprocesses, disconnect hardware.

        Returns a short human-readable summary of what was stopped.
        """
        import os
        import signal
        parts: list[str] = []
        base = getattr(agent, "_base", None)
        arm = getattr(agent, "_arm", None)

        # Go2: kill launched subprocess group (nav stack + bridge + MuJoCo)
        if base is not None:
            proc = getattr(base, "_sim_subprocess", None)
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=5)
                    parts.append("sim subprocess stopped")
                except Exception:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        parts.append("sim subprocess force-killed")
                    except Exception as exc:
                        parts.append(f"subprocess kill failed: {exc}")
            log_fh = getattr(base, "_sim_log_fh", None)
            if log_fh is not None:
                try:
                    log_fh.close()
                except Exception:
                    pass
            try:
                base.disconnect()
                parts.append(f"{type(base).__name__} disconnected")
            except Exception:
                pass

        # Arm + gripper (SO-101 arm-only sim, OR PiperROS2Proxy in go2-with-arm)
        gripper = getattr(agent, "_gripper", None)
        if gripper is not None:
            try:
                gripper.disconnect()
                parts.append(f"{type(gripper).__name__} disconnected")
            except Exception:
                pass
        if arm is not None:
            try:
                arm.disconnect()
                parts.append(f"{type(arm).__name__} disconnected")
            except Exception:
                pass

        return "; ".join(parts) or "nothing to stop"

    @staticmethod
    def _start_arm(gui: bool = True) -> Any:
        from vector_os_nano.core.agent import Agent  # type: ignore[import]
        from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm  # type: ignore[import]
        arm = MuJoCoArm(gui=gui)
        arm.connect()
        return Agent(arm=arm)

    # ------------------------------------------------------------------
    # Pickable-object discovery: populate world_model from MJCF
    # ------------------------------------------------------------------

    @staticmethod
    def _populate_pickables_from_mjcf(world_model: Any, scene_xml_path: str) -> int:
        """Register every body whose name starts with 'pickable_' as an ObjectState.

        Loads the MJCF locally in the main process (independent of the
        bridge subprocess's MuJoCo instance). Uses the MJCF's DEFAULT
        body positions — i.e. what's written in the XML, not the post-
        physics-settled state. Sim-to-sim the drift is <1 cm, fine for
        grasp targeting since the skill has its own grasp-z offset.
        """
        import mujoco  # local import to avoid hard dep at module load
        from vector_os_nano.core.world_model import ObjectState

        model = mujoco.MjModel.from_xml_path(scene_xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        def _label(body_name: str) -> str:
            stem = body_name[len("pickable_"):]
            parts = stem.split("_")
            if len(parts) == 2:
                return f"{parts[1]} {parts[0]}"
            return stem.replace("_", " ")

        count = 0
        for bid in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if not name or not name.startswith("pickable_"):
                continue
            pos = data.body(bid).xpos
            world_model.add_object(ObjectState(
                object_id=name,
                label=_label(name),
                x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
                confidence=1.0,
                state="on_table",
                properties={"source": "mjcf_scan"},
            ))
            count += 1
        return count

    @staticmethod
    def _start_go2(gui: bool = True, with_arm: bool = False) -> Any:
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

        # Propagate mode to the sim subprocess via environment variable.
        # MuJoCoGo2._build_room_scene_xml reads VECTOR_SIM_WITH_ARM to pick
        # the scene (go2_piper vs bare go2). Subprocess inherits os.environ.
        child_env = os.environ.copy()
        child_env["VECTOR_SIM_WITH_ARM"] = "1" if with_arm else "0"

        log_fh = open("/tmp/vector_vnav.log", "w")
        vnav_proc = subprocess.Popen(
            ["bash", vnav_script] + gui_flag,
            stdout=log_fh, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            env=child_env,
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
        # Stash subprocess handles on the base so SimStopTool can clean up
        # mid-session without waiting for atexit.
        base._sim_subprocess = vnav_proc  # type: ignore[attr-defined]
        base._sim_log_fh = log_fh         # type: ignore[attr-defined]

        pos = base.get_position()
        if pos == (0.0, 0.0, 0.28):
            # Default position — wait more for odom
            _time.sleep(5)
            pos = base.get_position()

        # Load config for API key
        cfg_path = os.path.join(repo, "config", "user.yaml")
        cfg = load_config(cfg_path) if os.path.exists(cfg_path) else {}
        api_key = cfg.get("llm", {}).get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")

        # Piper arm + gripper proxies — bridge advertises /piper/* topics
        # when VECTOR_SIM_WITH_ARM=1 was set in child_env above.
        piper_arm = None
        piper_gripper = None
        if with_arm:
            try:
                from vector_os_nano.hardware.sim.mujoco_go2 import _build_room_scene_xml
                scene_xml = str(_build_room_scene_xml(with_arm=True))

                from vector_os_nano.hardware.sim.piper_ros2_proxy import (
                    PiperROS2Proxy, PiperGripperROS2Proxy,
                )
                piper_arm = PiperROS2Proxy(base_proxy=base, scene_xml_path=scene_xml)
                piper_arm.connect()
                piper_gripper = PiperGripperROS2Proxy()
                piper_gripper.connect()
                logger.info("[sim_tool] Piper proxies connected (arm + gripper)")
            except Exception as exc:
                logger.error("[sim_tool] Piper proxy setup failed: %s", exc)
                piper_arm = None
                piper_gripper = None

        agent = Agent(base=base, arm=piper_arm, gripper=piper_gripper,
                      llm_api_key=api_key, config=cfg)

        # Populate world_model with pickable_* scene objects (MJCF scan in
        # main process — independent of the bridge subprocess's MuJoCo).
        if with_arm:
            try:
                from vector_os_nano.hardware.sim.mujoco_go2 import _build_room_scene_xml
                scene_xml = str(_build_room_scene_xml(with_arm=True))
                n = SimStartTool._populate_pickables_from_mjcf(agent._world_model, scene_xml)
                logger.info("[sim_tool] registered %d pickable objects", n)
            except Exception as exc:
                logger.warning("[sim_tool] pickable scan failed: %s", exc)

        # Go2 skills
        from vector_os_nano.skills.go2 import get_go2_skills  # type: ignore[import]
        for skill in get_go2_skills():
            agent._skill_registry.register(skill)
        # Piper manipulation skill — only useful when arm proxy connected
        if piper_arm is not None:
            from vector_os_nano.skills.pick_top_down import PickTopDownSkill
            agent._skill_registry.register(PickTopDownSkill())

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

    @staticmethod
    def _start_isaac_go2() -> Any:
        """Connect to Go2 in Isaac Sim Docker (must already be running).

        Uses IsaacSimProxy over ROS2 topics — identical interface to
        Go2ROS2Proxy but assumes Isaac Sim is already live. No subprocess
        launch or sleep(20) needed.
        """
        import os
        from vector_os_nano.hardware.sim.isaac_sim_proxy import IsaacSimProxy  # type: ignore[import]
        from vector_os_nano.core.agent import Agent  # type: ignore[import]
        from vector_os_nano.core.config import load_config

        proxy = IsaacSimProxy()
        proxy.connect()

        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))))
        cfg_path = os.path.join(repo, "config", "user.yaml")
        cfg = load_config(cfg_path) if os.path.exists(cfg_path) else {}
        api_key = cfg.get("llm", {}).get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")

        agent = Agent(base=proxy, llm_api_key=api_key, config=cfg)

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

        # Scene graph — persistent
        import os as _os
        from vector_os_nano.core.scene_graph import SceneGraph
        _persist_path = _os.path.expanduser("~/.vector_os_nano/scene_graph.yaml")
        _os.makedirs(_os.path.dirname(_persist_path), exist_ok=True)
        sg = SceneGraph(persist_path=_persist_path)
        sg.load()
        agent._spatial_memory = sg
        proxy._scene_graph = agent._spatial_memory

        return agent

    @staticmethod
    def _start_gazebo_go2() -> Any:
        """Start Go2 in Gazebo Harmonic via launch script + connect proxy.

        1. Launches Gazebo via scripts/launch_gazebo.sh (subprocess)
        2. Waits for /state_estimation topic (up to 60s)
        3. Connects GazeboGo2Proxy
        4. Builds Agent with skills + VLM + SceneGraph
        """
        import os
        import signal
        import subprocess
        import atexit
        import time as _time
        from vector_os_nano.hardware.sim.gazebo_go2_proxy import GazeboGo2Proxy  # type: ignore[import]
        from vector_os_nano.core.agent import Agent  # type: ignore[import]
        from vector_os_nano.core.config import load_config

        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))))

        # Launch Gazebo via script (handles sourcing quadruped_ros2_control)
        launch_script = os.path.join(repo, "scripts", "launch_gazebo.sh")
        log_fh = open("/tmp/vector_gazebo.log", "w")
        gz_proc = subprocess.Popen(
            ["bash", launch_script, "--world", "apartment"],
            stdout=log_fh, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        def _cleanup():
            try:
                os.killpg(os.getpgid(gz_proc.pid), signal.SIGTERM)
                gz_proc.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(gz_proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            log_fh.close()

        atexit.register(_cleanup)

        # Wait for Gazebo to start (poll for /clock topic)
        import logging as _logging
        logger = _logging.getLogger(__name__)
        logger.info("[Gazebo] Waiting for Gazebo to start...")
        for i in range(60):
            if GazeboGo2Proxy.is_gazebo_running():
                logger.info("[Gazebo] Gazebo ready after %ds", i)
                break
            _time.sleep(1)
        else:
            raise ConnectionError(
                "Gazebo did not start within 60s. Check /tmp/vector_gazebo.log"
            )

        # Extra wait for controllers to activate
        _time.sleep(5)

        # Connect proxy
        proxy = GazeboGo2Proxy()
        proxy.connect()

        cfg_path = os.path.join(repo, "config", "user.yaml")
        cfg = load_config(cfg_path) if os.path.exists(cfg_path) else {}
        api_key = cfg.get("llm", {}).get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")

        agent = Agent(base=proxy, llm_api_key=api_key, config=cfg)

        # Go2 skills
        from vector_os_nano.skills.go2 import get_go2_skills  # type: ignore[import]
        for skill in get_go2_skills():
            agent._skill_registry.register(skill)

        # VLM perception
        if api_key:
            try:
                from vector_os_nano.perception.vlm_go2 import Go2VLMPerception
                agent._vlm = Go2VLMPerception(config={"api_key": api_key})
            except Exception:
                agent._vlm = None

        # Scene graph — persistent
        from vector_os_nano.core.scene_graph import SceneGraph
        _persist_path = os.path.expanduser("~/.vector_os_nano/scene_graph.yaml")
        os.makedirs(os.path.dirname(_persist_path), exist_ok=True)
        sg = SceneGraph(persist_path=_persist_path)
        sg.load()
        agent._spatial_memory = sg
        proxy._scene_graph = agent._spatial_memory

        return agent

    @staticmethod
    def _start_isaac_arm() -> Any:
        """Connect to a 6-DOF arm in Isaac Sim Docker (must already be running).

        Uses IsaacSimArmProxy over ROS2 topics. Isaac Sim must be running
        before calling this method.
        """
        from vector_os_nano.hardware.sim.isaac_sim_arm_proxy import IsaacSimArmProxy  # type: ignore[import]
        from vector_os_nano.core.agent import Agent  # type: ignore[import]

        arm = IsaacSimArmProxy()
        arm.connect()
        return Agent(arm=arm)

    def check_permissions(
        self, params: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        return PermissionResult(behavior="allow", reason="Simulation startup")


@tool(
    name="stop_simulation",
    description=(
        "Stop a currently running robot simulation. Kills the MuJoCo + ROS2 "
        "bridge + nav stack subprocess, disconnects hardware, unregisters Go2 "
        "skills from the tool set. Call this when the user says '关闭仿真' / "
        "'stop sim' / 'shutdown simulation' etc."
    ),
    read_only=False,
    permission="ask",
)
class SimStopTool:
    """Stop the running simulation and clear the agent's hardware."""

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
    }

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        app = context.app_state
        if app is None:
            return ToolResult(content="No app state available", is_error=True)

        agent = app.get("agent")
        if agent is None:
            return ToolResult(content="No simulation is running.")

        # Tear down hardware and subprocesses
        summary = SimStartTool._shutdown_agent(agent)

        # Unregister all go2/arm skill tools so the LLM stops offering them
        registry = app.get("registry")
        skills_dropped = 0
        if registry is not None and hasattr(registry, "list_tools"):
            for tool_name in list(registry.list_tools()):
                t = registry.get(tool_name)
                if t is not None and getattr(t, "_is_skill_wrapper", False):
                    try:
                        registry.unregister(tool_name)
                        skills_dropped += 1
                    except Exception:
                        pass

        # Clear app state references
        app["agent"] = None
        app["scene_graph"] = None
        app["skill_registry"] = None

        # Rebuild system prompt without hardware context
        engine = app.get("engine")
        if engine is not None:
            try:
                from vector_os_nano.vcli.prompt import build_system_prompt
                engine._system_prompt = build_system_prompt(agent=None, cwd=context.cwd)
            except Exception:
                pass

        return ToolResult(
            content=f"Simulation stopped. {summary}. Dropped {skills_dropped} skill tools."
        )

    def check_permissions(
        self, params: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        return PermissionResult(behavior="allow", reason="Simulation shutdown")
