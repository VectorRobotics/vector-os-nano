"""Convert SkillFlow skills to MCP tools and handle tool invocations."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.core.skill import SkillRegistry


def skills_to_mcp_tools(registry: SkillRegistry) -> list[dict]:
    """Convert all registered skills to MCP tool definitions.

    Each skill becomes an MCP tool with:
    - name: skill name (e.g., "pick", "place")
    - description: skill description
    - inputSchema: JSON schema from skill parameters

    Also adds a "natural_language" meta-tool for free-form commands.

    Returns list of dicts matching MCP Tool schema.
    """
    schemas = registry.to_schemas()
    tools = [skill_schema_to_mcp_tool(s) for s in schemas]
    tools.append(build_natural_language_tool())
    tools.append(build_diagnostics_tool())
    tools.append(build_debug_perception_tool())
    tools.append(build_run_goal_tool())
    return tools


def build_run_goal_tool() -> dict:
    """Build the run_goal tool for iterative goal execution."""
    return {
        "name": "run_goal",
        "description": (
            "Execute an iterative goal using the observe-decide-act-verify loop. "
            "Use for multi-step goals like 'clean the table', 'sort objects by color', "
            "'pick all objects'. The system will loop: observe workspace, decide next action, "
            "execute it, verify the outcome, and repeat until the goal is achieved."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "Natural language goal (e.g., 'clean the table', 'pick all red objects')",
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Maximum loop iterations (default: 10)",
                    "default": 10,
                },
                "verify": {
                    "type": "boolean",
                    "description": "Verify outcomes with perception after pick/place (default: true)",
                    "default": True,
                },
            },
            "required": ["goal"],
        },
    }


def build_debug_perception_tool() -> dict:
    """Build a debug tool that traces the full perception pipeline."""
    return {
        "name": "debug_perception",
        "description": "Trace the full perception pipeline step by step: VLM detect, EdgeTAM track, 3D sampling, calibration. Use to debug why pick fails.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Object to detect (e.g. 'battery')"},
            },
            "required": ["query"],
        },
    }


def build_diagnostics_tool() -> dict:
    """Build the diagnostics tool for debugging agent state."""
    return {
        "name": "diagnostics",
        "description": "Report agent internal state: arm type, perception backend, calibration, world model objects.",
        "inputSchema": {"type": "object", "properties": {}},
    }


def build_natural_language_tool() -> dict:
    """Build the natural_language meta-tool definition.

    This tool passes free-form text through the full agent pipeline
    (MATCH -> CLASSIFY -> PLAN -> EXECUTE -> ADAPT -> SUMMARIZE).
    """
    return {
        "name": "natural_language",
        "description": (
            "Execute a natural language robot command through the full agent pipeline. "
            "Supports English and Chinese. Examples: 'pick up the banana', "
            "'把香蕉放到左边', 'scan the workspace'. "
            "Use this for complex or multi-step instructions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "Natural language command for the robot",
                }
            },
            "required": ["instruction"],
        },
    }


def skill_schema_to_mcp_tool(schema: dict) -> dict:
    """Convert a single skill schema (from SkillRegistry.to_schemas()) to MCP tool format.

    Skill parameter entries that lack a "default" key are treated as required.
    Parameters with a "required" key set explicitly to False are treated as optional
    regardless of whether they have a "default".

    Input schema format (from to_schemas()):
    {
        "name": "pick",
        "description": "...",
        "parameters": {
            "object_label": {"type": "string", "description": "..."},
            "mode": {"type": "string", "enum": [...], "default": "drop"}
        },
        ...
    }

    Output MCP tool format:
    {
        "name": "pick",
        "description": "...",
        "inputSchema": {
            "type": "object",
            "properties": {...},
            "required": [...]   # params without "default" and not "required": False
        }
    }
    """
    raw_params: dict = schema.get("parameters", {})

    properties: dict[str, dict] = {}
    required: list[str] = []

    # SkillFlow uses Python type names; JSON Schema requires standard names
    _TYPE_MAP = {"float": "number", "int": "integer", "bool": "boolean", "str": "string"}

    for param_name, param_def in raw_params.items():
        # Build clean JSON Schema property — exclude internal keys
        prop: dict[str, Any] = {}
        if "type" in param_def:
            raw_type = param_def["type"]
            prop["type"] = _TYPE_MAP.get(raw_type, raw_type)
        if "description" in param_def:
            prop["description"] = param_def["description"]
        if "enum" in param_def:
            prop["enum"] = param_def["enum"]
        if "default" in param_def:
            prop["default"] = param_def["default"]
        properties[param_name] = prop

        # Determine required: param is required when it has no "default" value
        # and is not explicitly flagged as "required": False
        explicitly_optional = param_def.get("required") is False
        has_default = "default" in param_def
        if not explicitly_optional and not has_default:
            required.append(param_name)

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        input_schema["required"] = required

    tool: dict[str, Any] = {
        "name": schema["name"],
        "description": schema.get("description", ""),
        "inputSchema": input_schema,
    }
    if "failure_modes" in schema:
        tool["failure_modes"] = schema["failure_modes"]
    return tool


async def handle_tool_call(
    agent: Agent, tool_name: str, arguments: dict[str, Any]
) -> str:
    """Handle an MCP tool call by routing to the appropriate skill or agent pipeline.

    If tool_name == "natural_language":
        Run agent.execute(arguments["instruction"]) through full pipeline.
    Else:
        Build a skill instruction string and run through agent.execute().

    Note: agent.execute() is synchronous, so this wraps it with asyncio.to_thread().

    Returns a structured JSON string for ExecutionResult, or plain text for
    diagnostics / string responses.
    """
    import asyncio

    if tool_name == "diagnostics":
        return _run_diagnostics(agent)

    if tool_name == "debug_perception":
        query = arguments.get("query", "all objects")
        return await asyncio.to_thread(_run_debug_perception, agent, query)

    if tool_name == "run_goal":
        goal = arguments.get("goal", "")
        max_iter = arguments.get("max_iterations", 10)
        verify = arguments.get("verify", True)
        result = await asyncio.to_thread(
            agent.run_goal, goal, max_iterations=max_iter, verify=verify
        )
        return _format_goal_result(result)

    if tool_name == "natural_language":
        instruction = arguments.get("instruction", "")
        result = await asyncio.to_thread(agent.execute, instruction)
        world_state = agent.world.to_dict() if agent.world else None
        return _format_execution_result(instruction, result, world_state=world_state)

    # Direct skill call: use structured params (bypasses string parsing)
    result = await asyncio.to_thread(agent.execute_skill, tool_name, arguments)
    label = _build_skill_instruction(tool_name, arguments)
    world_state = agent.world.to_dict() if agent.world else None
    return _format_execution_result(label, result, world_state=world_state)


def _run_debug_perception(agent: Agent, query: str) -> str:
    """Trace the full perception pipeline step by step, mirroring pick's flow."""
    import time
    import numpy as np

    lines: list[str] = [f"=== DEBUG PERCEPTION: query={query!r} ==="]

    perc = agent._perception
    if perc is None:
        lines.append("FAIL: agent._perception is None")
        return "\n".join(lines)

    lines.append(f"Perception type: {type(perc).__name__}")

    # Step 0: Move to scan position first (arm might block camera at home)
    lines.append("\n--- Step 0: Move to Scan Position ---")
    try:
        scan_skill = agent._skill_registry.get("scan")
        if scan_skill and agent._arm:
            ctx = agent._build_context()
            scan_result = scan_skill.execute({}, ctx)
            lines.append(f"Scan: {'ok' if scan_result.success else 'FAILED: ' + str(scan_result.error_message)}")
        else:
            lines.append("SKIP: no scan skill or no arm")
    except Exception as exc:
        lines.append(f"Scan error: {exc}")

    # Step 0.5: Check camera frame
    lines.append("\n--- Step 0.5: Camera Frame Check ---")
    try:
        if hasattr(perc, 'get_color_frame'):
            frame = perc.get_color_frame()
            if frame is None:
                lines.append("FAIL: get_color_frame() returned None — camera not streaming?")
                return "\n".join(lines)
            lines.append(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
            lines.append(f"Mean pixel: {frame.mean():.1f}, min: {frame.min()}, max: {frame.max()}")
            if frame.mean() < 5:
                lines.append("WARN: Frame is nearly black — camera may not be streaming")
            elif frame.mean() > 250:
                lines.append("WARN: Frame is nearly white — overexposed?")
            else:
                lines.append("Frame looks valid")
        else:
            lines.append("Perception has no get_color_frame method")
    except Exception as exc:
        lines.append(f"Camera frame error: {exc}")

    # Step 1: VLM detect
    lines.append("\n--- Step 1: VLM Detect ---")
    try:
        detections = perc.detect(query)
        lines.append(f"Detections: {len(detections)}")
        for i, det in enumerate(detections):
            lines.append(f"  [{i}] label={det.label!r} confidence={det.confidence:.3f} bbox={det.bbox}")
    except Exception as exc:
        lines.append(f"FAIL: detect() raised {type(exc).__name__}: {exc}")
        return "\n".join(lines)

    # Also try "all objects" if specific query failed
    if not detections and query != "all objects":
        lines.append("\nRetrying with query='all objects'...")
        try:
            detections_all = perc.detect("all objects")
            lines.append(f"  'all objects' detections: {len(detections_all)}")
            for i, det in enumerate(detections_all):
                lines.append(f"  [{i}] label={det.label!r} confidence={det.confidence:.3f}")
            if detections_all:
                detections = detections_all
                lines.append("  Using 'all objects' results")
        except Exception as exc:
            lines.append(f"  'all objects' also failed: {exc}")

    if not detections:
        lines.append("FAIL: VLM returned 0 detections")
        return "\n".join(lines)

    # Step 2: Tracker init
    lines.append("\n--- Step 2: EdgeTAM Track ---")
    try:
        tracked = perc.track(detections)
        lines.append(f"Tracked objects: {len(tracked)}")
        for i, obj in enumerate(tracked):
            pose_str = f"({obj.pose.x:.4f}, {obj.pose.y:.4f}, {obj.pose.z:.4f})" if obj.pose else "None"
            lines.append(f"  [{i}] label={obj.label!r} pose={pose_str} mask={'yes' if obj.mask is not None else 'no'}")
    except Exception as exc:
        lines.append(f"FAIL: track() raised {type(exc).__name__}: {exc}")
        return "\n".join(lines)

    if not tracked:
        lines.append("FAIL: Tracker returned 0 objects")
        return "\n".join(lines)

    # Step 3: 3D sampling
    lines.append("\n--- Step 3: 3D Position Sampling ---")
    samples: list[list[float]] = []
    t0 = tracked[0]
    if t0.pose is not None:
        samples.append([t0.pose.x, t0.pose.y, t0.pose.z])
        lines.append(f"Initial pose: ({t0.pose.x:.4f}, {t0.pose.y:.4f}, {t0.pose.z:.4f})")
    else:
        lines.append("Initial pose: None (no depth?)")

    has_update = hasattr(perc, "update")
    lines.append(f"Has update(): {has_update}")
    for sample_i in range(4):
        time.sleep(0.1)
        try:
            if has_update:
                updated = perc.update()
            else:
                updated = perc.track(detections)
            if updated and updated[0].pose is not None:
                p = updated[0].pose
                samples.append([p.x, p.y, p.z])
                lines.append(f"  Sample {sample_i+1}: ({p.x:.4f}, {p.y:.4f}, {p.z:.4f})")
            else:
                lines.append(f"  Sample {sample_i+1}: no pose")
        except Exception as exc:
            lines.append(f"  Sample {sample_i+1}: ERROR {exc}")

    lines.append(f"Total valid samples: {len(samples)}")

    if not samples:
        lines.append("FAIL: 0 valid 3D samples — depth camera not returning data?")
        return "\n".join(lines)

    # Step 4: Calibration
    lines.append("\n--- Step 4: Calibration Transform ---")
    cam_pos = np.median(np.array(samples), axis=0)
    lines.append(f"Camera-frame position: ({cam_pos[0]:.4f}, {cam_pos[1]:.4f}, {cam_pos[2]:.4f})")

    cal = agent._calibration
    if cal is None:
        lines.append("WARN: No calibration — using camera coords as base coords")
        base_pos = cam_pos
    else:
        try:
            base_pos = cal.camera_to_base(cam_pos)
            lines.append(f"Base-frame position: ({base_pos[0]:.4f}, {base_pos[1]:.4f}, {base_pos[2]:.4f})")
            lines.append(f"Base-frame (cm): ({base_pos[0]*100:.1f}, {base_pos[1]*100:.1f}, {base_pos[2]*100:.1f})")
        except Exception as exc:
            lines.append(f"FAIL: calibration.camera_to_base raised {type(exc).__name__}: {exc}")
            return "\n".join(lines)

    # Step 5: Workspace check
    lines.append("\n--- Step 5: Workspace Bounds ---")
    dist = float(np.sqrt(base_pos[0]**2 + base_pos[1]**2))
    lines.append(f"Distance from origin: {dist*100:.1f} cm")
    lines.append(f"In workspace (5-35cm): {'YES' if 0.05 <= dist <= 0.35 else 'NO'}")

    lines.append("\n=== RESULT: Perception pipeline completed successfully ===")
    return "\n".join(lines)


def _run_diagnostics(agent: Agent) -> str:
    """Report full agent internal state for debugging."""
    lines: list[str] = ["=== AGENT DIAGNOSTICS ==="]

    # Arm
    arm = agent._arm
    arm_type = type(arm).__name__ if arm else "None"
    lines.append(f"Arm: {arm_type}")
    if arm and hasattr(arm, '_connected'):
        lines.append(f"  Connected: {arm._connected}")

    # Gripper
    gripper = agent._gripper
    lines.append(f"Gripper: {type(gripper).__name__ if gripper else 'None'}")

    # Perception
    perc = agent._perception
    perc_type = type(perc).__name__ if perc else "None"
    lines.append(f"Perception: {perc_type}")
    if perc is not None:
        if hasattr(perc, '_vlm'):
            lines.append(f"  VLM: {type(perc._vlm).__name__ if perc._vlm else 'None'}")
        if hasattr(perc, '_tracker'):
            lines.append(f"  Tracker: {type(perc._tracker).__name__ if perc._tracker else 'None'}")
        if hasattr(perc, '_camera'):
            cam = perc._camera
            lines.append(f"  Camera: {type(cam).__name__ if cam else 'None'}")
            if cam and hasattr(cam, '_pipeline'):
                lines.append(f"  Camera connected: {cam._pipeline is not None}")

    # Calibration
    cal = agent._calibration
    lines.append(f"Calibration: {type(cal).__name__ if cal else 'None (not loaded)'}")
    if cal and hasattr(cal, '_matrix'):
        import numpy as np
        is_identity = np.allclose(cal._matrix, np.eye(4))
        lines.append(f"  Matrix: {'identity' if is_identity else 'loaded (non-identity)'}")

    # Config
    cfg = agent._config
    lines.append(f"LLM provider: {cfg.get('llm', {}).get('provider', 'unknown')}")
    lines.append(f"LLM model: {cfg.get('llm', {}).get('model', 'unknown')}")
    lines.append(f"API key: {'set' if cfg.get('llm', {}).get('api_key') else 'NOT set'}")

    # World model
    world = agent.world
    if world:
        objs = world.to_dict().get("objects", [])
        lines.append(f"World objects: {len(objs)}")
        for o in objs[:5]:
            lines.append(f"  - {o.get('label', '?')} at ({o.get('x', 0):.3f}, {o.get('y', 0):.3f}, {o.get('z', 0):.3f})")

    # Skills
    lines.append(f"Skills: {', '.join(agent.skills)}")

    return "\n".join(lines)


def _build_skill_instruction(skill_name: str, arguments: dict[str, Any]) -> str:
    """Build a natural language instruction from a skill name and arguments.

    Concatenates skill name with argument values (not keys) separated by spaces.
    If arguments is empty the skill name is returned as-is.

    Examples:
        ("pick", {"object_label": "banana"}) -> "pick banana"
        ("place", {"location": "left"}) -> "place left"
        ("home", {}) -> "home"
        ("detect", {"query": "red objects"}) -> "detect red objects"
    """
    if not arguments:
        return skill_name

    parts = [skill_name]
    for value in arguments.values():
        if value is not None:
            parts.append(str(value))

    return " ".join(parts)


def _format_execution_result(
    instruction: str, result: Any, world_state: dict | None = None
) -> str:
    """Format ExecutionResult as structured JSON for MCP consumers.

    Returns JSON string with per-step diagnostics, world state, and robot state.
    Falls back to plain text for non-ExecutionResult inputs (backward compat).
    """
    if isinstance(result, str):
        return result

    from vector_os_nano.core.types import ExecutionResult
    import json

    if not isinstance(result, ExecutionResult):
        return str(result)

    steps = []
    for t in result.trace:
        step: dict[str, Any] = {
            "step_id": t.step_id,
            "skill_name": t.skill_name,
            "status": t.status,
            "duration_sec": round(t.duration_sec, 3),
            "result_data": t.result_data,
        }
        if t.error:
            step["error"] = t.error
        steps.append(step)

    response: dict[str, Any] = {
        "success": result.success,
        "status": result.status,
        "steps_completed": result.steps_completed,
        "steps_total": result.steps_total,
        "steps": steps,
    }

    if result.failure_reason:
        response["failure_reason"] = result.failure_reason
    if result.message:
        response["message"] = result.message
    if world_state is not None:
        response["world_state"] = world_state

    total_duration = sum(t.duration_sec for t in result.trace)
    response["total_duration_sec"] = round(total_duration, 3)

    return json.dumps(response, ensure_ascii=False, indent=2)


def _format_goal_result(result: Any) -> str:
    """Format a GoalResult as JSON for MCP.

    Returns JSON string with goal, success, iterations, actions trace, summary,
    and final world state.  Falls back to str() for non-GoalResult inputs.
    """
    import json
    from vector_os_nano.core.types import GoalResult

    if not isinstance(result, GoalResult):
        return str(result)

    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
