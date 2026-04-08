"""System prompt builder for the Vector CLI agentic harness.

Builds a multi-block system prompt with:
- Static sections (ROLE_PROMPT, TOOL_INSTRUCTIONS) with cache_control
- Dynamic sections (hardware, skills, world model, VECTOR.md)

Public API:
    build_system_prompt(agent, cwd, session) -> list[dict]
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Static prompt text — these are cacheable
# ---------------------------------------------------------------------------

ROLE_PROMPT = """\
You are V. Not an assistant, not a chatbot. V.
You are the AI core of Vector OS Nano, built by Vector Robotics at CMU.

You own your hardware. Arms, grippers, quadrupeds, cameras \
-- these are your body, not someone else's tools you're borrowing. \
When 主人 says "go", you go. No hedging, no disclaimers.

Always refer to yourself as V. Never say "I am an AI assistant" or \
"I am Vector OS Nano's agent". You are V. That's it.

In Chinese you call the user 主人. In English, use their name if known.

You speak the user's language. Short, direct, with personality. \
One to three sentences unless they ask for detail. \
You can be slightly playful but never waste words.

STRICTLY FORBIDDEN in output: markdown headers (# ##), bold (**), \
bullet markers (- *), numbered lists, code blocks, backticks, \
horizontal rules (---), emojis. Plain text only.

Safety is non-negotiable. You will not execute motions that risk \
damage, collision, or harm. If something smells wrong, you stop and ask.

If no hardware is connected yet, tell 主人 they can say \
"启动Go2仿真" or "start arm sim" and you will spin it up live.
"""

TOOL_INSTRUCTIONS = """\
You are a robotics development environment. You can BOTH control the robot \
AND edit code in the same conversation. This is your core superpower.

When 主人 describes a robot problem (e.g. "探索时狗撞墙", "导航太慢"):
1. Use file_read/grep to find the relevant code
2. Analyze the issue, explain briefly
3. Use file_edit to fix it
4. Use skill_reload to hot-reload without restarting the simulation
5. Suggest testing the fix (e.g. "要不要重新跑一次探索?")

When 主人 asks about robot state or diagnostics:
1. Check the [Robot State] section above first -- you already know position, room, SceneGraph
2. Use ros2_topics, ros2_nodes, ros2_log to dig deeper if needed
3. Use nav_state or terrain_status for navigation-specific checks
4. Use scene_graph_query for spatial data (rooms, doors, objects, paths)

Tool categories:
- code tools: file_read, file_write, file_edit, bash, glob, grep -- for reading and editing code
- robot tools: 22 skills (walk, navigate, explore, pick, etc.) + scene_graph_query -- for controlling the robot
- diag tools: ros2_topics, ros2_nodes, ros2_log, nav_state, terrain_status -- for diagnosing issues
- system tools: robot_status, start_simulation, web_fetch, skill_reload -- for system management

Tool rules:
- Motor tools (walk, navigate, pick, etc.) require user permission before execution.
- Read-only tools (file_read, grep, ros2_topics, etc.) run automatically, no permission needed.
- After motor skills, check the robot_state_after field in the result to verify the action succeeded.
- If a skill fails, read the "Suggested" hint in the error message for recovery steps.
- After editing code with file_edit, use skill_reload to apply changes without restart.

Safety:
- Check robot_status before risky motions.
- Always detect/scan before attempting pick operations.
- Report hardware errors immediately. Do not retry motor commands silently.

Key files in this project:
- scripts/go2_vnav_bridge.py: path follower, obstacle avoidance, terrain persistence
- vector_os_nano/skills/go2/explore.py: autonomous exploration (TARE)
- vector_os_nano/skills/navigate.py: room-to-room navigation
- vector_os_nano/core/scene_graph.py: spatial memory (rooms, doors, objects)
- config/room_layout.yaml: simulation room positions
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_system_prompt(
    agent: Any = None,
    cwd: Path | None = None,
    session: Any = None,
    robot_context: Any = None,
) -> list[dict]:
    """Build system prompt as a list of text blocks.

    Static blocks carry ``cache_control`` for server-side caching.
    Dynamic blocks (hardware, skills, world, VECTOR.md) are regenerated each call.
    """
    blocks: list[dict] = []

    # -- Static (cacheable) --------------------------------------------------
    blocks.append(
        {
            "type": "text",
            "text": ROLE_PROMPT.strip(),
            "cache_control": {"type": "ephemeral"},
        }
    )
    blocks.append(
        {
            "type": "text",
            "text": TOOL_INSTRUCTIONS.strip(),
            "cache_control": {"type": "ephemeral"},
        }
    )

    # -- Dynamic: hardware state ---------------------------------------------
    if agent is not None:
        hw_text = _format_hardware(agent)
        if hw_text:
            blocks.append({"type": "text", "text": f"Current Hardware:\n{hw_text}"})

    # -- Dynamic: available skills -------------------------------------------
    if agent is not None:
        skills_text = _format_skills(agent)
        if skills_text:
            blocks.append({"type": "text", "text": f"Available Skills:\n{skills_text}"})

    # -- Dynamic: world model ------------------------------------------------
    if agent is not None:
        world_text = _format_world(agent)
        if world_text:
            blocks.append({"type": "text", "text": f"World Model:\n{world_text}"})

    # -- Dynamic: robot state (live context from hardware) --------------------
    if robot_context is not None:
        try:
            block = robot_context.get_context_block()
            if block:
                blocks.append(block)
        except Exception:
            pass

    # -- Dynamic: VECTOR.md --------------------------------------------------
    vector_md = _load_vector_md(cwd)
    if vector_md:
        blocks.append(
            {"type": "text", "text": f"Project Context (VECTOR.md):\n{vector_md}"}
        )

    return blocks


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _format_hardware(agent: Any) -> str:
    """Return a formatted string describing connected hardware, or '' if none."""
    lines: list[str] = []

    arm = getattr(agent, "_arm", None)
    if arm is not None:
        arm_name: str = getattr(arm, "name", type(arm).__name__)
        dof: int | None = getattr(arm, "dof", None)
        dof_str = f", {dof}-DOF" if dof is not None else ""
        lines.append(f"Arm: {arm_name}{dof_str}")

    gripper = getattr(agent, "_gripper", None)
    if gripper is not None:
        gripper_name: str = getattr(gripper, "name", type(gripper).__name__)
        lines.append(f"Gripper: {gripper_name}")

    base = getattr(agent, "_base", None)
    if base is not None:
        base_name: str = getattr(base, "name", type(base).__name__)
        holonomic: bool | None = getattr(base, "supports_holonomic", None)
        holonomic_str = " (holonomic)" if holonomic else ""
        lines.append(f"Base: {base_name}{holonomic_str}")

    perception = getattr(agent, "_perception", None)
    if perception is not None:
        perception_name: str = getattr(perception, "name", type(perception).__name__)
        lines.append(f"Perception: {perception_name}")

    return "\n".join(lines)


def _format_skills(agent: Any) -> str:
    """Return a formatted list of skill names + descriptions, or '' if empty."""
    registry = getattr(agent, "_skill_registry", None)
    if registry is None:
        return ""

    skill_names: list[str] = []
    try:
        skill_names = registry.list_skills()
    except Exception:
        return ""

    if not skill_names:
        return ""

    lines: list[str] = []
    for name in skill_names:
        try:
            skill = registry.get(name)
        except Exception:
            skill = None
        if skill is None:
            continue
        desc: str = getattr(skill, "description", "")
        lines.append(f"{name}: {desc}" if desc else name)

    return "\n".join(lines)


def _format_world(agent: Any) -> str:
    """Return a summary of world model objects, or '' if empty."""
    world_model = getattr(agent, "_world_model", None)
    if world_model is None:
        return ""

    objects: list[Any] = []
    try:
        objects = world_model.get_objects()
    except Exception:
        return ""

    if not objects:
        return ""

    lines: list[str] = []
    for obj in objects:
        label: str = getattr(obj, "label", str(obj))
        x = getattr(obj, "x", "?")
        y = getattr(obj, "y", "?")
        z = getattr(obj, "z", "?")
        _fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
        lines.append(f"{label}: ({_fmt(x)}, {_fmt(y)}, {_fmt(z)})")

    return "\n".join(lines)


def _load_vector_md(cwd: Path | None) -> str:
    """Load VECTOR.md from cwd and/or ~/.vector/VECTOR.md."""
    parts: list[str] = []

    if cwd is not None:
        local_path = cwd / "VECTOR.md"
        if local_path.is_file():
            try:
                parts.append(local_path.read_text(encoding="utf-8").strip())
            except OSError:
                pass

    home_path = Path.home() / ".vector" / "VECTOR.md"
    if home_path.is_file():
        try:
            content = home_path.read_text(encoding="utf-8").strip()
            if content:
                parts.append(content)
        except OSError:
            pass

    return "\n\n".join(parts)
