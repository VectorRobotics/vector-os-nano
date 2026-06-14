# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""SkillWrapperTool — wraps Vector OS Nano @skill instances as vcli Tool objects.

Any class that implements the Skill protocol (name, description, parameters,
preconditions, effects, execute) can be wrapped without importing real hardware.

Public API:
    SkillWrapperTool   — wraps a single skill instance
    wrap_skills(agent) — wrap all skills in agent._skill_registry
"""
from __future__ import annotations

import math
from typing import Any

from vector_os_nano.vcli.tools.base import PermissionResult, ToolContext, ToolResult

# Keywords that indicate a skill actuates motors / moves the robot.
# If any of these appear in the skill's preconditions or effects text,
# the skill is treated as a motor skill (requires permission, not concurrency-safe).
MOTOR_KEYWORDS: frozenset[str] = frozenset(
    {"arm", "gripper", "base", "motor", "joint", "move", "navigate"}
)

# The package all SIMULATED hardware adapters live under. A connected component
# whose module is this package (or a sub-module of it) is a simulation; anything
# else is real hardware. Precise package match (exact, or prefix + ".") so a name
# like "vector_os_nano.hardware.simulated_real" never false-matches "...sim".
_SIM_HW_PKG: str = "vector_os_nano.hardware.sim"

# JSON Schema type mapping from Python / skill type names.
_TYPE_MAP: dict[str, str] = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
}


class SkillWrapperTool:
    """Wraps a Vector OS Nano @skill instance as a vcli Tool.

    The wrapper is intentionally thin — it does not import any concrete skill
    class, so it works with any object that satisfies the Skill protocol.
    """

    # Marker so the registry/SimStopTool can identify (and unregister) skill tools.
    _is_skill_wrapper: bool = True

    def __init__(self, skill: Any, agent: Any) -> None:
        self.name: str = skill.name
        self.description: str = getattr(skill, "description", skill.name)
        self.input_schema: dict[str, Any] = self._build_schema(
            getattr(skill, "parameters", {})
        )
        self._skill = skill
        self._agent = agent
        self._is_motor: bool = self._detect_motor(skill)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_motor(skill: Any) -> bool:
        """Return True if skill involves motor/actuator operations.

        Scans both preconditions (list) and effects (dict) for motor keywords.
        """
        # Batch 3 #14: an explicit declaration beats keyword sniffing — the
        # Skill protocol's ``is_motor`` is authoritative when present.
        declared = getattr(skill, "is_motor", None)
        if declared is not None:
            return bool(declared)
        preconditions_text = " ".join(str(p) for p in getattr(skill, "preconditions", []))
        effects_text = str(getattr(skill, "effects", {}))
        description_text = str(getattr(skill, "description", ""))
        combined = (preconditions_text + " " + effects_text + " " + description_text).lower()
        return any(kw in combined for kw in MOTOR_KEYWORDS)

    @staticmethod
    def _build_schema(parameters: dict[str, Any]) -> dict[str, Any]:
        """Convert a skill parameters dict to a JSON Schema object."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, info in parameters.items():
            prop: dict[str, Any] = {}

            if isinstance(info, dict):
                raw_type = info.get("type", "string")
                prop["type"] = _TYPE_MAP.get(str(raw_type), "string")
                if "description" in info:
                    prop["description"] = info["description"]
                # Batch 3 #13: the LLM must SEE the legal value set and the
                # default — stripping them invited invalid enum values that
                # the skills then silently coerced.
                if "enum" in info:
                    prop["enum"] = list(info["enum"])
                if "default" in info:
                    prop["default"] = info["default"]
                # A parameter is required when it has no default and is not
                # explicitly marked required=False.
                has_default = "default" in info
                explicitly_required = info.get("required", True)
                if not has_default and explicitly_required:
                    required.append(param_name)
            else:
                # Bare type string (e.g. parameters = {"x": "float"})
                prop["type"] = _TYPE_MAP.get(str(info), "string")
                required.append(param_name)

            properties[param_name] = prop

        return {"type": "object", "properties": properties, "required": required}

    # ------------------------------------------------------------------
    # Tool Protocol implementation
    # ------------------------------------------------------------------

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute the wrapped skill and translate SkillResult -> ToolResult."""
        agent = context.agent if context.agent is not None else self._agent
        auto_steps = getattr(self._skill, "__skill_auto_steps__", None)
        if auto_steps:
            # Multi-step skill (e.g. pick = scan->detect->pick): run through the
            # agent so the auto_steps prerequisites execute. Returns ExecutionResult.
            result = agent.execute_skill(self.name, params)
        else:
            skill_ctx = agent._build_context()
            result = self._skill.execute(params, skill_ctx)
        agent._sync_robot_state()

        if result.success:
            content = f"Skill '{self.name}' succeeded."
            result_data: dict[str, Any] = getattr(result, "result_data", None) or {}
            if not result_data:
                # auto_steps skills return an ExecutionResult; the skill's own
                # output lives in the last trace step, not on the top-level result.
                _trace = getattr(result, "trace", None)
                if _trace:
                    result_data = getattr(_trace[-1], "result_data", None) or {}
            if result_data:
                content += f"\nData: {result_data}"
            # Append robot state after motor skills for verification
            if self._is_motor:
                post = self._get_post_state(agent)
                if post:
                    result_data["robot_state_after"] = post
                    pos = post.get("position", [])
                    room = post.get("room", "")
                    if pos:
                        content += f"\nState: pos=({pos[0]}, {pos[1]})"
                    if room:
                        content += f" room={room}"
            return ToolResult(content=content, metadata=result_data)

        # Failure with diagnosis + recovery hint
        error_msg = (
            getattr(result, "error_message", None)
            or getattr(result, "failure_reason", None)
            or f"Skill '{self.name}' failed."
        )
        diag = getattr(result, "diagnosis_code", None)
        if diag:
            error_msg += f" ({diag})"
            hint = _RECOVERY_HINTS.get(diag)
            if hint:
                error_msg += f"\nSuggested: {hint}"
        if self._is_motor:
            post = self._get_post_state(agent)
            if post:
                error_msg += f"\nCurrent state: {post}"
        return ToolResult(content=error_msg, is_error=True)

    def _get_post_state(self, agent: Any) -> dict[str, Any] | None:
        """Snapshot robot state after skill execution."""
        base = getattr(agent, "_base", None)
        if base is None:
            return None
        try:
            pos = base.get_position()
            heading = base.get_heading()
            state: dict[str, Any] = {
                "position": [round(pos[0], 1), round(pos[1], 1), round(pos[2], 2)],
                "heading_deg": round(math.degrees(heading)),
            }
            sg = getattr(agent, "_spatial_memory", None)
            if sg and hasattr(sg, "nearest_room"):
                state["room"] = sg.nearest_room(pos[0], pos[1])
            return state
        except Exception:
            return None

    def _robot_is_simulated(self) -> bool:
        """Return True when every connected hardware module is a simulated adapter.

        R2-5 (auto-allow motor skills in sim): a sim robot has no real-world
        consequence, so motor skills are safe to auto-allow. Real-hardware adapters
        live outside ``vector_os_nano.hardware.sim.*`` — if any connected component
        resolves to a non-sim module path, we return False and motor skills keep
        their confirmation requirement. World-agnostic: duck-types the agent's arm,
        base, and gripper attributes without importing any concrete class.
        """
        agent = self._agent
        if agent is None:
            return False
        # SAFETY: require ALL present hardware to be simulated — return False on the
        # FIRST real (non-sim) component. ANY-semantics would auto-allow a motor skill
        # on a mixed agent (e.g. sim arm + real base), actuating real hardware without
        # confirmation. Also require >=1 component present (a bare agent is not "sim").
        found_sim = False
        for attr in ("_arm", "_base", "_gripper"):
            hw = getattr(agent, attr, None)
            if hw is None:
                continue
            mod = type(hw).__module__
            if not (mod == _SIM_HW_PKG or mod.startswith(_SIM_HW_PKG + ".")):
                return False  # any real component -> require confirmation
            found_sim = True
        return found_sim

    def check_permissions(
        self, params: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        """Motor skills require confirmation on real hardware; read-only are always allowed.

        R2-5: when the connected robot is simulated (all hardware paths start with
        ``vector_os_nano.hardware.sim``), motor skills are auto-allowed — a sim action
        has no real-world consequence. On real hardware the confirmation requirement
        is preserved.
        """
        # Batch 3 #14: E-STOP-class skills (confirm_exempt=True) are NEVER
        # gated behind a prompt — a stop waiting for user confirmation
        # violates the E-stop-independence safety rule.
        if getattr(self._skill, "confirm_exempt", False):
            return PermissionResult("allow")
        if self._is_motor and not self._robot_is_simulated():
            return PermissionResult("ask")
        return PermissionResult("allow")

    def is_read_only(self, params: dict[str, Any]) -> bool:
        return not self._is_motor

    def is_concurrency_safe(self, params: dict[str, Any]) -> bool:
        return not self._is_motor


# Recovery hints for known diagnosis codes (shown to LLM on failure)
_RECOVERY_HINTS: dict[str, str] = {
    "no_base": "No robot connected. Use start_simulation tool first.",
    "unknown_room": "Room not found. Use scene_graph_query(query_type='rooms') to list available rooms.",
    "room_not_explored": "Room not explored yet. Run the explore skill first.",
    "navigation_failed": "Navigation failed. Use nav_state tool to check if nav stack is running.",
    "no_vlm": "VLM not available. Check if Ollama is running (bash: pgrep ollama).",
    "camera_failed": "Camera not connected. Use robot_status tool to check hardware.",
    "unknown_skill": "Skill not found. Use robot_status to list available skills.",
}


# ---------------------------------------------------------------------------
# Bulk factory
# ---------------------------------------------------------------------------


def wrap_skills(agent: Any) -> list[SkillWrapperTool]:
    """Wrap all skills registered in *agent._skill_registry* as Tool instances.

    Skills for which the registry returns None are silently skipped.
    """
    tools: list[SkillWrapperTool] = []
    registry = agent._skill_registry
    for skill_name in registry.list_skills():
        skill = registry.get(skill_name)
        if skill is not None:
            tools.append(SkillWrapperTool(skill, agent))
    return tools
