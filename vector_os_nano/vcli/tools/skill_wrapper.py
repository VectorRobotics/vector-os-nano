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

import json
import math
from typing import Any

from vector_os_nano.vcli.tools.base import PermissionResult, ToolContext, ToolResult, live_agent

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
    "list": "array",
    "tuple": "array",
    "array": "array",
    "dict": "object",
    "object": "object",
}

# Declared parameter keys that are JSON Schema too: passed to the model unchanged.
_SCHEMA_KEYS: tuple[str, ...] = ("enum", "default", "items", "minItems", "maxItems")


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
        # The skill's GT post-condition template; the native runner auto-verifies it.
        self.verify_hint: str = str(getattr(skill, "verify_hint", "") or "")
        # The skill's declared world-model effects (the runner reads its manipulation intent).
        self.effects: dict[str, Any] = dict(getattr(skill, "effects", None) or {})
        # A declared housekeeping action (stop): it changes no task state.
        self.housekeeping: bool = getattr(skill, "housekeeping", False) is True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_motor(skill: Any) -> bool:
        """True if the skill moves the robot: its declared ``@skill(actuates=...)``.

        Fail closed: a skill that declares nothing actuates.
        """
        return bool(getattr(skill, "__skill_actuates__", True))

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
                prop.update({k: info[k] for k in _SCHEMA_KEYS if k in info})
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

    def _live(self, context: ToolContext) -> tuple[Any, Any] | str:
        """(agent, skill) to run NOW, or why there is none.

        The agent is the REPL's live one (``app_state['agent']``): a sim switch inside
        a turn rebinds it, while ``context.agent`` was fixed when the turn started.
        The skill is the agent's registered instance, so a ``skill_reload`` takes
        effect at once.
        """
        agent = live_agent(context)
        if agent is None:
            if "agent" in (context.app_state or {}):
                return "No robot is connected (the simulation was stopped). Use start_simulation first."
            agent = self._agent
        registry = getattr(agent, "_skill_registry", None)
        skill = registry.get(self.name) if registry is not None else None
        if skill is None and agent is self._agent:
            skill = self._skill
        if skill is None:
            return f"Skill '{self.name}' is not available on the connected robot."
        return agent, skill

    def _coerce(self, params: dict[str, Any]) -> dict[str, Any]:
        """A JSON-string array for an array parameter becomes the list it spells."""
        out = dict(params)
        for key, prop in self.input_schema.get("properties", {}).items():
            if prop.get("type") == "array" and isinstance(out.get(key), str):
                try:
                    value = json.loads(out[key])
                except ValueError:
                    continue
                if isinstance(value, list):
                    out[key] = value
        return out

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute the wrapped skill and translate SkillResult -> ToolResult."""
        live = self._live(context)
        if isinstance(live, str):
            return ToolResult(content=live, is_error=True)
        agent, skill = live
        result = skill.execute(self._coerce(params), agent._build_context())
        agent._sync_robot_state()

        # A copy: the skill's own (frozen) result is never mutated.
        result_data: dict[str, Any] = dict(getattr(result, "result_data", None) or {})

        if result.success:
            content = f"Skill '{self.name}' succeeded."
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
        error_msg = result.error_message or f"Skill '{self.name}' failed."
        diag = getattr(result, "diagnosis_code", None) or result_data.get("diagnosis")
        if diag:
            error_msg += f" ({diag})"
            hint = _RECOVERY_HINTS.get(diag)
            if hint:
                error_msg += f"\nSuggested: {hint}"
        if result_data:
            error_msg += f"\nData: {result_data}"
        if self._is_motor:
            post = self._get_post_state(agent)
            if post:
                result_data["robot_state_after"] = post
                error_msg += f"\nCurrent state: {post}"
        return ToolResult(content=error_msg, is_error=True, metadata=result_data)

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
        """True iff the agent is the body of a live in-process SimSession.

        A sim action has no real-world consequence, so motor skills are auto-allowed
        there (R2-5). Anything else — a ROS proxy that may drive a real robot, a bare
        agent, real hardware — keeps the confirmation.
        """
        from vector_os_nano.sim.session import is_sim_agent

        return is_sim_agent(self._agent)

    def check_permissions(
        self, params: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        """Motor skills require confirmation on real hardware; read-only are always allowed.

        R2-5: when the agent is a live SimSession's body, motor skills are
        auto-allowed — a sim action has no real-world consequence. Otherwise the
        confirmation requirement is preserved.
        """
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
