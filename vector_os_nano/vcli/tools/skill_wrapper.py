"""SkillWrapperTool — wraps Vector OS Nano @skill instances as vcli Tool objects.

Any class that implements the Skill protocol (name, description, parameters,
preconditions, effects, execute) can be wrapped without importing real hardware.

Public API:
    SkillWrapperTool   — wraps a single skill instance
    wrap_skills(agent) — wrap all skills in agent._skill_registry
"""
from __future__ import annotations

from typing import Any

from vector_os_nano.vcli.tools.base import PermissionResult, ToolContext, ToolResult

# Keywords that indicate a skill actuates motors / moves the robot.
# If any of these appear in the skill's preconditions or effects text,
# the skill is treated as a motor skill (requires permission, not concurrency-safe).
MOTOR_KEYWORDS: frozenset[str] = frozenset(
    {"arm", "gripper", "base", "motor", "joint", "move", "navigate"}
)

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
        preconditions_text = " ".join(str(p) for p in getattr(skill, "preconditions", []))
        effects_text = str(getattr(skill, "effects", {}))
        combined = (preconditions_text + " " + effects_text).lower()
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
        skill_ctx = agent._build_context()
        result = self._skill.execute(params, skill_ctx)
        agent._sync_robot_state()

        if result.success:
            content = f"Skill '{self.name}' succeeded."
            result_data: dict[str, Any] = result.result_data or {}
            if result_data:
                content += f"\nData: {result_data}"
            return ToolResult(content=content, metadata=result_data)

        error_msg = result.error_message or f"Skill '{self.name}' failed."
        return ToolResult(content=error_msg, is_error=True)

    def check_permissions(
        self, params: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        """Motor skills require confirmation; read-only skills are auto-allowed."""
        return PermissionResult("ask" if self._is_motor else "allow")

    def is_read_only(self, params: dict[str, Any]) -> bool:
        return not self._is_motor

    def is_concurrency_safe(self, params: dict[str, Any]) -> bool:
        return not self._is_motor


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
