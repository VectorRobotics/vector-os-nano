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
    return tools


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

    for param_name, param_def in raw_params.items():
        # Build clean JSON Schema property — exclude internal keys
        prop: dict[str, Any] = {}
        if "type" in param_def:
            prop["type"] = param_def["type"]
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

    return {
        "name": schema["name"],
        "description": schema.get("description", ""),
        "inputSchema": input_schema,
    }


async def handle_tool_call(
    agent: Agent, tool_name: str, arguments: dict[str, Any]
) -> str:
    """Handle an MCP tool call by routing to the appropriate skill or agent pipeline.

    If tool_name == "natural_language":
        Run agent.execute(arguments["instruction"]) through full pipeline.
    Else:
        Build a skill instruction string and run through agent.execute().

    Note: agent.execute() is synchronous, so this wraps it with asyncio.to_thread().

    Returns a text description of the result.
    """
    import asyncio

    if tool_name == "natural_language":
        instruction = arguments.get("instruction", "")
        result = await asyncio.to_thread(agent.execute, instruction)
        return _format_execution_result(instruction, result)

    instruction = _build_skill_instruction(tool_name, arguments)
    result = await asyncio.to_thread(agent.execute, instruction)
    return _format_execution_result(instruction, result)


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


def _format_execution_result(instruction: str, result: Any) -> str:
    """Format an ExecutionResult (or string) into a human-readable MCP response.

    If result is a string (chat/query response), return it directly.

    If result is an ExecutionResult, format as:
        Executed: <instruction>
        Status: <status>
        Steps: skill1(ok) -> skill2(ok) -> skill3(failed)
        Duration: 4.2s
    """
    if isinstance(result, str):
        return result

    # ExecutionResult
    from vector_os_nano.core.types import ExecutionResult

    if not isinstance(result, ExecutionResult):
        return str(result)

    lines: list[str] = [
        f"Executed: {instruction}",
        f"Status: {result.status}",
    ]

    if result.trace:
        step_parts: list[str] = []
        for trace in result.trace:
            ok = "ok" if trace.status == "success" else "failed"
            step_parts.append(f"{trace.skill_name}({ok})")
        lines.append("Steps: " + " -> ".join(step_parts))

        total_duration = sum(t.duration_sec for t in result.trace)
        lines.append(f"Duration: {total_duration:.1f}s")

    if result.failure_reason:
        lines.append(f"Failure: {result.failure_reason}")

    if result.message:
        lines.append(f"Message: {result.message}")

    return "\n".join(lines)
