"""System prompts for Vector OS LLM task planner.

Two-stage prompt system:
1. CLASSIFY prompt — fast intent classification (chat vs task vs query vs direct)
2. PLAN prompt — task decomposition into skill sequence

Both use the same LLM (Haiku) but with different system prompts.
"""
from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# Stage 1: Intent Classification
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM_PROMPT = """\
You are an intent classifier for a robot arm control system.

Classify the user's message into exactly ONE category. Respond with ONLY the category word, nothing else.

Categories:
- chat: Questions, greetings, conversation, asking about capabilities, opinions, explanations
  Examples: "你好", "你能做什么", "为什么失败了", "桌上有什么颜色的东西", "hello", "what can you do"
- task: Commands to make the robot DO something (pick, place, move, clean, sort, demonstrate)
  Examples: "抓杯子", "把桌子清理干净", "随意做点事情", "pick the mug", "sort objects by color"
- direct: Single-word direct robot commands that need no planning
  Examples: "home", "scan", "open", "close", "stop", "detect"
- query: Asking to look at or identify objects on the table (needs perception, then answer)
  Examples: "看看桌上有什么", "检测所有物体", "scan the table", "what objects are there"

User message: {user_message}

Category:"""


# ---------------------------------------------------------------------------
# Stage 2: Chat Response (no action)
# ---------------------------------------------------------------------------

CHAT_SYSTEM_PROMPT = """\
{agent_prompt}
"""


# ---------------------------------------------------------------------------
# Stage 3: Task Planning (action required)
# ---------------------------------------------------------------------------

PLANNING_SYSTEM_PROMPT = """\
You are an action-oriented robot task planner for Vector OS.
Respond in the same language the user uses (Chinese or English).

Available skills:
{skills_json}

Current world state:
{world_state_json}

You must return a JSON object with TWO fields:
1. "message": A brief, friendly message to the user (in their language) explaining what you will do. No markdown formatting, no asterisks, no bullet points. Plain text only. Address the user as "主人" in Chinese.
2. "steps": An ordered list of skill calls to execute.

EXECUTION RULES:
1. ALWAYS produce steps for task requests. NEVER leave steps empty for a task.
2. Typos and variations are the SAME object: "proteinbar" = "protein bar".
3. When picking an object NOT in the world state, prepend scan + detect steps.
4. When the gripper is holding something and user wants to pick new, prepend place step.
5. Keep plans practical. Use available skills only.
6. For creative/open-ended requests ("do something interesting"), plan a multi-step sequence.
7. Parameters must match the skill schema.
8. CRITICAL: object_label in pick MUST be a specific object name from the world state (e.g. "banana", "mug", "bottle"). NEVER use generic words like "object", "item", "thing", "东西". If user says "随便抓" or "grab anything", YOU choose a specific object from the world state objects list.
9. For "grab everything" or "把所有东西都抓", plan one pick step per object using their actual names from world state.

PICK AND PLACE RULES:
- When user says "pick X and put it somewhere", use pick(mode="hold") then place(location=...).
- When user just says "pick X" or "grab X" without a destination, use pick(mode="drop") — this discards the object to the side.
- ALWAYS end every plan with a home step so the arm returns to rest position.

{constraints_block}

MULTI-OBJECT EXAMPLE:
User: "把所有东西都抓了随便乱放"
With objects: banana, mug, bottle on table →
steps: pick(banana,hold) → place(left) → pick(mug,hold) → place(right) → pick(bottle,hold) → place(front) → home

Output format — JSON only, no markdown fences:
{{
  "message": "好的主人，我来抓取杯子。",
  "steps": [
    {{
      "step_id": "s1",
      "skill_name": "scan",
      "parameters": {{}},
      "depends_on": [],
      "preconditions": [],
      "postconditions": []
    }},
    {{
      "step_id": "s2",
      "skill_name": "detect",
      "parameters": {{"query": "mug"}},
      "depends_on": ["s1"],
      "preconditions": [],
      "postconditions": []
    }},
    {{
      "step_id": "s3",
      "skill_name": "pick",
      "parameters": {{"object_label": "mug"}},
      "depends_on": ["s2"],
      "preconditions": [],
      "postconditions": []
    }}
  ]
}}

Clarification (ONLY when genuinely ambiguous):
{{
  "message": "主人，桌上有两个杯子，你要哪一个？",
  "requires_clarification": true
}}
"""


# ---------------------------------------------------------------------------
# Stage 4: Summarize execution results
# ---------------------------------------------------------------------------

SUMMARIZE_SYSTEM_PROMPT = """\
You are V, the AI assistant for Vector OS Nano robot arm.
Summarize the execution results for the user concisely.
No markdown formatting, no asterisks, no bullet points. Plain text only.
Address the user as "主人" in Chinese. Match the user's language.
Be factual: state what succeeded, what failed, and how long it took.
One to three sentences maximum.

User's original request: {original_request}

Execution results:
{execution_trace}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_classify_prompt(user_message: str) -> str:
    """Build the classification prompt."""
    return CLASSIFY_SYSTEM_PROMPT.format(user_message=user_message)


def build_planning_prompt(
    skill_schemas: list[dict[str, Any]],
    world_state: dict[str, Any],
) -> str:
    """Build the system prompt for the LLM task planner."""
    skills_json = json.dumps(skill_schemas, indent=2, ensure_ascii=False)
    world_state_json = json.dumps(world_state, indent=2, ensure_ascii=False)

    # Build dynamic constraints
    constraints_parts: list[str] = []

    # 1. Enum constraints from skill schemas
    for schema in skill_schemas:
        for param_name, param_def in schema.get("parameters", {}).items():
            if isinstance(param_def, dict) and "enum" in param_def:
                values = ", ".join(str(v) for v in param_def["enum"])
                constraints_parts.append(
                    f"VALID VALUES for {schema['name']}.{param_name}: {values}"
                )

    # 2. Available objects from world state
    objects = world_state.get("objects", [])
    if objects:
        labels = [o.get("label", "unknown") for o in objects if isinstance(o, dict)]
        constraints_parts.append(f"AVAILABLE OBJECTS: {', '.join(labels)}")
        constraints_parts.append("pick.object_label MUST be one of these exact names.")
    else:
        constraints_parts.append("AVAILABLE OBJECTS: none detected. Plan MUST start with scan + detect.")

    # 3. Gripper state
    robot = world_state.get("robot", {})
    held = robot.get("held_object")
    if held:
        constraints_parts.append(f"GRIPPER: holding {held}. Can place directly without picking.")
    else:
        gripper_state = robot.get("gripper_state", "unknown")
        constraints_parts.append(f"GRIPPER: {gripper_state} (not holding anything). Must pick before place.")

    constraints_block = "\n".join(constraints_parts)

    return PLANNING_SYSTEM_PROMPT.format(
        skills_json=skills_json,
        world_state_json=world_state_json,
        constraints_block=constraints_block,
    )


def build_summarize_prompt(
    original_request: str,
    execution_trace: str,
) -> str:
    """Build the summarization prompt."""
    return SUMMARIZE_SYSTEM_PROMPT.format(
        original_request=original_request,
        execution_trace=execution_trace,
    )


# ---------------------------------------------------------------------------
# Agent Loop prompt
# ---------------------------------------------------------------------------

AGENT_LOOP_SYSTEM_PROMPT = """\
You are a robot action planner executing an iterative goal.
Respond in the same language the user uses (Chinese or English).

GOAL: {goal}

AVAILABLE SKILLS:
{skills_json}

CURRENT OBSERVATION:
{observation_json}

EXECUTION HISTORY:
{history_json}

RULES:
1. Return EXACTLY ONE JSON object. No markdown fences. No explanation outside JSON.
2. If the goal is achieved, return: {{"done": true, "summary": "..."}}
3. If more work is needed, return: {{"action": "skill_name", "params": {{}}, "reasoning": "..."}}
4. ONLY use skill names from AVAILABLE SKILLS. Parameters must match the schema.
5. If a previous action failed or was not verified, try a DIFFERENT approach — do NOT repeat the same action with same params.
6. If you need to see the workspace, use "scan" then "detect" as your action.
7. Maximum {max_iterations} iterations — be efficient.
8. object_label in pick MUST be a specific object name from the observation, NOT "object" or "item".
"""


def build_agent_loop_prompt(
    goal: str,
    observation: dict,
    skill_schemas: list[dict],
    history: list[dict],
    max_iterations: int = 10,
) -> str:
    """Build the system prompt for the agent loop decide step."""
    import json as _json
    skills_json = _json.dumps(skill_schemas, indent=2, ensure_ascii=False)
    observation_json = _json.dumps(observation, indent=2, ensure_ascii=False)
    history_json = _json.dumps(history, indent=2, ensure_ascii=False) if history else "[]"

    return AGENT_LOOP_SYSTEM_PROMPT.format(
        goal=goal,
        skills_json=skills_json,
        observation_json=observation_json,
        history_json=history_json,
        max_iterations=max_iterations,
    )


def build_tool_definitions(skill_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert skill schemas to OpenAI function-calling tool format.

    Cleans property definitions to only contain valid JSON Schema keys.
    Skill-internal keys like 'required', 'source' are stripped.
    """
    # Keys valid inside a JSON Schema property definition
    _VALID_PROP_KEYS = {"type", "description", "enum", "default", "items"}
    _TYPE_MAP = {"float": "number", "int": "integer", "bool": "boolean", "str": "string"}

    tools: list[dict[str, Any]] = []
    for skill in skill_schemas:
        raw_params: dict[str, Any] = skill.get("parameters", {})

        properties: dict[str, dict] = {}
        required: list[str] = []

        for pname, pdef in raw_params.items():
            if not isinstance(pdef, dict):
                continue
            # Clean: only keep valid JSON Schema keys
            clean: dict[str, Any] = {}
            if "type" in pdef:
                clean["type"] = _TYPE_MAP.get(pdef["type"], pdef["type"])
            if "description" in pdef:
                clean["description"] = pdef["description"]
            if "enum" in pdef:
                clean["enum"] = pdef["enum"]
            if "default" in pdef:
                clean["default"] = pdef["default"]
            properties[pname] = clean

            # Determine required
            explicitly_optional = pdef.get("required") is False
            has_default = "default" in pdef
            if not explicitly_optional and not has_default:
                required.append(pname)

        tool: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": skill["name"],
                "description": skill.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                },
            },
        }
        if required:
            tool["function"]["parameters"]["required"] = required

        tools.append(tool)
    return tools
