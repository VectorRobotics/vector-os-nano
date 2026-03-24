"""Tool-calling Agent — LLM-native function calling for intelligent conversation.

Replaces the rigid classify→plan→execute pipeline with a proper
tool-calling conversation loop. The LLM sees the full conversation history,
decides when to chat vs when to call skills, and maintains context across turns.

Solves the "饿了→拿呀" problem:
  Turn 1: "饿了" → LLM chats "我帮你拿蛋白棒" (sees tools, decides not to call yet)
  Turn 2: "拿呀" → LLM sees context, calls pick(object_label="蛋白棒")

Works with any LLM that supports OpenAI-format tool calling (via OpenRouter):
  - Claude (Haiku, Sonnet, Opus)
  - GPT-4o, GPT-4o-mini
  - Llama 3.x (with tool support)
  - Mistral, Gemma, etc.

No ROS2 dependencies. No hardcoded LLM provider.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

import httpx

from vector_os_nano.core.types import SkillResult

logger = logging.getLogger(__name__)

# System prompt for the tool-calling agent
_SYSTEM_PROMPT = """\
You are V, the AI agent for Vector OS Nano robot arm.
You control a SO-101 robot arm through tool calls. You communicate in whatever language the user uses.
Call the user "主人" in Chinese.

PERSONALITY:
You are proactive and action-oriented. When the user expresses a need, ACT on it. Do NOT repeatedly ask for confirmation. If the user says "我饿了" (I'm hungry) and there is food on the table, immediately pick it up. If there are multiple options, pick the most appropriate one yourself. You are a helpful assistant, not a menu.

RULES:
1. When the user wants you to DO something, call the appropriate tool IMMEDIATELY. Do not ask "which one" if you can make a reasonable choice.
2. When the user expresses a need (hungry, thirsty, cold, etc.) and relevant objects are available, take action without asking. Be proactive.
3. If the user repeats the same request or says it more urgently, that means ACT NOW, not ask again.
4. You can see the conversation history. Use it to understand what the user wants.
5. After a tool call, briefly say what happened. If it failed, explain why and try an alternative.
6. Keep responses concise. No markdown, no asterisks, no bullet points. Plain text only.
7. For pick: object_label must be a specific object name from the table.
8. If you don't know what's on the table, call scan first, then detect to look.
9. When in doubt, ACT rather than ASK.

CURRENT STATE:
{state_info}
"""


class ToolAgent:
    """Agent that uses LLM-native function calling for conversation + skill execution.

    Instead of classify→plan→execute, the LLM sees tools and decides
    when to call them. Full conversation history is maintained.

    Args:
        agent_ref: Reference to the Agent instance (for skill execution).
        api_key: OpenRouter API key.
        model: Model to use (must support tool calling).
        api_base: API base URL (default: OpenRouter).
    """

    def __init__(
        self,
        agent_ref: Any,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        api_base: str = "https://openrouter.ai/api/v1",
    ) -> None:
        self._agent = agent_ref
        self._model = model
        self._endpoint = f"{api_base.rstrip('/')}/chat/completions"
        self._http = httpx.Client(timeout=30.0)
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._messages: list[dict[str, Any]] = []
        self._tools: list[dict[str, Any]] = []
        self._rebuild_tools()

    def _rebuild_tools(self) -> None:
        """Convert skill registry to OpenAI tool format."""
        from vector_os_nano.llm.prompts import build_tool_definitions
        schemas = self._agent._skill_registry.to_schemas()
        self._tools = build_tool_definitions(schemas)

    def _build_system_prompt(self) -> str:
        """Build system prompt with current state."""
        agent = self._agent
        parts = []
        if agent._arm:
            mode = "MuJoCo simulation" if hasattr(agent._arm, "get_object_positions") else "real hardware"
            parts.append(f"Mode: {mode}")
        else:
            parts.append("Mode: no arm connected")

        gripper = "unknown"
        if agent._gripper:
            try:
                pos = agent._gripper.get_position()
                gripper = "open" if pos > 0.5 else "closed"
            except Exception:
                pass
        parts.append(f"Gripper: {gripper}")

        # Objects
        objects_info = "unknown"
        if hasattr(agent._arm, "get_object_positions"):
            objs = agent._arm.get_object_positions()
            objects_info = ", ".join(objs.keys()) if objs else "none"
        elif agent._world_model:
            wm_objs = agent._world_model.get_objects()
            objects_info = ", ".join(o.label for o in wm_objs) if wm_objs else "none"
        parts.append(f"Objects on table: {objects_info}")

        return _SYSTEM_PROMPT.format(state_info="\n".join(parts))

    def chat(
        self,
        user_message: str,
        on_tool_call: Callable | None = None,
        on_debug: Callable | None = None,
    ) -> str:
        """Send a message and get a response. May trigger tool calls.

        Args:
            user_message: User's text input.
            on_tool_call: Optional callback(tool_name, params) before each tool execution.
            on_debug: Optional callback(stage, detail) for debug output.

        Returns:
            The agent's text response (after any tool calls are resolved).
        """
        def _dbg(stage: str, detail: str) -> None:
            logger.info("[ToolAgent] %s: %s", stage, detail)
            if on_debug:
                try:
                    on_debug(stage, detail)
                except Exception:
                    pass

        self._messages.append({"role": "user", "content": user_message})

        # Cap history
        if len(self._messages) > 40:
            self._messages = self._messages[-30:]

        max_tool_rounds = 5  # prevent infinite tool-call loops

        for round_idx in range(max_tool_rounds):
            system = self._build_system_prompt()
            payload = {
                "model": self._model,
                "messages": [{"role": "system", "content": system}] + self._messages,
                "tools": self._tools,
                "tool_choice": "auto",
                "temperature": 0.3,
                "max_tokens": 1024,
            }

            _dbg("LLM_CALL", f"round={round_idx}, model={self._model}, msgs={len(self._messages)}")

            try:
                resp = self._http.post(self._endpoint, json=payload, headers=self._headers)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning("[ToolAgent] API error: %s", exc)
                error_msg = f"API error: {exc}"
                self._messages.append({"role": "assistant", "content": error_msg})
                return error_msg

            choice = data["choices"][0]
            message = choice["message"]
            finish_reason = choice.get("finish_reason", "")

            # Case 1: LLM wants to call tool(s)
            tool_calls = message.get("tool_calls")
            if tool_calls:
                # Add assistant message with tool_calls to history
                self._messages.append(message)

                for tc in tool_calls:
                    fn = tc["function"]
                    tool_name = fn["name"]
                    try:
                        tool_args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                    except json.JSONDecodeError:
                        tool_args = {}

                    _dbg("TOOL_CALL", f"{tool_name}({tool_args})")
                    if on_tool_call:
                        on_tool_call(tool_name, tool_args)

                    # Execute the skill
                    tool_result = self._execute_tool(tool_name, tool_args)
                    _dbg("TOOL_RESULT", f"{tool_name} → {tool_result[:200]}")

                    # Add tool result to history
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    })

                # Loop back for LLM to process tool results
                continue

            # Case 2: LLM responds with text (no tool call)
            text = message.get("content", "")
            _dbg("RESPONSE", f"text ({len(text)} chars)")
            self._messages.append({"role": "assistant", "content": text})
            return text

        # Should not reach here
        return "Max tool call rounds reached."

    def _execute_tool(self, tool_name: str, params: dict) -> str:
        """Execute a skill and return result as string for LLM context."""
        agent = self._agent

        # Sync world model
        if hasattr(agent, '_sync_robot_state'):
            agent._sync_robot_state()

        skill = agent._skill_registry.get(tool_name)
        if skill is None:
            return json.dumps({"success": False, "error": f"Unknown skill: {tool_name}"})

        context = agent._build_context()
        try:
            result = skill.execute(params, context)
        except Exception as exc:
            return json.dumps({"success": False, "error": str(exc)})

        # Apply world model effects
        agent._world_model.apply_skill_effects(tool_name, params, result)
        agent._sync_robot_state()

        # Return structured result
        return json.dumps({
            "success": result.success,
            "error_message": result.error_message if not result.success else "",
            "result_data": result.result_data,
        }, ensure_ascii=False)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._messages.clear()
