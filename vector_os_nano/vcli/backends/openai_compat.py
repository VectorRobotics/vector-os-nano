# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""OpenAI-compatible LLM backend for Vector CLI.

Works with any provider that implements the OpenAI chat completions API:
- OpenRouter (https://openrouter.ai/api/v1)
- Ollama (http://localhost:11434/v1)
- vLLM (http://localhost:8000/v1)
- Any OpenAI-compatible local server

Handles conversion between Anthropic-canonical format (used internally)
and OpenAI chat format (used by these providers).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

import openai

from vector_os_nano.vcli.backends.types import LLMResponse, LLMToolCall
from vector_os_nano.vcli.session import TokenUsage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop reason mapping: OpenAI → canonical
# ---------------------------------------------------------------------------

_STOP_REASON_MAP: dict[str, str] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "end_turn",
}


# ---------------------------------------------------------------------------
# Message format converters: Anthropic-canonical → OpenAI
# ---------------------------------------------------------------------------


def convert_system(system_blocks: list[dict[str, Any]]) -> str:
    """Convert Anthropic system blocks to a single OpenAI system message string."""
    parts: list[str] = []
    for block in system_blocks:
        text = block.get("text", "")
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def convert_tools(anthropic_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic tool schemas to OpenAI function tool format.

    Anthropic: {"name": "...", "description": "...", "input_schema": {...}}
    OpenAI:    {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    """
    openai_tools: list[dict[str, Any]] = []
    for t in anthropic_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return openai_tools


def convert_messages(
    anthropic_messages: list[dict[str, Any]],
    system_text: str,
) -> list[dict[str, Any]]:
    """Convert Anthropic message format to OpenAI chat format.

    Handles:
    - user text messages
    - assistant text + tool_use blocks → assistant message + tool_calls
    - user tool_result blocks → one "tool" message per result
    - system prompt as first system message
    """
    openai_msgs: list[dict[str, Any]] = []

    # System message first
    if system_text:
        openai_msgs.append({"role": "system", "content": system_text})

    for msg in anthropic_messages:
        role = msg["role"]
        content = msg.get("content")

        if role == "user":
            if isinstance(content, str):
                openai_msgs.append({"role": "user", "content": content})
            elif isinstance(content, list):
                # Could be tool_result blocks or mixed content
                tool_results = [
                    b for b in content
                    if isinstance(b, dict) and b.get("type") == "tool_result"
                ]
                if tool_results:
                    for tr in tool_results:
                        openai_msgs.append({
                            "role": "tool",
                            "tool_call_id": tr["tool_use_id"],
                            "content": tr.get("content", ""),
                        })
                else:
                    # Plain text blocks
                    text = " ".join(
                        b.get("text", str(b)) if isinstance(b, dict) else str(b)
                        for b in content
                    )
                    openai_msgs.append({"role": "user", "content": text})

        elif role == "assistant":
            if isinstance(content, str):
                openai_msgs.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                # Extract text and tool_use blocks
                text_parts: list[str] = []
                tool_calls: list[dict[str, Any]] = []

                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(
                                    block.get("input", {}), ensure_ascii=False
                                ),
                            },
                        })

                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": "".join(text_parts) or None,
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                openai_msgs.append(assistant_msg)

    return openai_msgs


# ---------------------------------------------------------------------------
# Response parser: OpenAI → canonical
# ---------------------------------------------------------------------------


def parse_usage(raw_usage: Any) -> TokenUsage:
    """Extract token usage from an OpenAI response."""
    if raw_usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
        cache_read_tokens=getattr(raw_usage, "prompt_tokens_details", None)
        and getattr(raw_usage.prompt_tokens_details, "cached_tokens", 0) or 0,
    )


# ---------------------------------------------------------------------------
# OpenAICompatBackend
# ---------------------------------------------------------------------------


class OpenAICompatBackend:
    """Backend for any OpenAI-compatible API endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        max_retries: int = 3,
    ) -> None:
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._max_retries = max_retries

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: list[dict[str, Any]],
        max_tokens: int,
        on_text: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Call the OpenAI-compatible API with streaming and retry."""
        system_text = convert_system(system)
        oai_messages = convert_messages(messages, system_text)
        oai_tools = convert_tools(tools) if tools else None

        return self._call_with_retry(
            oai_messages, oai_tools, max_tokens, on_text, on_reasoning
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        on_text: Callable[[str], None] | None,
        on_reasoning: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Make the API call with exponential backoff retry."""
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                return self._call_streaming(
                    messages, tools, max_tokens, on_text, on_reasoning
                )
            except openai.RateLimitError as exc:
                last_exc = exc
                delay = 2**attempt
                logger.warning(
                    "Rate limited (attempt %d/%d), retrying in %ds",
                    attempt + 1, self._max_retries, delay,
                )
                time.sleep(delay)
            except openai.APIConnectionError as exc:
                last_exc = exc
                delay = 2**attempt
                logger.warning(
                    "Connection error (attempt %d/%d), retrying in %ds",
                    attempt + 1, self._max_retries, delay,
                )
                time.sleep(delay)
            except openai.InternalServerError as exc:
                last_exc = exc
                delay = 2**attempt
                logger.warning(
                    "Server error (attempt %d/%d), retrying in %ds",
                    attempt + 1, self._max_retries, delay,
                )
                time.sleep(delay)
            except openai.APIStatusError:
                raise  # Non-retryable

        raise last_exc  # type: ignore[misc]

    def _call_streaming(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        on_text: Callable[[str], None] | None,
        on_reasoning: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Make a streaming API call, accumulate text + tool calls, return LLMResponse."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools

        stream = self._client.chat.completions.create(**kwargs)

        # Accumulators
        text_parts: list[str] = []
        tool_call_acc: dict[int, dict[str, Any]] = {}  # index → {id, name, arguments}
        finish_reason: str | None = None
        usage_data: Any = None

        for chunk in stream:
            if not chunk.choices:
                # Usage-only chunk (some providers send this at the end)
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    usage_data = chunk.usage
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            # Finish reason
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

            # Reasoning content (reasoning models: DeepSeek `reasoning_content`,
            # some providers `reasoning`). Hidden trace — surfaced ONLY as a live
            # "thinking…" heartbeat, never accumulated into the response text.
            if on_reasoning is not None and delta is not None:
                reasoning_chunk = getattr(delta, "reasoning_content", None) or getattr(
                    delta, "reasoning", None
                )
                if reasoning_chunk:
                    on_reasoning(reasoning_chunk)

            # Text content
            if delta and delta.content:
                text_parts.append(delta.content)
                if on_text is not None:
                    on_text(delta.content)

            # Tool calls (streamed in chunks — accumulate by index)
            if delta and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_acc:
                        tool_call_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    acc = tool_call_acc[idx]
                    if tc_delta.id:
                        acc["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            acc["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            acc["arguments"] += tc_delta.function.arguments

        # Build canonical tool calls
        llm_tool_calls: list[LLMToolCall] = []
        for idx in sorted(tool_call_acc.keys()):
            acc = tool_call_acc[idx]
            parsed_input: Any = {}
            error = ""
            try:
                parsed_input = json.loads(acc["arguments"]) if acc["arguments"] else {}
            except json.JSONDecodeError as exc:
                error = f"arguments are not valid JSON ({exc.msg})"
            if not isinstance(parsed_input, dict):
                parsed_input, error = {}, "arguments are not a JSON object"
            if error:
                logger.warning("tool call %s: %s", acc["name"], error)
            llm_tool_calls.append(
                LLMToolCall(id=acc["id"], name=acc["name"], input=parsed_input, parse_error=error)
            )

        stop = _STOP_REASON_MAP.get(finish_reason or "stop", "end_turn")
        usage = parse_usage(usage_data)

        return LLMResponse(
            text="".join(text_parts),
            tool_calls=llm_tool_calls,
            stop_reason=stop,
            usage=usage,
        )
