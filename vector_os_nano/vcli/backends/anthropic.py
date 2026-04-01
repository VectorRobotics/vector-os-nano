"""Anthropic API backend for Vector CLI.

Uses the anthropic Python SDK with streaming and prompt caching.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

import anthropic

from vector_os_nano.vcli.backends.types import LLMResponse, LLMToolCall
from vector_os_nano.vcli.session import TokenUsage

logger = logging.getLogger(__name__)


class AnthropicBackend:
    """Backend for Anthropic's native Messages API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        base_url: str | None = None,
        max_retries: int = 3,
    ) -> None:
        # OAuth tokens (sk-ant-oat*) work as regular api_key via x-api-key header
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**kwargs)
        self._model = model
        self._max_retries = max_retries

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: list[dict[str, Any]],
        max_tokens: int,
        on_text: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Call Anthropic Messages API with streaming and retry."""
        return self._call_with_retry(messages, tools, system, max_tokens, on_text)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: list[dict[str, Any]],
        max_tokens: int,
        on_text: Callable[[str], None] | None,
    ) -> LLMResponse:
        """Make the API call with exponential backoff retry."""
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                return self._call_streaming(messages, tools, system, max_tokens, on_text)
            except anthropic.RateLimitError as exc:
                last_exc = exc
                delay = 2**attempt
                logger.warning("Rate limited (attempt %d/%d), retrying in %ds", attempt + 1, self._max_retries, delay)
                time.sleep(delay)
            except anthropic.APIConnectionError as exc:
                last_exc = exc
                delay = 2**attempt
                logger.warning("Connection error (attempt %d/%d), retrying in %ds", attempt + 1, self._max_retries, delay)
                time.sleep(delay)
            except anthropic.InternalServerError as exc:
                last_exc = exc
                delay = 2**attempt
                logger.warning("Server error (attempt %d/%d), retrying in %ds", attempt + 1, self._max_retries, delay)
                time.sleep(delay)
            except anthropic.APIStatusError:
                raise

        raise last_exc  # type: ignore[misc]

    def _call_streaming(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: list[dict[str, Any]],
        max_tokens: int,
        on_text: Callable[[str], None] | None,
    ) -> LLMResponse:
        """Stream from Anthropic API, accumulate text + tool_use, return LLMResponse."""
        text_parts: list[str] = []
        tool_use_blocks: list[Any] = []

        with self._client.messages.stream(
            model=self._model,
            messages=messages,
            tools=tools,
            system=system,
            max_tokens=max_tokens,
        ) as stream:
            for event in stream:
                if hasattr(event, "type") and event.type == "content_block_delta":
                    delta = event.delta
                    if hasattr(delta, "text"):
                        text_parts.append(delta.text)
                        if on_text is not None:
                            on_text(delta.text)

            final_message = stream.get_final_message()
            for block in final_message.content:
                if block.type == "tool_use":
                    tool_use_blocks.append(block)

        # Build canonical tool calls
        llm_tool_calls = [
            LLMToolCall(id=b.id, name=b.name, input=b.input)
            for b in tool_use_blocks
        ]

        # Usage
        raw_usage = final_message.usage
        usage = TokenUsage(
            input_tokens=raw_usage.input_tokens,
            output_tokens=raw_usage.output_tokens,
            cache_read_tokens=getattr(raw_usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_tokens=getattr(raw_usage, "cache_creation_input_tokens", 0) or 0,
        )

        stop = final_message.stop_reason or "end_turn"

        return LLMResponse(
            text="".join(text_parts),
            tool_calls=llm_tool_calls,
            stop_reason=stop,
            usage=usage,
        )
