# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Public model-call contract and canonical response values; no provider or CLI imports."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    def add(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TokenUsage":
        return cls(
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            cache_read_tokens=d.get("cache_read_tokens", 0),
            cache_creation_tokens=d.get("cache_creation_tokens", 0),
        )

@dataclass(frozen=True)
class LLMToolCall:
    """A single tool call requested by the model."""

    id: str
    name: str
    input: dict[str, Any]
    # Why the model's arguments could not be read ('' when they could): such a call
    # is answered with an error, never run with defaults.
    parse_error: str = ""

@dataclass(frozen=True)
class LLMResponse:
    """Canonical response from any LLM backend.

    The engine consumes this without knowing which provider produced it.
    """

    text: str
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"  # "end_turn" | "tool_use" | "max_tokens"
    usage: TokenUsage = field(default_factory=TokenUsage)

@runtime_checkable
class LLMBackend(Protocol):
    """Structural interface for LLM API backends.

    Each backend handles:
    - Message format conversion (canonical Anthropic-like → native)
    - API call with streaming
    - Response parsing (native → canonical LLMResponse)
    - Retry logic for transient errors
    """

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: list[dict[str, Any]],
        max_tokens: int,
        on_text: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Make an LLM API call and return a canonical response.

        Args:
            messages: Conversation history in Anthropic message format.
            tools:    Tool definitions in Anthropic tool schema format.
            system:   System prompt blocks in Anthropic format.
            max_tokens: Max output tokens.
            on_text:  Called incrementally with text chunks during streaming.
            on_reasoning: Called incrementally with hidden reasoning/think chunks
                during streaming (reasoning models only). Used purely for a live
                "thinking…" heartbeat; never accumulated into the response text.

        Returns:
            LLMResponse with text, tool_calls, stop_reason, and usage.
        """
        ...
