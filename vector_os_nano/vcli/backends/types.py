# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Canonical LLM response types shared by all backends.

These frozen dataclasses are the interface between backends and the engine.
Backends convert their native response format into these types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vector_os_nano.vcli.session import TokenUsage


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
