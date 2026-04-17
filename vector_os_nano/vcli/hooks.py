"""Tool execution hooks for Vector CLI.

Pre/post callbacks that fire around every tool execution. Use cases:
- Auto-verify robot state after motor skills
- Log tool call telemetry
- Chain side-effects (auto-format after file edit)

Hooks must not raise — exceptions are caught and logged.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from vector_os_nano.vcli.tools.base import ToolResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolHookContext:
    """Context passed to hook callbacks."""

    tool_name: str
    params: dict[str, Any]
    result: ToolResult | None = None   # None for pre-hooks
    duration: float = 0.0              # 0.0 for pre-hooks


ToolHook = Callable[[ToolHookContext], None]


class ToolHookRegistry:
    """Register and fire pre/post tool execution hooks."""

    def __init__(self) -> None:
        self._pre: list[ToolHook] = []
        self._post: list[ToolHook] = []

    def add_pre_hook(self, hook: ToolHook) -> None:
        self._pre.append(hook)

    def add_post_hook(self, hook: ToolHook) -> None:
        self._post.append(hook)

    def fire_pre(self, ctx: ToolHookContext) -> None:
        for hook in self._pre:
            try:
                hook(ctx)
            except Exception as exc:
                logger.debug("Pre-hook error: %s", exc)

    def fire_post(self, ctx: ToolHookContext) -> None:
        for hook in self._post:
            try:
                hook(ctx)
            except Exception as exc:
                logger.debug("Post-hook error: %s", exc)
