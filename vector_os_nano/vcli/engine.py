"""VectorEngine — core Anthropic tool_use agent loop for Vector CLI.

Mirrors Claude Code's query.ts / toolOrchestration.ts pattern.

Phase 1: Uses client.messages.create() (non-streaming) for simplicity.
Phase 2 will switch to client.messages.stream() for token-by-token rendering.

Public exports:
    ToolCall    — frozen record of a single tool execution
    TurnResult  — frozen result of one full user turn (may span N API calls)
    ToolBatch   — internal grouping for concurrent vs sequential execution
    VectorEngine — the stateful agent loop
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import anthropic

from vector_os_nano.vcli.permissions import PermissionContext
from vector_os_nano.vcli.session import Session, TokenUsage
from vector_os_nano.vcli.tools.base import (
    PermissionResult,
    ToolContext,
    ToolRegistry,
    ToolResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """Immutable record of a single tool invocation within a turn."""

    tool_name: str
    params: dict[str, Any]
    result: ToolResult
    duration_sec: float
    permission_action: str  # "allowed" | "denied" | "asked_allowed" | "asked_denied"


@dataclass(frozen=True)
class TurnResult:
    """Immutable result of one full user turn (may include multiple API round-trips)."""

    text: str
    tool_calls: list[ToolCall]
    stop_reason: str  # "end_turn" | "max_tokens" | "tool_use"
    usage: TokenUsage


# ---------------------------------------------------------------------------
# Internal batching type
# ---------------------------------------------------------------------------


@dataclass
class ToolBatch:
    """A group of tool_use blocks to execute together."""

    concurrent: bool
    blocks: list[Any]  # list of tool_use content blocks (MagicMock-able in tests)


# ---------------------------------------------------------------------------
# VectorEngine
# ---------------------------------------------------------------------------


class VectorEngine:
    """Core agent loop: user message → API call → tool execution → repeat until end_turn.

    Thread-safety: a single VectorEngine instance should not be shared across
    concurrent threads. Create one instance per agent session.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        registry: ToolRegistry | None = None,
        system_prompt: list[dict[str, Any]] | None = None,
        permissions: PermissionContext | None = None,
        max_turns: int = 50,
        max_tokens: int = 16384,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._registry: ToolRegistry = registry or ToolRegistry()
        self._system_prompt: list[dict[str, Any]] = system_prompt or []
        self._permissions: PermissionContext = permissions or PermissionContext()
        self._max_turns: int = max_turns
        self._max_tokens: int = max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_turn(
        self,
        user_message: str,
        session: Session,
        agent: Any = None,
        on_text: Callable[[str], None] | None = None,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None = None,
        on_tool_end: Callable[[str, ToolResult], None] | None = None,
        ask_permission: Callable[[str, dict[str, Any]], bool] | None = None,
    ) -> TurnResult:
        """Run one user turn through the tool_use agent loop.

        Algorithm:
        1. Append user message to session
        2. Call API (non-streaming, Phase 1)
        3. Collect text blocks + tool_use blocks
        4. If tool_use blocks present: execute tools, append results to session, loop
        5. If no tool_use blocks: return TurnResult

        Args:
            user_message:   The user's input text for this turn.
            session:        Mutable session object; updated in-place.
            agent:          Optional back-reference to the outer Agent (passed to ToolContext).
            on_text:        Called with each text block's content.
            on_tool_start:  Called before each tool execution with (tool_name, params).
            on_tool_end:    Called after each tool execution with (tool_name, result).
            ask_permission: For "ask"-level permissions, called with (tool_name, params).
                            Returns True to allow, False to deny.

        Returns:
            TurnResult with the final assistant text, all tool calls, stop reason, and
            cumulative token usage across all API round-trips in this turn.
        """
        session.append_user(user_message)

        all_tool_calls: list[ToolCall] = []
        total_usage = TokenUsage()
        final_text = ""
        stop_reason = "end_turn"
        turns = 0
        abort_event = threading.Event()

        tool_context = ToolContext(
            agent=agent,
            cwd=Path.cwd(),
            session=session,
            permissions=self._permissions,
            abort=abort_event,
        )

        while turns < self._max_turns:
            messages = session.to_messages()

            response = self._client.messages.create(
                model=self._model,
                messages=messages,
                tools=self._registry.to_anthropic_schemas(),
                system=self._system_prompt,
                max_tokens=self._max_tokens,
            )

            # Extract content blocks
            text_parts: list[str] = []
            tool_use_blocks: list[Any] = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                    if on_text is not None:
                        on_text(block.text)
                elif block.type == "tool_use":
                    tool_use_blocks.append(block)

            assistant_text = "".join(text_parts)
            final_text = assistant_text
            stop_reason = response.stop_reason

            # Track usage
            raw_usage = response.usage
            step_usage = TokenUsage(
                input_tokens=raw_usage.input_tokens,
                output_tokens=raw_usage.output_tokens,
                cache_read_tokens=getattr(raw_usage, "cache_read_input_tokens", 0) or 0,
                cache_creation_tokens=getattr(raw_usage, "cache_creation_input_tokens", 0) or 0,
            )
            total_usage = total_usage.add(step_usage)

            # Append assistant message to session (include tool_use dicts if any)
            tool_use_dicts: list[dict[str, Any]] | None = None
            if tool_use_blocks:
                tool_use_dicts = [
                    {"id": b.id, "name": b.name, "input": b.input, "type": "tool_use"}
                    for b in tool_use_blocks
                ]
            session.append_assistant(assistant_text, tool_use_dicts)

            if not tool_use_blocks:
                break  # end_turn — no tools called, conversation complete

            # Execute tools and collect results
            raw_results = self._dispatch_tools(
                tool_use_blocks, tool_context, on_tool_start, on_tool_end, ask_permission
            )

            # Unpack (result_dict, ToolCall) pairs
            result_dicts: list[dict[str, Any]] = []
            for result_dict, tool_call in raw_results:
                result_dicts.append(result_dict)
                all_tool_calls.append(tool_call)

            # Append tool results to session so next API call sees them
            session.append_tool_results(result_dicts)

            turns += 1

        session.add_usage(total_usage)

        return TurnResult(
            text=final_text,
            tool_calls=all_tool_calls,
            stop_reason=stop_reason,
            usage=total_usage,
        )

    # ------------------------------------------------------------------
    # Internal: tool partitioning and dispatch
    # ------------------------------------------------------------------

    def _partition_tools(self, tool_use_blocks: list[Any]) -> list[ToolBatch]:
        """Partition tool_use blocks into concurrent (read-only) and sequential batches.

        Consecutive blocks whose tools are concurrency-safe are merged into one
        concurrent ToolBatch. Any write/unsafe tool starts a new sequential batch.

        Mirrors Claude Code's toolOrchestration.ts grouping logic.
        """
        batches: list[ToolBatch] = []
        for block in tool_use_blocks:
            tool = self._registry.get(block.name)
            is_safe = bool(
                tool is not None
                and hasattr(tool, "is_concurrency_safe")
                and tool.is_concurrency_safe(block.input)
            )
            if is_safe and batches and batches[-1].concurrent:
                # Extend the existing concurrent batch
                batches[-1].blocks.append(block)
            else:
                batches.append(ToolBatch(concurrent=is_safe, blocks=[block]))
        return batches

    def _dispatch_tools(
        self,
        tool_use_blocks: list[Any],
        tool_context: ToolContext,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None,
        on_tool_end: Callable[[str, ToolResult], None] | None,
        ask_permission: Callable[[str, dict[str, Any]], bool] | None,
    ) -> list[tuple[dict[str, Any], ToolCall]]:
        """Dispatch all tool_use blocks, respecting concurrency partitioning."""
        results: list[tuple[dict[str, Any], ToolCall]] = []
        batches = self._partition_tools(tool_use_blocks)

        for batch in batches:
            if batch.concurrent and len(batch.blocks) > 1:
                batch_results = self._run_concurrent(
                    batch.blocks, tool_context, on_tool_start, on_tool_end, ask_permission
                )
            else:
                batch_results = self._run_sequential(
                    batch.blocks, tool_context, on_tool_start, on_tool_end, ask_permission
                )
            results.extend(batch_results)

        return results

    def _execute_single_tool(
        self,
        block: Any,
        tool_context: ToolContext,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None,
        on_tool_end: Callable[[str, ToolResult], None] | None,
        ask_permission: Callable[[str, dict[str, Any]], bool] | None,
    ) -> tuple[dict[str, Any], ToolCall]:
        """Execute one tool_use block with full permission checking.

        Returns:
            Tuple of (result_dict_for_session, ToolCall record).
        """
        tool_name: str = block.name
        params: dict[str, Any] = block.input
        tool = self._registry.get(tool_name)

        if tool is None:
            result = ToolResult(content=f"Unknown tool: {tool_name}", is_error=True)
            logger.warning("Tool %r not found in registry", tool_name)
            return (
                {"tool_use_id": block.id, "content": result.content, "is_error": True},
                ToolCall(
                    tool_name=tool_name,
                    params=params,
                    result=result,
                    duration_sec=0.0,
                    permission_action="denied",
                ),
            )

        # Permission check
        perm: PermissionResult = self._permissions.check(tool, params, tool_context)

        if perm.behavior == "deny":
            reason = perm.reason or f"Permission denied for {tool_name}"
            result = ToolResult(content=f"Permission denied: {reason}", is_error=True)
            logger.info("Permission denied for tool %r: %s", tool_name, reason)
            return (
                {"tool_use_id": block.id, "content": result.content, "is_error": True},
                ToolCall(
                    tool_name=tool_name,
                    params=params,
                    result=result,
                    duration_sec=0.0,
                    permission_action="denied",
                ),
            )

        if perm.behavior == "ask":
            if ask_permission is None or not ask_permission(tool_name, params):
                denial = f"Permission denied by user for {tool_name}"
                result = ToolResult(content=denial, is_error=True)
                logger.info("User denied permission for tool %r", tool_name)
                return (
                    {"tool_use_id": block.id, "content": result.content, "is_error": True},
                    ToolCall(
                        tool_name=tool_name,
                        params=params,
                        result=result,
                        duration_sec=0.0,
                        permission_action="asked_denied",
                    ),
                )
            perm_action = "asked_allowed"
        else:
            perm_action = "allowed"

        # Execute the tool
        if on_tool_start is not None:
            on_tool_start(tool_name, params)

        start = time.monotonic()
        try:
            result = tool.execute(params, tool_context)
        except Exception as exc:
            result = ToolResult(content=f"Tool error: {exc}", is_error=True)
            logger.error("Tool %r raised %r", tool_name, exc, exc_info=True)
        duration = time.monotonic() - start

        if on_tool_end is not None:
            on_tool_end(tool_name, result)

        return (
            {"tool_use_id": block.id, "content": result.content, "is_error": result.is_error},
            ToolCall(
                tool_name=tool_name,
                params=params,
                result=result,
                duration_sec=duration,
                permission_action=perm_action,
            ),
        )

    def _run_sequential(
        self,
        blocks: list[Any],
        tool_context: ToolContext,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None,
        on_tool_end: Callable[[str, ToolResult], None] | None,
        ask_permission: Callable[[str, dict[str, Any]], bool] | None,
    ) -> list[tuple[dict[str, Any], ToolCall]]:
        """Execute blocks one-by-one in order."""
        return [
            self._execute_single_tool(
                block, tool_context, on_tool_start, on_tool_end, ask_permission
            )
            for block in blocks
        ]

    def _run_concurrent(
        self,
        blocks: list[Any],
        tool_context: ToolContext,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None,
        on_tool_end: Callable[[str, ToolResult], None] | None,
        ask_permission: Callable[[str, dict[str, Any]], bool] | None,
    ) -> list[tuple[dict[str, Any], ToolCall]]:
        """Execute read-only blocks concurrently using a thread pool.

        Results are returned in the original order of *blocks*.
        """
        max_workers = min(len(blocks), 10)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    self._execute_single_tool,
                    block,
                    tool_context,
                    on_tool_start,
                    on_tool_end,
                    ask_permission,
                )
                for block in blocks
            ]
            return [f.result() for f in futures]
