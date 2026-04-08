"""VectorEngine — core tool_use agent loop for Vector CLI.

Mirrors Claude Code's query.ts / toolOrchestration.ts pattern.
Backend-agnostic: works with any LLMBackend (Anthropic, OpenRouter, local).

Public exports:
    ToolCall     — frozen record of a single tool execution
    TurnResult   — frozen result of one full user turn (may span N API calls)
    ToolBatch    — internal grouping for concurrent vs sequential execution
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

from vector_os_nano.vcli.backends import LLMBackend
from vector_os_nano.vcli.backends.types import LLMResponse, LLMToolCall
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
    """A group of tool calls to execute together."""

    concurrent: bool
    tool_calls: list[LLMToolCall]


# ---------------------------------------------------------------------------
# VectorEngine
# ---------------------------------------------------------------------------


class VectorEngine:
    """Core agent loop: user message -> backend call -> tool execution -> repeat.

    Backend-agnostic: accepts any LLMBackend implementation.
    Thread-safety: a single VectorEngine instance should not be shared across
    concurrent threads. Create one instance per agent session.
    """

    def __init__(
        self,
        backend: LLMBackend,
        registry: ToolRegistry | None = None,
        system_prompt: list[dict[str, Any]] | None = None,
        permissions: PermissionContext | None = None,
        max_turns: int = 50,
        max_tokens: int = 16384,
        intent_router: Any = None,
        hooks: Any = None,
    ) -> None:
        self._backend: LLMBackend = backend
        self._registry: ToolRegistry = registry or ToolRegistry()
        self._system_prompt: list[dict[str, Any]] = system_prompt or []
        self._permissions: PermissionContext = permissions or PermissionContext()
        self._max_turns: int = max_turns
        self._max_tokens: int = max_tokens
        self._intent_router = intent_router  # IntentRouter or None
        self._hooks = hooks                  # ToolHookRegistry or None

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
        ask_permission: Callable[[str, dict[str, Any]], str] | None = None,
        app_state: dict[str, Any] | None = None,
    ) -> TurnResult:
        """Run one user turn through the tool_use agent loop.

        Algorithm:
        1. Append user message to session
        2. Call backend (handles streaming + format conversion)
        3. If tool_calls present: execute tools, append results, loop
        4. If no tool_calls: return TurnResult

        Args:
            user_message:   The user's input text for this turn.
            session:        Mutable session object; updated in-place.
            agent:          Optional back-reference to the outer Agent (passed to ToolContext).
            on_text:        Called with each text chunk as it streams.
            on_tool_start:  Called before each tool execution with (tool_name, params).
            on_tool_end:    Called after each tool execution with (tool_name, result).
            ask_permission: For "ask"-level permissions, called with (tool_name, params).
                            Returns "y" (allow once), "a" (always allow), or "n" (deny).

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
            app_state=app_state,
        )

        while turns < self._max_turns:
            if abort_event.is_set():
                break

            messages = session.to_messages()

            # Intent routing: select relevant tool categories
            if self._intent_router is not None and hasattr(self._registry, "to_anthropic_schemas"):
                categories = self._intent_router.route(user_message)
                if categories is not None:
                    tools = self._registry.to_anthropic_schemas(categories=categories)
                else:
                    tools = self._registry.to_anthropic_schemas()
            else:
                tools = self._registry.to_anthropic_schemas()

            # Backend handles streaming, format conversion, and retry
            response: LLMResponse = self._backend.call(
                messages=messages,
                tools=tools,
                system=self._system_prompt,
                max_tokens=self._max_tokens,
                on_text=on_text,
            )

            final_text = response.text
            stop_reason = response.stop_reason
            total_usage = total_usage.add(response.usage)

            # Append assistant message to session
            tool_use_dicts: list[dict[str, Any]] | None = None
            if response.tool_calls:
                tool_use_dicts = [
                    {"id": tc.id, "name": tc.name, "input": tc.input, "type": "tool_use"}
                    for tc in response.tool_calls
                ]
            session.append_assistant(response.text, tool_use_dicts)

            if not response.tool_calls:
                break  # end_turn — no tools called, conversation complete

            # Execute tools and collect results
            raw_results = self._dispatch_tools(
                response.tool_calls, tool_context, on_tool_start, on_tool_end, ask_permission
            )

            result_dicts: list[dict[str, Any]] = []
            for result_dict, tool_call in raw_results:
                result_dicts.append(result_dict)
                all_tool_calls.append(tool_call)

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

    def _partition_tools(self, tool_calls: list[LLMToolCall]) -> list[ToolBatch]:
        """Partition tool calls into concurrent (read-only) and sequential batches."""
        batches: list[ToolBatch] = []
        for tc in tool_calls:
            tool = self._registry.get(tc.name)
            is_safe = bool(
                tool is not None
                and hasattr(tool, "is_concurrency_safe")
                and tool.is_concurrency_safe(tc.input)
            )
            if is_safe and batches and batches[-1].concurrent:
                batches[-1].tool_calls.append(tc)
            else:
                batches.append(ToolBatch(concurrent=is_safe, tool_calls=[tc]))
        return batches

    def _dispatch_tools(
        self,
        tool_calls: list[LLMToolCall],
        tool_context: ToolContext,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None,
        on_tool_end: Callable[[str, ToolResult], None] | None,
        ask_permission: Callable[[str, dict[str, Any]], str] | None,
    ) -> list[tuple[dict[str, Any], ToolCall]]:
        """Dispatch all tool calls, respecting concurrency partitioning."""
        results: list[tuple[dict[str, Any], ToolCall]] = []
        batches = self._partition_tools(tool_calls)

        for batch in batches:
            if batch.concurrent and len(batch.tool_calls) > 1:
                batch_results = self._run_concurrent(
                    batch.tool_calls, tool_context, on_tool_start, on_tool_end, ask_permission
                )
            else:
                batch_results = self._run_sequential(
                    batch.tool_calls, tool_context, on_tool_start, on_tool_end, ask_permission
                )
            results.extend(batch_results)

        return results

    def _execute_single_tool(
        self,
        tc: LLMToolCall,
        tool_context: ToolContext,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None,
        on_tool_end: Callable[[str, ToolResult], None] | None,
        ask_permission: Callable[[str, dict[str, Any]], str] | None,
    ) -> tuple[dict[str, Any], ToolCall]:
        """Execute one tool call with full permission checking."""
        tool_name = tc.name
        params = tc.input
        tool = self._registry.get(tool_name)

        if tool is None:
            result = ToolResult(content=f"Unknown tool: {tool_name}", is_error=True)
            logger.warning("Tool %r not found in registry", tool_name)
            return (
                {"tool_use_id": tc.id, "content": result.content, "is_error": True},
                ToolCall(tool_name=tool_name, params=params, result=result, duration_sec=0.0, permission_action="denied"),
            )

        # Permission check
        perm: PermissionResult = self._permissions.check(tool, params, tool_context)

        if perm.behavior == "deny":
            reason = perm.reason or f"Permission denied for {tool_name}"
            result = ToolResult(content=f"Permission denied: {reason}", is_error=True)
            logger.info("Permission denied for tool %r: %s", tool_name, reason)
            return (
                {"tool_use_id": tc.id, "content": result.content, "is_error": True},
                ToolCall(tool_name=tool_name, params=params, result=result, duration_sec=0.0, permission_action="denied"),
            )

        if perm.behavior == "ask":
            response = ask_permission(tool_name, params) if ask_permission else "n"
            if response == "n":
                denial = f"Permission denied by user for {tool_name}"
                result = ToolResult(content=denial, is_error=True)
                logger.info("User denied permission for tool %r", tool_name)
                return (
                    {"tool_use_id": tc.id, "content": result.content, "is_error": True},
                    ToolCall(tool_name=tool_name, params=params, result=result, duration_sec=0.0, permission_action="asked_denied"),
                )
            if response == "a":
                self._permissions.add_always_allow(tool_name)
            perm_action = "asked_allowed"
        else:
            perm_action = "allowed"

        # Execute the tool
        if on_tool_start is not None:
            on_tool_start(tool_name, params)

        # Pre-hook
        if self._hooks is not None:
            from vector_os_nano.vcli.hooks import ToolHookContext
            self._hooks.fire_pre(ToolHookContext(tool_name=tool_name, params=params))

        start = time.monotonic()
        try:
            result = tool.execute(params, tool_context)
        except Exception as exc:
            result = ToolResult(content=f"Tool error: {exc}", is_error=True)
            logger.error("Tool %r raised %r", tool_name, exc, exc_info=True)
        duration = time.monotonic() - start

        # Post-hook
        if self._hooks is not None:
            from vector_os_nano.vcli.hooks import ToolHookContext
            self._hooks.fire_post(ToolHookContext(
                tool_name=tool_name, params=params, result=result, duration=duration,
            ))

        if on_tool_end is not None:
            on_tool_end(tool_name, result)

        return (
            {"tool_use_id": tc.id, "content": result.content, "is_error": result.is_error},
            ToolCall(tool_name=tool_name, params=params, result=result, duration_sec=duration, permission_action=perm_action),
        )

    def _run_sequential(
        self,
        tool_calls: list[LLMToolCall],
        tool_context: ToolContext,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None,
        on_tool_end: Callable[[str, ToolResult], None] | None,
        ask_permission: Callable[[str, dict[str, Any]], str] | None,
    ) -> list[tuple[dict[str, Any], ToolCall]]:
        """Execute tool calls one-by-one in order."""
        return [
            self._execute_single_tool(tc, tool_context, on_tool_start, on_tool_end, ask_permission)
            for tc in tool_calls
        ]

    def _run_concurrent(
        self,
        tool_calls: list[LLMToolCall],
        tool_context: ToolContext,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None,
        on_tool_end: Callable[[str, ToolResult], None] | None,
        ask_permission: Callable[[str, dict[str, Any]], str] | None,
    ) -> list[tuple[dict[str, Any], ToolCall]]:
        """Execute read-only tool calls concurrently using a thread pool."""
        max_workers = min(len(tool_calls), 10)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    self._execute_single_tool, tc, tool_context, on_tool_start, on_tool_end, ask_permission
                )
                for tc in tool_calls
            ]
            return [f.result() for f in futures]
