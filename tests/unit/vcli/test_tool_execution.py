# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The permission gate + execution of one tool (``vcli.tool_execution``).

Moved from tests/vcli/test_dispatch_seam_parity.py when its two callers (the ReAct
loop and the VGG ToolDispatcher) were deleted: the fixed tool corpus and the gate
outcomes it pinned now pin the one remaining caller's seam (the native loop).
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from vector_os_nano.vcli.permissions import PermissionContext
from vector_os_nano.vcli.tool_execution import (
    DECISION_ALLOW,
    DECISION_ASK_ALLOW,
    DECISION_ASK_DENY,
    DECISION_CHECK_ERROR,
    DECISION_DENY,
    execute_resolved_tool,
    resolve_permission,
)
from vector_os_nano.vcli.tools.base import (
    PermissionResult,
    ToolContext,
    ToolRegistry,
    ToolResult,
)


class _BaseTool:
    name = "base"
    description = "base"
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    def is_read_only(self, params: dict[str, Any]) -> bool:
        return False

    def is_concurrency_safe(self, params: dict[str, Any]) -> bool:
        return False


class AllowReadOnlyTool(_BaseTool):
    """Read-only with no check of its own: auto-allowed by layer 6 of the gate,
    never prompts. (Since R3 a tool's own explicit ``ask`` is NOT overridden by
    the read-only flag — tests/unit/vcli/test_permissions.py pins that.)"""

    name = "allow_ro"

    def is_read_only(self, params: dict[str, Any]) -> bool:
        return True

    def execute(self, params, context) -> ToolResult:
        return ToolResult(content="ro-ok", is_error=False)


class AskTool(_BaseTool):
    """Side-effecting: requires an interactive ``ask`` decision to run."""

    name = "ask_tool"

    def check_permissions(self, params, context) -> PermissionResult:
        return PermissionResult(behavior="ask", reason="Allow ask_tool?")

    def execute(self, params, context) -> ToolResult:
        return ToolResult(content="ask-ran", is_error=False)


class HardDenyTool(_BaseTool):
    """Intrinsic ``deny`` from ``check_permissions`` (e.g. the bash deny-list)."""

    name = "hard_deny"

    def check_permissions(self, params, context) -> PermissionResult:
        return PermissionResult(behavior="deny", reason="blocked by safety rail")

    def execute(self, params, context) -> ToolResult:  # pragma: no cover
        return ToolResult(content="should-never-run", is_error=False)


class RaisingTool(_BaseTool):
    """Allowed, but ``execute`` raises — the error tail must produce an error result."""

    name = "raiser"

    def is_read_only(self, params: dict[str, Any]) -> bool:
        return True  # auto-allow so we reach execute

    def check_permissions(self, params, context) -> PermissionResult:
        return PermissionResult(behavior="allow")

    def execute(self, params, context) -> ToolResult:
        raise RuntimeError("boom")


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    for cls in (AllowReadOnlyTool, AskTool, HardDenyTool, RaisingTool):
        reg.register(cls())
    return reg


class _BrokenPermissions(PermissionContext):
    def check(self, tool, params, tool_context=None):  # type: ignore[override]
        raise ValueError("permission machinery exploded")


_ALLOW = lambda _n, _p: "y"  # noqa: E731
_ALWAYS = lambda _n, _p: "a"  # noqa: E731
_DENY = lambda _n, _p: "n"  # noqa: E731


def _decide(tool_name: str, ask: Any = None, perms: PermissionContext | None = None):
    perms = perms or PermissionContext()
    ctx = ToolContext(agent=None, cwd=Path.cwd(), session=None, permissions=perms,
                      abort=threading.Event(), app_state=None)
    return resolve_permission(_registry().get(tool_name), {}, ctx, perms, ask)


@pytest.mark.parametrize("tool_name, ask, kind", [
    ("allow_ro", None, DECISION_ALLOW),
    ("ask_tool", _ALLOW, DECISION_ASK_ALLOW),
    ("ask_tool", _DENY, DECISION_ASK_DENY),
    ("ask_tool", None, DECISION_ASK_DENY),                   # no resolver: deny by default
    ("ask_tool", lambda _n, _p: "maybe", DECISION_ASK_DENY),  # a non-y/a answer fails closed
    ("ask_tool", lambda _n, _p: None, DECISION_ASK_DENY),
    ("hard_deny", _ALLOW, DECISION_DENY),                    # an intrinsic deny is never asked
])
def test_decision_matrix(tool_name: str, ask: Any, kind: str) -> None:
    decision = _decide(tool_name, ask)
    assert decision.kind == kind
    assert decision.allowed is (kind in (DECISION_ALLOW, DECISION_ASK_ALLOW))


def test_hard_deny_carries_the_tools_reason() -> None:
    assert "blocked by safety rail" in _decide("hard_deny", _ALLOW).reason


def test_no_permission_allows_a_side_effecting_tool_without_asking() -> None:
    assert _decide("ask_tool", None, PermissionContext(no_permission=True)).kind == DECISION_ALLOW


def test_always_allow_persists() -> None:
    """An 'a' resolver records always-allow; the next call needs no prompt."""
    perms = PermissionContext()
    assert _decide("ask_tool", _ALWAYS, perms).kind == DECISION_ASK_ALLOW
    assert "ask_tool" in perms.session_allow
    assert _decide("ask_tool", _DENY, perms).kind == DECISION_ALLOW


def test_a_raising_permission_check_fails_closed() -> None:
    decision = _decide("allow_ro", None, _BrokenPermissions())
    assert decision.kind == DECISION_CHECK_ERROR and decision.allowed is False
    assert "exploded" in decision.reason


def test_a_raising_tool_becomes_an_error_result() -> None:
    errors: list[str] = []
    ctx = ToolContext(agent=None, cwd=Path.cwd(), session=None, permissions=None,
                      abort=threading.Event(), app_state=None)
    res = execute_resolved_tool(_registry().get("raiser"), {}, ctx, error_prefix="Skill 'raiser' raised",
                                on_error=lambda name, exc: errors.append(f"{name}:{exc}"))
    assert res.is_error and res.content == "Skill 'raiser' raised: boom"
    assert errors == ["raiser:boom"]
