# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""S5.1 parity tests for the shared tool-dispatch seam (``vcli.tool_execution``).

Stage 5 extracts ONE permission-gate + execute core that BOTH planning paths use:

- the ReAct tool loop (``VectorEngine._execute_single_tool``), and
- the VGG ``tool`` sub-goal path (``ToolDispatcher.dispatch``).

These tests drive a FIXED corpus of tools (an allowed read-only tool, a
side-effecting ask-tool that is allowed / denied, a hard-deny via
``check_permissions``, and a tool that raises) through BOTH callers on a mock
backend (no live LLM) and assert they reach the SAME allow/deny outcome and the
SAME ``ToolResult.content`` / ``is_error`` for the underlying tool. They are the
safety net that the additive extraction did not let any tool gain or lose a gate.

The two callers legitimately differ in their OUTPUT SHAPE (ReAct returns a rich
``ToolCall`` + permission-action label; VGG returns ``(success, error)``) and in a
few caller-specific concerns (VGG's allowlist, ReAct's hooks/streaming). Parity is
asserted on the shared decision + the tool result, not on those differences.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from vector_os_nano.vcli.backends.types import LLMToolCall
from vector_os_nano.vcli.cognitive.tool_dispatcher import ToolDispatcher
from vector_os_nano.vcli.engine import VectorEngine
from vector_os_nano.vcli.hooks import ToolHookRegistry
from vector_os_nano.vcli.permissions import PermissionContext
from vector_os_nano.vcli.tool_execution import (
    DECISION_ALLOW,
    DECISION_ASK_ALLOW,
    DECISION_ASK_DENY,
    DECISION_DENY,
    PermissionDecision,
    resolve_permission,
)
from vector_os_nano.vcli.tools.base import (
    PermissionResult,
    ToolContext,
    ToolRegistry,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Fixed tool corpus — plain classes (not MagicMock) so the Tool protocol and
# every distinguishing behaviour is exercised identically through both callers.
# ---------------------------------------------------------------------------


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


_ALLOWLIST = frozenset({"allow_ro", "ask_tool", "hard_deny", "raiser"})


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    for cls in (AllowReadOnlyTool, AskTool, HardDenyTool, RaisingTool):
        reg.register(cls())
    return reg


# ---------------------------------------------------------------------------
# Caller adapters — run ONE tool through each caller, returning a normalized
# (success, error_or_content) pair plus, for ReAct, the permission_action.
# ---------------------------------------------------------------------------


def _run_react(
    tool_name: str,
    args: dict[str, Any],
    *,
    permissions: PermissionContext,
    ask: Any = None,
    hooks: ToolHookRegistry | None = None,
) -> tuple[bool, str, str, ToolResult]:
    """Drive the ReAct caller's per-tool unit (``_execute_single_tool``)."""
    engine = VectorEngine(
        backend=object(),  # never called — we invoke the dispatch unit directly
        registry=_registry(),
        permissions=permissions,
        hooks=hooks,
    )
    ctx = ToolContext(
        agent=None,
        cwd=Path.cwd(),
        session=None,
        permissions=permissions,
        abort=threading.Event(),
        app_state=None,
    )
    tc = LLMToolCall(id="tc-1", name=tool_name, input=args)
    result_dict, tool_call = engine._execute_single_tool(tc, ctx, None, None, ask)
    res = tool_call.result
    success = not res.is_error
    return success, res.content, tool_call.permission_action, res


def _run_vgg(
    tool_name: str,
    args: dict[str, Any],
    *,
    permissions: PermissionContext,
    ask: Any = None,
) -> tuple[bool, str]:
    """Drive the VGG caller (``ToolDispatcher.dispatch``)."""
    disp = ToolDispatcher(
        _registry(), permissions, allowlist=_ALLOWLIST, ask_permission=ask,
    )
    return disp.dispatch(tool_name, args)


# Resolver constants
_ALLOW = lambda _n, _p: "y"  # noqa: E731
_ALWAYS = lambda _n, _p: "a"  # noqa: E731
_DENY = lambda _n, _p: "n"  # noqa: E731


# ---------------------------------------------------------------------------
# Parity: same allow/deny OUTCOME for both callers across the corpus
# ---------------------------------------------------------------------------


def test_parity_allowed_read_only_tool() -> None:
    """A read-only tool runs (no prompt) on BOTH callers."""
    rt_ok, _, rt_action, rt_res = _run_react(
        "allow_ro", {}, permissions=PermissionContext()
    )
    vgg_ok, vgg_err = _run_vgg("allow_ro", {}, permissions=PermissionContext())

    assert rt_ok is True and vgg_ok is True
    assert rt_action == "allowed"
    assert vgg_err == ""
    assert rt_res.content == "ro-ok"


def test_parity_ask_tool_allowed() -> None:
    """An ``ask`` tool with a 'y' resolver runs on BOTH callers."""
    rt_ok, _, rt_action, rt_res = _run_react(
        "ask_tool", {}, permissions=PermissionContext(), ask=_ALLOW
    )
    vgg_ok, vgg_err = _run_vgg(
        "ask_tool", {}, permissions=PermissionContext(), ask=_ALLOW
    )

    assert rt_ok is True and vgg_ok is True
    assert rt_action == "asked_allowed"
    assert rt_res.content == "ask-ran"
    assert vgg_err == ""


def test_parity_ask_tool_denied() -> None:
    """An ``ask`` tool with an 'n' resolver is blocked on BOTH callers."""
    rt_ok, rt_content, rt_action, _ = _run_react(
        "ask_tool", {}, permissions=PermissionContext(), ask=_DENY
    )
    vgg_ok, vgg_err = _run_vgg(
        "ask_tool", {}, permissions=PermissionContext(), ask=_DENY
    )

    assert rt_ok is False and vgg_ok is False
    assert rt_action == "asked_denied"
    assert "denied" in rt_content.lower()
    assert "denied" in vgg_err.lower()


def test_parity_ask_tool_none_resolver_fails_closed() -> None:
    """No resolver -> deny-by-default on BOTH callers (the gate never auto-allows)."""
    rt_ok, _, rt_action, _ = _run_react(
        "ask_tool", {}, permissions=PermissionContext(), ask=None
    )
    vgg_ok, _ = _run_vgg("ask_tool", {}, permissions=PermissionContext(), ask=None)

    assert rt_ok is False and vgg_ok is False
    assert rt_action == "asked_denied"


def test_parity_ask_tool_unexpected_resolver_fails_closed() -> None:
    """A non-y/a resolver answer fails closed identically on BOTH callers."""
    weird = lambda _n, _p: "maybe"  # noqa: E731
    rt_ok, _, _, _ = _run_react(
        "ask_tool", {}, permissions=PermissionContext(), ask=weird
    )
    vgg_ok, _ = _run_vgg("ask_tool", {}, permissions=PermissionContext(), ask=weird)

    assert rt_ok is False and vgg_ok is False


def test_parity_hard_deny_tool() -> None:
    """An intrinsic ``check_permissions`` deny blocks BOTH callers (never runs)."""
    rt_ok, rt_content, rt_action, _ = _run_react(
        "hard_deny", {}, permissions=PermissionContext(), ask=_ALLOW
    )
    vgg_ok, vgg_err = _run_vgg(
        "hard_deny", {}, permissions=PermissionContext(), ask=_ALLOW
    )

    assert rt_ok is False and vgg_ok is False
    assert rt_action == "denied"
    assert "blocked by safety rail" in rt_content
    assert "blocked by safety rail" in vgg_err


def test_parity_raising_tool_becomes_error_result() -> None:
    """A tool that raises yields an error outcome (not an exception) on BOTH callers."""
    rt_ok, rt_content, _, rt_res = _run_react(
        "raiser", {}, permissions=PermissionContext()
    )
    vgg_ok, vgg_err = _run_vgg("raiser", {}, permissions=PermissionContext())

    assert rt_ok is False and vgg_ok is False
    assert rt_res.is_error is True
    assert "boom" in rt_content
    assert "boom" in vgg_err


def test_parity_no_permission_allows_side_effecting_tool() -> None:
    """``no_permission`` lets a normally-ask tool run on BOTH callers (no prompt)."""
    rt_ok, _, rt_action, _ = _run_react(
        "ask_tool", {}, permissions=PermissionContext(no_permission=True)
    )
    vgg_ok, vgg_err = _run_vgg(
        "ask_tool", {}, permissions=PermissionContext(no_permission=True)
    )

    assert rt_ok is True and vgg_ok is True
    assert rt_action == "allowed"
    assert vgg_err == ""


# ---------------------------------------------------------------------------
# The shared seam itself returns the SAME PermissionDecision for both callers
# (the decision is what S5.1 deduplicated; assert it directly).
# ---------------------------------------------------------------------------


def _decide(tool_name: str, *, ask: Any, swallow: bool) -> PermissionDecision:
    tool = _registry().get(tool_name)
    perms = PermissionContext()
    ctx = ToolContext(
        agent=None, cwd=Path.cwd(), session=None, permissions=perms,
        abort=threading.Event(), app_state=None,
    )
    return resolve_permission(
        tool, {}, ctx, perms, ask,
        swallow_check_errors=swallow,
        swallow_always_allow_errors=swallow,
    )


def test_seam_decision_matrix() -> None:
    """resolve_permission yields the documented decision kind for each branch."""
    assert _decide("allow_ro", ask=None, swallow=False).kind == DECISION_ALLOW
    assert _decide("ask_tool", ask=_ALLOW, swallow=False).kind == DECISION_ASK_ALLOW
    assert _decide("ask_tool", ask=_DENY, swallow=False).kind == DECISION_ASK_DENY
    assert _decide("hard_deny", ask=None, swallow=False).kind == DECISION_DENY


def test_seam_decision_identical_for_both_swallow_modes() -> None:
    """The DECISION is independent of the per-caller failure-isolation flags."""
    for tool_name, ask in (
        ("allow_ro", None),
        ("ask_tool", _ALLOW),
        ("ask_tool", _DENY),
        ("hard_deny", None),
    ):
        react_mode = _decide(tool_name, ask=ask, swallow=False)
        vgg_mode = _decide(tool_name, ask=ask, swallow=True)
        assert react_mode.kind == vgg_mode.kind
        assert react_mode.allowed == vgg_mode.allowed


def test_seam_always_allow_persists_for_both() -> None:
    """An 'a' resolver records always-allow; the next call needs no prompt."""
    tool = _registry().get("ask_tool")
    perms = PermissionContext()
    ctx = ToolContext(
        agent=None, cwd=Path.cwd(), session=None, permissions=perms,
        abort=threading.Event(), app_state=None,
    )
    first = resolve_permission(tool, {}, ctx, perms, _ALWAYS)
    assert first.kind == DECISION_ASK_ALLOW
    assert "ask_tool" in perms.session_allow
    # Second call: session-allow short-circuits the ask (layer 5).
    second = resolve_permission(tool, {}, ctx, perms, _DENY)
    assert second.kind == DECISION_ALLOW


# ---------------------------------------------------------------------------
# Failure-isolation flags: the ReAct path propagates a buggy permission object;
# the VGG path fails closed. This is a LEGITIMATE difference the seam preserves.
# ---------------------------------------------------------------------------


class _BrokenPermissions(PermissionContext):
    def check(self, tool, params, tool_context=None):  # type: ignore[override]
        raise ValueError("permission machinery exploded")


def test_check_error_propagates_for_react_but_fails_closed_for_vgg() -> None:
    """A raised ``permissions.check`` surfaces on ReAct, fails closed on VGG."""
    tool = _registry().get("allow_ro")
    perms = _BrokenPermissions()
    ctx = ToolContext(
        agent=None, cwd=Path.cwd(), session=None, permissions=perms,
        abort=threading.Event(), app_state=None,
    )

    # ReAct path: swallow_check_errors=False -> the bug propagates.
    raised = False
    try:
        resolve_permission(tool, {}, ctx, perms, None, swallow_check_errors=False)
    except ValueError:
        raised = True
    assert raised is True

    # VGG path: swallow_check_errors=True -> a closed deny, no exception.
    decision = resolve_permission(
        tool, {}, ctx, perms, None, swallow_check_errors=True
    )
    assert decision.allowed is False
    assert "exploded" in decision.reason


# ---------------------------------------------------------------------------
# Hooks fire on the ReAct path (its concern); the seam stays hook-free so the
# VGG path is unaffected. Assert the ReAct caller still fires pre/post hooks.
# ---------------------------------------------------------------------------


def test_react_hooks_fire_around_execution() -> None:
    """Pre/post hooks fire on a successful ReAct dispatch through the seam."""
    fired: list[str] = []
    hooks = ToolHookRegistry()
    hooks.add_pre_hook(lambda ctx: fired.append(f"pre:{ctx.tool_name}"))
    hooks.add_post_hook(lambda ctx: fired.append(f"post:{ctx.tool_name}"))

    rt_ok, _, _, _ = _run_react(
        "allow_ro", {}, permissions=PermissionContext(), hooks=hooks
    )
    assert rt_ok is True
    assert fired == ["pre:allow_ro", "post:allow_ro"]


def test_react_hooks_do_not_fire_on_deny() -> None:
    """A denied tool never runs, so no pre/post hooks fire (matches pre-seam)."""
    fired: list[str] = []
    hooks = ToolHookRegistry()
    hooks.add_pre_hook(lambda ctx: fired.append("pre"))
    hooks.add_post_hook(lambda ctx: fired.append("post"))

    rt_ok, _, _, _ = _run_react(
        "hard_deny", {}, permissions=PermissionContext(), hooks=hooks
    )
    assert rt_ok is False
    assert fired == []
