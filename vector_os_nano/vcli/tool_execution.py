# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""The permission gate and the execution of one resolved tool (the native loop's).

``resolve_permission`` runs the 7-layer ``PermissionContext.check`` plus the
deny-by-default ``ask`` resolution and returns a ``PermissionDecision``;
``execute_resolved_tool`` runs the tool once and turns a raised error into an
error result. Both fail closed: the loop runs autonomously, so a buggy permission
object or resolver never allows a tool.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Callable

from vector_os_nano.vcli.tools.base import PermissionResult, ToolResult

logger = logging.getLogger(__name__)

# Decision-kind labels returned by ``resolve_permission``: WHAT the gate decided; the
# caller words them.
DECISION_ALLOW = "allow"            # tool was already allowed (no prompt)
DECISION_ASK_ALLOW = "ask_allow"    # tool was ``ask`` and the user allowed it
DECISION_DENY = "deny"              # hard deny from ``permissions.check``
DECISION_ASK_DENY = "ask_deny"      # tool was ``ask`` and the resolver denied it
DECISION_CHECK_ERROR = "check_error"  # ``permissions.check`` raised


@dataclass(frozen=True)
class PermissionDecision:
    """Outcome of the shared permission gate for one tool.

    ``kind`` is one of the ``DECISION_*`` constants. ``allowed`` is the bool the
    caller branches on. ``reason`` carries the underlying detail (the tool/user deny
    reason, the resolver response repr, or the check-error text).
    """

    kind: str
    allowed: bool
    reason: str = ""


def resolve_permission(
    tool: Any,
    params: dict[str, Any],
    ctx: Any,
    permissions: Any,
    ask_permission: Callable[[str, dict[str, Any]], str] | None,
) -> PermissionDecision:
    """Run the 7-layer gate + deny-by-default ``ask`` resolution for one tool.

    The ``ask`` resolution is deny-by-default: only an explicit ``"y"``/``"a"``
    allows; ``"a"`` also records an always-allow (a failure to record it does not
    undo the yes). Any other resolver output (``None``, ``""``, unexpected) fails
    closed, and so does a raised ``permissions.check`` (``DECISION_CHECK_ERROR``).
    """
    tool_name: str = getattr(tool, "name", "") or getattr(tool, "__tool_name__", "")

    try:
        perm: PermissionResult = permissions.check(tool, params, ctx)
    except Exception as exc:  # noqa: BLE001
        return PermissionDecision(
            kind=DECISION_CHECK_ERROR, allowed=False, reason=str(exc)
        )

    if perm.behavior == "deny":
        return PermissionDecision(kind=DECISION_DENY, allowed=False, reason=perm.reason)

    if perm.behavior == "ask":
        response = ask_permission(tool_name, params) if ask_permission else "n"
        if response == "a":
            try:
                permissions.add_always_allow(tool_name)
            except Exception:  # noqa: BLE001
                logger.debug("permissions: could not record always-allow for %s", tool_name)
        elif response != "y":
            return PermissionDecision(
                kind=DECISION_ASK_DENY, allowed=False, reason=repr(response)
            )
        return PermissionDecision(kind=DECISION_ASK_ALLOW, allowed=True)

    return PermissionDecision(kind=DECISION_ALLOW, allowed=True)


def execute_resolved_tool(
    tool: Any,
    params: dict[str, Any],
    ctx: Any,
    *,
    error_prefix: str = "Tool error",
    on_error: Callable[[str, Exception], None] | None = None,
) -> ToolResult:
    """Invoke ``tool.execute`` once, turning any raised error into an error result.

    ``error_prefix`` heads the error message of a raised tool; ``on_error`` is an
    optional logging callback ``(tool_name, exc)``. Always returns a ``ToolResult``
    (never raises): ``is_error`` is the caller's one success signal.
    """
    try:
        result = tool.execute(params, ctx)
    except Exception as exc:  # noqa: BLE001
        if on_error is not None:
            on_error(getattr(tool, "name", "") or "", exc)
        result = ToolResult(content=f"{error_prefix}: {exc}", is_error=True)
    session = (getattr(ctx, "app_state", None) or {}).get("sim_session")
    if callable(getattr(session, "metadata", None)):
        try:
            info = session.metadata()
        except Exception as exc:  # provenance failure must not erase the tool's actual outcome
            logger.warning("Could not read session provenance: %s", exc)
            info = {}
        result = replace(result, metadata={**result.metadata, "session_info": info})
    return result
