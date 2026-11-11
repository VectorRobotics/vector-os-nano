# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Multi-layer permission system for Vector CLI's agentic harness.

Public exports:
    PermissionContext — stateful permission checker for a single agent session

Check order (mirrors Claude Code's hasPermissionsToUseTool):
    1. Tool check_permissions() returning "deny" → deny (unconditional hard stop;
       bypasses all modes, including --no-permission — a safety rail such as the
       bash deny-list or a dangerous-path write is never disabled by a flag)
    2. no_permission flag  → allow everything not intrinsically denied above
    3. User deny_tools     → deny (a softer per-session preference that
       --no-permission may override)
    4. Tool check_permissions() returning "allow" → allow
    5. Session always-allow → allow
    6. Read-only auto-allow — only for a tool WITHOUT its own check_permissions();
       a tool's explicit "ask" (e.g. a read outside the project root) is never
       overridden by its read-only flag
    7. Tool check_permissions() returning "ask" → ask
    8. Default             → ask
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vector_os_nano.vcli.tools.base import PermissionResult


# ---------------------------------------------------------------------------
# PermissionContext
# ---------------------------------------------------------------------------


@dataclass
class PermissionContext:
    """Stateful per-session permission checker.

    Attributes:
        deny_tools:    Tool names that are always denied.
        session_allow: Tool names the user approved with "always" this session.
        no_permission: When True, every tool is allowed (--no-permission flag).
    """

    deny_tools: set[str] = field(default_factory=set)
    session_allow: set[str] = field(default_factory=set)
    no_permission: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        tool: Any,
        params: dict[str, Any],
        tool_context: Any = None,
    ) -> PermissionResult:
        """Return the effective PermissionResult for *tool* with *params*.

        Args:
            tool:         Any object with a ``name`` attribute. Optionally
                          implements ``check_permissions()``, ``is_read_only()``.
            params:       The parameters that will be passed to ``tool.execute()``.
            tool_context: Optional ``ToolContext`` forwarded to tool methods.

        Returns:
            A ``PermissionResult`` with behavior ``"allow"``, ``"deny"``, or
            ``"ask"``.
        """
        tool_name: str = getattr(tool, "name", "") or getattr(tool, "__tool_name__", "")

        # 1. Tool intrinsic safety check FIRST — an intrinsic "deny" (e.g. the
        #    bash deny-list, a dangerous-path write) is an unconditional hard stop
        #    evaluated even before no_permission, so a safety rail can never be
        #    disabled by --no-permission. (User-configured deny_tools is a softer
        #    preference, handled at step 3, that --no-permission may override.)
        tool_perm: PermissionResult | None = None
        if hasattr(tool, "check_permissions"):
            tool_perm = tool.check_permissions(params, tool_context)
            if tool_perm.behavior == "deny":
                return tool_perm

        # 2. no-permission mode — allow everything not intrinsically denied above.
        if self.no_permission:
            return PermissionResult("allow")

        # 3. User deny rules
        if tool_name in self.deny_tools:
            return PermissionResult("deny", f"Tool '{tool_name}' is denied")

        # 4. Tool-specific explicit allow
        if tool_perm is not None and tool_perm.behavior == "allow":
            return tool_perm

        # 5. Session always-allow
        if tool_name in self.session_allow:
            return PermissionResult("allow")

        # 6. Read-only auto-allow — never over the tool's own explicit "ask".
        if tool_perm is None and hasattr(tool, "is_read_only") and tool.is_read_only(params):
            return PermissionResult("allow")

        # 7. Propagate tool-specific "ask" (reuse result from step 2)
        if tool_perm is not None and tool_perm.behavior == "ask":
            return tool_perm

        # 8. Default: ask
        return PermissionResult("ask", f"Allow {tool_name}?")

    def add_always_allow(self, tool_name: str) -> None:
        """Add *tool_name* to the session always-allow set.

        Called when the user responds "a" (always) to a permission prompt.
        """
        self.session_allow.add(tool_name)

    def add_deny(self, tool_name: str) -> None:
        """Add *tool_name* to the permanent deny set for this session."""
        self.deny_tools.add(tool_name)
